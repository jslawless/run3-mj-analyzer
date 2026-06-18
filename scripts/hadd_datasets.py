#!/usr/bin/env python3
"""hadd_datasets.py - Merge each dataset's ROOT files into ONE file on EOS.

Given a JSON of files (the same JSONs the rest of the pipeline already emits -
see "Input JSON" below), this hadds every file of a slice/dataset into a single
ROOT file and delivers it to one EOS *input* directory, named <dataset>.root.

Why: the analyzer makes one ``from_root`` call per slice, so a slice spread over
many per-job files means many xrootd round-trips. Pre-merging to one file per
slice makes the analyzer's load fast and its fileset trivial (one file each).

The merge preserves everything the analyzer needs:
  * hadd SUMS the ``cutflow`` histogram across ALL files of a slice, so the
    merged file's ``cutflow[0]`` equals n_original (the xsec-normalisation
    denominator) - exactly what fileset.py computes by summing cutflow[0] over
    every file, including the events-less ones.
  * Low-HT QCD slices where nothing passed the slimmer's HT cut produce
    cutflow-only files (no ``events`` tree; see the slimmer's empty-extend
    guard). hadd unions keys, so those still contribute their cutflow while the
    ``events`` trees from the non-empty files merge together. (hadd prints a
    warning about the differing structure; that is expected and harmless.)

EOS compatibility (per the LPC EOS rules) - this script is careful to:
  * NEVER hadd directly to EOS. The LPC docs forbid keeping a file open while
    writing to EOS ("Do not ... hadd directly to EOS"). We hadd to a LOCAL temp
    file (--tmpdir, default CWD - keep it on NFS/scratch, never /eos/uscms),
    then deliver it with ``xrdcp -f``, the only blessed EOS write path.
  * READ the source files through the ``root://cmseos.fnal.gov//store/...``
    redirector (the docs explicitly bless ``hadd target.root `xrdfsls -u ...` ``)
    - never via the /eos/uscms FUSE mount, which is gone on worker nodes and
    degrades the interactive nodes.
  * Create the EOS input dir with ``xrdfs <host> mkdir -p`` (no FUSE, no ls/cp).
  * Enumerate every source explicitly from the JSON - EOS rejects wildcards.

Needs a ROOT environment (``hadd``, e.g. after ``cmsenv``) and a grid proxy
(``voms-proxy-init -voms cms``) for the xrootd reads/writes. Run it on an
interactive LPC node, not from a condor job.

Input JSON - any of the three shapes the pipeline already writes is accepted:
  * make_dataset_json.py : {"metadata": {...}, "datasets": {name: [paths]}}
  * a coffea fileset      : {name: {"files": {path: tree, ...}}}
  * a bare mapping        : {name: [paths]}
Paths may be full ``root://...`` URLs or bare ``/store/...`` (the redirector is
added automatically).

Usage:
    python scripts/hadd_datasets.py dataset_evaluated.json -o /store/user/you/merged
    python scripts/hadd_datasets.py fileset.json -o /store/user/you/merged \\
        --skip-existing --tmpdir $PWD --jobs 4
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

# Personal/group EOS (/store/user, /store/group) is streamed through the EOS
# redirector; XCache (root://xcache/) only serves the global CMS namespace.
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov/"


def host_of(redirector):
    """'root://cmseos.fnal.gov/' -> 'root://cmseos.fnal.gov' (for xrdfs/xrdcp)."""
    return redirector.rstrip("/")


def strip_redirector(path):
    """Reduce a full URL to its bare ``/store/...`` LFN; pass /store paths through.

    ``root://host//store/x`` -> ``/store/x``. Used to build the xrdcp/xrdfs
    destination so the redirector is applied exactly once (never doubled).
    """
    return re.sub(r"^root://[^/]+/+", "/", path)


def to_xrd(path, redirector):
    """Prefix a bare ``/store`` path with the redirector for WAN streaming.

    XRootD needs a DOUBLE slash between host and an absolute path
    (``root://host//store/...``); a single slash makes ``store/...`` relative,
    which the server rejects. Full ``root://`` URLs are returned unchanged.
    """
    if path.startswith("root://"):
        return path
    return f"{host_of(redirector)}//{path.lstrip('/')}"


def extract_datasets(blob):
    """Normalise any accepted JSON shape to ``{dataset: [source_path, ...]}``."""
    if not isinstance(blob, dict):
        sys.exit("Top-level JSON must be an object (dataset -> files).")
    datasets = blob.get("datasets")
    if datasets is None:  # coffea fileset or bare map; ignore a stray metadata key
        datasets = {k: v for k, v in blob.items() if k != "metadata"}

    out = {}
    for name, val in datasets.items():
        if isinstance(val, list):
            paths = val
        elif isinstance(val, dict) and "files" in val:
            files = val["files"]
            paths = list(files) if isinstance(files, (dict, list)) else None
        else:
            paths = None
        if paths is None:
            sys.exit(
                f"Dataset '{name}': expected a list of paths or a coffea "
                f"{{'files': {{path: tree}}}} entry, got {type(val).__name__}."
            )
        if not paths:
            print(f"  skip (no files listed): {name}", file=sys.stderr)
            continue
        out[name] = paths
    return out


def have_proxy():
    """True if a valid grid proxy exists (``voms-proxy-info -e``); None if the
    tool is missing. xrootd reads/writes to cmseos generally need one."""
    try:
        return subprocess.run(
            ["voms-proxy-info", "-e"], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        return None


def eos_exists(dest_lfn, host):
    """True if ``dest_lfn`` already exists on EOS (``xrdfs <host> stat``)."""
    return subprocess.run(
        ["xrdfs", host, "stat", dest_lfn], capture_output=True
    ).returncode == 0


def run(cmd, dry_run):
    """Echo a command and (unless --dry-run) run it, returning the exit code."""
    print("  $ " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(
        description="hadd each dataset's files into one ROOT file on EOS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "json", help="JSON of files (make_dataset_json / coffea fileset / bare map)."
    )
    parser.add_argument(
        "-o", "--outdir", required=True,
        help="EOS input dir for the merged files (bare /store/... path or a full "
             "root://host//store/... URL). Created with 'xrdfs mkdir -p'.",
    )
    parser.add_argument(
        "--redirector", default=DEFAULT_REDIRECTOR,
        help="XRootD redirector for reading sources and writing the merged file.",
    )
    parser.add_argument(
        "--prefix", default="",
        help="Prefix for each merged filename (output is <prefix><dataset>.root).",
    )
    parser.add_argument(
        "--tmpdir", default=".",
        help="Local dir for the hadd output before xrdcp (NFS/scratch - NEVER "
             "/eos/uscms). The merged file is hadd'd here, then copied to EOS.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="DATASET",
        help="Merge only these dataset(s) (default: all in the JSON).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a dataset whose merged file already exists on EOS "
             "(default: overwrite it, via xrdcp -f).",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="hadd -j parallelism (parallel source merging).",
    )
    parser.add_argument(
        "--hadd", default="hadd",
        help="hadd executable (needs a ROOT environment, e.g. after cmsenv).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the hadd/xrdcp/xrdfs commands without running them.",
    )
    args = parser.parse_args()

    with open(args.json) as f:
        blob = json.load(f)
    redirector = blob.get("metadata", {}).get("redirector", args.redirector) \
        if isinstance(blob, dict) else args.redirector
    host = host_of(redirector)

    datasets = extract_datasets(blob)
    if args.only:
        missing = [d for d in args.only if d not in datasets]
        if missing:
            sys.exit(f"--only dataset(s) not in JSON: {', '.join(missing)}")
        datasets = {d: datasets[d] for d in args.only}
    if not datasets:
        sys.exit("No datasets to merge.")

    out_lfn = strip_redirector(args.outdir).rstrip("/")  # bare /store dir for EOS
    os.makedirs(args.tmpdir, exist_ok=True)

    if not args.dry_run:
        proxy = have_proxy()
        if proxy is None:
            print("WARNING: voms-proxy-info not found - cannot verify a grid "
                  "proxy; xrootd reads/writes may fail.", file=sys.stderr)
        elif not proxy:
            print("WARNING: no valid grid proxy (run 'voms-proxy-init -voms "
                  "cms'); xrootd reads/writes to cmseos will likely fail.",
                  file=sys.stderr)

    # Create the EOS input directory (FUSE-free; harmless if it already exists).
    print(f"EOS input dir: {host}/{out_lfn}")
    if run(["xrdfs", host, "mkdir", "-p", out_lfn], args.dry_run) != 0:
        sys.exit(f"Failed to create EOS dir {out_lfn} (proxy? permissions?).")

    merged, skipped, failed = [], [], []
    for name in sorted(datasets):
        sources = [to_xrd(p, redirector) for p in datasets[name]]
        out_name = f"{args.prefix}{name}.root"
        dest_lfn = f"{out_lfn}/{out_name}"
        dest_url = f"{host}/{dest_lfn}"          # host already ends without '/'
        local = os.path.join(args.tmpdir, out_name)

        print(f"\n[{name}] {len(sources)} file(s) -> {dest_url}")
        if args.skip_existing and not args.dry_run and eos_exists(dest_lfn, host):
            print("  already on EOS, skipping (--skip-existing).")
            skipped.append(name)
            continue

        # 1) hadd the sources into a LOCAL file (-f overwrites a stale temp).
        hadd_cmd = [args.hadd, "-f"]
        if args.jobs > 1:
            hadd_cmd += ["-j", str(args.jobs)]
        hadd_cmd += [local] + sources
        try:
            rc = run(hadd_cmd, args.dry_run)
        except FileNotFoundError:
            sys.exit(f"'{args.hadd}' not found - set up a ROOT environment "
                     "(e.g. cmsenv) or pass --hadd.")
        if rc != 0:
            print(f"  ERROR: hadd failed (exit {rc}); not delivering.",
                  file=sys.stderr)
            failed.append(name)
            if not args.dry_run and os.path.exists(local):
                os.remove(local)
            continue

        # 2) Deliver the merged file to EOS with xrdcp (-f overwrites), then drop
        #    the local copy. This is the only EOS write the docs permit here.
        try:
            rc = run(["xrdcp", "-f", local, dest_url], args.dry_run)
        finally:
            if not args.dry_run and os.path.exists(local):
                os.remove(local)
        if rc != 0:
            print(f"  ERROR: xrdcp failed (exit {rc}); EOS file NOT written.",
                  file=sys.stderr)
            failed.append(name)
            continue
        merged.append(name)

    print(f"\nDone: {len(merged)} merged, {len(skipped)} skipped, "
          f"{len(failed)} failed -> {host}/{out_lfn}")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
