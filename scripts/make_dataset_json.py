#!/usr/bin/env python3
"""make_dataset_json.py - Build a single dataset JSON from an EOS output path.

Given an EOS directory of run3-mj-slimmer (or run3-mj-evaluator) output, walks
the per-dataset subdirectories and records every .root file under each dataset
into ONE JSON file. Unlike the evaluator's make_filesets.py (one JSON per
dataset, for condor), this writes a single combined JSON meant to be loaded
locally for analysis via run3_mj_analyzer.fileset.load_fileset().

The slimmer/evaluator write their output as <base>/<dataset>/*.root (one
subdirectory per dataset; see the slimmer's run_all.sh, which submits with
-o <base>/<dataset>). If the path instead contains .root files directly, it is
treated as a single dataset.

Output JSON layout:
    {
        "metadata": {
            "eos_path":   "/store/.../slimmed",
            "redirector": "root://cmseos.fnal.gov/",
            "tree":       "events"
        },
        "datasets": {
            "<dataset>": [
                "root://cmseos.fnal.gov//store/.../slimmed_xxx.root",
                ...
            ],
            ...
        }
    }

Usage:
    python scripts/make_dataset_json.py <eos_path> -o datasets.json
    python scripts/make_dataset_json.py /store/user/jlawless/slimmed -o datasets.json
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

# Slimmer/evaluator outputs live on personal/group EOS and must be streamed
# through the EOS redirector (XCache only serves the global CMS namespace).
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov/"


def _natural_key(name):
    """Sort key that orders embedded numbers numerically, so HT-binned slice
    names sort by HT (HT-100to200 < HT-200to400 < ... < HT-1000to1200 < HT-2000)
    instead of lexically (which would put HT-1000to1200 before HT-100to200).
    Each token is tagged by type so digit and non-digit positions never compare
    across types.
    """
    return [(0, int(tok)) if tok.isdigit() else (1, tok)
            for tok in re.split(r"(\d+)", name)]


def eos_ls(path, ls_cmd):
    """Return the entry names under an EOS directory by running `ls_cmd <path>`.

    `ls_cmd` may be a multi-word command (e.g. 'eos root://cmseos.fnal.gov ls');
    it is tokenized and the path appended. Note the interactive `eosls` is
    usually a shell *function* wrapping `eos ... ls`, which cannot be invoked
    from a subprocess -- so we call the `eos` binary directly.
    """
    cmd = shlex.split(ls_cmd) + [path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit(
            f"'{cmd[0]}' not found on PATH. ('eosls' is often a shell function "
            "wrapping 'eos root://cmseos.fnal.gov ls' and can't be called from a "
            "script.) Run where the EOS client is available (e.g. cmslpc), or "
            "pass --ls-cmd."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"`{' '.join(cmd)}` failed:\n{e.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def to_xrd(eos_dir, fname, redirector):
    """Prefix a /store path with the XRootD redirector for WAN streaming.

    XRootD needs a DOUBLE slash between the host and an absolute path:
    'root://host//store/...'. A single slash makes 'store/...' a relative path,
    which the server rejects ('Locating relative path ... is disallowed').
    """
    full = f"{eos_dir.rstrip('/')}/{fname}"
    if full.startswith("root://"):
        return full
    host = redirector.rstrip("/")          # e.g. root://cmseos.fnal.gov
    return f"{host}//{full.lstrip('/')}"   # -> root://cmseos.fnal.gov//store/...


def discover_datasets(eos_path, ls_cmd):
    """Map dataset name -> EOS directory (one subdir per dataset, or flat)."""
    entries = eos_ls(eos_path, ls_cmd)
    subdirs = [e for e in entries if not e.endswith(".root")]
    has_root = any(e.endswith(".root") for e in entries)

    if subdirs:
        return {d: f"{eos_path}/{d}" for d in subdirs}
    if has_root:
        return {os.path.basename(eos_path): eos_path}
    sys.exit(f"No dataset subdirectories or .root files found under {eos_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a single combined dataset JSON from a slimmer/"
                    "evaluator EOS output directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "eos_path",
        help="EOS dir of slimmer/evaluator output (one subdir per dataset, or "
             ".root files directly for a single dataset).",
    )
    parser.add_argument(
        "-o", "--output", default="datasets.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--redirector", default=DEFAULT_REDIRECTOR,
        help="XRootD redirector prefix applied to /store paths.",
    )
    parser.add_argument(
        "--tree", default="events",
        help="Tree name recorded in the JSON metadata (slimmer/evaluator output "
             "tree is 'events').",
    )
    parser.add_argument(
        "--ls-cmd", default="eos root://cmseos.fnal.gov ls",
        help="Command used to list an EOS directory (the path is appended). "
             "Defaults to the 'eos' binary, since the interactive 'eosls' is a "
             "shell function that scripts can't call.",
    )
    args = parser.parse_args()

    eos_path = args.eos_path.rstrip("/")
    datasets = discover_datasets(eos_path, args.ls_cmd)

    out = {
        "metadata": {
            "eos_path": eos_path,
            "redirector": args.redirector,
            "tree": args.tree,
        },
        "datasets": {},
    }

    total = 0
    for name, dsdir in sorted(datasets.items(), key=lambda kv: _natural_key(kv[0])):
        files = [
            to_xrd(dsdir, f, args.redirector)
            for f in eos_ls(dsdir, args.ls_cmd)
            if f.endswith(".root")
        ]
        if not files:
            print(f"  skip (no .root files): {dsdir}", file=sys.stderr)
            continue
        out["datasets"][name] = files
        total += len(files)
        print(f"  {name}: {len(files)} files")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=4)

    print(f"\nWrote {len(out['datasets'])} datasets ({total} files) -> {args.output}")


if __name__ == "__main__":
    main()
