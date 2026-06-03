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
import subprocess
import sys

# Slimmer/evaluator outputs live on personal/group EOS and must be streamed
# through the EOS redirector (XCache only serves the global CMS namespace).
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov/"


def eos_ls(path, ls_cmd):
    """Return the entry names under an EOS directory via `ls_cmd` (e.g. eosls)."""
    try:
        proc = subprocess.run(
            [ls_cmd, path], check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        sys.exit(
            f"'{ls_cmd}' not found on PATH. Run this where the EOS client is "
            "available (e.g. cmslpc), or pass --ls-cmd."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"`{ls_cmd} {path}` failed:\n{e.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def to_xrd(eos_dir, fname, redirector):
    """Prefix a /store path with the XRootD redirector for WAN streaming."""
    full = f"{eos_dir.rstrip('/')}/{fname}"
    if full.startswith("root://"):
        return full
    return redirector.rstrip("/") + "/" + full.lstrip("/")


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
        "--ls-cmd", default="eosls",
        help="Command used to list an EOS directory.",
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
    for name, dsdir in sorted(datasets.items()):
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
