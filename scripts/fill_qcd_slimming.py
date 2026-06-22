#!/usr/bin/env python3
"""fill_qcd_slimming.py - fill xsec-weighted spectra for ONE set of slimmed QCD
and write them to a ROOT file.

Takes a single dataset JSON (one combined JSON per set, from
scripts/make_dataset_json.py, listing that set's QCD HT slices). Each slice is
filled with the standard xsec weight ``lumi * xs_pb / n_original`` - ``xs_pb``
from ``--xs-json`` (default: the shared aux repo's mj_samples_xs.json) keyed by
the slice's dataset name, ``n_original`` summed from each file's cutflow[0] by
run3_mj_analyzer.load_fileset. Pass ``--unweighted`` to fill with weight 1
instead.

It fills three event-/jet-level histograms straight off the slimmer branches
(no candidate collections needed):

  * ht       - per-event ``HT``
  * njet     - per-event jet multiplicity, ``ak.num(ScoutingPFJet_pt)``
  * jet_pt   - inclusive jet ``p_T`` (every jet in every event)

and writes them to a single ROOT file (one TH1 per observable). Overlaying /
comparing different sets is a separate downstream step that reads these ROOT
files; this script only fills and writes.

Binning is set by the constants below (HT_BINS/HT_RANGE/...), not the command
line, so every set is filled with identical axes and the outputs line up when
they are compared later.

Example:
    python scripts/fill_qcd_slimming.py setA.json -o setA_spectra.root
    python scripts/fill_qcd_slimming.py new.json          # -> new_spectra.root
"""

import argparse
import json
import sys
import time
from pathlib import Path

import awkward as ak
import hist
import uproot

try:
    from tqdm import tqdm
except ImportError:  # progress bars are cosmetic - everything runs without them
    tqdm = None

# Make the package importable without `pip install -e .` (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run3_mj_analyzer.fileset import load_fileset

# Cross sections live in the shared aux repo, assumed checked out next to
# run3-mj-analyzer (same convention as make_histograms.py).
DEFAULT_XS_JSON = (
    Path(__file__).resolve().parents[2] / "run3-mj-pass-the-aux" / "mj_samples_xs.json"
)

# Only branches the three observables need, so the iterate stays cheap.
BRANCHES = ["HT", "ScoutingPFJet_pt"]

# ---------------------------------------------------------------------------
# Histogram binning. Edit these to change the axes; they are intentionally
# script constants (not CLI flags) so every set is filled with identical
# binning and the ROOT outputs can be overlaid later without re-binning.
# ---------------------------------------------------------------------------
HT_BINS, HT_RANGE = 60, (0.0, 3000.0)
PT_BINS, PT_RANGE = 100, (0.0, 2000.0)
MAX_NJET = 20  # upper edge of the integer jet-multiplicity axis


def echo(msg):
    """print() that doesn't tear through active tqdm bars."""
    (tqdm.write if tqdm is not None else print)(msg)


def build_axes():
    """The three histogram axes, from the module-level binning constants."""
    return {
        "ht": hist.axis.Regular(HT_BINS, *HT_RANGE, name="ht", label=r"$H_T$ [GeV]"),
        "njet": hist.axis.Integer(
            0, MAX_NJET + 1, name="njet", label="number of jets"
        ),
        "jet_pt": hist.axis.Regular(
            PT_BINS, *PT_RANGE, name="pt", label=r"jet $p_T$ [GeV]"
        ),
    }


# How to turn a chunk of events into each observable's flat fill values. HT and
# njet are per-event (one value per event); jet_pt is per-jet (every jet).
FILLS = {
    "ht": lambda ev: ak.to_numpy(ev["HT"]),
    "njet": lambda ev: ak.to_numpy(ak.num(ev["ScoutingPFJet_pt"], axis=1)),
    "jet_pt": lambda ev: ak.to_numpy(ak.flatten(ev["ScoutingPFJet_pt"], axis=1)),
}


def weighted_fileset(json_path, args):
    """Load the dataset JSON and return ``(fileset, weights)``.

    The inspection pass (in load_fileset) drops slices whose files have no
    events tree - e.g. a low-HT slice where nothing passed the slimmer cuts -
    while still counting their cutflow[0] toward ``n_original``.
    """
    print(
        f"inspecting {json_path} for events trees + cutflow n_original "
        f"({args.inspect_workers} threads)...",
        flush=True,
    )
    fileset = load_fileset(
        json_path, tree=args.tree, workers=args.inspect_workers,
        progress=not args.no_progress,
    )
    if args.unweighted:
        return fileset, {name: 1.0 for name in fileset}

    with open(args.xs_json) as f:
        xs = json.load(f)
    missing = [name for name in fileset if name not in xs]
    if missing:
        raise SystemExit(
            f"No cross section in {args.xs_json} for: {missing}. "
            "Add them, or pass --unweighted."
        )
    weights = {
        name: args.lumi * xs[name]["xs_pb"] / ds["metadata"]["n_original"]
        for name, ds in fileset.items()
    }
    return fileset, weights


def fill_histograms(json_path, label, args, axes):
    """Fill the three histograms for the dataset; return ``{obs: hist.Hist}``."""
    fileset, weights = weighted_fileset(json_path, args)
    hists = {
        name: hist.Hist(ax, storage=hist.storage.Weight())
        for name, ax in axes.items()
    }
    for name, ds in fileset.items():
        print(f"  [{label}/{name}] {len(ds['files'])} file(s), "
              f"weight {weights[name]:.4g}")

    jobs = [
        (path, tree, weights[name])
        for name, ds in fileset.items()
        for path, tree in ds["files"].items()
    ]
    use_bars = tqdm is not None and not args.no_progress
    n_events = 0
    t0 = time.monotonic()
    job_bar = tqdm(jobs, unit="file", desc=label) if use_bars else jobs
    for path, tree_name, weight in job_bar:
        with uproot.open(path) as f:
            if tree_name not in f:
                echo(f"[skip] {path}: no '{tree_name}' tree")
                continue
            for ev in f[tree_name].iterate(
                filter_name=BRANCHES, step_size=args.step_size
            ):
                for obs, fill in FILLS.items():
                    hists[obs].fill(fill(ev), weight=weight)
                n_events += len(ev)
    if use_bars:
        job_bar.close()

    elapsed = time.monotonic() - t0
    rate = n_events / elapsed if elapsed else 0.0
    print(f"  [{label}] {n_events:,} events in {elapsed:.1f} s "
          f"({rate:,.0f} ev/s)")
    return hists


def main():
    parser = argparse.ArgumentParser(
        description="Fill xsec-weighted HT, jet-multiplicity and inclusive "
        "jet-pt spectra for one set of slimmed QCD and write them to a ROOT file."
    )
    parser.add_argument("input",
                        help="one dataset JSON (from "
                        "scripts/make_dataset_json.py)")
    parser.add_argument("-o", "--output", default=None,
                        help="output ROOT file (default: <input stem>_spectra.root)")
    parser.add_argument("--label", default=None,
                        help="name for log lines (default: input JSON's stem)")
    parser.add_argument("--xs-json", default=str(DEFAULT_XS_JSON),
                        help="cross-section JSON (default: %(default)s)")
    parser.add_argument("--lumi", type=float, default=1.0,
                        help="integrated luminosity in pb^-1; 1.0 keeps pure "
                        "xs/N weights (default: %(default)s)")
    parser.add_argument("--unweighted", action="store_true",
                        help="fill with weight 1 instead of xsec weights")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: JSON metadata or 'events')")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress tqdm progress bars")
    parser.add_argument("--inspect-workers", type=int, default=16,
                        help="threads for the up-front per-file inspection "
                        "(default: %(default)s)")
    args = parser.parse_args()

    label = args.label or Path(args.input).stem
    output = Path(args.output) if args.output \
        else Path(f"{Path(args.input).stem}_spectra.root")

    axes = build_axes()
    print(f"=== set '{label}': {args.input} ===")
    hists = fill_histograms(args.input, label, args, axes)

    print("\nweighted entries:")
    sums = ", ".join(f"{obs}={h.sum().value:.4g}" for obs, h in hists.items())
    print(f"  {label}: {sums}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(output) as fout:
        for obs, h in hists.items():
            fout[obs] = h
    print(f"wrote {len(hists)} histograms ({', '.join(hists)}) to {output}")


if __name__ == "__main__":
    main()
