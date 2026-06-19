#!/usr/bin/env python3
"""compare_qcd_slimming.py - overlay xsec-weighted spectra from two (or more)
sets of slimmed QCD.

Takes the same dataset JSON(s) as the rest of the analyzer (one combined JSON
per set, from scripts/make_dataset_json.py, listing that set's QCD HT slices).
Each slice is filled with the standard xsec weight ``lumi * xs_pb / n_original``
- ``xs_pb`` from ``--xs-json`` (default: the shared aux repo's mj_samples_xs.json)
keyed by the slice's dataset name, ``n_original`` summed from each file's
cutflow[0] by run3_mj_analyzer.load_fileset. Pass ``--unweighted`` to fill with
weight 1 instead.

For every input set it fills three event-/jet-level histograms straight off the
slimmer branches (no candidate collections needed):

  * ht       - per-event ``HT``
  * njet     - per-event jet multiplicity, ``ak.num(ScoutingPFJet_pt)``
  * jet_pt   - inclusive jet ``p_T`` (every jet in every event)

then draws one overlay figure per observable - the sets superimposed as step
histograms (log-y by default, since the QCD HT spectrum spans many orders of
magnitude) - saved as PNGs in ``--outdir``.

Example:
    python scripts/compare_qcd_slimming.py setA.json setB.json -o qcd_compare/
    python scripts/compare_qcd_slimming.py old.json new.json --labels old new
"""

import argparse
import json
import sys
import time
from pathlib import Path

import awkward as ak
import hist
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

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

# Distinct colours per set; cycles if more sets than entries.
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]


def echo(msg):
    """print() that doesn't tear through active tqdm bars."""
    (tqdm.write if tqdm is not None else print)(msg)


def build_axes(args):
    """The three shared axes (same binning across sets so edges line up)."""
    return {
        "ht": hist.axis.Regular(
            args.ht_bins, *args.ht_range, name="ht", label=r"$H_T$ [GeV]"
        ),
        "njet": hist.axis.Integer(
            0, args.max_njet + 1, name="njet", label="number of jets"
        ),
        "jet_pt": hist.axis.Regular(
            args.pt_bins, *args.pt_range, name="pt", label=r"jet $p_T$ [GeV]"
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
    """Load one set's dataset JSON and return ``(fileset, weights)``.

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


def process_set(json_path, label, args, axes):
    """Fill the three histograms for one set; return ``{obs: hist.Hist}``."""
    fileset, weights = weighted_fileset(json_path, args)
    hists = {
        name: hist.Hist(ax, storage=hist.storage.Weight())
        for name, ax in axes.items()
    }
    jobs = [
        (path, tree, weights[name])
        for name, ds in fileset.items()
        for path, tree in ds["files"].items()
    ]
    for name, ds in fileset.items():
        print(f"  [{label}/{name}] {len(ds['files'])} file(s), "
              f"weight {weights[name]:.4g}")

    use_bars = tqdm is not None and not args.no_progress
    n_events = 0
    t0 = time.monotonic()
    job_bar = tqdm(jobs, unit="file", desc=label) if use_bars else jobs
    for path, tree_name, weight in job_bar:
        import uproot

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


def plot_overlay(obs, per_set, args, outpath):
    """Overlay one observable across all sets and save to ``outpath``."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ylabel = "events" if args.unweighted else f"weighted events (pb @ {args.lumi:g})"
    xlabel = None
    for i, (label, hists) in enumerate(per_set.items()):
        h = hists[obs]
        edges = h.axes[0].edges
        xlabel = h.axes[0].label
        ax.stairs(
            h.values(), edges, label=f"{label} ({h.sum().value:.3g})",
            color=COLORS[i % len(COLORS)], linewidth=1.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"QCD {obs} - xsec weighted" if not args.unweighted
                 else f"QCD {obs} - unweighted")
    if args.logy:
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay xsec-weighted HT, jet-multiplicity and inclusive "
        "jet-pt spectra from two (or more) sets of slimmed QCD dataset JSONs."
    )
    parser.add_argument("inputs", nargs="+",
                        help="one dataset JSON per set (from "
                        "scripts/make_dataset_json.py)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="legend label per set (default: each JSON's stem)")
    parser.add_argument("-o", "--outdir", default="qcd_compare_plots",
                        help="directory for the output PNGs (default: %(default)s)")
    parser.add_argument("--xs-json", default=str(DEFAULT_XS_JSON),
                        help="cross-section JSON (default: %(default)s)")
    parser.add_argument("--lumi", type=float, default=1.0,
                        help="integrated luminosity in pb^-1; 1.0 keeps pure "
                        "xs/N weights (default: %(default)s)")
    parser.add_argument("--unweighted", action="store_true",
                        help="fill with weight 1 instead of xsec weights")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: JSON metadata or 'events')")
    parser.add_argument("--ht-bins", type=int, default=60)
    parser.add_argument("--ht-range", type=float, nargs=2, default=(0.0, 3000.0),
                        metavar=("LO", "HI"))
    parser.add_argument("--pt-bins", type=int, default=100)
    parser.add_argument("--pt-range", type=float, nargs=2, default=(0.0, 2000.0),
                        metavar=("LO", "HI"))
    parser.add_argument("--max-njet", type=int, default=20,
                        help="upper edge of the jet-multiplicity axis "
                        "(default: %(default)s)")
    parser.add_argument("--no-logy", dest="logy", action="store_false",
                        help="linear y axis (default: log)")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress tqdm progress bars")
    parser.add_argument("--inspect-workers", type=int, default=16,
                        help="threads for the up-front per-file inspection "
                        "(default: %(default)s)")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.inputs]
    if len(labels) != len(args.inputs):
        raise SystemExit(
            f"got {len(args.inputs)} input(s) but {len(labels)} label(s)"
        )

    axes = build_axes(args)
    per_set = {}
    for json_path, label in zip(args.inputs, labels):
        print(f"\n=== set '{label}': {json_path} ===")
        per_set[label] = process_set(json_path, label, args, axes)

    print("\nweighted entries per set:")
    for label, hists in per_set.items():
        sums = ", ".join(f"{obs}={h.sum().value:.4g}" for obs, h in hists.items())
        print(f"  {label}: {sums}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for obs in axes:
        plot_overlay(obs, per_set, args, outdir / f"{obs}.png")


if __name__ == "__main__":
    main()
