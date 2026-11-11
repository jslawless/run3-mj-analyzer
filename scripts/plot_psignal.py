#!/usr/bin/env python3
"""plot_psignal.py - per-model pSignal distributions, for choosing a cut.

Inputs are ``make_histograms.py`` outputs; every ``h_psignal_<Model>`` (filled
after the mass-asymmetry cut, for models that write a pSignal - the PairNets)
is overlaid across inputs, one figure per model, each curve normalised to unit
area. The x-axis is ``-log10(1 - pSignal)``: signal piles up within 1e-2 of 1,
which a linear axis squeezes into a sliver at the right edge.

The terminal table is the part to choose a cut from: for each model, input and
candidate cut, the fraction of that input's events kept and the weighted yield
left (pb for xsec-weighted inputs). Cuts must be pSignal bin edges.

    python scripts/plot_psignal.py hists_qcd.root hists_M300.root \\
        hists_M1000.root --label QCD M300 M1000 -o plots/psignal
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HIST_KEY_RE = re.compile(r"^h_psignal_(?!mass_)(.+)$")
COLOURS = plt.get_cmap("tab10").colors
DEFAULT_CUTS = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999)


def load_psignal_hists(path):
    """``{model: hist.Hist}`` for every 1D ``h_psignal_<Model>``."""
    with uproot.open(path) as f:
        out = {m.group(1): f[k].to_hist() for k in f.keys(cycle=False)
               if (m := HIST_KEY_RE.match(k))}
    if not out:
        raise SystemExit(f"No h_psignal_* histograms in {path} - was it made "
                         "by a make_histograms.py with pSignal support, from "
                         "inputs that have a PairNet?")
    return out


def to_x(p):
    """Plot coordinate for pSignal: -log10(1 - p), capped at 6 so the last
    bin (up to p = 1) ends at 1 - 1e-6 instead of running off to infinity."""
    return -np.log10(np.clip(1.0 - np.asarray(p), 1e-6, None))


def kept(h, cut):
    """(weighted sum at pSignal >= cut, that sum as a fraction of the total)."""
    edges = h.axes[0].edges
    i = np.flatnonzero(np.isclose(edges, cut, rtol=0, atol=1e-9))
    if not len(i):
        raise SystemExit(f"cut {cut:g} is not a pSignal bin edge")
    v = h.values()
    total = v.sum()
    above = v[int(i[0]):].sum()
    return above, (above / total if total > 0 else float("nan"))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="make_histograms.py output files to overlay")
    p.add_argument("--label", nargs="+", default=None, metavar="NAME",
                   help="legend name per input (default: the file stem)")
    p.add_argument("--cuts", nargs="+", type=float, default=DEFAULT_CUTS,
                   metavar="P", help="cuts to tabulate (default: %(default)s)")
    p.add_argument("-o", "--outdir", default=".",
                   help="directory for the figures (default: %(default)s)")
    p.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    n = len(args.inputs)
    if args.label and len(args.label) != n:
        sys.exit(f"--label takes one name per input ({n} given)")
    labels = args.label or [Path(f).stem for f in args.inputs]
    hists = [load_psignal_hists(f) for f in args.inputs]
    models = sorted(set.intersection(*(set(h) for h in hists)))
    if not models:
        sys.exit("No model has an h_psignal_* histogram in every input.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    width = max(len(s) for s in labels)
    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, hs) in enumerate(zip(labels, hists)):
            h = hs[model]
            x = to_x(h.axes[0].edges)
            v = h.values()
            # Unit area in the plotted coordinate, so the variable pSignal
            # binning doesn't show up as steps in the curve.
            dens = v / (v.sum() * np.diff(x)) if v.sum() > 0 else v
            ax.stairs(dens, x, label=label, color=COLOURS[i % len(COLOURS)])
        ticks = [0.0, 0.9, 0.99, 0.999, 0.9999, 0.99999]
        ax.set_xticks(to_x(ticks), [f"{t:g}" for t in ticks])
        ax.set_yscale("log")
        ax.set_xlabel("pSignal  (axis is -log10(1 - pSignal))")
        ax.set_ylabel("density")
        ax.set_title(f"{model}: pSignal after mass-asymmetry cut")
        ax.legend()
        out = outdir / f"psignal_{model}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{model}   (fraction kept / weighted yield left)")
        print(f"  {'cut':>8}  " + "  ".join(f"{s:>{max(width, 20)}}"
                                            for s in labels))
        for cut in args.cuts:
            cells = []
            for hs in hists:
                above, frac = kept(hs[model], cut)
                cells.append(f"{frac:>9.4f} / {above:<9.4g}".rjust(
                    max(width, 20)))
            print(f"  {cut:>8g}  " + "  ".join(cells))
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
