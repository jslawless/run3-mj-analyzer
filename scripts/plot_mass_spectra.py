#!/usr/bin/env python3
"""plot_mass_spectra.py - overlay tri-jet mass spectra, one figure per model.

The script form of notebooks/mass_spectra.ipynb. Inputs are output files of
``make_histograms.py``; every ``h_mass_<Model>`` histogram in them is read back
with its Sumw2 errors intact, and the models common to all inputs get one
overlay figure each.

``make_histograms_both_candidates.py`` output works too - its keys are
``h_mass_cand0_<Model>`` / ``h_mass_cand1_<Model>``, so they simply appear as
models named ``cand0_<Model>`` and ``cand1_<Model>``, and overlaying one file
against itself is not possible but overlaying the two candidates across files
is. Mixing the two scripts' outputs in one call gives no common model, which
the script reports rather than drawing an empty figure.

Any number of inputs, rather than the notebook's hard-coded four:

    python scripts/plot_mass_spectra.py qcd_spectra.root mixed_spectra.root \\
        --label "QCD 6 jets" "Pseudo-events" -o plots/

``--density`` normalises each curve to unit area, which is what you want for
comparing *shapes* - and it is the honest default for anything involving mixed
pseudo-events, whose weight is pb^2/event^2 and so is not on the same footing
as an xsec-weighted QCD spectrum until a global rescale is applied. Without it
the y-axis is weighted entries, and curves are comparable only if their weights
already were.

``--scale`` multiplies a curve by a constant, which is how an unweighted
sample is put on the same footing as the weighted ones: the notebook's
``TTBAR_WEIGHT = LUMI_PB * XS_TTBAR_PB / N_ORIGINAL_TTBAR``, now per input.

    python scripts/plot_mass_spectra.py qcd.root ttbar.root \\
        --label QCD ttbar --scale 1.0 3.2e-5 -o plots/
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")           # write files; never needs a display
import matplotlib.pyplot as plt  # noqa: E402

HIST_KEY_RE = re.compile(r"^h_mass_(.+)$")

#: matplotlib's tab10, in its own order - one colour per input, assigned in the
#: order the inputs are given, so a curve keeps its colour when another input is
#: added after it.
COLOURS = plt.get_cmap("tab10").colors


def load_mass_hists(path):
    """``{model: hist.Hist}`` for every ``h_mass_<Model>`` histogram."""
    out = {}
    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            m = HIST_KEY_RE.match(key)
            if m:
                out[m.group(1)] = f[key].to_hist()
    if not out:
        raise SystemExit(
            f"No h_mass_* histograms in {path} - is it make_histograms.py "
            f"output? Keys found: {sorted(uproot.open(path).keys(cycle=False))}"
        )
    return out


def integral(h):
    """Sum of a histogram, for Weight storage or plain."""
    s = h.sum()
    return getattr(s, "value", s)   # Weight storage -> WeightedSum.value


def step_with_band(ax, h, scale, label, colour, density=False):
    """Step outline plus a stat-error band. Returns the plotted values."""
    edges = h.axes[0].edges
    values = h.values() * scale
    variances = h.variances()
    # No Sumw2 means unweighted fills, where the variance is the count itself.
    variances = (variances if variances is not None else h.values()) * scale**2
    errors = np.sqrt(variances)
    if density:
        widths = np.diff(edges)
        if not np.allclose(widths, widths[0]):
            raise SystemExit(
                f"{label}: --density assumes uniform bins, but this axis has "
                "varying widths. Divide by np.diff(edges) instead."
            )
        norm = values.sum() * widths[0]
        if norm > 0:
            values, errors = values / norm, errors / norm
    ax.stairs(values, edges, label=label, color=colour)
    pad = lambda a: np.append(a, a[-1])   # step="post" needs len(edges) points
    ax.fill_between(edges, pad(values - errors), pad(values + errors),
                    step="post", color=colour, alpha=0.2, linewidth=0)
    return values


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="make_histograms.py output files to overlay")
    p.add_argument("--label", nargs="+", default=None, metavar="NAME",
                   help="legend name per input (default: the file stem)")
    p.add_argument("--scale", nargs="+", type=float, default=None,
                   metavar="F",
                   help="constant multiplier per input (default: 1.0 each)")
    p.add_argument("-o", "--outdir", default=".",
                   help="directory for the figures (default: %(default)s)")
    p.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"),
                   help="figure format (default: %(default)s)")
    p.add_argument("--density", action="store_true",
                   help="normalise each curve to unit area: compares shapes, "
                        "and the only sound comparison when the samples' "
                        "weights are not on a common footing")
    p.add_argument("--linear", action="store_true",
                   help="linear y (default is log: the xsec-weighted QCD "
                        "spectrum falls steeply)")
    p.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                   help="only these models (default: all common to every input)")
    p.add_argument("--xlabel", default="tri-jet invariant mass [GeV]")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    n = len(args.inputs)
    if args.label and len(args.label) != n:
        sys.exit(f"--label takes one name per input ({n} given)")
    if args.scale and len(args.scale) != n:
        sys.exit(f"--scale takes one factor per input ({n} given)")
    if n > len(COLOURS):
        sys.exit(f"{n} inputs is more than the {len(COLOURS)} colours "
                 "available; a repeated colour would read as one curve.")
    labels = args.label or [Path(f).stem for f in args.inputs]
    scales = args.scale or [1.0] * n

    hists = [load_mass_hists(f) for f in args.inputs]
    common = set(hists[0])
    for h in hists[1:]:
        common &= set(h)
    everything = set().union(*(set(h) for h in hists))
    models = sorted(common)
    missing = sorted(everything - common)
    if missing:
        print(f"[skip] models not in every input: {missing}")
    if args.models:
        unknown = sorted(set(args.models) - common)
        if unknown:
            sys.exit(f"requested model(s) {unknown} are not in every input; "
                     f"available: {models}")
        models = [m for m in args.models]
    if not models:
        sys.exit("No model is present in every input - nothing to overlay.")
    print(f"models: {models}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The integrals are the reason to read the terminal as well as the figure:
    # under --density every curve is 1.0 by construction, so the only place the
    # actual normalisation is visible is here.
    width = max(len(s) for s in labels)
    print(f"\n{'model':<20} {'input':<{width}} {'integral':>14}")
    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, hist_set, scale) in enumerate(
                zip(labels, hists, scales)):
            step_with_band(ax, hist_set[model], scale, label, COLOURS[i],
                           args.density)
            print(f"{model:<20} {label:<{width}} "
                  f"{integral(hist_set[model]) * scale:>14.6g}")
        if not args.linear:
            ax.set_yscale("log")
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("density" if args.density else "weighted entries")
        ax.set_title(model)
        ax.legend()
        out = outdir / f"mass_{model}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
