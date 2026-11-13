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

``--ratio`` adds a lower panel holding every other input divided by the
first, bin by bin, from the curves exactly as drawn (plot_slimmed_spectra.py's
``draw_ratio``), and prints a chi2/ndf of each against it. That is the mixing
closure test: the exactly-6-jet QCD reference first, pseudo-events after.

    python scripts/plot_mass_spectra.py qcd_njet6.root mixed.root \\
        --label "QCD, 6 jets" "Pseudo-events" --density --linear --ratio \\
        --rebin 2 -o plots/closure/

The chi2 adds the two inputs' errors in quadrature over the bins where both
are filled. Under ``--density`` it compares shapes and ``ndf`` is one less
than that bin count, for the normalisation the area scaling fixed. It ignores
any bin-to-bin correlation, and the pseudo-events have some: one hemisphere is
never reused, but one parent event contributes both of its hemispheres to
different pseudo-events. Read it as a guide, not a p-value.
"""

import argparse
import re
import sys
from pathlib import Path

import hist
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")           # write files; never needs a display
import matplotlib.pyplot as plt  # noqa: E402

# The ratio panel is plot_slimmed_spectra.py's, imported rather than re-typed so
# the two scripts' ratios cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_slimmed_spectra import draw_ratio  # noqa: E402

HIST_KEY_RE = re.compile(r"^h_mass_(.+)$")

#: matplotlib's tab10, in its own order - one colour per input, assigned in the
#: order the inputs are given, so a curve keeps its colour when another input is
#: added after it.
COLOURS = plt.get_cmap("tab10").colors

#: Models whose spectrum came from the pSignal projection, for the titles.
PSIGNAL_CUT_APPLIED = set()


def psignal_projection(h2, cut):
    """Mass spectrum of events with pSignal >= ``cut``, from the
    ``h_psignal_mass_<Model>`` 2D histogram (mass x pSignal). ``cut`` must sit
    on a pSignal bin edge, or the cut would be silently rounded to one."""
    edges = h2.axes[1].edges
    i = np.flatnonzero(np.isclose(edges, cut, rtol=0, atol=1e-9))
    if not len(i):
        raise SystemExit(
            f"--psignal-cut {cut:g} is not a pSignal bin edge; nearest are "
            f"{edges[np.searchsorted(edges, cut) - 1]:g} and "
            f"{edges[np.searchsorted(edges, cut)]:g}")
    return h2[:, int(i[0])::sum]


def load_mass_hists(path, psignal_cut=None):
    """``{model: hist.Hist}`` for every ``h_mass_<Model>`` histogram. With
    ``psignal_cut``, models that have an ``h_psignal_mass_<Model>`` get the
    projection at that cut instead; the others keep their ``h_mass``."""
    out = {}
    with uproot.open(path) as f:
        keys = set(f.keys(cycle=False))
        for key in sorted(keys):
            m = HIST_KEY_RE.match(key)
            if m:
                model, key2d = m.group(1), f"h_psignal_mass_{m.group(1)}"
                if psignal_cut is not None and key2d in keys:
                    out[model] = psignal_projection(f[key2d].to_hist(),
                                                    psignal_cut)
                    PSIGNAL_CUT_APPLIED.add(model)
                else:
                    out[model] = f[key].to_hist()
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
    """Step outline plus a stat-error band. Returns ``(edges, values, errors)``
    as plotted, normalisation included, so a ratio divides what is drawn."""
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
    return edges, values, errors


def shape_chi2(base, other, density):
    """``(chi2, ndf)`` of ``other`` against ``base``, two ``step_with_band``
    return values, over the bins where both are filled. Under ``density`` the
    area normalisation is a fitted constraint, so it costs one degree of
    freedom."""
    _, b, eb = base
    _, v, ev = other
    used = (b > 0) & (v > 0)
    chi2 = float(np.sum((v[used] - b[used]) ** 2
                        / (eb[used] ** 2 + ev[used] ** 2)))
    return chi2, int(used.sum()) - (1 if density else 0)


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
    p.add_argument("--ylabel", default=None,
                   help="y-axis label (default: 'density' or 'weighted entries')")
    p.add_argument("--ymin", type=float, default=None,
                   help="y-axis lower limit (default: matplotlib's choice)")
    p.add_argument("--psignal-cut", type=float, default=None, metavar="P",
                   help="for models with an h_psignal_mass_* histogram (the "
                        "PairNets), plot only events with pSignal >= P. P must "
                        "be a pSignal bin edge. Other models are unchanged.")
    p.add_argument("--ratio", action="store_true",
                   help="lower panel: every other input divided by the first, "
                        "plus a printed chi2/ndf of each against it")
    p.add_argument("--ratio-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="ratio-panel y limits (default: from the bulk of the "
                        "ratio, always keeping 1 in view)")
    p.add_argument("--rebin", type=int, default=1, metavar="N",
                   help="merge every N mass bins before plotting (default: "
                        "%(default)s); the low-stat tails of xsec-weighted QCD "
                        "are otherwise a few events per bin")
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
    if args.ratio and n < 2:
        sys.exit("--ratio divides the other inputs by the first, so it needs "
                 "at least two")
    labels = args.label or [Path(f).stem for f in args.inputs]
    scales = args.scale or [1.0] * n

    hists = [load_mass_hists(f, args.psignal_cut) for f in args.inputs]
    if args.rebin > 1:
        hists = [{m: h[::hist.rebin(args.rebin)] for m, h in hs.items()}
                 for hs in hists]
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
    chi2_rows = []
    for model in models:
        if args.ratio:
            fig, (ax, ax_ratio) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True,
                gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_ratio = None
        drawn = []
        for i, (label, hist_set, scale) in enumerate(
                zip(labels, hists, scales)):
            drawn.append(step_with_band(ax, hist_set[model], scale, label,
                                        COLOURS[i], args.density))
            print(f"{model:<20} {label:<{width}} "
                  f"{integral(hist_set[model]) * scale:>14.6g}")
        if ax_ratio is not None:
            if not all(len(d[0]) == len(drawn[0][0])
                       and np.allclose(d[0], drawn[0][0]) for d in drawn):
                sys.exit(f"{model}: the inputs bin the mass differently, so "
                         "there is no bin-by-bin ratio")
            text = []
            for i in range(1, n):
                draw_ratio(ax_ratio, drawn[0], drawn[i], COLOURS[i])
                chi2, ndf = shape_chi2(drawn[0], drawn[i], args.density)
                chi2_rows.append((model, labels[i], chi2, ndf))
                text.append(f"{labels[i]}: $\\chi^2$/ndf = {chi2:.1f}/{ndf}")
            if args.ratio_range:
                ax_ratio.set_ylim(*args.ratio_range)
            ax_ratio.set_ylabel(f"ratio to\n{labels[0]}", fontsize=9)
            ax_ratio.text(0.01, 0.95, "\n".join(text), fontsize=8,
                          transform=ax_ratio.transAxes, va="top")
        if not args.linear:
            ax.set_yscale("log")
        if args.ymin is not None:
            ax.set_ylim(bottom=args.ymin)
        (ax_ratio or ax).set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel or
                      ("density" if args.density else "weighted entries"))
        ax.set_title(model + (f"  (pSignal >= {args.psignal_cut:g})"
                              if model in PSIGNAL_CUT_APPLIED else ""))
        ax.legend()
        out = outdir / f"mass_{model}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")

    if chi2_rows:
        print(f"\n{'model':<20} {'input':<{width}} {'chi2/ndf':>14}  "
              f"(against {labels[0]})")
        for model, label, chi2, ndf in chi2_rows:
            print(f"{model:<20} {label:<{width}} "
                  f"{f'{chi2:.1f}/{ndf}':>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
