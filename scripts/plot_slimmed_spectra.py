#!/usr/bin/env python3
"""plot_slimmed_spectra.py - overlay slimmer-level spectra, one figure each.

The script form of notebooks/qcd_5_v_6.ipynb: QCD at exactly 5 jets, QCD at
exactly 6 jets, and mixed pseudo-events, compared in ``ht`` / ``njet`` /
``jet_pt``. Inputs are the ROOT files written by ``fill_qcd_slimming.py`` (the
QCD sets) and ``make_stitched_histograms.py`` (the pseudo-events), which fill
the same axes under the same bare names so their histograms line up.

The interesting part is the normalisation. Pseudo-event weights are
pb^2/event^2 - a product of two slice weights - so their absolute scale means
nothing and they can only be compared to QCD after being scaled to it. That is
what ``--normalise`` does: the named curves are multiplied by
``integral(reference) / integral(curve)``, exactly the notebook's
``f6['njet'].sum() / mix['njet'].sum()``.

The factor is computed once, from ``--norm-from`` (default ``njet``), and then
applied to *every* observable - not recomputed per plot. That is deliberate: a
per-plot factor would force every overlay to match in area and so hide the
question being asked, which is whether one scale factor makes the pseudo-events
agree with QCD everywhere at once.

    # the notebook's first three figures
    python scripts/plot_slimmed_spectra.py \\
        qcd6_spectra.root qcd5_spectra.root mixed_spectra.root \\
        --label "6 jets" "5 jets" "Pseudo-events" \\
        --normalise "Pseudo-events" -o plots/

    # its fourth: HT above a cut, renormalised over that range alone
    python scripts/plot_slimmed_spectra.py ... --xmin ht=550 --norm-from ht
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Observables that are counts of things and read better on a linear y. Every
#: other spectrum here falls steeply enough to want log.
LINEAR_OBSERVABLES = {"njet"}

#: Axis labels for the observables these two fillers write; anything else falls
#: back to its own key.
XLABELS = {
    "ht": "$H_T$ [GeV]",
    "njet": "jet multiplicity",
    "jet_pt": "jet $p_T$ [GeV]",
    "jet_eta": "jet $\\eta$",
    "jet_phi": "jet $\\phi$",
    "h_match_distance": "match distance in (directed $\\phi$, partner $\\eta$)",
    "h_m6j": "6-jet invariant mass [GeV]",
    "h_thrust": "transverse thrust",
}

COLOURS = plt.get_cmap("tab10").colors


def load_hists(path):
    """``{key: hist.Hist}`` for every TH1 in the file."""
    out = {}
    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            obj = f[key]
            try:
                h = obj.to_hist()
            except (AttributeError, TypeError):
                continue          # TObjString metadata, not a histogram
            if len(h.axes) == 1:
                out[key] = h
    if not out:
        raise SystemExit(f"No 1-D histograms in {path}.")
    return out


def integral(h, lo=None):
    """Sum of ``h``, optionally only over bins whose centre is >= ``lo``."""
    if lo is not None:
        h = h[complex(0, lo):]     # hist's data-coordinate slice, e.g. 550j
    s = h.sum()
    return getattr(s, "value", s)


def draw(ax, h, scale, label, colour, lo=None):
    """Step outline plus a stat-error band."""
    if lo is not None:
        h = h[complex(0, lo):]
    edges = h.axes[0].edges
    values = h.values() * scale
    variances = h.variances()
    # No Sumw2 means unweighted fills, where the variance is the count itself.
    variances = (variances if variances is not None else h.values()) * scale**2
    errors = np.sqrt(variances)
    ax.stairs(values, edges, label=label, color=colour)
    pad = lambda a: np.append(a, a[-1])   # step="post" needs len(edges) points
    ax.fill_between(edges, pad(values - errors), pad(values + errors),
                    step="post", color=colour, alpha=0.2, linewidth=0)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="fill_qcd_slimming.py / make_stitched_histograms.py "
                        "output files to overlay")
    p.add_argument("--label", nargs="+", default=None, metavar="NAME",
                   help="legend name per input (default: the file stem)")
    p.add_argument("--normalise", nargs="+", default=(), metavar="LABEL",
                   help="scale these curves to the reference's integral. Use "
                        "it for anything whose absolute normalisation is not "
                        "comparable - pseudo-events above all.")
    p.add_argument("--ref", default=None, metavar="LABEL",
                   help="the curve --normalise scales to (default: the first "
                        "input)")
    p.add_argument("--norm-from", default="njet", metavar="OBS",
                   help="observable whose integral sets the scale factor "
                        "(default: %(default)s). One factor per curve, applied "
                        "to every observable.")
    p.add_argument("--observables", nargs="+", default=None, metavar="OBS",
                   help="which to plot (default: every histogram present in at "
                        "least two inputs)")
    p.add_argument("--xmin", nargs="+", default=(), metavar="OBS=VALUE",
                   help="restrict an observable's range, e.g. ht=550. When it "
                        "is also --norm-from, the factor is computed over the "
                        "restricted range - the notebook's ht[9:] variant.")
    p.add_argument("--linear", nargs="+", default=(), metavar="OBS",
                   help=f"force linear y (default linear: "
                        f"{sorted(LINEAR_OBSERVABLES)})")
    p.add_argument("--logy", nargs="+", default=(), metavar="OBS",
                   help="force log y")
    p.add_argument("-o", "--outdir", default=".")
    p.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    n = len(args.inputs)
    if args.label and len(args.label) != n:
        sys.exit(f"--label takes one name per input ({n} given)")
    if n > len(COLOURS):
        sys.exit(f"{n} inputs is more than the {len(COLOURS)} colours "
                 "available; a repeated colour would read as one curve.")
    labels = args.label or [Path(f).stem for f in args.inputs]
    if len(set(labels)) != n:
        sys.exit(f"labels must be unique, got {labels}")

    xmin = {}
    for item in args.xmin:
        if "=" not in item:
            sys.exit(f"--xmin wants OBS=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        xmin[key] = float(value)

    hists = [load_hists(f) for f in args.inputs]
    by_label = dict(zip(labels, hists))

    ref_label = args.ref or labels[0]
    if ref_label not in by_label:
        sys.exit(f"--ref {ref_label!r} is not one of {labels}")
    unknown = [l for l in args.normalise if l not in by_label]
    if unknown:
        sys.exit(f"--normalise names {unknown}, which are not among {labels}")

    # Scale factors, from one observable, applied everywhere.
    scales = {label: 1.0 for label in labels}
    if args.normalise:
        obs = args.norm_from
        lo = xmin.get(obs)
        for label in [ref_label, *args.normalise]:
            if obs not in by_label[label]:
                sys.exit(f"--norm-from {obs!r} is not in {label!r}; it has "
                         f"{sorted(by_label[label])}")
        ref_total = integral(by_label[ref_label][obs], lo)
        for label in args.normalise:
            total = integral(by_label[label][obs], lo)
            if total <= 0:
                sys.exit(f"{label!r} has no entries in {obs!r} - cannot "
                         "normalise to the reference.")
            scales[label] = ref_total / total
        where = f"{obs}" + (f" above {lo:g}" if lo is not None else "")
        print(f"scale factors (to {ref_label!r}, from {where}):")
        for label in args.normalise:
            print(f"  {label:<32} x {scales[label]:.6g}")

    if args.observables:
        observables = list(args.observables)
    else:
        counts = {}
        for h in hists:
            for key in h:
                counts[key] = counts.get(key, 0) + 1
        observables = sorted(k for k, c in counts.items() if c >= 2)
        singles = sorted(k for k, c in counts.items() if c < 2)
        if singles:
            print(f"[skip] in only one input, nothing to compare: {singles}")
    if not observables:
        sys.exit("No observable appears in two or more inputs.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for obs in observables:
        have = [label for label in labels if obs in by_label[label]]
        if not have:
            print(f"[skip] {obs}: in none of the inputs")
            continue
        if len(have) < len(labels):
            print(f"[note] {obs}: only in {have}")
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in have:
            draw(ax, by_label[label][obs], scales[label], label,
                 COLOURS[labels.index(label)], xmin.get(obs))
        log = obs not in LINEAR_OBSERVABLES
        if obs in args.linear:
            log = False
        if obs in args.logy:
            log = True
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(XLABELS.get(obs, obs))
        ax.set_ylabel("weighted entries")
        ax.set_title(obs + (f"  ($\\geq$ {xmin[obs]:g})" if obs in xmin else ""))
        ax.legend()
        out = outdir / f"{obs}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
