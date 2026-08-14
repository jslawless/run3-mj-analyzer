#!/usr/bin/env python3
"""plot_slimmed_spectra.py - overlay slimmer-level spectra, one figure each.

The script form of notebooks/qcd_5_v_6.ipynb: QCD at exactly 5 jets, QCD at
exactly 6 jets, and mixed pseudo-events, compared in ``ht`` / ``njet`` /
``jet_pt``. Inputs are the ROOT files written by ``fill_qcd_slimming.py`` (the
QCD sets) and ``make_stitched_histograms.py`` (the pseudo-events), which fill
the same axes under the same bare names so their histograms line up.

**Every curve is area-normalised to unit area**, so the figures compare shapes.
That is the only sound default here: pseudo-event weights are pb^2/event^2 - a
product of two slice weights - so their absolute scale is not on the same
footing as an xsec-weighted QCD spectrum, and putting the raw numbers on shared
axes would invite reading a difference that is pure normalisation.

    python scripts/plot_slimmed_spectra.py \\
        qcd6_spectra.root qcd5_spectra.root mixed_spectra.root \\
        --label "6 jets" "5 jets" "Pseudo-events" -o plots/

    # HT above a cut; each curve is then normalised over that range alone
    python scripts/plot_slimmed_spectra.py ... --xmin ht=550

``--absolute`` turns that off and plots weighted entries, which is when the
notebook's other normalisation matters: ``--normalise`` scales the named curves
by ``integral(reference) / integral(curve)``, exactly its
``f6['njet'].sum() / mix['njet'].sum()``. That factor is computed once, from
``--norm-from`` (default ``njet``), and applied to *every* observable rather
than recomputed per plot - which is the whole point of it, since it asks
whether one scale factor makes the pseudo-events agree with QCD everywhere at
once. Under area normalisation a constant factor cancels, so the two options
are mutually exclusive and saying both is an error rather than a silent no-op.

    python scripts/plot_slimmed_spectra.py ... --absolute \\
        --normalise "Pseudo-events" --norm-from njet

Mass spectra are deliberately not plotted here - this script is kinematics.
``h_m6j`` and ``h_hemi_mass`` are skipped unless named explicitly in
``--observables``; for tri-jet candidate masses use plot_mass_spectra.py.
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

#: Mass spectra, skipped by default: this script is kinematics, and the mass
#: distributions have their own script (plot_mass_spectra.py) with the model
#: splitting they need. Naming one in --observables still plots it.
MASS_OBSERVABLES = {"h_m6j", "h_hemi_mass", "h_mass"}

#: Cutflows are counters over a StrCategory axis, not spectra: overlaying them
#: on shared axes says nothing, and normalising one to unit area says less.
#: Named explicitly in --observables they are still plotted.
def _is_cutflow(key):
    return "cutflow" in key

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


def draw(ax, h, scale, label, colour, lo=None, density=True):
    """Step outline plus a stat-error band."""
    if lo is not None:
        h = h[complex(0, lo):]
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
                f"{label}: area normalisation assumes uniform bins, but this "
                "axis has varying widths. Divide by np.diff(edges) instead."
            )
        norm = values.sum() * widths[0]
        if norm > 0:
            values, errors = values / norm, errors / norm
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
    p.add_argument("--absolute", action="store_true",
                   help="plot weighted entries instead of normalising each "
                        "curve to unit area. Only meaningful when the samples' "
                        "weights are already on a common footing, or together "
                        "with --normalise.")
    p.add_argument("--normalise", nargs="+", default=(), metavar="LABEL",
                   help="with --absolute: scale these curves to the "
                        "reference's integral, by one factor from --norm-from "
                        "applied to every observable. Cancels under area "
                        "normalisation, so it requires --absolute.")
    p.add_argument("--ref", default=None, metavar="LABEL",
                   help="the curve --normalise scales to (default: the first "
                        "input)")
    p.add_argument("--norm-from", default="njet", metavar="OBS",
                   help="observable whose integral sets the scale factor "
                        "(default: %(default)s). One factor per curve, applied "
                        "to every observable.")
    p.add_argument("--observables", nargs="+", default=None, metavar="OBS",
                   help="which to plot (default: every histogram present in at "
                        f"least two inputs, except the mass spectra "
                        f"{sorted(MASS_OBSERVABLES)}, which belong to "
                        "plot_mass_spectra.py. Naming one here plots it.)")
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
    if args.normalise and not args.absolute:
        sys.exit(
            "--normalise scales curves by a constant, which cancels when every "
            "curve is normalised to unit area - it would do nothing. Add "
            "--absolute to plot weighted entries, or drop --normalise."
        )

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
        shared = sorted(k for k, c in counts.items() if c >= 2)
        observables = [k for k in shared
                       if k not in MASS_OBSERVABLES and not _is_cutflow(k)]
        masses = [k for k in shared if k in MASS_OBSERVABLES]
        if masses:
            print(f"[skip] mass spectra, not kinematics: {masses} "
                  "(plot_mass_spectra.py, or name them in --observables)")
        cutflows = [k for k in shared if _is_cutflow(k)]
        if cutflows:
            print(f"[skip] cutflows, not spectra: {cutflows}")
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
                 COLOURS[labels.index(label)], xmin.get(obs),
                 density=not args.absolute)
        log = obs not in LINEAR_OBSERVABLES
        if obs in args.linear:
            log = False
        if obs in args.logy:
            log = True
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(XLABELS.get(obs, obs))
        ax.set_ylabel("weighted entries" if args.absolute
                      else "a.u. (unit area)")
        ax.set_title(obs + (f"  ($\\geq$ {xmin[obs]:g})" if obs in xmin else ""))
        ax.legend()
        out = outdir / f"{obs}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
