#!/usr/bin/env python3
"""plot_injection_scan.py - how much injected signal moves the mixed background.

Input is one ``make_histograms.py --signal-file-id-min`` output for one
signal-injection study (signal hemispheres added to the mixing pool). Its
histograms are split by pair type: ``h_mass_<Model>_QQ`` (both halves QCD),
``_SQ`` (one half signal) and ``_SS`` (both signal), all filled with the
nominal signal cross section. Scaling the signal cross section by a strength
``mu`` scales each half's xs_weight factor by ``mu``, and the fill weight is
``xs_weight ** p`` (p from the file's ``pair_split``; 0.5 for sqrt weights),
so the mixed background at strength mu is

    QQ + mu^p * SQ + mu^(2p) * SS

The pairing itself does not depend on the weights, so one set of histograms
covers every mu. For each model this draws that shape (unit area) at each mu
in ``--mu``, with a ratio panel to mu = 0, and prints per mu:

  * the signal-containing fraction of the weighted sample after the cuts;
  * chi2/ndf of the mu shape against the mu = 0 shape, using the mu = 0
    statistical errors only. The two shapes share all their QQ events, so the
    difference is purely the signal-containing pairs, and the mu = 0 error is
    the precision the background estimate itself has. Read it as "would this
    shift be visible over the background's own stat noise", not a p-value.

    python scripts/plot_injection_scan.py hists_inject_M300_x1.root \\
        --label "M300 x1" --mu 0 1 3 10 30 100 --rebin 2 -o plots/M300_x1/

``--psignal-cut`` reads the PairNets' spectra from their
``h_psignal_mass_<Model>_<type>`` 2D histograms at that cut, as in
plot_mass_spectra.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import hist
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_mass_spectra import psignal_projection  # noqa: E402
from plot_slimmed_spectra import draw_ratio  # noqa: E402

TYPES = ("QQ", "SQ", "SS")
HIST_KEY_RE = re.compile(r"^h_mass_(.+)_QQ$")
#: viridis, so increasing mu reads as an ordered sequence
CMAP = plt.get_cmap("viridis")


def load(path, psignal_cut):
    """``({model: {type: hist}}, pair_split)`` from one split file."""
    with uproot.open(path) as f:
        if "pair_split" not in f:
            raise SystemExit(f"{path} has no pair_split - make it with "
                             "make_histograms.py --signal-file-id-min")
        split = json.loads(str(f["pair_split"]))
        keys = set(f.keys(cycle=False))
        out = {}
        for key in sorted(keys):
            m = HIST_KEY_RE.match(key)
            if not m:
                continue
            model = m.group(1)
            out[model] = {}
            for t in TYPES:
                k2d = f"h_psignal_mass_{model}_{t}"
                if psignal_cut is not None and k2d in keys:
                    out[model][t] = psignal_projection(f[k2d].to_hist(),
                                                       psignal_cut)
                else:
                    out[model][t] = f[f"h_mass_{model}_{t}"].to_hist()
    if not out:
        raise SystemExit(f"no h_mass_<Model>_QQ histograms in {path}")
    return out, split


def combine(parts, mu, p):
    """``(values, variances)`` of QQ + mu^p SQ + mu^(2p) SS."""
    scale = {"QQ": 1.0, "SQ": mu ** p, "SS": mu ** (2 * p)}
    v = sum(parts[t].values() * scale[t] for t in TYPES)
    var = sum(parts[t].variances() * scale[t] ** 2 for t in TYPES)
    return v, var


def unit_area(edges, v, var):
    """Density-normalised values and errors (any bin widths)."""
    norm = v.sum()
    widths = np.diff(edges)
    if norm <= 0:
        return v * 0, np.sqrt(var) * 0
    return v / norm / widths, np.sqrt(var) / norm / widths


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="make_histograms.py --signal-file-id-min output")
    p.add_argument("--label", default=None,
                   help="study name for titles (default: the file stem)")
    p.add_argument("--mu", type=float, nargs="+",
                   default=[0, 1, 3, 10, 30, 100],
                   help="signal strengths to draw; 0 is always included as "
                        "the reference (default: %(default)s)")
    p.add_argument("--psignal-cut", type=float, default=None, metavar="P",
                   help="PairNet spectra at pSignal >= P (a bin edge)")
    p.add_argument("--rebin", type=int, default=1, metavar="N")
    p.add_argument("--models", nargs="+", default=None, metavar="MODEL")
    p.add_argument("--ratio-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--log", action="store_true", help="log y")
    p.add_argument("-o", "--outdir", default=".")
    p.add_argument("--format", default="png", choices=("pdf", "png", "svg"))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    data, split = load(args.input, args.psignal_cut)
    power = split["weight_power"]
    label = args.label or Path(args.input).stem
    mus = sorted(set([0.0] + list(args.mu)))
    models = args.models or sorted(data)
    unknown = sorted(set(models) - set(data))
    if unknown:
        sys.exit(f"model(s) {unknown} not in {args.input}; have {sorted(data)}")
    if args.rebin > 1:
        data = {m: {t: h[::hist.rebin(args.rebin)] for t, h in d.items()}
                for m, d in data.items()}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{label}: fill weight xs_weight^{power:g}, so SQ scales as "
          f"mu^{power:g} and SS as mu^{2 * power:g}"
          + (f" | pSignal >= {args.psignal_cut:g}"
             if args.psignal_cut is not None else ""))
    print(f"\n{'model':<20} {'mu':>8} {'signal frac':>12} {'chi2/ndf vs mu=0':>18}")

    for model in models:
        parts = data[model]
        edges = parts["QQ"].axes[0].edges
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
        ref = None
        for i, mu in enumerate(mus):
            v, var = combine(parts, mu, power)
            d, e = unit_area(edges, v, var)
            colour = CMAP(i / max(len(mus) - 1, 1))
            ax.stairs(d, edges, color=colour, label=f"$\\mu$ = {mu:g}")
            drawn = (edges, d, e)
            sig = v.sum() - parts["QQ"].values().sum()
            frac = sig / v.sum() if v.sum() > 0 else 0.0
            if ref is None:
                ref = drawn
                pad = lambda a: np.append(a, a[-1])
                ax.fill_between(edges, pad(d - e), pad(d + e), step="post",
                                color=colour, alpha=0.2, linewidth=0)
                chi2_s = "-"
            else:
                draw_ratio(ax_ratio, ref, drawn, colour)
                used = (ref[1] > 0) & (ref[2] > 0)
                chi2 = float(np.sum((d[used] - ref[1][used]) ** 2
                                    / ref[2][used] ** 2))
                chi2_s = f"{chi2:.1f}/{int(used.sum()) - 1}"
            print(f"{model:<20} {mu:>8g} {frac:>12.3e} {chi2_s:>18}")
        if args.ratio_range:
            ax_ratio.set_ylim(*args.ratio_range)
        if args.log:
            ax.set_yscale("log")
        ax.set_ylabel("density")
        ax_ratio.set_ylabel("ratio to\n$\\mu$ = 0", fontsize=9)
        ax_ratio.set_xlabel("tri-jet invariant mass [GeV]")
        ax.set_title(f"{label}: {model}"
                     + (f"  (pSignal >= {args.psignal_cut:g})"
                        if args.psignal_cut is not None else ""))
        ax.legend(fontsize=8)
        out = outdir / f"scan_{model}.{args.format}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"\nfigures in {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
