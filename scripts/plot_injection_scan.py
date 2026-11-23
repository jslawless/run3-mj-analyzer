#!/usr/bin/env python3
"""plot_injection_scan.py - how much injected signal moves the mixed background.

Input is one ``make_histograms.py --signal-file-id-min`` output for one
signal-injection study (signal hemispheres added to the mixing pool). Make it
without a pSignal cut (``--psignal-cut -1``). Its histograms are split by pair
type: ``h_<obs>_QQ`` (both halves QCD), ``_SQ`` (one half signal) and ``_SS``
(both signal), all filled at the nominal signal cross section. Scaling the
signal cross section by a strength mu scales each signal half's xs_weight
factor by mu, and the fill weight is ``xs_weight ** p`` (p from the file's
``pair_split``; 0.5 for sqrt weights), so the mixed background at strength mu is

    QQ + mu^p * SQ + mu^(2p) * SS

The pairing itself does not depend on the weights, so one set of histograms
covers every mu.

**Duplicated-signal studies (--mu-scale).** In an xK study every signal
hemisphere appears K times, and the file table already gives each copy 1/K of
the weight. Under sqrt filling that does not keep the normalisation: there are
~K times as many SQ pairs, each filled with sqrt(1/K) of the weight, so SQ comes
out ~sqrt(K) too large (and SS ~K). Applying mu/K instead matches the x1 study
at mu for both terms: K * sqrt(mu/K * 1/K) = sqrt(mu). So pass
``--mu-scale 0.25`` for x4: every mu given with ``--mu`` (and every label) is
the x1-equivalent strength, and mu * 0.25 is what is applied. The printed
signal fractions let you check the x1/x4 equivalence directly.

Observables: ``mass_<Model>`` (average tri-jet candidate mass, one per model),
``ht`` and ``leadjet_pt``. Outputs, one PDF each, in ``--outdir``:

  scan_<obs>.pdf       unit-area shape at every mu, ratio to mu = 0. The mu = 0
                       curve carries its statistical band, in both panels; the
                       nominal cross section (mu = 1) is drawn thick and black.
  decomp_<obs>.pdf     QQ and SQ at mu = 1 with their true normalisation (log
                       y), raw entry counts in the legend, and the per-bin
                       signal-containing fraction below. SS is quoted in the
                       legend and only drawn if it has --ss-min-entries or more.
  summary_chi2.pdf     chi2/ndf of each mu shape against mu = 0, vs mu, one
                       curve per observable.
  summary_sigfrac.pdf  signal-containing fraction of the weighted sample vs mu.
  scan_table.json      the numbers behind both summaries.
  sqcompare_<obs>.pdf  only with --compare-sq OTHER: this file's SQ shape vs
                       OTHER's (e.g. x1 vs x4), unit area, ratio below.
  noninj_<obs>.pdf     only with --reference FILE: the non-injected mixed
                       sample (a plain make_histograms.py output, no pair
                       split) against this study's mu = 0, i.e. its QQ pairs.
                       Ideally identical; adding signal rows to the index
                       changes the matcher's draw sequence and uses up some
                       QCD hemispheres on signal partners, so the QQ pairings
                       differ. Absolute weighted entries (not unit area),
                       ratio below.

chi2/ndf uses the mu = 0 statistical errors only. The mu shapes share all of
their QQ events with mu = 0, so the difference is purely the signal-containing
pairs, and the mu = 0 error is the precision the background estimate itself
has. Read it as "would this shift be visible over the background's own stat
noise", not as a p-value (bins are also correlated, since each parent event
supplies both of its hemispheres).

    python scripts/plot_injection_scan.py hists_inject_M300_x1.root \\
        --label "M300 x1" --rebin 2 -o plots/M300_x1/
    python scripts/plot_injection_scan.py hists_inject_M300_x4.root \\
        --label "M300 x4" --mu-scale 0.25 --rebin 2 -o plots/M300_x4/ \\
        --compare-sq hists_inject_M300_x1.root --compare-label "M300 x1" \\
        --reference hists_mixed_noninjected.root
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

TYPES = ("QQ", "SQ", "SS")
HIST_KEY_RE = re.compile(r"^h_(.+)_QQ$")
#: observables never scanned: pSignal itself, and its 2D companion
SKIP_RE = re.compile(r"^psignal")
DEFAULT_MU = [0, 1, 3, 10, 100, 1000, 10000]
#: viridis for the scanned mu values, so increasing mu reads as a sequence
CMAP = plt.get_cmap("viridis")
REF_COLOUR = "0.45"
NOMINAL_COLOUR = "black"


def xlabel(obs):
    if obs.startswith("mass_"):
        return "average tri-jet candidate mass [GeV]"
    return {"ht": "$H_T$ [GeV]",
            "leadjet_pt": "leading jet $p_T$ [GeV]"}.get(obs, obs)


def raw_count(split, obs, t):
    """Unweighted entries of pair type ``t`` behind ``obs``, or None."""
    rc = split.get("raw_counts", {}).get(t)
    if rc is None:
        return None
    if obs.startswith("mass_"):
        return rc["mass"].get(obs[len("mass_"):])
    return rc["events"]


def load(path):
    """``({obs: {type: hist}}, pair_split)`` from one split file."""
    with uproot.open(path) as f:
        if "pair_split" not in f:
            raise SystemExit(f"{path} has no pair_split - make it with "
                             "make_histograms.py --signal-file-id-min")
        split = json.loads(str(f["pair_split"]))
        keys = set(f.keys(cycle=False))
        out = {}
        for key in sorted(keys):
            m = HIST_KEY_RE.match(key)
            if not m or SKIP_RE.match(m.group(1)):
                continue
            obs = m.group(1)
            if f[key].to_hist().ndim != 1:
                continue
            out[obs] = {t: f[f"h_{obs}_{t}"].to_hist() for t in TYPES}
    if not out:
        raise SystemExit(f"no h_<obs>_QQ histograms in {path}")
    cut = split.get("psignal_cut")
    if cut is not None and cut >= 0:
        print(f"[warn] {path} was made with a pSignal cut ({cut:g}) on the "
              "PairNet masses; remake it with --psignal-cut -1 for the "
              "uncut spectra")
    return out, split


def combine(parts, mu, p, types=TYPES):
    """``(values, variances)`` of QQ + mu^p SQ + mu^(2p) SS (over ``types``)."""
    scale = {"QQ": 1.0, "SQ": mu ** p, "SS": mu ** (2 * p)}
    v = sum(parts[t].values() * scale[t] for t in types)
    var = sum(parts[t].variances() * scale[t] ** 2 for t in types)
    return v, var


def unit_area(edges, v, var):
    """Density-normalised values and errors (any bin widths)."""
    norm = v.sum()
    widths = np.diff(edges)
    if norm <= 0:
        return v * 0, np.sqrt(var) * 0
    return v / norm / widths, np.sqrt(var) / norm / widths


def pad(a):
    return np.append(a, a[-1])


def style(mu, i, n):
    """(colour, linewidth, zorder) for the i-th of n nonzero mu values."""
    if mu == 1:
        return NOMINAL_COLOUR, 2.4, 5
    # stop short of viridis' pale yellow end, which vanishes on white
    return CMAP(0.85 * i / max(n - 1, 1)), 1.2, 3


def mu_label(mu):
    if mu == 1:
        return "$\\mu$ = 1 ($\\sigma_{nominal}$)"
    if mu >= 1000:
        return f"$\\mu$ = $10^{{{int(round(np.log10(mu)))}}}$" \
            if np.isclose(np.log10(mu), round(np.log10(mu))) else f"$\\mu$ = {mu:g}"
    return f"$\\mu$ = {mu:g}"


def scan_plot(obs, parts, mus, power, mu_scale, label, outpath, logy):
    """One scan_<obs> figure; returns {mu: (signal fraction, chi2, ndf)}."""
    edges = parts["QQ"].axes[0].edges
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
    v0, var0 = combine(parts, 0.0, power)
    d0, e0 = unit_area(edges, v0, var0)
    ax.stairs(d0, edges, color=REF_COLOUR, linewidth=1.2, zorder=4,
              label="$\\mu$ = 0 (QCD only)")
    ax.fill_between(edges, pad(d0 - e0), pad(d0 + e0), step="post",
                    color=REF_COLOUR, alpha=0.25, linewidth=0)
    # Ratio panel: the mu = 0 relative error as a band around 1, and each mu
    # as a bare line - its error is mostly the same QQ fluctuation, so
    # propagating both as independent would double count.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel0 = np.where(d0 > 0, e0 / d0, np.nan)
    axr.fill_between(edges, pad(1 - rel0), pad(1 + rel0), step="post",
                     color=REF_COLOUR, alpha=0.25, linewidth=0)
    axr.axhline(1.0, color=REF_COLOUR, linewidth=0.8, linestyle="--")

    used = (d0 > 0) & (e0 > 0)
    results = {0.0: (0.0, 0.0, int(used.sum()) - 1)}
    nonzero = [m for m in mus if m > 0]
    ratios = []
    for i, mu in enumerate(nonzero):
        v, var = combine(parts, mu * mu_scale, power)
        d, _ = unit_area(edges, v, var)
        colour, lw, z = style(mu, i, len(nonzero))
        ax.stairs(d, edges, color=colour, linewidth=lw, zorder=z,
                  label=mu_label(mu))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d0 > 0, d / d0, np.nan)
        axr.stairs(r, edges, color=colour, linewidth=lw, zorder=z)
        ratios.append(r)
        frac = (v.sum() - v0.sum()) / v.sum() if v.sum() > 0 else 0.0
        chi2 = float(np.sum((d[used] - d0[used]) ** 2 / e0[used] ** 2))
        results[mu] = (frac, chi2, int(used.sum()) - 1)

    # Ratio limits from the bulk, so one wild tail bin doesn't squash the rest.
    finite = np.concatenate([r[np.isfinite(r)] for r in ratios]
                            + [(1 + rel0)[np.isfinite(rel0)],
                               (1 - rel0)[np.isfinite(rel0)]])
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        padding = 0.1 * max(hi - lo, 0.05)
        axr.set_ylim(min(lo - padding, 0.95), max(hi + padding, 1.05))
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("density")
    axr.set_ylabel("ratio to\n$\\mu$ = 0", fontsize=9)
    axr.set_xlabel(xlabel(obs))
    ax.set_title(f"{label}: {obs}" + (f"   (applied $\\mu$ = {mu_scale:g} "
                                      "$\\times$ label)" if mu_scale != 1
                                      else ""), fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return results


def decomp_plot(obs, parts, split, power, mu_scale, label, outpath,
                ss_min):
    """decomp_<obs>: QQ and SQ (and SS if populated) at mu = 1."""
    edges = parts["QQ"].axes[0].edges
    mu = 1.0 * mu_scale
    scale = {"QQ": 1.0, "SQ": mu ** power, "SS": mu ** (2 * power)}
    colours = {"QQ": "tab:blue", "SQ": "tab:red", "SS": "tab:green"}
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
    total = np.zeros(len(edges) - 1)
    for t in TYPES:
        n = raw_count(split, obs, t)
        v = parts[t].values() * scale[t]
        e = np.sqrt(parts[t].variances()) * scale[t]
        total += v
        n_s = f"{n:,} entries" if n is not None else "entries n/a"
        if t == "SS" and (n is None or n < ss_min):
            ax.plot([], [], " ", label=f"SS: {n_s} (not drawn)")
            continue
        ax.stairs(v, edges, color=colours[t], linewidth=1.4,
                  label=f"{t}: {n_s}, $\\Sigma w$ = {v.sum():.3g}")
        ax.fill_between(edges, pad(np.maximum(v - e, 0)), pad(v + e),
                        step="post", color=colours[t], alpha=0.2,
                        linewidth=0)
    sig = total - parts["QQ"].values()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(total > 0, sig / total, np.nan)
    axr.stairs(f, edges, color=colours["SQ"])
    axr.set_ylabel("signal-pair\nfraction", fontsize=9)
    axr.set_yscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("weighted entries / bin")
    axr.set_xlabel(xlabel(obs))
    ax.set_title(f"{label}: {obs} by pair type at $\\mu$ = 1 "
                 "($\\sigma_{nominal}$)", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def sqcompare_plot(obs, mine, other, label, other_label, outpath):
    """sqcompare_<obs>: unit-area SQ of this file vs another study's."""
    edges = mine["SQ"].axes[0].edges
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
    curves = []
    for parts, lab, colour in ((other, other_label, "tab:blue"),
                               (mine, label, "tab:red")):
        d, e = unit_area(edges, parts["SQ"].values(), parts["SQ"].variances())
        ax.stairs(d, edges, color=colour, label=f"{lab} SQ")
        ax.fill_between(edges, pad(d - e), pad(d + e), step="post",
                        color=colour, alpha=0.2, linewidth=0)
        curves.append((d, e))
    (d0, e0), (d1, e1) = curves
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d0 > 0, d1 / d0, np.nan)
        # Independent samples (different signal copies, different partners),
        # so the errors add in quadrature here.
        er = np.abs(r) * np.sqrt(np.where(d1 > 0, (e1 / d1) ** 2, 0)
                                 + np.where(d0 > 0, (e0 / d0) ** 2, 0))
    axr.axhline(1.0, color=REF_COLOUR, linewidth=0.8, linestyle="--")
    axr.stairs(r, edges, color="tab:red")
    axr.fill_between(edges, pad(r - er), pad(r + er), step="post",
                     color="tab:red", alpha=0.2, linewidth=0)
    axr.set_ylim(0, 2)
    axr.set_ylabel(f"{label} /\n{other_label}", fontsize=9)
    axr.set_xlabel(xlabel(obs))
    ax.set_ylabel("density")
    ax.set_title(f"SQ shape: {label} vs {other_label} - {obs}", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def load_reference(path):
    """``{obs: hist}`` from a plain (non-split) make_histograms.py output."""
    with uproot.open(path) as f:
        if "pair_split" in f:
            raise SystemExit(f"{path} is a pair-split file; --reference wants "
                             "the non-injected mixed sample")
        out = {}
        for key in f.keys(cycle=False):
            if key.startswith("h_") and not key.startswith("h_psignal"):
                h = f[key].to_hist()
                if h.ndim == 1:
                    out[key[2:]] = h
        # A pSignal cut shows up as "pSignal >= P [Model]" cutflow bins.
        if "cutflow" in f:
            labels = f["cutflow"].axis().labels() or []
            if any(str(lab).startswith("pSignal") for lab in labels):
                print(f"[warn] {path} was made with a pSignal cut on the "
                      "PairNet masses; remake it with --psignal-cut -1")
    return out


def noninj_plot(obs, parts, ref, label, ref_label, outpath, logy):
    """noninj_<obs>: non-injected mixed sample vs this study's mu = 0 (QQ).

    The two share most hemispheres but not their pairings, so they are
    neither independent nor identical: the ratio errors are added in
    quadrature (as if independent), which overstates them somewhat. The
    chi2/ndf printed alongside uses the same errors.
    """
    edges = parts["QQ"].axes[0].edges
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05})
    q, qvar = parts["QQ"].values(), parts["QQ"].variances()
    r_, rvar = ref.values(), ref.variances()
    for v, var, lab, colour in ((r_, rvar, ref_label, "tab:blue"),
                                (q, qvar, f"{label} $\\mu$ = 0 (QQ)",
                                 REF_COLOUR)):
        e = np.sqrt(var)
        ax.stairs(v, edges, color=colour, linewidth=1.4,
                  label=f"{lab}: $\\Sigma w$ = {v.sum():.4g}")
        ax.fill_between(edges, pad(np.maximum(v - e, 0)), pad(v + e),
                        step="post", color=colour, alpha=0.2, linewidth=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(r_ > 0, q / r_, np.nan)
        er = np.abs(r) * np.sqrt(np.where(q > 0, qvar / q ** 2, 0)
                                 + np.where(r_ > 0, rvar / r_ ** 2, 0))
    axr.axhline(1.0, color=REF_COLOUR, linewidth=0.8, linestyle="--")
    axr.stairs(r, edges, color="black")
    axr.fill_between(edges, pad(r - er), pad(r + er), step="post",
                     color="black", alpha=0.15, linewidth=0)
    finite = r[np.isfinite(r)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        padding = 0.1 * max(hi - lo, 0.05)
        axr.set_ylim(min(lo - padding, 0.95), max(hi + padding, 1.05))
    axr.set_ylabel("$\\mu$ = 0 /\nnon-injected", fontsize=9)
    axr.set_xlabel(xlabel(obs))
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("weighted entries / bin")
    used = (r_ > 0) & (q > 0)
    chi2 = float(np.sum((q[used] - r_[used]) ** 2
                        / (qvar[used] + rvar[used])))
    ndf = int(used.sum())
    ax.set_title(f"{label} $\\mu$ = 0 vs non-injected mixed: {obs}\n"
                 f"yield ratio {q.sum() / r_.sum():.4f}, "
                 f"$\\chi^2$/ndf = {chi2:.1f}/{ndf}", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return q.sum() / r_.sum(), chi2, ndf


def summary_plots(table, label, outdir, fmt):
    obs_list = sorted(table)
    cmap = plt.get_cmap("tab10")
    for what, ylab, fname, logy in (
            ("chi2", "$\\chi^2$/ndf vs $\\mu$ = 0", "summary_chi2", True),
            ("frac", "signal-containing fraction (weighted)",
             "summary_sigfrac", True)):
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, obs in enumerate(obs_list):
            rows = [(mu, r) for mu, r in sorted(table[obs].items()) if mu > 0]
            x = [mu for mu, _ in rows]
            if what == "chi2":
                y = [r["chi2"] / max(r["ndf"], 1) for _, r in rows]
            else:
                y = [r["frac"] for _, r in rows]
            ax.plot(x, y, marker="o", color=cmap(i % 10), label=obs)
        ax.axvline(1.0, color=NOMINAL_COLOUR, linewidth=1.5, linestyle=":",
                   label="$\\sigma_{nominal}$ ($\\mu$ = 1)")
        if what == "chi2":
            ax.axhline(1.0, color=REF_COLOUR, linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("$\\mu$ (x1-equivalent signal strength)")
        ax.set_ylabel(ylab)
        ax.set_title(label)
        ax.legend(fontsize=8)
        fig.savefig(outdir / f"{fname}.{fmt}", bbox_inches="tight")
        plt.close(fig)


def rebinned(data, n):
    if n <= 1:
        return data
    return {o: {t: h[::hist.rebin(n)] for t, h in d.items()}
            for o, d in data.items()}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="make_histograms.py --signal-file-id-min output")
    p.add_argument("--label", default=None,
                   help="study name for titles (default: the file stem)")
    p.add_argument("--mu", type=float, nargs="+", default=DEFAULT_MU,
                   help="x1-equivalent signal strengths; 0 is always "
                        "included as the reference (default: %(default)s)")
    p.add_argument("--mu-scale", type=float, default=1.0, metavar="S",
                   help="applied strength = S * mu; 1/K for an xK duplicated-"
                        "signal study, e.g. 0.25 for x4 (default: 1)")
    p.add_argument("--rebin", type=int, default=1, metavar="N")
    p.add_argument("--obs", nargs="+", default=None, metavar="OBS",
                   help="observables to plot, e.g. mass_PairNetV0_9 ht "
                        "(default: all)")
    p.add_argument("--log", action="store_true", help="log y on the scans")
    p.add_argument("--ss-min-entries", type=int, default=20, metavar="N",
                   help="draw SS in the decomposition only with at least N "
                        "raw entries (default: %(default)s)")
    p.add_argument("--compare-sq", default=None, metavar="FILE",
                   help="another study's split file: also draw this file's SQ "
                        "shape against that one's")
    p.add_argument("--compare-label", default=None)
    p.add_argument("--reference", default=None, metavar="FILE",
                   help="non-injected mixed sample (plain make_histograms.py "
                        "output, same cuts): also draw it against mu = 0")
    p.add_argument("--reference-label", default="non-injected mixed")
    p.add_argument("-o", "--outdir", default=".")
    p.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    args = p.parse_args(argv)

    data, split = load(args.input)
    data = rebinned(data, args.rebin)
    power = split["weight_power"]
    label = args.label or Path(args.input).stem
    mus = sorted(set([0.0] + list(args.mu)))
    obs_list = args.obs or sorted(data)
    unknown = sorted(set(obs_list) - set(data))
    if unknown:
        sys.exit(f"observable(s) {unknown} not in {args.input}; "
                 f"have {sorted(data)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{label}: fill weight xs_weight^{power:g}, so SQ scales as "
          f"mu^{power:g} and SS as mu^{2 * power:g}; applied mu = "
          f"{args.mu_scale:g} x label")
    rc = split.get("raw_counts")
    if rc:
        print("raw entries after event cuts: "
              + ", ".join(f"{t} {rc[t]['events']:,}" for t in TYPES))
    print(f"\n{'observable':<26} {'mu':>8} {'signal frac':>12} "
          f"{'chi2/ndf vs mu=0':>18}")

    table = {}
    for obs in obs_list:
        res = scan_plot(obs, data[obs], mus, power, args.mu_scale, label,
                        outdir / f"scan_{obs}.{args.format}", args.log)
        table[obs] = {mu: {"frac": f, "chi2": c, "ndf": n}
                      for mu, (f, c, n) in res.items()}
        for mu, (f, c, n) in res.items():
            chi2_s = "-" if mu == 0 else f"{c:.1f}/{n}"
            print(f"{obs:<26} {mu:>8g} {f:>12.3e} {chi2_s:>18}")
        decomp_plot(obs, data[obs], split, power, args.mu_scale, label,
                    outdir / f"decomp_{obs}.{args.format}",
                    args.ss_min_entries)

    summary_plots(table, label, outdir, args.format)
    with open(outdir / "scan_table.json", "w") as f:
        json.dump({"label": label, "mu_scale": args.mu_scale,
                   "weight_power": power,
                   "table": {o: {f"{mu:g}": r for mu, r in t.items()}
                             for o, t in table.items()}}, f, indent=1)

    if args.compare_sq:
        other, _ = load(args.compare_sq)
        other = rebinned(other, args.rebin)
        olabel = args.compare_label or Path(args.compare_sq).stem
        for obs in obs_list:
            if obs in other:
                sqcompare_plot(obs, data[obs], other[obs], label, olabel,
                               outdir / f"sqcompare_{obs}.{args.format}")
    if args.reference:
        ref = load_reference(args.reference)
        if args.rebin > 1:
            ref = {o: h[::hist.rebin(args.rebin)] for o, h in ref.items()}
        print(f"\n{'observable':<26} {'mu=0 / non-inj yield':>21} "
              f"{'chi2/ndf':>14}")
        for obs in obs_list:
            if obs not in ref:
                print(f"{obs:<26} (not in {args.reference})")
                continue
            yr, c, n = noninj_plot(obs, data[obs], ref[obs], label,
                                   args.reference_label,
                                   outdir / f"noninj_{obs}.{args.format}",
                                   args.log)
            print(f"{obs:<26} {yr:>21.4f} {f'{c:.1f}/{n}':>14}")
    print(f"\nfigures in {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
