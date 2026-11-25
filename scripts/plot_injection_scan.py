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
  summary_chi2.pdf     Delta chi2 of each mu shape against mu = 0, vs mu, one
                       curve per observable, with lines at 4 and 25 (2 and 5
                       sigma for a shift of known shape).
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

**Reading the chi2.** It uses the mu = 0 statistical errors only. The mu
shapes share all of their QQ events with mu = 0, so the difference contains no
noise: it is purely the shift the signal-containing pairs cause. The sum is
therefore not an ordinary goodness-of-fit chi2 (which would be ~ndf from noise
alone) but Delta chi2, the noncentrality a real comparison would gain on top
of ~ndf. Two conversions are printed:

  Z_tmpl  = sqrt(Delta chi2): the significance of a fit for this particular
            shift shape (a one-parameter template fit).
  Z_gen   = the median significance of a generic chi2 test that assumes no
            shape: median of noncentral chi2(ndf, Delta chi2), as a p-value
            under chi2(ndf). Always below Z_tmpl, and binning dependent.

Delta chi2 is noise-corrected: the signal pairs are a small MC sample (a few
hundred SQ events per PairNet), and their own fluctuations add, on average,
sum(var_signal / sigma0^2) to the raw sum of pull^2 - for the masses at 30 GeV
bins about half of it. That expectation is subtracted; scan_table.json keeps
the raw sum and the subtracted part.

Both are against the mixed-MC statistics, not data, and both are optimistic:
bins are correlated (each parent event supplies both of its hemispheres), so
the mu = 0 errors are underestimated.

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
#: set by --mu95: {mu: (legend label, colour)} for the excluded-limit multiples
MU95_NAMES = {}
MU95_COLOURS = ("tab:blue", "tab:red", "tab:green")


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


def style(mu, i, n, highlight=True):
    """(colour, linewidth, zorder) for the i-th of n nonzero mu values.

    With ``highlight`` off every curve gets the same width; a short list of mu
    values then uses distinct tab10 colours rather than a viridis ramp. With
    --mu95, the limit multiples get fixed colours and mu = 1 a thin black line."""
    if MU95_NAMES:
        if mu in MU95_NAMES:
            return MU95_NAMES[mu][1], 1.4, 4
        if mu == 1:
            return NOMINAL_COLOUR, 1.4, 5
    if not highlight:
        if n <= 4:
            return ("tab:blue", "tab:red", "tab:green", "tab:purple")[i], 1.4, 3
        return CMAP(0.85 * i / max(n - 1, 1)), 1.4, 3
    if mu == 1:
        return NOMINAL_COLOUR, 2.4, 5
    # stop short of viridis' pale yellow end, which vanishes on white
    return CMAP(0.85 * i / max(n - 1, 1)), 1.2, 3


def mu_label(mu):
    if mu in MU95_NAMES:
        return MU95_NAMES[mu][0]
    if mu == 1:
        return "$\\mu$ = 1 ($\\sigma_{nominal}$)"
    if mu >= 1000:
        return f"$\\mu$ = $10^{{{int(round(np.log10(mu)))}}}$" \
            if np.isclose(np.log10(mu), round(np.log10(mu))) else f"$\\mu$ = {mu:g}"
    return f"$\\mu$ = {mu:g}"


def scan_plot(obs, parts, mus, power, mu_scale, label, outpath, logy,
              highlight=True, xlim=None):
    """One scan_<obs> figure; returns {mu: (signal fraction, Delta chi2
    (noise-corrected), ndf, raw sum of pull^2, its signal-MC noise part)}."""
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
    results = {0.0: (0.0, 0.0, int(used.sum()) - 1, 0.0, 0.0)}
    nonzero = [m for m in mus if m > 0]
    ratios = []
    for i, mu in enumerate(nonzero):
        v, var = combine(parts, mu * mu_scale, power)
        d, _ = unit_area(edges, v, var)
        colour, lw, z = style(mu, i, len(nonzero), highlight)
        ax.stairs(d, edges, color=colour, linewidth=lw, zorder=z,
                  label=mu_label(mu))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d0 > 0, d / d0, np.nan)
        axr.stairs(r, edges, color=colour, linewidth=lw, zorder=z)
        ratios.append(r)
        frac = (v.sum() - v0.sum()) / v.sum() if v.sum() > 0 else 0.0
        raw = float(np.sum((d[used] - d0[used]) ** 2 / e0[used] ** 2))
        # The signal pairs are a finite MC sample too: their own fluctuations
        # add sum(var_signal / sigma0^2) to the raw sum on average, which is
        # not a shift. Subtract that expectation (density units, like d).
        app = mu * mu_scale
        var_sig = (app ** (2 * power) * parts["SQ"].variances()
                   + app ** (4 * power) * parts["SS"].variances())
        var_sig = var_sig / (v.sum() * np.diff(edges)) ** 2
        noise = float(np.sum(var_sig[used] / e0[used] ** 2))
        results[mu] = (frac, max(raw - noise, 0.0), int(used.sum()) - 1,
                       raw, noise)

    # Ratio limits from the bulk, so one wild tail bin doesn't squash the rest.
    # With xlim, only the bins inside the window count (for the y range too).
    centres = 0.5 * (edges[:-1] + edges[1:])
    inwin = np.ones_like(centres, dtype=bool) if xlim is None else \
        (centres >= xlim[0]) & (centres <= xlim[1])
    finite = np.concatenate([r[inwin & np.isfinite(r)] for r in ratios]
                            + [(1 + rel0)[inwin & np.isfinite(rel0)],
                               (1 - rel0)[inwin & np.isfinite(rel0)]])
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        padding = 0.1 * max(hi - lo, 0.05)
        floor = 0.005 if xlim is not None else 0.05
        axr.set_ylim(min(lo - padding, 1 - floor), max(hi + padding, 1 + floor))
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        axr.set_xlim(*xlim)
        vis = d0[inwin & (d0 > 0)]
        if vis.size:
            ax.set_ylim(*((vis.min() / 2, vis.max() * 2) if logy
                          else (0, vis.max() * 1.15)))
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


def sqcompare_plot(obs, mine, other, label, other_label, outpath,
                   logy=False):
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
    if logy:
        ax.set_yscale("log")
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


def z_generic(dchi2, ndf):
    """Median significance of a generic shape test (no assumed shift shape).

    A real comparison against an independent sample would give a chi2 drawn
    from a noncentral chi2(ndf, lambda = dchi2); its median, read as a p-value
    under the no-signal chi2(ndf), converted to Gaussian sigma. None without
    scipy.
    """
    try:
        from scipy import stats
    except ImportError:
        return None
    if dchi2 <= 0 or ndf <= 0:
        return 0.0
    median = stats.ncx2.median(ndf, dchi2)
    return float(stats.norm.isf(stats.chi2.sf(median, ndf)))


def mu_crossing(mus, dchi2, level):
    """Smallest mu where dchi2 reaches ``level`` (log-log interpolation;
    below the smallest scanned mu, extrapolated as Delta chi2 ~ mu)."""
    pts = [(m, c) for m, c in zip(mus, dchi2) if m > 0 and c > 0]
    for (m0, c0), (m1, c1) in zip(pts, pts[1:]):
        if c0 < level <= c1:
            t = (np.log(level) - np.log(c0)) / (np.log(c1) - np.log(c0))
            return float(np.exp(np.log(m0) + t * (np.log(m1) - np.log(m0))))
    if pts and pts[0][1] >= level:
        # Already above at the smallest mu: extrapolate down along
        # Delta chi2 ~ mu (the shift grows as sqrt(mu) when SQ dominates).
        return float(pts[0][0] * level / pts[0][1])
    return None


#: Delta chi2 reference lines: sqrt(Delta chi2) = Z for a known-shape shift
DCHI2_LINES = ((4.0, "2$\\sigma$"), (25.0, "5$\\sigma$"))


def summary_plots(table, label, outdir, fmt):
    """summary_chi2 (Delta chi2 vs mu) and summary_sigfrac."""
    obs_list = sorted(table)
    cmap = plt.get_cmap("tab10")
    mus_all = [mu for t in table.values() for mu in t if mu > 0]
    for what, ylab, fname in (
            ("chi2", "$\\Delta\\chi^2$ of the $\\mu$ shape vs $\\mu$ = 0",
             "summary_chi2"),
            ("frac", "signal-containing fraction (weighted)",
             "summary_sigfrac")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, obs in enumerate(obs_list):
            rows = [(mu, r) for mu, r in sorted(table[obs].items()) if mu > 0]
            x = [mu for mu, _ in rows]
            y = [r["chi2"] if what == "chi2" else r["frac"] for _, r in rows]
            ax.plot(x, y, marker="o", color=cmap(i % 10), label=obs)
        ax.axvline(1.0, color=NOMINAL_COLOUR, linewidth=1.5, linestyle=":",
                   label="$\\sigma_{nominal}$ ($\\mu$ = 1)")
        for mu, (name, colour) in MU95_NAMES.items():
            ax.axvline(mu, color=colour, linewidth=1.2, linestyle=":",
                       label=name)
        if what == "chi2":
            for level, name in DCHI2_LINES:
                ax.axhline(level, color=REF_COLOUR, linewidth=0.9,
                           linestyle="--")
                ax.text(max(mus_all) * 1.05, level * 1.12,
                        f"{name} (known shape)", fontsize=8, ha="right",
                        color=REF_COLOUR)
        ax.set_xscale("log")
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
    p.add_argument("--mu95", type=float, default=None, metavar="MU",
                   help="x1-equivalent excluded (95%% CL) signal strength: "
                        "scan mu = MU, 10 MU, 100 MU and 1 (theory) instead "
                        "of --mu, with those labels and fixed colours, and "
                        "mark them on summary_chi2")
    p.add_argument("--mu95-label", default="$\\mu_{95}^{obs}$",
                   help="legend name for MU (default: %(default)s)")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="zoom the scan plots to LO..HI (y ranges follow the "
                        "visible bins)")
    p.add_argument("--rebin", type=int, default=1, metavar="N")
    p.add_argument("--obs", nargs="+", default=None, metavar="OBS",
                   help="observables to plot, e.g. mass_PairNetV0_9 ht "
                        "(default: all)")
    p.add_argument("--log", action="store_true",
                   help="log y on the scan, noninj and sqcompare plots (the "
                        "decompositions are always log)")
    p.add_argument("--no-highlight", action="store_true",
                   help="draw mu = 1 like every other mu (no thick black line)")
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
    if args.mu95:
        for k, colour in zip((1, 10, 100), MU95_COLOURS):
            mu = args.mu95 * k
            name = args.mu95_label if k == 1 else f"{k} $\\times$ {args.mu95_label}"
            MU95_NAMES[mu] = (f"{name} = {mu:.3g}", colour)
        mus = sorted({0.0, 1.0, *MU95_NAMES})
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
          f"{'dchi2/ndf':>18} {'Z_tmpl':>7} {'Z_gen':>7}")

    table = {}
    for obs in obs_list:
        res = scan_plot(obs, data[obs], mus, power, args.mu_scale, label,
                        outdir / f"scan_{obs}.{args.format}", args.log,
                        not args.no_highlight, args.xlim)
        table[obs] = {mu: {"frac": f, "chi2": c, "ndf": n,
                           "chi2_raw": raw, "chi2_signal_noise": nz,
                           "z_template": float(np.sqrt(c)),
                           "z_generic": z_generic(c, n)}
                      for mu, (f, c, n, raw, nz) in res.items()}
        for mu, (f, c, n, raw, nz) in res.items():
            r = table[obs][mu]
            chi2_s = "-" if mu == 0 else f"{c:.1f}/{n}"
            zg = "-" if mu == 0 or r["z_generic"] is None \
                else f"{r['z_generic']:.1f}"
            zt = "-" if mu == 0 else f"{r['z_template']:.1f}"
            print(f"{obs:<26} {mu:>8g} {f:>12.3e} {chi2_s:>18} {zt:>7} {zg:>7}")
        decomp_plot(obs, data[obs], split, power, args.mu_scale, label,
                    outdir / f"decomp_{obs}.{args.format}",
                    args.ss_min_entries)

    summary_plots(table, label, outdir, args.format)
    crossings = {}
    print(f"\n{'observable':<26} {'mu at dchi2=4 (2 sigma)':>24} "
          f"{'mu at dchi2=25 (5 sigma)':>25}")
    for obs in obs_list:
        mus_sorted = sorted(m for m in table[obs] if m > 0)
        dc = [table[obs][m]["chi2"] for m in mus_sorted]
        c2, c5 = (mu_crossing(mus_sorted, dc, lv) for lv, _ in DCHI2_LINES)
        crossings[obs] = {"mu_2sigma": c2, "mu_5sigma": c5}
        fmt_mu = lambda v: "-" if v is None else f"{v:.3g}"
        print(f"{obs:<26} {fmt_mu(c2):>24} {fmt_mu(c5):>25}")
    with open(outdir / "scan_table.json", "w") as f:
        json.dump({"label": label, "mu_scale": args.mu_scale,
                   "weight_power": power,
                   "crossings": crossings,
                   "table": {o: {f"{mu:g}": r for mu, r in t.items()}
                             for o, t in table.items()}}, f, indent=1)

    if args.compare_sq:
        other, _ = load(args.compare_sq)
        other = rebinned(other, args.rebin)
        olabel = args.compare_label or Path(args.compare_sq).stem
        for obs in obs_list:
            if obs in other:
                sqcompare_plot(obs, data[obs], other[obs], label, olabel,
                               outdir / f"sqcompare_{obs}.{args.format}",
                               args.log)
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
