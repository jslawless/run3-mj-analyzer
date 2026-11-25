#!/usr/bin/env python3
"""Figures for chi2_explainer.tex, from the real M300 x1 PairNetV0_9 mass.

    ~/analysis/run3-mj-analyzer/pkg-env/bin/python make_figures.py
"""

import json
import os
from pathlib import Path

import hist
import matplotlib
import numpy as np
import uproot
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("INJECT_WORKDIR",
                           "~/analysis/histograms/inject_26_11_23")).expanduser()
ROOT = WORK / "hists_inject_M300_x1.root"
ROOT_X4 = WORK / "hists_inject_M300_x4.root"
MODEL = "PairNetV0_9"
REBIN = 2
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
BLUE, RED, GREY = "tab:blue", "tab:red", "0.45"
XMAX = 1000.0
plt.rcParams.update({"font.size": 12})


def load(rebin=REBIN, path=ROOT, model=None):
    model = model or MODEL
    with uproot.open(path) as f:
        split = json.loads(str(f["pair_split"]))
        h = {t: f[f"h_mass_{model}_{t}"].to_hist()[::hist.rebin(rebin)]
             for t in ("QQ", "SQ", "SS")}
    return h, split["weight_power"]


def shapes(h, p, mu):
    edges = h["QQ"].axes[0].edges
    w = np.diff(edges)
    v0, var0 = h["QQ"].values(), h["QQ"].variances()
    v = v0 + mu ** p * h["SQ"].values() + mu ** (2 * p) * h["SS"].values()
    d0, e0 = v0 / v0.sum() / w, np.sqrt(var0) / v0.sum() / w
    d = v / v.sum() / w
    return edges, d0, e0, d


def dchi2_of(h, p, mu):
    """(raw sum of pull^2, its expected signal-MC noise part, ndf)."""
    edges, d0, e0, d = shapes(h, p, mu)
    used = (d0 > 0) & (e0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = float(np.sum(((d - d0) / e0)[used] ** 2))
        v = (h["QQ"].values() + mu ** p * h["SQ"].values()
             + mu ** (2 * p) * h["SS"].values())
        var_sig = (mu ** (2 * p) * h["SQ"].variances()
                   + mu ** (4 * p) * h["SS"].variances())
        var_sig = var_sig / (v.sum() * np.diff(edges)) ** 2
        noise = float(np.sum((var_sig / e0 ** 2)[used]))
    return raw, noise, int(used.sum()) - 1


def pad(a):
    return np.append(a, a[-1])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


h, p = load()
edges, d0, e0, d1 = shapes(h, p, 1.0)
centres = 0.5 * (edges[1:] + edges[:-1])
used = (d0 > 0) & (e0 > 0)
pull = np.where(used, (d1 - d0) / np.where(e0 > 0, e0, 1), 0.0)
raw, noise, ndf = dchi2_of(h, p, 1.0)
dchi2 = raw - noise   # what the significance slides use
print(f"M300 x1 {MODEL} mu=1: raw {raw:.2f}, noise {noise:.2f}, "
      f"Delta chi2 {dchi2:.2f}, ndf {ndf}")

# 1. the two shapes, with the mu = 0 band -------------------------------------
fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                              gridspec_kw={"height_ratios": (3, 1),
                                           "hspace": 0.05})
ax.fill_between(edges, pad(d0 - e0), pad(d0 + e0), step="post", color=GREY,
                alpha=0.3, linewidth=0, label="$\\mu$ = 0 stat. error $\\sigma_i$")
ax.stairs(d0, edges, color=GREY, linewidth=1.5, label="$\\mu$ = 0: QCD pairs only")
ax.stairs(d1, edges, color=RED, linewidth=1.5, label="$\\mu$ = 1: with signal pairs")
ax.set_ylabel("fraction of events / GeV")
ax.legend()
with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.where(d0 > 0, e0 / d0, np.nan)
    r = np.where(d0 > 0, d1 / d0, np.nan)
axr.fill_between(edges, pad(1 - rel), pad(1 + rel), step="post", color=GREY,
                 alpha=0.3, linewidth=0)
axr.axhline(1, color=GREY, linestyle="--", linewidth=0.8)
axr.stairs(r, edges, color=RED, linewidth=1.5)
axr.set_ylim(0.9, 1.1)
axr.set_ylabel("ratio")
axr.set_xlabel("average tri-jet mass [GeV]")
axr.set_xlim(0, XMAX)
save(fig, "f1_shapes")

# 2. per-bin pulls --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(centres[used], pull[used], width=np.diff(edges)[used] * 0.9,
       color=np.where(pull[used] > 0, RED, BLUE))
ax.axhspan(-1, 1, color=GREY, alpha=0.15, label="$\\pm 1$: one standard error")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlim(0, XMAX)
ax.set_ylim(-3, 3)
ax.set_xlabel("average tri-jet mass [GeV]")
ax.set_ylabel("pull = (B$_1$ $-$ B$_0$) / $\\sigma$")
ax.legend(loc="upper right")
save(fig, "f2_pulls")

# 3. running sum of pull^2 ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
cum = np.cumsum(np.where(used, pull ** 2, 0.0))
ax.stairs(cum, edges, color=RED, linewidth=2,
          label=f"running $\\Sigma$ pull$^2$ (total {raw:.1f})")
ax.axhline(raw, color=RED, linestyle=":", linewidth=1)
ax.set_xlim(0, edges[-1])
ax.set_xlabel("average tri-jet mass [GeV]  (summing bins left to right)")
ax.set_ylabel("$\\Sigma$ pull$^2$")
ax.legend(loc="lower right")
save(fig, "f3_cumulative")

# 4. what pure noise looks like: one toy, same errors -------------------------
rng = np.random.default_rng(7)
toy_pull = np.where(used, rng.standard_normal(len(d0)), 0.0)
toy_chi2 = float(np.sum(toy_pull[used] ** 2))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pl, title in ((axes[0], toy_pull,
                       f"noise only (a toy): $\\Sigma$ pull$^2$ = {toy_chi2:.0f}"),
                      (axes[1], pull,
                       f"our $\\mu$ = 1 shift: $\\Sigma$ pull$^2$ = {raw:.0f}")):
    ax.bar(centres[used], pl[used], width=np.diff(edges)[used] * 0.9,
           color=np.where(pl[used] > 0, RED, BLUE))
    ax.axhspan(-1, 1, color=GREY, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-3, 3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("average tri-jet mass [GeV]")
axes[0].set_ylabel("pull")
save(fig, "f4_noise_vs_shift")

# 5. generic test: chi2(ndf) vs noncentral chi2(ndf, dchi2) --------------------
x = np.linspace(0, 140, 800)
null = stats.chi2.pdf(x, ndf)
alt = stats.ncx2.pdf(x, ndf, dchi2)
median_alt = stats.ncx2.median(ndf, dchi2)
pval = stats.chi2.sf(median_alt, ndf)
z_gen = stats.norm.isf(pval)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, null, color=GREY, linewidth=2,
        label=f"no signal: $\\chi^2$ with {ndf} dof (mean {ndf})")
ax.plot(x, alt, color=RED, linewidth=2,
        label=f"$\\mu$ = 1: mean {ndf} + {dchi2:.0f} = {ndf + dchi2:.0f}")
ax.fill_between(x, 0, null, where=x >= median_alt, color=GREY, alpha=0.4,
                label=f"p = {pval:.3f}  ($Z \\approx$ {z_gen:.1f}$\\sigma$)")
ax.axvline(median_alt, color=RED, linestyle=":", linewidth=1.2)
ax.set_xlabel("$\\chi^2$ you would measure comparing two independent samples")
ax.set_ylabel("probability density")
ax.legend()
save(fig, "f5_generic_test")

# 6. known-shape test: the fitted amount of shift ------------------------------
z_t = np.sqrt(dchi2)
x = np.linspace(-4, z_t + 4, 600)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, stats.norm.pdf(x, 0, 1), color=GREY, linewidth=2,
        label="no signal: fitted shift centred on 0, width 1")
ax.plot(x, stats.norm.pdf(x, z_t, 1), color=RED, linewidth=2,
        label=f"$\\mu$ = 1: centred on $\\sqrt{{\\Delta\\chi^2}}$ = {z_t:.1f}")
ax.annotate("", xy=(z_t, 0.42), xytext=(0, 0.42),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text(z_t / 2, 0.44, f"{z_t:.1f} standard errors", ha="center")
ax.set_ylim(0, 0.62)
ax.set_xlabel("fitted size of the signal-shaped shift, in units of its own error")
ax.set_ylabel("probability density")
ax.legend(loc="upper left", fontsize=10)
save(fig, "f6_template_test")

# 7. binning: raw sum of pull^2 drifts with binning, corrected Delta chi2 not --
rows = []
for rb in (1, 2, 4, 5, 10, 20):
    hr, _ = load(rb)
    rw, nz, n = dchi2_of(hr, p, 1.0)
    c = rw - nz
    zg = stats.norm.isf(stats.chi2.sf(stats.ncx2.median(n, c), n))
    rows.append((15 * rb, n, rw, nz, c, np.sqrt(c), zg))
fig, ax = plt.subplots(figsize=(8, 4.2))
nb = [r[1] + 1 for r in rows]
ax.plot(nb, [r[2] for r in rows], marker="o", color=GREY, linestyle="--",
        label="raw $\\Sigma$ pull$^2$")
ax.plot(nb, [r[3] for r in rows], marker="v", color=BLUE, linestyle=":",
        label="expected from signal-MC noise")
ax.plot(nb, [r[4] for r in rows], marker="o", color=RED, linewidth=2,
        label="corrected $\\Delta\\chi^2$ = raw $-$ noise")
ax.set_xscale("log")
ax.set_xlabel("number of non-empty bins (finer binning $\\rightarrow$)")
ax.set_ylabel("value at $\\mu$ = 1")
ax.set_ylim(0, None)
ax.legend()
save(fig, "f7_binning")
with open(OUT / "binning.txt", "w") as f:
    for bw, n, rw, nz, c, zt, zg in rows:
        f.write(f"bin width {bw} GeV  bins {n + 1}  raw {rw:.1f}  noise {nz:.1f}"
                f"  dchi2 {c:.1f}  Z_tmpl {zt:.2f}  Z_gen {zg:.2f}\n")
print(open(OUT / "binning.txt").read())

# 8. x1 vs x4, raw and corrected, per PairNet -----------------------------------
models = ["PairNetV0_9", "PairnetV1", "PairNetInRepo", "PairNetGivenByLarry"]
vals = {}
for m in models:
    h1, p1 = load(REBIN, ROOT, m)
    h4, p4 = load(REBIN, ROOT_X4, m)
    r1, n1, _ = dchi2_of(h1, p1, 1.0)
    r4, n4, _ = dchi2_of(h4, p4, 0.25)
    vals[m] = (r1, r4, r1 - n1, r4 - n4)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
xs = np.arange(len(models))
for ax, (i1, i4), title in ((axes[0], (0, 1), "raw $\\Sigma$ pull$^2$"),
                            (axes[1], (2, 3), "corrected $\\Delta\\chi^2$")):
    ax.bar(xs - 0.2, [vals[m][i1] for m in models], 0.4, color=BLUE,
           label="x1 (~300 SQ events)")
    ax.bar(xs + 0.2, [vals[m][i4] for m in models], 0.4, color=RED,
           label="x4 (~1300 SQ events)")
    ax.set_xticks(xs, models, rotation=15, fontsize=10)
    ax.set_title(title + " at $\\mu$ = 1")
axes[0].set_ylim(0, 32)
axes[0].legend(loc="upper left", ncol=2, fontsize=9)
save(fig, "f8_x1_vs_x4")
print(f"generic test: median chi2 {median_alt:.1f}, p {pval:.4f}, Z {z_gen:.2f}")
