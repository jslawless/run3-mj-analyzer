"""mu95 vs gluino mass from CMS-EXO-21-004 (HEPData ins2774247, Figure 4c)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
sys.path.insert(0, str(here / "../../excluded_xs"))
from mu95_table import mu95_table  # noqa: E402

t = mu95_table()
M, A = t["mass"], t["A"]
mu = lambda k: t["mu95 " + k]
obs, exp = mu("Observed XSCN"), mu("Expected XSCN")
m1, p1 = mu("-1 sigma Expected XSCN"), mu("+1 sigma Expected XSCN")
m2, p2 = mu("-2 sigma Expected XSCN"), mu("+2 sigma Expected XSCN")

fig, ax = plt.subplots(figsize=(7, 4.6))
# the two search regions overlap at 500 and 900; draw each region separately
regions = [A < 0.01, A >= 0.01]
# second 900 row belongs to the upper (>900) region: split high at it
idx = np.arange(len(M))
dup900 = idx[M == 900]
high_a = regions[1] & (idx <= dup900[0])
high_b = regions[1] & (idx >= dup900[1])
for i, sel in enumerate((regions[0], high_a, high_b)):
    ax.fill_between(M[sel], m2[sel], p2[sel], color="#F5BB54", lw=0,
                    label="expected $\\pm2\\sigma$" if i == 0 else None)
    ax.fill_between(M[sel], m1[sel], p1[sel], color="#607641", lw=0,
                    label="expected $\\pm1\\sigma$" if i == 0 else None)
    ax.plot(M[sel], exp[sel], "k--", lw=1.2, label="expected" if i == 0 else None)
    ax.plot(M[sel], obs[sel], "k-o", ms=3, lw=1.4, label="observed" if i == 0 else None)
ax.axhline(1, color="tab:red", lw=1.2)
ax.text(1180, 1.25, "theory cross section ($\\mu$ = 1)", color="tab:red", ha="right", fontsize=9)
for m, mu in ((300, 3.39e-3), (1000, 3.00e-2)):
    ax.plot(m, mu, "*", color="tab:blue", ms=14, zorder=6)
    ax.annotate(f"M{m}: $\\mu_{{95}}^{{obs}}$ = {mu:.3g}", (m, mu),
                xytext=(10, -18), textcoords="offset points", fontsize=9,
                color="tab:blue")
ax.set_yscale("log")
ax.set_xlabel("gluino mass [GeV]")
ax.set_ylabel("$\\mu_{95}$ = limit on $\\sigma B A$ / theory $\\sigma A$")
ax.set_title("CMS-EXO-21-004, 128 fb$^{-1}$ (13 TeV), resolved trijets", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
(here / "figs").mkdir(exist_ok=True)
fig.savefig(here / "figs/mu95_vs_mass.pdf", bbox_inches="tight")
