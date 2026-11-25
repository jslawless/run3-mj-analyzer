"""Excluded signal strength mu95 at every gluino mass, from CMS-EXO-21-004.

Input: HEPData ins2774247, table "Figure 4c" (resolved trijets), as CSV:
https://www.hepdata.net/download/submission/ins2774247/1/csv

The table gives 95% CL limits on sigma*B*A (pb), the acceptance A and the
theory sigma*A (pb). mu95 = limit / (theory sigma*A): the acceptance cancels.

    python mu95_table.py [CSV]  > mu95_table.txt
"""
import sys
from pathlib import Path

import numpy as np

DEFAULT_CSV = Path(__file__).parent / "CMS-EXO-21-004_HEPData_Figure4c.csv"
LIMITS = ("Observed XSCN", "Expected XSCN",
          "-2 sigma Expected XSCN", "-1 sigma Expected XSCN",
          "+1 sigma Expected XSCN", "+2 sigma Expected XSCN")


def read_blocks(path):
    """{column name: [(mass, value), ...]} from the HEPData CSV, which stacks
    one block per column, each headed 'Gluino Mass [GeV],<column name>'."""
    blocks, cur = {}, None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Gluino Mass"):
            cur = line.split(",", 1)[1]
            blocks[cur] = []
            continue
        m, v = line.split(",")
        blocks[cur].append((int(m), float(v)))
    return blocks


def mu95_table(path=DEFAULT_CSV):
    """Arrays per row: mass, acceptance, theory sigma*A, theory sigma, every
    limit, and mu95 for every limit (as 'mu95 <limit name>')."""
    blocks = read_blocks(path)
    col = lambda k: np.array([v for _, v in blocks[k]])
    t = {"mass": np.array([m for m, _ in blocks["Acceptance"]]),
         "A": col("Acceptance"),
         "sigmaA_th": col("SignalXscn_X_acceptance")}
    t["sigma_th"] = t["sigmaA_th"] / t["A"]
    for k in LIMITS:
        t[k] = col(k)
        t["mu95 " + k] = t[k] / t["sigmaA_th"]
    return t


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t = mu95_table(argv[0] if argv else DEFAULT_CSV)
    print(f"{'M':>5} {'region':>6} {'A':>9} {'sig_th[pb]':>10} "
          f"{'sigA[pb]':>9} {'lim obs':>9} {'lim exp':>9} | {'mu95 obs':>9} "
          f"{'mu95 exp':>9} {'-2sig':>9} {'-1sig':>9} {'+1sig':>9} {'+2sig':>9}")
    for i, m in enumerate(t["mass"]):
        region = "low" if t["A"][i] < 0.01 else "high"
        mus = " ".join(f"{t['mu95 ' + k][i]:9.2e}" for k in LIMITS)
        print(f"{m:>5} {region:>6} {t['A'][i]:9.2e} {t['sigma_th'][i]:10.4g} "
              f"{t['sigmaA_th'][i]:9.3g} {t[LIMITS[0]][i]:9.2e} "
              f"{t[LIMITS[1]][i]:9.2e} | {mus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
