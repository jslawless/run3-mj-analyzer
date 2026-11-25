"""Per-dataset weights of the injection studies, read from the mixer's file
tables (copied from cmslpc ~/nobackup/run3-mj-mixer/inject/ into
<work dir>/weights/file_tables/; 15 MB, so not kept in git), and the weight a
mixed event of each kind ends up with. Work dir: $INJECT_WORKDIR, default
~/analysis/histograms/inject_26_11_23.

    w_rel        = xs_pb / N                 (mixer, filetable.py; per dataset)
    xs_weight    = w_rel[seed] * w_rel[match] (mixer, assemble.py; per event)
    fill weight  = xs_weight ** 0.5           (analyzer, make_histograms.py)

    python weights_table.py > weights_table.txt
"""
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("INJECT_WORKDIR",
                           "~/analysis/histograms/inject_26_11_23")).expanduser()
STUDIES = ("M300_x1", "M300_x4", "M1000_x1", "M1000_x4")
QCD = "QCD-4Jets_HT-{}_TuneCP5_13p6TeV_madgraphMLM-pythia8"


def short(ds):
    return ds.replace("QCD-4Jets_", "QCD ").split("_TuneCP5")[0]


def main():
    tables = {st: json.load(open(WORK / f"weights/file_tables/file_table_{st}.json"))
              for st in STUDIES}

    print("Per-dataset weight w_rel = xs_pb / N  (N = summed cutflow[0])\n")
    print(f"{'dataset':<22} {'files':>5} {'xs_pb':>11} {'N':>12} {'w_rel':>11}")
    t = tables["M300_x1"]
    for ds, r in t["datasets"].items():
        if ds.startswith("QCD"):
            print(f"{short(ds):<22} {r['n_files']:>5} {r['xs_pb']:>11.6g} "
                  f"{r['n_original_sum']:>12,.0f} {r['w_rel']:>11.4g}")
    for st in STUDIES:
        sig = {ds: r for ds, r in tables[st]["datasets"].items()
               if not ds.startswith("QCD")}
        for ds, r in sig.items():
            print(f"{ds + ' (' + st[-2:] + ')':<22} {r['n_files']:>5} "
                  f"{r['xs_pb']:>11.6g} {r['n_original_sum']:>12,.0f} "
                  f"{r['w_rel']:>11.4g}")

    print("\nExample mixed events, M300 x1 (fill weight = sqrt(xs_weight))\n")
    d = tables["M300_x1"]["datasets"]
    w = {k: d[QCD.format(k)]["w_rel"] for k in ("400to600", "600to800")}
    w_sig = d["RPV_M300"]["w_rel"]
    print(f"{'pair':<34} {'xs_weight':>11} {'fill weight':>12}")
    for name, a, b in (("QQ: HT-400to600 x HT-600to800", w["400to600"], w["600to800"]),
                       ("SQ: RPV_M300 x HT-400to600", w_sig, w["400to600"])):
        print(f"{name:<34} {a * b:>11.4g} {(a * b) ** 0.5:>12.4g}")
    for mu in (0.00339, 1.0):
        a, b = mu * w_sig, w["400to600"]
        print(f"{'SQ at mu = %g' % mu:<34} {a * b:>11.4g} {(a * b) ** 0.5:>12.4g}")


if __name__ == "__main__":
    main()
