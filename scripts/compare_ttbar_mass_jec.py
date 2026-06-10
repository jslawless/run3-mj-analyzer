#!/usr/bin/env python3
"""compare_ttbar_mass_jec.py - truth-matched m(ttbar) spectrum, pre- vs post-JEC.

Runs run3_mj_analyzer.truth_matching twice per event chunk on evaluator output
(which carries both the JEC-corrected ``ScoutingPFJet_pt`` / ``_m`` and the
uncorrected ``ScoutingPFJet_pt_raw`` / ``_m_raw``):

  * "jec": jet four-vectors from ``ScoutingPFJet_pt``     / ``ScoutingPFJet_m``
  * "raw": jet four-vectors from ``ScoutingPFJet_pt_raw`` / ``ScoutingPFJet_m_raw``

For each variant, events where BOTH tops are truth-matched (exactly two
tri-jets) enter the spectrum, and m(ttbar) is the invariant mass of the sum of
the two tri-jet four-vectors. Note the selections are independent: the jet
preselection pT cut acts on each variant's own pT, so an event can pass in one
variant and not the other.

The two spectra are written as TH1D (``h_mtt_jec``, ``h_mtt_raw``) to a ROOT
file, with an overlay PNG saved next to it.

Example:
    python scripts/compare_ttbar_mass_jec.py evaluated_TTto4Q_*.root -o ttbar_mtt_jec.root
    python scripts/compare_ttbar_mass_jec.py ttbar_dataset.json -o ttbar_mtt_jec.root
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import hist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the package importable without `pip install -e .` (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run3_mj_analyzer.fileset import load_fileset
from run3_mj_analyzer.truth_matching import truth_matched_trijets

# Everything the matcher needs, both JEC variants included. Files missing any
# of these (plain slimmer output, or data without GenPart) are rejected loudly
# rather than silently producing an empty "raw" or "jec" histogram.
BRANCHES = [
    "ScoutingPFJet_pt",
    "ScoutingPFJet_eta",
    "ScoutingPFJet_phi",
    "ScoutingPFJet_m",
    "ScoutingPFJet_pt_raw",
    "ScoutingPFJet_m_raw",
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_status",
    "GenPart_genPartIdxMother",
]


def pair_masses(events, pt_branch, m_branch, **match_kwargs):
    """m(ttbar) for events whose two tops are both truth-matched (numpy array)."""
    trijets = truth_matched_trijets(
        events, jet_pt_branch=pt_branch, jet_m_branch=m_branch, **match_kwargs
    )
    both = trijets[ak.num(trijets, axis=1) == 2]
    return ak.to_numpy((both[:, 0] + both[:, 1]).mass)


def make_hist(bins, lo, hi):
    return hist.Hist.new.Reg(
        bins, lo, hi, name="mtt", label=r"$m_{t\bar{t}}$ [GeV]"
    ).Double()


def hist_mean(h):
    """Mean of a 1D hist from bin centers/values (nan when empty)."""
    values = h.values()
    if values.sum() <= 0:
        return float("nan")
    return float(np.average(h.axes[0].centers, weights=values))


def resolve_inputs(inputs, tree):
    """Flatten the inputs into ``[(path, tree_name), ...]``.

    Accepts either evaluated ROOT files directly or a single dataset JSON from
    scripts/make_dataset_json.py (resolved via load_fileset, which also drops
    files whose events tree is empty). The spectra are filled unweighted either
    way - this is a shape comparison within one sample.
    """
    json_inputs = [p for p in inputs if p.endswith(".json")]
    root_inputs = [p for p in inputs if not p.endswith(".json")]
    if json_inputs and root_inputs:
        raise SystemExit("Pass either ROOT files or one dataset JSON, not both.")
    if len(json_inputs) > 1:
        raise SystemExit("Pass at most one dataset JSON.")
    if json_inputs:
        fileset = load_fileset(json_inputs[0], tree=tree)
        return [
            (path, tree_name)
            for ds in fileset.values()
            for path, tree_name in ds["files"].items()
        ]
    return [(path, tree or "events") for path in root_inputs]


def main():
    parser = argparse.ArgumentParser(
        description="Truth-matched m(ttbar) spectrum before (raw) and after (jec) "
        "jet energy corrections, written as TH1D to a ROOT file."
    )
    parser.add_argument("inputs", nargs="+",
                        help="evaluator output ROOT file(s), or one dataset JSON "
                        "from scripts/make_dataset_json.py")
    parser.add_argument("-o", "--output", default="ttbar_mtt_jec.root",
                        help="output ROOT file (default: %(default)s); the overlay "
                        "PNG is saved next to it")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: 'events', or the "
                        "dataset JSON's metadata)")
    parser.add_argument("--bins", type=int, default=100,
                        help="histogram bins (default: %(default)s)")
    parser.add_argument("--range", type=float, nargs=2, default=(0.0, 2000.0),
                        metavar=("LO", "HI"),
                        help="histogram range in GeV (default: 0 2000)")
    parser.add_argument("--dr-max", type=float, default=0.4,
                        help="jet-parton match dR (default: %(default)s)")
    parser.add_argument("--jet-pt-min", type=float, default=30.0,
                        help="jet preselection pT in GeV, applied per variant "
                        "(default: %(default)s)")
    parser.add_argument("--jet-abseta-max", type=float, default=2.4,
                        help="jet preselection |eta| (default: %(default)s)")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    args = parser.parse_args()

    match_kwargs = dict(
        dr_max=args.dr_max,
        jet_pt_min=args.jet_pt_min,
        jet_abseta_max=args.jet_abseta_max,
    )

    h_jec = make_hist(args.bins, *args.range)
    h_raw = make_hist(args.bins, *args.range)
    n_events = 0

    for path, tree_name in resolve_inputs(args.inputs, args.tree):
        with uproot.open(path) as f:
            if tree_name not in f:
                print(f"[skip] {path}: no '{tree_name}' tree")
                continue
            tree = f[tree_name]
            missing = [b for b in BRANCHES if b not in tree]
            if missing:
                raise SystemExit(
                    f"{path} is missing branches {missing} - this script needs "
                    "evaluator output (JEC-corrected + raw jets and GenPart truth)."
                )
            for events in tree.iterate(BRANCHES, step_size=args.step_size):
                n_events += len(events)
                h_jec.fill(
                    pair_masses(events, "ScoutingPFJet_pt", "ScoutingPFJet_m",
                                **match_kwargs)
                )
                h_raw.fill(
                    pair_masses(events, "ScoutingPFJet_pt_raw", "ScoutingPFJet_m_raw",
                                **match_kwargs)
                )
        print(f"[done] {path}")

    print(
        f"\n{n_events} events read | truth-matched ttbar pairs: "
        f"jec {int(h_jec.sum())} (mean {hist_mean(h_jec):.1f} GeV), "
        f"raw {int(h_raw.sum())} (mean {hist_mean(h_raw):.1f} GeV)"
    )
    if n_events and not h_jec.sum():
        print("warning: no truth-matched pairs - is this a ttbar sample?")

    with uproot.recreate(args.output) as fout:
        fout["h_mtt_jec"] = h_jec
        fout["h_mtt_raw"] = h_raw
    print(f"histograms written to {args.output}")

    fig, ax = plt.subplots(figsize=(8, 5))
    edges = h_jec.axes[0].edges
    ax.stairs(h_raw.values(), edges,
              label=f"raw (no JEC), {int(h_raw.sum())} pairs", color="tab:red")
    ax.stairs(h_jec.values(), edges,
              label=f"JEC, {int(h_jec.sum())} pairs", color="tab:blue")
    ax.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Truth-matched ttbar invariant mass")
    ax.legend()
    png = Path(args.output).with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"overlay plot written to {png}")


if __name__ == "__main__":
    main()
