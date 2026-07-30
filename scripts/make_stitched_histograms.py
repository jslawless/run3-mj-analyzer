#!/usr/bin/env python3
"""make_stitched_histograms.py - QA + physics spectra from stitcher output.

Companion to make_histograms.py / make_hemisphere_histograms.py, adapted to
the run3-mj-stitch output format (``stitched_*.root``: 6-jet pseudo-events
built from two 3-jet hemispheres, flat NanoAOD-style branches, all weight 1).
Fills two families of histograms into a single output ROOT file as TH1D
(with Sumw2), named ``h_<histogram>``:

  physics validation (compare against the same spectra in real >=6-jet data):
    m6j (6-jet invariant mass), thrust, jet{1..6}_pt (pT-ranked),
    hemi_mass (per-hemisphere tri-jet mass)

  fill_qcd_slimming.py mirror set, written with the SAME bare names and the
  SAME binning (axes + fills imported from that script, so they cannot
  drift) for direct overlay with its *_spectra.root outputs:
    ht, njet, jet_pt, jet_eta, jet_phi

  stitching QA (artifacts of the mixing itself):
    match_distance   - goodness of the (directed phi, partner eta) match
    hemi_dphi        - |dphi| between the two stitched halves (piles up
                       near pi if the naturally-opposite query works)
    hemi_pt_asym     - (pt0-pt1)/(pt0+pt1), bounded by the +-10% pT window
    mht, mht_over_ht - event pT imbalance created by the seam
    min_dr_inter     - min dR between jets from DIFFERENT halves: overlap
                       spikes at low dR are jets a real reconstruction
                       would have merged
    min_dr_intra     - min dR within a half (reference for min_dr_inter)
    lead_jet_hemi    - which half owns the leading jet (0 = seed, 1 = match)

Two input modes, both reading stitcher output - including the chunked files
from run3-mj-evaluator's scripts/split_fileset.py:

  * one or more stitched ``.root`` files (local paths or root:// URLs).
  * a single dataset ``.json`` (from scripts/make_dataset_json.py), whose
    ``datasets`` lists the stitched files; every dataset in it is processed
    into the same histograms, and the tree name defaults to the JSON's
    ``metadata.tree``.

Either way every fill is weighted by the per-event ``xs_weight`` branch (all
1.0 for the current stitcher, so effectively unweighted) - so, unlike
make_histograms.py, the JSON needs no cross sections and is expanded without
opening any file. A ``stitch_cutflow`` summed over every input file that
carries one is written through to the output (chunked inputs only carry it in
chunk 0).

Adding histograms: add an entry to ``histogram_defs()`` below and list any
newly-needed branches in BRANCHES.

Example:
    python scripts/make_stitched_histograms.py stitched.root -o stitched_hists.root
    python scripts/make_stitched_histograms.py dataset_stitched.json -o stitched_hists.root
    python scripts/make_stitched_histograms.py 'root://cmseos.fnal.gov//store/user/.../stitched_chunk*.root' -o stitched_hists.root
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import uproot

# fill_qcd_slimming.py owns the binning for every observable the two scripts
# share: its build_axes()/FILLS supply ht, njet and jet_pt/eta/phi outright, and
# its PT_BINS/PT_RANGE also bin the pT-ranked jet{1..6}_pt axes below. Editing
# the constants there moves both scripts together - nothing is re-typed here, so
# the spectra cannot drift out of overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_qcd_slimming as qcd_slimming

# Make the package importable without `pip install -e .` (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run3_mj_analyzer.fileset import DEFAULT_TREE, load_fileset

try:
    from tqdm import tqdm
except ImportError:  # progress bars are cosmetic - everything runs without them
    tqdm = None


def echo(msg):
    """print() that doesn't tear through active tqdm bars."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


# Branches read per chunk. The stitcher writes flat NanoAOD-style names.
BRANCHES = [
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "ScoutingPFJet_m", "ScoutingPFJet_hemisphere",
    "HT", "thrust", "xs_weight", "match_distance",
    "Hemisphere_pt", "Hemisphere_phi", "Hemisphere_mass",
]

N_JET_RANKS = 6  # jets per pseudo-event (two 3-jet hemispheres)


def _fold_dphi(dphi):
    """|delta phi| folded onto [0, pi]."""
    return np.abs(np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi)


def derive(events):
    """Per-chunk derived quantities shared by several histogram defs."""
    pt, eta = events["ScoutingPFJet_pt"], events["ScoutingPFJet_eta"]
    phi, m = events["ScoutingPFJet_phi"], events["ScoutingPFJet_m"]

    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    sum_e, sum_px = ak.sum(e, axis=1), ak.sum(px, axis=1)
    sum_py, sum_pz = ak.sum(py, axis=1), ak.sum(pz, axis=1)

    d = {}
    d["m6j"] = np.sqrt(np.maximum(
        ak.to_numpy(sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2), 0.0
    ))
    d["mht"] = ak.to_numpy(np.sqrt(sum_px**2 + sum_py**2))

    # min dR between jets of different halves (seam) vs the same half.
    jets = ak.zip({"eta": eta, "phi": phi,
                   "hemi": events["ScoutingPFJet_hemisphere"]})
    a, b = ak.unzip(ak.combinations(jets, 2))
    dr = np.sqrt(_fold_dphi(a.phi - b.phi) ** 2 + (a.eta - b.eta) ** 2)
    inter = a.hemi != b.hemi
    d["min_dr_inter"] = ak.drop_none(ak.min(dr[inter], axis=1))
    d["min_dr_intra"] = ak.drop_none(ak.min(dr[~inter], axis=1))
    return d


@dataclass
class HistDef:
    """One histogram: an axis plus how to compute its fill values.

    ``values(events, d)`` gets the chunk and the derive() dict and returns the
    flat fill array; ``weighted`` picks per-event xs_weight (broadcast to the
    fill's multiplicity via ``per_event``: 1 = one value/event, 2 = two, ...).
    """

    axis: object
    values: callable
    per_event: int = 1  # fill values per event (for xs_weight broadcasting)
    key: str = None  # output name; defaults to "h_<def name>"


def histogram_defs():
    """Every histogram this script produces. Add new entries here."""
    ax = hist.axis.Regular
    defs = {
        # -- physics validation ------------------------------------------
        "m6j": HistDef(ax(150, 0.0, 6000.0, name="m6j",
                          label="6-jet invariant mass [GeV]"),
                       lambda ev, d: d["m6j"]),
        "thrust": HistDef(ax(60, 0.5, 1.0, name="thrust",
                             label="transverse thrust"),
                          lambda ev, d: ev["thrust"]),
        # jet_eta / jet_phi come from the fill_qcd_slimming.py mirror set below
        # (same binning, bare output names) rather than being defined twice.
        "hemi_mass": HistDef(ax(100, 0.0, 2000.0, name="mass",
                                label="hemisphere (tri-jet) mass [GeV]"),
                             lambda ev, d: ak.flatten(ev["Hemisphere_mass"]),
                             per_event=2),
        # -- stitching QA ------------------------------------------------
        "match_distance": HistDef(
            ax(100, 0.0, 0.6, name="dist",
               label="match distance in (directed #phi, partner #eta)"),
            lambda ev, d: ev["match_distance"]),
        "hemi_dphi": HistDef(
            ax(64, 0.0, np.pi, name="dphi",
               label="|#Delta#phi| between stitched halves"),
            lambda ev, d: _fold_dphi(ak.to_numpy(
                ev["Hemisphere_phi"][:, 0] - ev["Hemisphere_phi"][:, 1]))),
        "hemi_pt_asym": HistDef(
            ax(60, -0.15, 0.15, name="asym",
               label="(p_{T}^{0} - p_{T}^{1}) / (p_{T}^{0} + p_{T}^{1})"),
            lambda ev, d: ak.to_numpy(
                (ev["Hemisphere_pt"][:, 0] - ev["Hemisphere_pt"][:, 1])
                / (ev["Hemisphere_pt"][:, 0] + ev["Hemisphere_pt"][:, 1]))),
        "mht": HistDef(ax(100, 0.0, 500.0, name="mht",
                          label="|#Sigma vec p_{T}| [GeV]"),
                       lambda ev, d: d["mht"]),
        "mht_over_ht": HistDef(ax(60, 0.0, 0.6, name="frac",
                                  label="|#Sigma vec p_{T}| / H_{T}"),
                               lambda ev, d: d["mht"] / ak.to_numpy(ev["HT"])),
        "min_dr_inter": HistDef(
            ax(100, 0.0, 5.0, name="dr",
               label="min #DeltaR (jets from different halves)"),
            lambda ev, d: d["min_dr_inter"]),
        "min_dr_intra": HistDef(
            ax(60, 0.0, 3.0, name="dr",
               label="min #DeltaR (jets within one half)"),
            lambda ev, d: d["min_dr_intra"]),
        "lead_jet_hemi": HistDef(
            ax(2, -0.5, 1.5, name="hemi",
               label="leading jet from: 0 = seed, 1 = match"),
            lambda ev, d: ak.to_numpy(ev["ScoutingPFJet_hemisphere"][:, 0])),
    }
    # pT-ranked jet spectra: jets are written pT-sorted, so rank = index. The
    # binning is fill_qcd_slimming.py's PT constants, so every pT axis in both
    # scripts (inclusive jet_pt and all six ranks) moves together when edited.
    for rank in range(N_JET_RANKS):
        defs[f"jet{rank + 1}_pt"] = HistDef(
            ax(qcd_slimming.PT_BINS, *qcd_slimming.PT_RANGE, name="pt",
               label=f"jet {rank + 1} p_{{T}} [GeV]"),
            lambda ev, d, r=rank: ak.to_numpy(ev["ScoutingPFJet_pt"][:, r]),
        )
    # fill_qcd_slimming.py's observables, with its axes/fills and its bare
    # output names (no "h_" prefix), for direct overlay with *_spectra.root.
    per_event = {
        "ht": 1, "njet": 1,
        "jet_pt": N_JET_RANKS, "jet_eta": N_JET_RANKS, "jet_phi": N_JET_RANKS,
    }
    for obs, axis in qcd_slimming.build_axes().items():
        if obs not in per_event:
            raise SystemExit(
                f"fill_qcd_slimming.py added observable '{obs}' - add its fill "
                f"multiplicity to per_event in {Path(__file__).name}"
            )
        defs[obs] = HistDef(
            axis,
            lambda ev, d, f=qcd_slimming.FILLS[obs]: f(ev),
            per_event=per_event[obs],
            key=obs,
        )
    return defs


class HistogramBook:
    """Lazily-created histograms keyed by def name."""

    def __init__(self, defs):
        self.defs = defs
        self.hists = {}

    def fill(self, events, derived):
        weight = ak.to_numpy(events["xs_weight"])
        for name, hdef in self.defs.items():
            h = self.hists.get(name)
            if h is None:
                h = self.hists[name] = hist.Hist(
                    hdef.axis, storage=hist.storage.Weight()
                )
            values = ak.to_numpy(hdef.values(events, derived))
            w = (np.repeat(weight, hdef.per_event)
                 if hdef.per_event > 1 else weight)
            # Defensive: derived quantities that drop events (min over empty
            # pair sets) fall back to unit weights rather than misaligning.
            if len(w) != len(values):
                w = None
            h.fill(values, weight=w)

    def out_name(self, name):
        return self.defs[name].key or f"h_{name}"

    def write(self, path, extra=None):
        with uproot.recreate(path) as f:
            for name in sorted(self.hists):
                f[self.out_name(name)] = self.hists[name]
            for key, value in (extra or {}).items():
                f[key] = value


def cutflow_labels(cf, n):
    """Best-effort stage labels for an uproot cutflow histogram."""
    try:
        return [str(s) for s in cf.to_hist().axes[0]]
    except Exception:
        return [f"stage{i}" for i in range(n)]


def resolve_inputs(args):
    """Flatten the CLI inputs into ``[(path, tree_name), ...]``.

    Bare ROOT paths all get ``--tree`` (or "events"); a dataset JSON is expanded
    with ``skip_missing_tree=False``, i.e. no file I/O and every path kept, for
    two reasons: load_fileset's inspection pass demands a ``cutflow`` histogram
    per file, which stitcher output does not have (it writes ``stitch_cutflow``,
    and chunked output only in chunk 0), and nothing here needs the
    ``n_original`` that pass computes - the fills are weighted by the per-event
    ``xs_weight`` branch. Files with no events tree are skipped by the main loop
    as they are opened.

    Every dataset in the JSON is filled into the same histograms: for stitched
    output the "datasets" are usually just chunks of one mixed sample.
    """
    json_inputs = [p for p in args.inputs if p.endswith(".json")]
    root_inputs = [p for p in args.inputs if not p.endswith(".json")]
    if json_inputs and root_inputs:
        raise SystemExit("Pass either ROOT files or one dataset JSON, not both.")
    if len(json_inputs) > 1:
        raise SystemExit("Pass at most one dataset JSON.")

    if not json_inputs:
        return [(p, args.tree or DEFAULT_TREE) for p in root_inputs]

    fileset = load_fileset(json_inputs[0], tree=args.tree,
                           skip_missing_tree=False)
    jobs = []
    for name, ds in fileset.items():
        print(f"[{name}] {len(ds['files'])} file(s)")
        jobs.extend(ds["files"].items())
    if not jobs:
        raise SystemExit(f"{json_inputs[0]} lists no files.")
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="QA + physics spectra from run3-mj-stitch output "
        "(6-jet pseudo-events), written as TH1D to a single ROOT file. Takes "
        "stitched ROOT files or one dataset JSON listing them."
    )
    parser.add_argument("inputs", nargs="+",
                        help="stitched ROOT file(s), local or root:// URLs, or "
                        "one dataset JSON from scripts/make_dataset_json.py")
    parser.add_argument("-o", "--output", default="stitched_hists.root",
                        help="output ROOT file (default: %(default)s)")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: 'events', or the "
                        "dataset JSON's metadata)")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress tqdm progress bars")
    args = parser.parse_args()
    use_bars = tqdm is not None and not args.no_progress

    jobs = resolve_inputs(args)

    defs = histogram_defs()
    book = HistogramBook(defs)
    print(f"{len(jobs)} file(s) to process | histograms: "
          f"{sorted(defs)} | output: {args.output}")

    n_events, n_skipped = 0, 0
    cutflow_total, cf_labels = None, None
    t0 = time.monotonic()

    file_bar = tqdm(jobs, unit="file", desc="files") if use_bars else jobs
    for path, tree_name in file_bar:
        t_file = time.monotonic()
        with uproot.open(path) as f:
            # Sum the stitcher's cutflow over every file that carries one
            # (split_fileset.py chunks carry it in chunk 0 only).
            if "stitch_cutflow" in f:
                vals = np.asarray(f["stitch_cutflow"].values(), dtype=np.float64)
                if cutflow_total is None:
                    cutflow_total = vals
                    cf_labels = cutflow_labels(f["stitch_cutflow"], len(vals))
                elif len(vals) == len(cutflow_total):
                    cutflow_total = cutflow_total + vals
                else:
                    echo(f"[warn] {path}: stitch_cutflow has {len(vals)} bins, "
                         f"expected {len(cutflow_total)} - not summed")
            if tree_name not in f:
                echo(f"[skip] {path}: no '{tree_name}' tree")
                n_skipped += 1
                continue
            tree = f[tree_name]
            missing = [b for b in BRANCHES if b not in tree]
            if missing:
                raise SystemExit(
                    f"{path} is missing {missing} - is this stitcher output?"
                )
            n_file = 0
            for events in tree.iterate(filter_name=BRANCHES,
                                       step_size=args.step_size):
                book.fill(events, derive(events))
                n_file += len(events)
        n_events += n_file
        dt = time.monotonic() - t_file
        echo(f"[done] {Path(path).name}: {n_file:,} events in {dt:.1f} s "
             f"({n_events:,} total)")
    if use_bars:
        file_bar.close()

    if not book.hists:
        raise SystemExit("No histograms filled - no usable input files?")

    elapsed = time.monotonic() - t0
    print(f"\n{n_events:,} events from {len(jobs) - n_skipped} file(s) "
          f"in {elapsed:.1f} s"
          + (f" | {n_skipped} skipped" if n_skipped else ""))

    print("\nhistograms (in-range weighted entries):")
    for name in sorted(book.hists):
        print(f"  {book.out_name(name)}: {book.hists[name].sum().value:.6g}")

    extra = {}
    if cutflow_total is not None:
        cutflow = hist.Hist(
            hist.axis.StrCategory(cf_labels), storage=hist.storage.Double()
        )
        cutflow.view()[:] = cutflow_total
        extra["stitch_cutflow"] = cutflow
        print("\nstitch cutflow (summed over inputs):")
        for label, value in zip(cf_labels, cutflow_total):
            print(f"  {label:<42} {value:>15,.0f}")
    else:
        echo("[warn] no input 'stitch_cutflow' found - none written to output")

    book.write(args.output, extra=extra)
    print(f"\nhistograms written to {args.output}")


if __name__ == "__main__":
    main()
