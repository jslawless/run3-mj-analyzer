#!/usr/bin/env python3
"""make_hemisphere_histograms.py - hemisphere kinematics from mixer output.

Companion to make_histograms.py, adapted to the run3-mj-mixer output format
(``mixed_*.root``: slimmed branches + a 2/event ``Hemisphere`` collection).
Fills, for every hemisphere kinematic variable, one histogram per hemisphere
jet-multiplicity class (**1-4 jets** - the sides of an exactly-5-jet event;
0- and 5-jet hemispheres are excluded), plus the event-level transverse thrust
split by 2+3 vs 1+4 topology. All histograms land in a single output ROOT file
as TH1D (with Sumw2), named ``h_<histogram>_njets<N>`` / ``h_thrust_split<S>``.

Weighting: every fill uses ``lumi * xs_pb / n_original`` recomputed here with
the slice-summed ``n_original`` (cutflow[0] over ALL files of the slice). The
per-file ``Hemisphere_weight`` baked into the mixed files is deliberately
ignored - it was computed with each file's own cutflow[0] as denominator, which
is only a per-file share of the normalisation.

Two input modes:

  * one or more mixed ``.root`` files: the HT slice of each file is inferred
    from its filename (mixed files keep the slimmed input's name, which embeds
    the dataset), an up-front pass sums cutflow[0] per slice, and each file is
    filled with its slice's xsec weight. Pass ``--unweighted`` to skip all of
    that and fill with weight 1.
  * a single dataset ``.json`` (from scripts/make_dataset_json.py) listing the
    mixed files per slice: identical to make_histograms.py, with n_original
    from run3_mj_analyzer.load_fileset.

In every mode the output file also gets an ``n_original`` TObjString (JSON map
``{dataset: cutflow[0] sum}``) and a ``cutflow`` histogram: the slimmer
preselection stages summed over every input file, with the mixer's
exactly-5-jet counts appended.

Adding histograms: add an entry to ``histogram_defs()`` below. ``cat="njets"``
definitions read the flat per-hemisphere arrays and are instantiated once per
jet-multiplicity class; ``cat="split"`` definitions read event-level arrays and
are instantiated per split topology (2+3 / 1+4).

Example:
    python scripts/make_hemisphere_histograms.py mixed_*.root -o hemi_hists.root
    python scripts/make_hemisphere_histograms.py dataset_mixed.json -o hemi_hists.root
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import uproot

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

# Make the package importable without `pip install -e .` (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run3_mj_analyzer.fileset import load_fileset

# The hemisphere multiplicity classes that get a histogram each. 0-jet and
# 5-jet hemispheres (degenerate 0+5 splits) are excluded everywhere.
NJETS_CLASSES = (1, 2, 3, 4)

# min(n_jets) of the event's two hemispheres -> split label.
SPLIT_CLASSES = {2: "23", 1: "14"}

# Hemisphere branches read from the mixed files (Hemisphere_<field>).
HEMI_FIELDS = ["pt", "eta", "phi", "mass", "energy", "pz",
               "pt_par", "pt_perp", "partner_eta", "n_jets"]
EXTRA_BRANCHES = ["thrust"]

# Cross sections live in the shared aux repo, assumed checked out next to
# run3-mj-analyzer (same convention as the notebooks).
DEFAULT_XS_JSON = (
    Path(__file__).resolve().parents[2] / "run3-mj-pass-the-aux" / "mj_samples_xs.json"
)


@dataclass
class HistDef:
    """One histogram family: an axis plus how to compute its fill values.

    ``cat="njets"``: ``values(hemi)`` on the flat per-hemisphere arrays, one
    instance per class in NJETS_CLASSES. ``cat="split"``: ``values(evt)`` on
    the event-level arrays, one instance per split topology.
    """

    axis: object
    values: callable
    cat: str = "njets"


def histogram_defs():
    """Every histogram this script produces. Add new entries here."""
    ax = hist.axis.Regular
    return {
        "pt": HistDef(ax(100, 0.0, 2000.0, name="pt",
                         label="hemisphere p_{T} [GeV]"),
                      values=lambda h: h["pt"]),
        "pz": HistDef(ax(150, -6000.0, 6000.0, name="pz",
                         label="hemisphere p_{z} [GeV]"),
                      values=lambda h: h["pz"]),
        "energy": HistDef(ax(150, 0.0, 6000.0, name="energy",
                             label="hemisphere energy [GeV]"),
                          values=lambda h: h["energy"]),
        "mass": HistDef(ax(100, 0.0, 2000.0, name="mass",
                           label="hemisphere invariant mass [GeV]"),
                        values=lambda h: h["mass"]),
        "eta": HistDef(ax(60, -3.0, 3.0, name="eta", label="hemisphere #eta"),
                       values=lambda h: h["eta"]),
        "phi": HistDef(ax(64, -np.pi, np.pi, name="phi",
                          label="hemisphere #phi"),
                       values=lambda h: h["phi"]),
        "abs_pt_par": HistDef(ax(100, 0.0, 2000.0, name="abs_pt_par",
                                 label="|p_{T,#parallel}| along n_{T} [GeV]"),
                              values=lambda h: np.abs(h["pt_par"])),
        "pt_perp": HistDef(ax(100, -500.0, 500.0, name="pt_perp",
                              label="p_{T,#perp} w.r.t. n_{T} [GeV]"),
                           values=lambda h: h["pt_perp"]),
        "partner_eta": HistDef(ax(60, -3.0, 3.0, name="partner_eta",
                                  label="partner hemisphere #eta"),
                               values=lambda h: h["partner_eta"]),
        "d_eta_partner": HistDef(ax(100, -5.0, 5.0, name="d_eta_partner",
                                    label="#eta - #eta_{partner}"),
                                 values=lambda h: h["eta"] - h["partner_eta"]),
        "thrust": HistDef(ax(60, 0.5, 1.0, name="thrust",
                             label="transverse thrust T"),
                          values=lambda e: e["thrust"], cat="split"),
    }


class HistogramBook:
    """Lazily-created histograms keyed ``<def>_njets<N>`` / ``<def>_split<S>``."""

    def __init__(self, defs):
        self.defs = defs
        self.hists = {}

    def fill(self, hemi, evt, weight):
        for dname, hdef in self.defs.items():
            if hdef.cat == "njets":
                values = np.asarray(hdef.values(hemi))
                masks = {f"njets{n}": hemi["n_jets"] == n for n in NJETS_CLASSES}
            else:
                values = np.asarray(hdef.values(evt))
                masks = {f"split{lab}": evt["split_min"] == s
                         for s, lab in SPLIT_CLASSES.items()}
            for label, mask in masks.items():
                key = f"{dname}_{label}"
                h = self.hists.get(key)
                if h is None:
                    h = self.hists[key] = hist.Hist(
                        hdef.axis, storage=hist.storage.Weight()
                    )
                h.fill(values[mask], weight=weight)

    def write(self, path, extra=None):
        """Write all histograms, plus any ``extra`` named objects (the
        ``n_original`` TObjString, the ``cutflow`` histogram, ...)."""
        with uproot.recreate(path) as f:
            for key in sorted(self.hists):
                f[f"h_{key}"] = self.hists[key]
            for key, value in (extra or {}).items():
                f[key] = value


def flatten_chunk(events):
    """Per-hemisphere flat arrays + event-level arrays from an iterate chunk."""
    hemi = {f: ak.to_numpy(ak.flatten(events[f"Hemisphere_{f}"]))
            for f in HEMI_FIELDS}
    hemi["n_jets"] = hemi["n_jets"].astype(int)
    nj2 = ak.to_numpy(
        ak.fill_none(ak.pad_none(events["Hemisphere_n_jets"], 2, clip=True), 0)
    ).astype(int)
    evt = {
        "thrust": ak.to_numpy(events["thrust"]),
        "split_min": nj2.min(axis=1),
    }
    return hemi, evt


def cutflow_labels(cf, n):
    """Best-effort stage labels for an uproot ``cutflow`` histogram (the slimmer
    writes a StrCategory); fall back to numbered stages if they can't be read."""
    try:
        return [str(s) for s in cf.to_hist().axes[0]]
    except Exception:
        return [f"stage{i}" for i in range(n)]


def infer_dataset(path, xs_keys):
    """The xs-JSON dataset whose name is embedded in the file's basename
    (mixed files keep the slimmed input's name), or None."""
    base = Path(path).name
    matches = [k for k in xs_keys if k in base]
    return max(matches, key=len) if matches else None


def scan_root_inputs(paths, xs_keys, tree, workers, progress):
    """Up-front pass over bare ROOT inputs for weighted mode.

    Infers each file's dataset from its name and sums cutflow[0] per dataset
    (the slice's xsec-normalisation denominator, cutflow-only files included).
    Fails loudly on a missing cutflow or an un-inferable dataset - silently
    skipping either would bias the weights.
    """
    datasets = {}
    for p in paths:
        ds = infer_dataset(p, xs_keys)
        if ds is None:
            raise SystemExit(
                f"Cannot infer the HT slice of {p} from its filename "
                "(no cross-section key matches). Pass --unweighted, or use a "
                "dataset JSON."
            )
        datasets[p] = ds

    def work(path):
        with uproot.open(path) as f:
            if "cutflow" not in f:
                raise ValueError(
                    f"{path} has no 'cutflow' histogram (partial/truncated "
                    "mixer output?) - re-run that job; silently including it "
                    "would undercount n_original."
                )
            return float(f["cutflow"].values()[0])

    pool = None
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=workers)
    iterator = pool.map(work, paths) if pool else map(work, paths)
    if progress and tqdm is not None:
        iterator = tqdm(iterator, total=len(paths), unit="file",
                        desc="inspecting files")
    n_original = {}
    for path, n0 in zip(paths, iterator):
        n_original[datasets[path]] = n_original.get(datasets[path], 0.0) + n0
    if pool is not None:
        pool.shutdown()
    return datasets, n_original


def resolve_jobs(args):
    """Flatten the inputs into ``([(path, tree, weight, dataset), ...], n_original)``.

    ``n_original`` maps dataset -> summed cutflow[0]. Where it is ``None``
    (unweighted bare-ROOT mode) the main loop accumulates it from each file's
    ``cutflow`` as the file is processed.
    """
    json_inputs = [p for p in args.inputs if p.endswith(".json")]
    root_inputs = [p for p in args.inputs if not p.endswith(".json")]
    if json_inputs and root_inputs:
        raise SystemExit("Pass either ROOT files or one dataset JSON, not both.")
    if len(json_inputs) > 1:
        raise SystemExit("Pass at most one dataset JSON.")

    if not json_inputs:
        tree = args.tree or "events"
        if args.unweighted:
            jobs = [(p, tree, 1.0, Path(p).stem) for p in root_inputs]
            return jobs, {Path(p).stem: None for p in root_inputs}
        with open(args.xs_json) as f:
            xs = json.load(f)
        print(f"inspecting {len(root_inputs)} file(s) for cutflow n_original "
              f"({args.inspect_workers} threads)...", flush=True)
        datasets, n_original = scan_root_inputs(
            root_inputs, list(xs), tree,
            workers=args.inspect_workers, progress=not args.no_progress,
        )
        weights = {ds: args.lumi * xs[ds]["xs_pb"] / n0
                   for ds, n0 in n_original.items()}
        for ds in sorted(n_original):
            n_files = sum(1 for d in datasets.values() if d == ds)
            print(f"[{ds}] {n_files} file(s), weight {weights[ds]:.4g}")
        jobs = [(p, tree, weights[datasets[p]], datasets[p]) for p in root_inputs]
        return jobs, n_original

    if args.unweighted:
        fileset = load_fileset(json_inputs[0], tree=args.tree,
                               skip_missing_tree=False)
        weights = {name: 1.0 for name in fileset}
    else:
        print(
            f"inspecting files in {json_inputs[0]} for events trees + "
            f"cutflow n_original ({args.inspect_workers} threads)...",
            flush=True,
        )
        fileset = load_fileset(
            json_inputs[0], tree=args.tree,
            workers=args.inspect_workers,
            progress=not args.no_progress,
        )
        with open(args.xs_json) as f:
            xs = json.load(f)
        missing = [name for name in fileset if name not in xs]
        if missing:
            raise SystemExit(
                f"No cross section in {args.xs_json} for: {missing}. "
                "Add them, or pass --unweighted."
            )
        weights = {
            name: args.lumi * xs[name]["xs_pb"] / ds["metadata"]["n_original"]
            for name, ds in fileset.items()
        }

    jobs = []
    n_original = {
        name: ds["metadata"].get("n_original") for name, ds in fileset.items()
    }
    for name, ds in fileset.items():
        print(f"[{name}] {len(ds['files'])} file(s), weight {weights[name]:.4g}")
        for path, tree in ds["files"].items():
            jobs.append((path, tree, weights[name], name))
    return jobs, n_original


def main():
    parser = argparse.ArgumentParser(
        description="Hemisphere kinematics from run3-mj-mixer output, split by "
        "hemisphere jet multiplicity (1-4), written as TH1D to a single ROOT "
        "file. Works on bare mixed ROOT files (HT slice inferred from each "
        "filename) or on a dataset JSON; xsec-weighted by default."
    )
    parser.add_argument("inputs", nargs="+",
                        help="mixed ROOT file(s), or one dataset JSON from "
                        "scripts/make_dataset_json.py")
    parser.add_argument("-o", "--output", default="hemisphere_histograms.root",
                        help="output ROOT file (default: %(default)s)")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: 'events', or the "
                        "dataset JSON's metadata)")
    parser.add_argument("--xs-json", default=str(DEFAULT_XS_JSON),
                        help="cross-section JSON (default: %(default)s)")
    parser.add_argument("--lumi", type=float, default=1.0,
                        help="integrated luminosity in pb^-1; 1.0 keeps pure "
                        "xs/N weights (default: %(default)s)")
    parser.add_argument("--unweighted", action="store_true",
                        help="fill with weight 1 instead of xsec weights")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress tqdm progress bars (e.g. when output is "
                        "redirected to a log file)")
    parser.add_argument("--inspect-workers", type=int, default=16,
                        help="threads for the up-front per-file cutflow "
                        "inspection in weighted mode (default: %(default)s)")
    args = parser.parse_args()
    use_bars = tqdm is not None and not args.no_progress

    jobs, n_original = resolve_jobs(args)
    # Datasets whose n_original wasn't known up front (unweighted bare-ROOT
    # mode) get it summed from each file's cutflow[0] as the file is read.
    accumulate = {name for name, n in n_original.items() if n is None}
    for name in accumulate:
        n_original[name] = 0.0
    defs = histogram_defs()
    book = HistogramBook(defs)
    print(
        f"\n{len(jobs)} file(s) to process | histograms: {sorted(defs)} "
        f"x njets {NJETS_CLASSES} | output: {args.output}"
    )

    branches = [f"Hemisphere_{f}" for f in HEMI_FIELDS] + EXTRA_BRANCHES
    n_events, n_skipped = 0, 0
    per_dataset = {}      # dataset -> [n_files, n_events, weight]
    n_hemis = {n: 0 for n in NJETS_CLASSES}   # unweighted hemisphere counts
    n_hemis_dropped = 0                       # 0- and 5-jet hemispheres
    cutflow_total = None  # np.ndarray of summed input cutflow bin values
    cf_labels = None      # stage labels of that input cutflow
    mixer_cf_total = None # summed mixer_cutflow (events read, exactly-5-jets)
    t0 = time.monotonic()

    file_bar = tqdm(jobs, unit="file", desc="files") if use_bars else jobs
    for path, tree_name, weight, dataset in file_bar:
        t_file = time.monotonic()
        with uproot.open(path) as f:
            if dataset in accumulate and n_original[dataset] is not None:
                if "cutflow" in f:
                    n_original[dataset] += float(f["cutflow"].values()[0])
                else:
                    echo(
                        f"[warn] {path}: no 'cutflow' histogram - n_original "
                        f"for '{dataset}' will be recorded as null"
                    )
                    n_original[dataset] = None
            # Sum the preselection cutflow across all files (cutflow-only files
            # included) for the output cutflow.
            if "cutflow" in f:
                vals = np.asarray(f["cutflow"].values(), dtype=np.float64)
                if cutflow_total is None:
                    cutflow_total = vals
                    cf_labels = cutflow_labels(f["cutflow"], len(vals))
                elif len(vals) == len(cutflow_total):
                    cutflow_total = cutflow_total + vals
                else:
                    echo(
                        f"[warn] {path}: cutflow has {len(vals)} bins, expected "
                        f"{len(cutflow_total)} - not summed into output cutflow"
                    )
            if "mixer_cutflow" in f:
                vals = np.asarray(f["mixer_cutflow"].values(), dtype=np.float64)
                mixer_cf_total = vals if mixer_cf_total is None else mixer_cf_total + vals
            if tree_name not in f:
                echo(f"[skip] {path}: no '{tree_name}' tree")
                n_skipped += 1
                continue
            tree = f[tree_name]
            if "Hemisphere_pt" not in tree:
                raise SystemExit(
                    f"{path} has no Hemisphere_* branches - is this mixer output?"
                )
            event_bar = (
                tqdm(total=tree.num_entries, unit="evt", leave=False,
                     desc=Path(path).name[:48])
                if use_bars else None
            )
            n_file = 0
            for events in tree.iterate(
                filter_name=branches, step_size=args.step_size
            ):
                hemi, evt = flatten_chunk(events)
                book.fill(hemi, evt, weight)
                for n in NJETS_CLASSES:
                    n_hemis[n] += int(np.count_nonzero(hemi["n_jets"] == n))
                n_hemis_dropped += int(
                    np.count_nonzero(~np.isin(hemi["n_jets"], NJETS_CLASSES))
                )
                n_file += len(events)
                if event_bar is not None:
                    event_bar.update(len(events))
            if event_bar is not None:
                event_bar.close()
        stats = per_dataset.setdefault(dataset, [0, 0, weight])
        stats[0] += 1
        stats[1] += n_file
        n_events += n_file
        dt_file = time.monotonic() - t_file
        per_event = 1e6 * dt_file / n_file if n_file else 0.0
        echo(
            f"[done] {dataset}: {Path(path).name} - "
            f"{n_file:,} events in {dt_file:.1f} s "
            f"({per_event:.1f} us/evt) ({n_events:,} total)"
        )
    if use_bars:
        file_bar.close()

    if not book.hists:
        raise SystemExit("No histograms filled - no usable input files?")

    elapsed = time.monotonic() - t0
    rate = n_events / elapsed if elapsed > 0 else 0.0
    per_event = 1e6 * elapsed / n_events if n_events else 0.0
    print(
        f"\n{n_events:,} events from {len(jobs) - n_skipped} file(s) in "
        f"{elapsed:.1f} s ({rate:,.0f} ev/s, {per_event:.1f} us/evt)"
        + (f" | {n_skipped} skipped" if n_skipped else "")
    )

    print(f"\n{'dataset':<58} {'files':>5} {'events':>12} {'weight':>10} "
          f"{'n_original':>12}")
    for name, (n_files, n_evts, weight) in per_dataset.items():
        n_orig = n_original.get(name)
        n_orig_s = f"{n_orig:,.0f}" if n_orig is not None else "null"
        print(f"{name[:58]:<58} {n_files:>5} {n_evts:>12,} {weight:>10.4g} "
              f"{n_orig_s:>12}")

    total_hemis = sum(n_hemis.values())
    print(f"\nhemispheres (unweighted): {total_hemis:,} in classes "
          + ", ".join(f"{n}j={n_hemis[n]:,}" for n in NJETS_CLASSES)
          + (f" | {n_hemis_dropped:,} dropped (0/5-jet)" if n_hemis_dropped else ""))

    print("\nhistograms (in-range weighted entries):")
    for key in sorted(book.hists):
        h = book.hists[key]
        print(f"  h_{key}: {h.sum().value:.6g}")

    extra = {"n_original": json.dumps(n_original)}

    # Output cutflow: the slimmer preselection stages, with the mixer's
    # exactly-5-jet selection appended (unweighted, like the rest).
    if cutflow_total is not None:
        labels = list(cf_labels)
        values = list(cutflow_total)
        if mixer_cf_total is not None:
            labels.append("exactly 5 jets [mixer]")
            values.append(float(mixer_cf_total[1]))
        cutflow = hist.Hist(
            hist.axis.StrCategory(labels), storage=hist.storage.Double()
        )
        cutflow.view()[:] = values
        extra["cutflow"] = cutflow
        print("\ncutflow (unweighted events):")
        for label, value in zip(labels, values):
            print(f"  {label:<42} {value:>15,.0f}")
    else:
        echo("[warn] no input 'cutflow' histogram found - none written to output")

    book.write(args.output, extra=extra)
    print(f"\nhistograms + n_original{' + cutflow' if cutflow_total is not None else ''} "
          f"written to {args.output}")


if __name__ == "__main__":
    main()
