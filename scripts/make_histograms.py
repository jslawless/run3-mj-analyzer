#!/usr/bin/env python3
"""make_histograms.py - per-model spectra from evaluator output, in one ROOT file.

The evaluator writes, for every model in its config, a two-candidate tri-jet
collection ``{Model}Candidate_{pt,eta,phi,mass,jetIdx0..2}`` (and, when run on
GenJets, ``{Model}GenCandidate_*``). This script *discovers* those collections
in the input files - no hardcoded model list - and fills, per model, the
tri-jet invariant mass spectrum (both candidates per event). All histograms
land in a single output ROOT file as TH1D (with Sumw2), named
``h_<histogram>_<Model>``, e.g. ``h_mass_SPANet``.

Two input modes:

  * one or more evaluated ``.root`` files (e.g. a TTto4Q sample): unweighted.
  * a single dataset ``.json`` (from scripts/make_dataset_json.py) listing the
    evaluated QCD HT slices: each slice is filled with weight
    ``lumi * xs_pb / n_original``, where ``xs_pb`` comes from ``--xs-json``
    (default: run3-mj-pass-the-aux/mj_samples_xs.json) and ``n_original`` is
    the slice's cutflow[0] sum computed by run3_mj_analyzer.load_fileset.
    Pass ``--unweighted`` to skip the weighting (e.g. a ttbar dataset JSON).

Adding histograms: add an entry to ``histogram_defs()`` below. A ``per_model``
definition is instantiated once per discovered candidate collection; a
non-per-model one (event-level quantities) is filled once per chunk - list any
extra branches it needs in ``EXTRA_BRANCHES``.

Example:
    python scripts/make_histograms.py evaluated_TTto4Q_*.root -o ttbar_spectra.root
    python scripts/make_histograms.py dataset_evaluated.json -o qcd_spectra.root
"""

import argparse
import json
import re
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

# A model's candidate collection is identified by its mass branch. The prefix is
# the evaluator's sanitized model label ("Mass Asymmetry" -> "Mass_Asymmetry");
# a GenJet pass shows up as its own "model" (e.g. "SPANetGen").
CANDIDATE_MASS_RE = re.compile(r"^(.+)Candidate_mass$")

# Event-level branches read in addition to the candidate collections. Extend
# this when a future (non-per-model) histogram def needs them.
EXTRA_BRANCHES = []

# Cross sections live in the shared aux repo, assumed checked out next to
# run3-mj-analyzer (same convention as the notebooks).
DEFAULT_XS_JSON = (
    Path(__file__).resolve().parents[2] / "run3-mj-pass-the-aux" / "mj_samples_xs.json"
)


@dataclass
class HistDef:
    """One histogram family: an axis plus how to compute its fill values."""

    axis: object  # hist axis, shared by every instance of this family
    values: callable  # (events, model) -> flat array of fill values
    per_model: bool = True  # instantiate per candidate collection vs. once


def histogram_defs(args):
    """Every histogram this script produces. Add new entries here."""
    return {
        "mass": HistDef(
            axis=hist.axis.Regular(
                args.bins, *args.range, name="mass",
                label="tri-jet invariant mass [GeV]",
            ),
            values=lambda events, model: ak.flatten(
                events[f"{model}Candidate_mass"], axis=1
            ),
        ),
    }


class HistogramBook:
    """Lazily-created histograms keyed ``<def>_<model>`` (or ``<def>``)."""

    def __init__(self, defs):
        self.defs = defs
        self.hists = {}

    def fill(self, events, models, weight):
        for dname, hdef in self.defs.items():
            for model in models if hdef.per_model else [None]:
                key = f"{dname}_{model}" if model else dname
                h = self.hists.get(key)
                if h is None:
                    h = self.hists[key] = hist.Hist(
                        hdef.axis, storage=hist.storage.Weight()
                    )
                values = ak.to_numpy(hdef.values(events, model))
                h.fill(values, weight=weight)

    def write(self, path):
        with uproot.recreate(path) as f:
            for key in sorted(self.hists):
                f[f"h_{key}"] = self.hists[key]


def discover_models(tree):
    """Candidate-collection prefixes present in ``tree`` (sorted)."""
    return sorted(
        m.group(1) for k in tree.keys() if (m := CANDIDATE_MASS_RE.match(k))
    )


def resolve_jobs(args):
    """Flatten the inputs into ``[(path, tree, weight, dataset), ...]``."""
    json_inputs = [p for p in args.inputs if p.endswith(".json")]
    root_inputs = [p for p in args.inputs if not p.endswith(".json")]
    if json_inputs and root_inputs:
        raise SystemExit("Pass either ROOT files or one dataset JSON, not both.")
    if len(json_inputs) > 1:
        raise SystemExit("Pass at most one dataset JSON.")

    if not json_inputs:
        tree = args.tree or "events"
        return [(path, tree, 1.0, Path(path).stem) for path in root_inputs]

    if args.unweighted:
        # No n_original needed, so skip the per-file inspection pass entirely;
        # files without an events tree are caught by the [skip] in the main
        # loop instead.
        fileset = load_fileset(json_inputs[0], tree=args.tree,
                               skip_missing_tree=False)
        weights = {name: 1.0 for name in fileset}
    else:
        # The xsec weights need n_original from every file's cutflow, so each
        # file is opened once up front. Over xrootd this is latency-bound -
        # hence the thread pool - but on thousands of files it still takes a
        # while; the progress bar is there so it doesn't look hung.
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
    for name, ds in fileset.items():
        print(f"[{name}] {len(ds['files'])} file(s), weight {weights[name]:.4g}")
        for path, tree in ds["files"].items():
            jobs.append((path, tree, weights[name], name))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Per-model tri-jet mass spectra from evaluator output, "
        "written as TH1D to a single ROOT file. Works on a plain sample "
        "(ttbar ROOT files, unweighted) or on xsec-weighted QCD HT slices "
        "(dataset JSON)."
    )
    parser.add_argument("inputs", nargs="+",
                        help="evaluated ROOT file(s), or one dataset JSON from "
                        "scripts/make_dataset_json.py")
    parser.add_argument("-o", "--output", default="histograms.root",
                        help="output ROOT file (default: %(default)s)")
    parser.add_argument("--tree", default=None,
                        help="events tree name (default: 'events', or the "
                        "dataset JSON's metadata)")
    parser.add_argument("--bins", type=int, default=100,
                        help="mass histogram bins (default: %(default)s)")
    parser.add_argument("--range", type=float, nargs=2, default=(0.0, 1500.0),
                        metavar=("LO", "HI"),
                        help="mass histogram range in GeV (default: 0 1500)")
    parser.add_argument("--xs-json", default=str(DEFAULT_XS_JSON),
                        help="cross-section JSON for dataset-JSON inputs "
                        "(default: %(default)s)")
    parser.add_argument("--lumi", type=float, default=1.0,
                        help="integrated luminosity in pb^-1; 1.0 keeps pure "
                        "xs/N weights (default: %(default)s)")
    parser.add_argument("--unweighted", action="store_true",
                        help="dataset-JSON mode: fill with weight 1 instead of "
                        "xsec weights")
    parser.add_argument("--step-size", default="500 MB",
                        help="uproot.iterate chunk size (default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress tqdm progress bars (e.g. when output is "
                        "redirected to a log file)")
    parser.add_argument("--inspect-workers", type=int, default=16,
                        help="threads for the up-front per-file tree/cutflow "
                        "inspection in dataset-JSON mode (default: %(default)s)")
    args = parser.parse_args()
    use_bars = tqdm is not None and not args.no_progress

    jobs = resolve_jobs(args)
    defs = histogram_defs(args)
    book = HistogramBook(defs)
    print(
        f"\n{len(jobs)} file(s) to process | histograms: {sorted(defs)} "
        f"(bins={args.bins}, range={args.range[0]:g}-{args.range[1]:g}) | "
        f"output: {args.output}"
    )

    all_models, n_events, n_skipped = set(), 0, 0
    per_dataset = {}  # dataset -> [n_files, n_events, weight]
    t0 = time.monotonic()

    file_bar = tqdm(jobs, unit="file", desc="files") if use_bars else jobs
    for path, tree_name, weight, dataset in file_bar:
        t_file = time.monotonic()
        with uproot.open(path) as f:
            if tree_name not in f:
                echo(f"[skip] {path}: no '{tree_name}' tree")
                n_skipped += 1
                continue
            tree = f[tree_name]
            models = discover_models(tree)
            if not models:
                raise SystemExit(
                    f"{path} has no <Model>Candidate_mass branches - "
                    "is this evaluator output?"
                )
            new_models = set(models) - all_models
            if new_models:
                echo(f"[models] found {sorted(new_models)} in {Path(path).name}")
                all_models.update(new_models)
            branches = [f"{m}Candidate_*" for m in models] + EXTRA_BRANCHES
            event_bar = (
                tqdm(total=tree.num_entries, unit="evt", leave=False,
                     desc=Path(path).name[:48])
                if use_bars else None
            )
            n_file = 0
            for events in tree.iterate(
                filter_name=branches, step_size=args.step_size
            ):
                book.fill(events, models, weight)
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
    print(f"models: {sorted(all_models)}")

    print(f"\n{'dataset':<58} {'files':>5} {'events':>12} {'weight':>10}")
    for name, (n_files, n_evts, weight) in per_dataset.items():
        print(f"{name[:58]:<58} {n_files:>5} {n_evts:>12,} {weight:>10.4g}")

    print("\nhistograms (in-range weighted entries):")
    for key in sorted(book.hists):
        h = book.hists[key]
        print(f"  h_{key}: {h.sum().value:.6g}")

    book.write(args.output)
    print(f"\nhistograms written to {args.output}")


if __name__ == "__main__":
    main()
