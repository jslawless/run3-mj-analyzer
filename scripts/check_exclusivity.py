#!/usr/bin/env python3
"""check_exclusivity.py - do a model's two tri-jet solutions share any jets?

The evaluator writes, per model, two candidate tri-jets per event as the index
branches ``{Model}Candidate_jetIdx{0,1,2}``, each shape ``(N, 2)``: column 0 is
solution t1, column 1 is solution t2. So the two solutions for an event are

    t1 = {jetIdx0[:,0], jetIdx1[:,0], jetIdx2[:,0]}
    t2 = {jetIdx0[:,1], jetIdx1[:,1], jetIdx2[:,1]}

This script reports, per model, the fraction of events whose two solutions
overlap, broken down by how many jets are shared, plus the rate of (degenerate)
repeated jets *within* a single solution.

Discovers the models present in the file(s) - no hardcoded model list - so it
also reports the GenJet pass (``{Model}Gen``) when present. Counts are
unweighted: exclusivity is a per-event property of a model, not a physics
spectrum, so raw event fractions are what you want.

Examples:
    python scripts/check_exclusivity.py evaluated_TTto4Q_*.root
    python scripts/check_exclusivity.py dataset_evaluated.json
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

try:
    from tqdm import tqdm
except ImportError:  # progress bars are cosmetic - everything runs without them
    tqdm = None

# A model's candidate collection is identified here by its first index branch
# (we need the indices, not the kinematics). The captured prefix is the
# evaluator's sanitized model label; a GenJet pass shows up as its own model.
CANDIDATE_IDX_RE = re.compile(r"^(.+)Candidate_jetIdx0$")


def echo(msg):
    """print() that doesn't tear through active tqdm bars."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


@dataclass
class Counts:
    """Running exclusivity tallies for one model."""

    n_events: int = 0
    # shared[k] = number of events whose two solutions share exactly k jets
    # (k in 0..3); shared[0] are the exclusive events.
    shared: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    # events with a repeated jet index *within* a single solution (t1 or t2).
    n_internal_dup: int = 0

    def add(self, idx0, idx1, idx2):
        """Accumulate one chunk. idx0/1/2 are (N, 2) int arrays."""
        # t{1,2} : (N, 3) - the three jet indices making up each solution.
        t1 = np.stack([idx0[:, 0], idx1[:, 0], idx2[:, 0]], axis=1)
        t2 = np.stack([idx0[:, 1], idx1[:, 1], idx2[:, 1]], axis=1)

        # Distinct shared jets between the two solutions. (N, 3, 3): does t1
        # column i equal t2 column j? A t2 column is "in t1" if its row has any
        # match; dedupe repeated values within t2 (first occurrence only) so a
        # degenerate t2 can't inflate the distinct-overlap count.
        t2_in_t1 = (t1[:, :, None] == t2[:, None, :]).any(axis=1)  # (N, 3)
        t2_first = np.ones_like(t2, dtype=bool)
        t2_first[:, 1] = t2[:, 1] != t2[:, 0]
        t2_first[:, 2] = (t2[:, 2] != t2[:, 0]) & (t2[:, 2] != t2[:, 1])
        n_shared = (t2_in_t1 & t2_first).sum(axis=1)  # (N,) in 0..3, exact

        self.shared += np.bincount(n_shared, minlength=4)

        # Repeated jet within a single solution (the other failure mode: an
        # argmax triplet need not have 3 distinct jets).
        def has_dup(t):
            return (
                (t[:, 0] == t[:, 1]) | (t[:, 0] == t[:, 2]) | (t[:, 1] == t[:, 2])
            )

        self.n_internal_dup += int(np.count_nonzero(has_dup(t1) | has_dup(t2)))
        self.n_events += len(t1)

    def merge(self, other):
        self.n_events += other.n_events
        self.shared += other.shared
        self.n_internal_dup += other.n_internal_dup


def discover_models(tree):
    """Candidate-collection prefixes present in ``tree`` (sorted)."""
    return sorted(
        m.group(1) for k in tree.keys() if (m := CANDIDATE_IDX_RE.match(k))
    )


def resolve_paths(inputs):
    """Flatten inputs into ``[(path, tree), ...]``.

    Accepts ROOT files or a single dataset JSON (from make_dataset_json.py).
    Always unweighted, so the JSON only needs its file list - no cutflow
    inspection pass.
    """
    json_inputs = [p for p in inputs if p.endswith(".json")]
    root_inputs = [p for p in inputs if not p.endswith(".json")]
    if json_inputs and root_inputs:
        raise SystemExit("Pass either ROOT files or one dataset JSON, not both.")
    if len(json_inputs) > 1:
        raise SystemExit("Pass at most one dataset JSON.")

    if not json_inputs:
        return [(p, "events") for p in root_inputs]

    # load_fileset is only needed for JSON input; import it lazily so ROOT-file
    # mode runs in any env with uproot/awkward/numpy (the analyzer package
    # __init__ also pulls in `vector`, which a lighter env may not have).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from run3_mj_analyzer.fileset import load_fileset

    fileset = load_fileset(json_inputs[0], tree=None, skip_missing_tree=False)
    jobs = []
    for name, ds in fileset.items():
        for path, tree in ds["files"].items():
            jobs.append((path, tree))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Report, per model, the fraction of events whose two "
        "tri-jet solutions share one or more jets (i.e. are NOT exclusive), "
        "from evaluator output."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="evaluated ROOT file(s), or one dataset JSON from "
        "scripts/make_dataset_json.py",
    )
    args = parser.parse_args()
    use_bars = tqdm is not None

    jobs = resolve_paths(args.inputs)
    print(f"{len(jobs)} file(s) to scan\n")

    totals = {}  # model -> Counts (global)
    n_skipped = 0
    t0 = time.monotonic()

    file_bar = tqdm(jobs, unit="file", desc="files") if use_bars else jobs
    for path, tree_name in file_bar:
        with uproot.open(path) as f:
            if tree_name not in f:
                echo(f"[skip] {path}: no '{tree_name}' tree")
                n_skipped += 1
                continue
            tree = f[tree_name]
            models = discover_models(tree)
            if not models:
                raise SystemExit(
                    f"{path} has no <Model>Candidate_jetIdx0 branches - "
                    "is this evaluator output?"
                )
            branches = [f"{m}Candidate_jetIdx*" for m in models]
            for events in tree.iterate(filter_name=branches):
                for m in models:
                    idx0 = ak.to_numpy(events[f"{m}Candidate_jetIdx0"])
                    idx1 = ak.to_numpy(events[f"{m}Candidate_jetIdx1"])
                    idx2 = ak.to_numpy(events[f"{m}Candidate_jetIdx2"])
                    totals.setdefault(m, Counts()).add(idx0, idx1, idx2)
    if use_bars:
        file_bar.close()

    if not totals:
        raise SystemExit("No events read - no usable input files?")

    elapsed = time.monotonic() - t0
    n_any = next(iter(totals.values())).n_events
    print(
        f"\nScanned {n_any:,} events from {len(jobs) - n_skipped} file(s) in "
        f"{elapsed:.1f} s"
        + (f" | {n_skipped} skipped (no tree)" if n_skipped else "")
    )

    def print_table(title, table):
        print(f"\n{title}")
        print(f"  {'model':<20} {'events':>12} {'shared%':>9} "
              f"{'=1':>8} {'=2':>8} {'=3':>8} {'dup%':>8}")
        for m in sorted(table):
            c = table[m]
            n = c.n_events or 1
            shared_any = c.shared[1:].sum()
            print(
                f"  {m:<20} {c.n_events:>12,} "
                f"{100.0 * shared_any / n:>8.3f}% "
                f"{100.0 * c.shared[1] / n:>7.3f}% "
                f"{100.0 * c.shared[2] / n:>7.3f}% "
                f"{100.0 * c.shared[3] / n:>7.3f}% "
                f"{100.0 * c.n_internal_dup / n:>7.3f}%"
            )

    print("\nExclusivity = the two tri-jet solutions share NO jet.")
    print("'shared%' = events whose solutions overlap (=1/=2/=3 jets shared); "
          "'dup%' = events with a repeated jet within one solution.")
    print_table("Per model (all files):", totals)


if __name__ == "__main__":
    main()
