"""fileset.py - Load a dataset JSON into a coffea-style fileset dictionary."""

import json
import warnings

DEFAULT_TREE = "events"

#: Cutflows written by stages downstream of the slimmer. A file carrying one of
#: these instead of ``cutflow`` is complete - it is just not a slimmed file, and
#: has no slimmer denominator to contribute. The mixer's assemble stage writes
#: ``chunk_cutflow`` (per-chunk counts only, so that hadd does not multiply the
#: global ones); ``stitch_cutflow`` and ``mixer_cutflow`` are its predecessors.
DOWNSTREAM_CUTFLOWS = ("chunk_cutflow", "stitch_cutflow", "mixer_cutflow",
                       "index_cutflow")


def _inspect_file(path, tree):
    """Open a slimmed file once and report ``(has_<tree>, cutflow[0])``.

    ``has_tree`` is whether the file contains the ``tree`` TTree; ``n_original``
    is the first ``cutflow`` bin (total events the slimmer read for this file =
    this file's contribution to the xsec-normalisation denominator).

    Raises ``ValueError`` if there is no ``cutflow`` histogram, distinguishing
    the two ways that happens: a truncated slimmer output (counting it would
    bias ``n_original``, so fail rather than skip), or a complete file from a
    later stage, which has no slimmer denominator to give and does not need one
    because it carries its own per-event weights.
    """
    import uproot

    with uproot.open(path) as f:
        has_tree = tree in f
        if "cutflow" not in f:
            downstream = [k for k in DOWNSTREAM_CUTFLOWS if k in f]
            if downstream:
                raise ValueError(
                    f"{path} has no 'cutflow' histogram, but it does have "
                    f"'{downstream[0]}': this is not slimmer output, it is a "
                    "complete file from a later stage. There is no xsec "
                    "denominator to read from it - and no need for one, since "
                    "these events carry their own per-event weight. Pass "
                    "--unweighted so no external weight is derived."
                )
            raise ValueError(
                f"{path} has no 'cutflow' histogram: the slimmer job that "
                "produced it did not finish (partial/truncated output). Re-run "
                "that job - silently including it would undercount n_original."
            )
        n_original = float(f["cutflow"].values()[0])
    return has_tree, n_original


def _inspect_all(pairs, tree, workers, progress):
    """Run ``_inspect_file`` over ``(dataset, path)`` pairs, in order.

    ``workers > 1`` uses a thread pool: the per-file cost is xrootd round-trip
    latency, not CPU, so threads give a near-linear speedup on large filesets.
    ``progress=True`` wraps the scan in a tqdm bar when tqdm is available.
    Exceptions from ``_inspect_file`` (e.g. a missing ``cutflow``) propagate.
    """
    def work(pair):
        return _inspect_file(pair[1], tree)

    pool = None
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=workers)
    iterator = pool.map(work, pairs) if pool else map(work, pairs)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator, total=len(pairs), unit="file", desc="inspecting files"
            )
        except ImportError:
            pass
    try:
        return list(iterator)
    finally:
        if pool is not None:
            pool.shutdown()


def load_fileset(json_path, tree=None, skip_missing_tree=True, workers=1,
                 progress=False):
    """Open a dataset JSON (from scripts/make_dataset_json.py) and return a
    coffea-style fileset dict ready for ``NanoEventsFactory.from_root`` or
    ``coffea.dataset_tools.apply_to_fileset``:

        {
            "<dataset>": {
                "files": {"root://.../file.root": "events", ...},
                "metadata": {"dataset": "<dataset>", "n_original": <float>},
            },
            ...
        }

    Parameters
    ----------
    json_path : str
        Path to the JSON written by make_dataset_json.py. The expected layout is
        ``{"metadata": {"tree": ...}, "datasets": {name: [paths]}}``, but a bare
        ``{name: [paths]}`` mapping is also accepted.
    tree : str, optional
        Tree name to associate with every file. Defaults to the JSON's
        ``metadata.tree`` if present, otherwise ``"events"``.
    skip_missing_tree : bool, default True
        Open each file once to:
          * drop files that have no ``<tree>`` TTree from the returned ``files``
            - e.g. a low-HT QCD slice where nothing passed the slimmer's HT cut,
            so the slimmer wrote a cutflow-only file with no events tree. This
            keeps downstream ``from_root`` from choking on a missing tree; and
          * sum each dataset's ``cutflow[0]`` over ALL files (including the
            dropped ones, whose original events still count toward the
            denominator) into ``metadata["n_original"]``.
        A file missing its ``cutflow`` (a partial slimmer output) raises rather
        than being silently skipped. Pass ``False`` for the old pure-JSON
        behaviour: no file I/O, every path kept, and no ``n_original``.
    workers : int, default 1
        Thread count for the per-file inspection. The cost is xrootd round-trip
        latency, so on large remote filesets ``workers=16`` is ~16x faster.
    progress : bool, default False
        Show a tqdm bar during the inspection (if tqdm is installed).
    """
    with open(json_path) as f:
        blob = json.load(f)

    metadata = blob.get("metadata", {})
    default_tree = tree or metadata.get("tree", DEFAULT_TREE)
    datasets = blob.get("datasets", blob)  # tolerate a bare {name: [paths]} map

    for name, paths in datasets.items():
        if not isinstance(paths, list):
            raise ValueError(
                f"Dataset '{name}' must map to a list of file paths, got "
                f"{type(paths).__name__}."
            )

    if not skip_missing_tree:
        return {
            name: {
                "files": {path: default_tree for path in paths},
                "metadata": {"dataset": name},
            }
            for name, paths in datasets.items()
        }

    # One flat inspection pass over every file (parallelizable), then
    # reassemble per dataset.
    pairs = [(name, path) for name, paths in datasets.items() for path in paths]
    results = _inspect_all(pairs, default_tree, workers, progress)

    acc = {name: {"files": {}, "n_original": 0.0, "skipped": []}
           for name in datasets}
    for (name, path), (has_tree, n_orig) in zip(pairs, results):
        acc[name]["n_original"] += n_orig
        if has_tree:
            acc[name]["files"][path] = default_tree
        else:
            acc[name]["skipped"].append(path)

    fileset = {}
    for name, paths in datasets.items():
        skipped = acc[name]["skipped"]
        if skipped:
            warnings.warn(
                f"[{name}] {len(skipped)}/{len(paths)} file(s) have no "
                f"'{default_tree}' tree (no events passed the slimmer cuts); "
                f"counted toward n_original but excluded from event reading. "
                f"First skipped: {skipped[0]}",
                stacklevel=2,
            )
        fileset[name] = {
            "files": acc[name]["files"],
            "metadata": {"dataset": name, "n_original": acc[name]["n_original"]},
        }
    return fileset
