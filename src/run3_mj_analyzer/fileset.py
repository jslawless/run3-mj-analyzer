"""fileset.py - Load a dataset JSON into a coffea-style fileset dictionary."""

import json
import warnings

DEFAULT_TREE = "events"


def _inspect_file(path, tree):
    """Open a slimmed file once and report ``(has_<tree>, cutflow[0])``.

    ``has_tree`` is whether the file contains the ``tree`` TTree; ``n_original``
    is the first ``cutflow`` bin (total events the slimmer read for this file =
    this file's contribution to the xsec-normalisation denominator).

    Raises ``ValueError`` if the file has no ``cutflow`` histogram: that means
    the slimmer job that wrote it did not finish (a truncated/partial output), so
    counting it would bias ``n_original`` - fail loudly rather than skip it.
    """
    import uproot

    with uproot.open(path) as f:
        has_tree = tree in f
        if "cutflow" not in f:
            raise ValueError(
                f"{path} has no 'cutflow' histogram: the slimmer job that "
                "produced it did not finish (partial/truncated output). Re-run "
                "that job - silently including it would undercount n_original."
            )
        n_original = float(f["cutflow"].values()[0])
    return has_tree, n_original


def load_fileset(json_path, tree=None, skip_missing_tree=True):
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
    """
    with open(json_path) as f:
        blob = json.load(f)

    metadata = blob.get("metadata", {})
    default_tree = tree or metadata.get("tree", DEFAULT_TREE)
    datasets = blob.get("datasets", blob)  # tolerate a bare {name: [paths]} map

    fileset = {}
    for name, paths in datasets.items():
        if not isinstance(paths, list):
            raise ValueError(
                f"Dataset '{name}' must map to a list of file paths, got "
                f"{type(paths).__name__}."
            )

        if not skip_missing_tree:
            fileset[name] = {
                "files": {path: default_tree for path in paths},
                "metadata": {"dataset": name},
            }
            continue

        files = {}
        n_original = 0.0
        skipped = []
        for path in paths:
            has_tree, n_orig = _inspect_file(path, default_tree)
            n_original += n_orig
            if has_tree:
                files[path] = default_tree
            else:
                skipped.append(path)

        if skipped:
            warnings.warn(
                f"[{name}] {len(skipped)}/{len(paths)} file(s) have no "
                f"'{default_tree}' tree (no events passed the slimmer cuts); "
                f"counted toward n_original but excluded from event reading. "
                f"First skipped: {skipped[0]}",
                stacklevel=2,
            )

        fileset[name] = {
            "files": files,
            "metadata": {"dataset": name, "n_original": n_original},
        }
    return fileset
