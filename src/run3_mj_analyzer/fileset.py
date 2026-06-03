"""fileset.py - Load a dataset JSON into a coffea-style fileset dictionary."""

import json

DEFAULT_TREE = "events"


def load_fileset(json_path, tree=None):
    """Open a dataset JSON (from scripts/make_dataset_json.py) and return a
    coffea-style fileset dict ready for ``NanoEventsFactory.from_root`` or
    ``coffea.dataset_tools.apply_to_fileset``:

        {
            "<dataset>": {
                "files": {"root://.../file.root": "events", ...},
                "metadata": {"dataset": "<dataset>"},
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
        fileset[name] = {
            "files": {path: default_tree for path in paths},
            "metadata": {"dataset": name},
        }
    return fileset
