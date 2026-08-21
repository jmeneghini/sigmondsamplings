"""
Consolidated per-dataset metadata table for Sigmond HDF5 files.

Historically each observable component dataset carried its energy annotation as
individual HDF5 attributes, costing one metadata read per attribute (thousands
per file, dominating load time). Instead we write a single ``ObsMeta`` dataset
beside ``Values``: a 1-D variable-length UTF-8 array holding one JSON object per
component, recording the dataset ``key``, its sample ``shape`` and ``dtype``,
plus optional annotations such as energy attrs or explicit LaTeX labels. One
read recovers everything, and the lazy index reports shapes without opening a
single dataset.

Files predating this table (and real Sigmond files) simply lack ``ObsMeta``;
:func:`read` returns an empty mapping so callers fall back to per-dataset attrs.
"""

import json

import h5py
import numpy as np

# Name of the consolidated table, written as a sibling of ``Values``.
DATASET = "ObsMeta"

# Reserved JSON keys. ``shape``/``dtype`` describe the component array; ``key``
# is the join back to its ``Values`` dataset. None collide with annotation names
# (obs_kind, irrep, psq, energy_type, level_index, ref_particle, ni_pairs,
# latex_str).
_KEY = "key"
_SHAPE = "shape"
_DTYPE = "dtype"


def _jsonable(value):
    """Coerce a numpy scalar/array attr value to JSON-serializable Python."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fields_for(arr: np.ndarray, attrs: dict | None) -> dict:
    """Metadata fields for one component: sample shape/dtype plus any energy attrs."""
    fields = {_SHAPE: list(arr.shape), _DTYPE: str(arr.dtype)}
    if attrs:
        fields.update({key: _jsonable(value) for key, value in attrs.items()})
    return fields


def shape_of(fields: dict | None) -> tuple[int, ...] | None:
    """Sample shape recorded in ``fields``, or None when unavailable (legacy files)."""
    if fields is not None and _SHAPE in fields:
        return tuple(fields[_SHAPE])
    return None


def read(group: h5py.Group) -> dict[str, dict]:
    """Read the ``ObsMeta`` table into ``{dataset_key: fields}`` (``{}`` if absent)."""
    if DATASET not in group:
        return {}
    meta: dict[str, dict] = {}
    for raw in group[DATASET][:]:
        row = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        meta[row.pop(_KEY)] = row
    return meta


def write(group: h5py.Group, meta: dict[str, dict]) -> None:
    """Write/replace the ``ObsMeta`` table for ``group`` from ``{key: fields}``."""
    if DATASET in group:
        del group[DATASET]
    if not meta:
        return
    rows = [json.dumps({_KEY: key, **fields}) for key, fields in meta.items()]
    group.create_dataset(DATASET, data=np.array(rows, dtype=object),
                         dtype=h5py.string_dtype("utf-8"))
