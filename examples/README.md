# Examples

These examples use the small HDF5 fixtures committed under `tests/data/`, so
they can be run from a source checkout without any external data.

```bash
python examples/load_hdf5.py
python examples/write_hdf5.py
```

Reading binary fstream (`.smp`) files requires `sigmond_query`, but these
examples use HDF5 files and are pure Python.

These examples are currently rather limited. I plan to extend these in the future.
