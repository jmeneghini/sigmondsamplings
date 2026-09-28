# sigmondsamplings

`sigmondsamplings` is a python package for manipulating and analyzing lattice QCD Monte-Carlo data in `sigmond` format. The package provides python-native plotting, statistical analysis, and data management features not present in the [C++ `sigmond` software suite](https://github.com/jmeneghini/sigmond)..  This package is also useful for those who wish to work with or convert `sigmond` files without the overhead of the full C++ `sigmond` library.

# Features

- **File Loading/Writing**: View and manipulate `sigmond` binary fstream (`.smp`) and HDF5 (`.hdf5`) file formats. Both Monte Carlo *bins* and *samplings* file types are supported. Note: only HDF5 files can be written to, and reading fstream files requires `sigmond_query`, which is packaged with `sigmond` (this is the only use of `sigmond` in this package).

- **NumPy-Style Arithmetic Ops**: Sampling/bin data is stored in `SigmondSampling`/`SigmondBins` objects, which wrap `np.ndarray` while preserving convenient `ufunc` arithmetic and enforcing observable/resampling compatibility checks. Complex observables are stored in `sigmond` files as separate real and imaginary parts, but are merged into a single NumPy array when loaded.

- **Statistical Ops**: Core `SigmondSampling`/`SigmondBins` objects provide basic statistical information such as resampling means, confidence intervals, and standard errors. Metrics involving multiple observables are computed with `SamplingStats` (e.g. covariance and $\chi^2$). Multi-ensemble analysis is also supported, with covariance between different ensembles automatically set to zero.

- **Convenient Many-Observable Data Structures**: In-memory `ObservableCollection` objects act as queryable containers, providing batched accessors along with `filter`, `find`, `group_by`, and `sort` methods. `EnergyObsInfo` can be constructed by flexibly parsing observable names and extracting level info, which can then be stored in specialized `Single/MultiEnsembleEnergyCollection` objects with accessors like `irreps`, `psq_values`, and `group_by_irrep()`.

- **Operators and Correlation Matrices**: Sigmond's BL/GI operator labels, `CorrT` correlator keys and VEV keys parse into immutable, hashable objects (`BasicOperator`, `GeneralOperator`, `CorrelatorKey`, `VEVKey`) that print Sigmond's canonical short form. Loaded correlators carry typed infos, so they filter and group like other observables. A metadata-only `CorrelatorIndex` (`loader.correlators`) finds them, and `index.matrix(operators)` assembles a complete, labeled `(time, sink, source)` matrix with Hermitian completion, VEV subtraction (bins are resampled jointly before subtracting) and complex basis changes. See `docs/operators_correlators_api_design.md`.

- **Plotting**: General observable plots, such as histograms and corner plots, are provided by `SamplingPlotter`. Spectrum plotting of energy levels is provided by `SpectrumPlotter`, and a WIP fit-result plotter lives in `fitting/fit_plotter.py`.

- **Runtime Configuration**: Global `rcparams` support package-level defaults for plotting behavior, metric selection, confidence levels, color/marker palettes, and the `KnownEnsembles` XML database path. Settings can be temporarily overridden or saved/loaded from TOML config files.

- **CLI Utilities**: The package installs a single `ss` command. `ss query` inspects/queries files (`groups`, `info`, `obs`, `energy`, `corr`); `ss convert`, `ss combine`, and `ss edit` write HDF5 (convert an fstream/HDF5 file, combine several sampling files, or edit an observable set).

  `ss edit` retags observables as energy levels, rewrites attributes, adds derived observables, and prunes, all on the way to a new file:

  ```bash
  # interpret energy attrs and attach non-interacting pairs
  ss edit in.h5 out.h5 --tag-energy --ni-yml ni.yml --ref-particle N

  # add E/M_N reference levels for every level
  ss edit in.h5 out.h5 --tag-energy --add-ref N

  # fix a mis-parsed observable and resync its canonical name
  ss edit in.h5 out.h5 --tag-energy -w name=badname_0 --set psq=2 --set irrep=E --rename

  # prune to a spectrum captured earlier by `ss query energy --save`
  ss edit in.h5 out.h5 --spec spectrum.toml --only
  ```

  Selection (`-w/--where`, `--contains`, `--regex`, `--spec`) is the same filter language `ss query` uses. A scope says which observables an operation *touches*; everything else passes through untouched, and only `--only`/`--drop` change what reaches the output. For multi-step edits that need different scopes per operation, write them to a TOML recipe and pass `--recipe`; `--save-recipe` records what a flag invocation ran, for provenance and replay.


## Installation

`sigmondsamplings` is currently intended to be installed from source.

### Using `pip`

For a normal editable install:

```bash
git clone <repository-url>
cd sigmondsamplings
pip install -e .
```

For development, install the optional developer tools as well:

```bash
pip install -e ".[dev]"
```

### Using `conda`

If you prefer to manage the Python environment with `conda`, create and activate
an environment first, then install the package with `pip` inside that environment:

```bash
conda create -n sigmondsamplings python=3.11
conda activate sigmondsamplings

git clone <repository-url>
cd sigmondsamplings
pip install -e .
```

For development:

```bash
conda create -n sigmondsamplings-dev python=3.11
conda activate sigmondsamplings-dev

git clone <repository-url>
cd sigmondsamplings
pip install -e ".[dev]"
```

The core Python dependencies are installed automatically by `pip` from
`pyproject.toml`:

- `numpy`
- `scipy`
- `uncertainties`
- `h5py`
- `xarray`

### `sigmond_query`

HDF5 workflows are pure Python. Reading binary fstream (`.smp`) files
requires the `sigmond_query` executable, which is packaged with
[`sigmond`](https://github.com/jmeneghini/sigmond/tree/pip). Make sure
`sigmond_query` is on your `PATH` before loading `.smp` files:

```bash
which sigmond_query
```

`sigmond_query` is only needed for reading fstream files; it is not required for
writing or manipulating HDF5 files.

## Quick Start

The examples below use the small HDF5 fixtures committed under `tests/data/`.
Replace the path with your own `.hdf5` file in normal use.

### Load and Inspect Data

```python
from pathlib import Path

from sigmondsamplings import SigmondLoader

filename = Path("tests/data/energy_levels_samplings.hdf5")

loader = SigmondLoader(str(filename))
observables = loader.observables

print(loader.file_kind)        # "samplings"
print(loader.group)            # auto-detected HDF5 root group, if unique
print(len(observables))

ensemble_info, sampling_info, observable_infos = loader.get_file_info()
print(ensemble_info)
print(sampling_info)
print(observable_infos[0])
```

`loader.observables` is an `ObservableCollection`, so it can be queried without
re-reading the file:

```python
energy = observables.find(name="K(0)_elab")
kaons = observables.filter(lambda obs: obs.name.startswith("K"))

print(len(kaons))
print(energy.pdg_format())
```

### Work with Individual Observables

Each element is a `SigmondSampling` or `SigmondBins` object, depending on the
file type. These objects wrap the underlying NumPy data while keeping the
observable and resampling metadata attached.

```python
sampling = observables[0]

print(sampling.observable_info.name)
print(sampling.full_sample_value)
print(sampling.mean)
print(sampling.error)
print(sampling.confidence_interval(0.68))

shifted = sampling - sampling.mean # creates a new observable
scaled = 2.0 * sampling
```

Arithmetic between observables checks that the resampling data are compatible.
For example, two samplings from the same file can be combined directly:

```python
ratio = observables[0] / observables[2]
print(ratio.mean, ratio.error)
```

### Compute Multi-Observable Statistics

Use `SamplingStats` for quantities that involve more than one observable, such
as covariance matrices, correlation matrices, residuals, and $\chi^2$ values.

```python
import numpy as np

from sigmondsamplings import SamplingStats

subset = [observables[i] for i in [0, 2, 3, 4, 5]] # any iterable, including collections
stats = SamplingStats(subset)

print(stats.cov_matrix)
print(stats.corr_matrix)

theory_values = np.array(stats.val.mean)
chi2 = stats.chi_squared(theory_values)
print(chi2)
```

Covariances between observables from different ensembles are automatically set
to zero, while same-ensemble observables use the usual resampling covariance.

### Use Energy-Level Metadata

Energy-level observables can be parsed into `EnergyObsInfo` and stored in
specialized energy collections:

```python
from sigmondsamplings import SingleEnsembleEnergyCollection

energy_levels = SingleEnsembleEnergyCollection.from_collection(observables)

print(energy_levels.irreps)
print(energy_levels.psqs)
print(energy_levels.group_by_irrep().keys())
print(energy_levels.filter(irrep = 'A1g', psq = 0))
print(energy_levels.filter(irrep = ['A1', 'A2'], psq = 1)) # returns collections

```

### Work with Correlation Matrices

The loader gives every `<CorrT>` correlator and `<VEV>` observable a typed
info (`CorrelatorObsInfo`, `VEVObsInfo`), so correlators can be queried like
any other observable (`filter(obs_kind="correlator", time=8)`,
`ss query obs f.hdf5 --where obs_kind=correlator psq=0`). Building the index and
listing its contents never reads samples. A matrix request reads only the
selected cells and times, and never zero-fills a missing cell.

```python
from sigmondsamplings import SigmondLoader

loader = SigmondLoader("corrs.hdf5", lazy=True)
at_t8 = loader.observables.filter(obs_kind="correlator", time=8)
index = loader.correlators                        # CorrelatorIndex, metadata only
print(index.channels)                             # candidate bases, by channel

ops = ["pion P=(0,0,0) A1um_1 SD_5 D4", "pion P=(0,0,0) A1um_1 SS_1"]
matrix = index.matrix(ops, times=range(4, 21))    # Hermitian by default
c01 = matrix.correlator(ops[0], ops[1])           # 1-D SigmondData over time
print(c01.sel(time=8).full_sample_value)
print(matrix.diagonal().mean)                     # xarray over (time, operator)
print(matrix.at(8).as_xarray())                   # the (sink, source) matrix at t=8
print(matrix.key_form, matrix.stored.sel(time=8).values)  # how the cells were read
```

`matrix.values` is an ordinary N-D `SigmondSampling`/`SigmondBins`, so
statistics, arithmetic, `.sel`/`.isel` and `.to_collection()` work as usual.
Operators may be given as short strings or as `BasicOperator`/`GeneralOperator`
objects. One stored key form (`"HermMat"`, `"plain"`, `"SubVEV"`, ...) is used
per request; if a file holds several for the basis, the request raises and
`key_form=` chooses. Connected correlators use `subtract_vev=True` (pass
`sampling_info=` when the file holds bins), basis changes use
`matrix.transform(T, new_gi_operators)` (`C_new = T^H C T`), and
`matrix.write("out.hdf5", group=...)` writes keys that Sigmond reads back. To
fit a correlator, pass the 1-D series to `SamplingFit.from_data` with a model of
`x`. It uses the series' `time` coord as `x` (`x=` picks the dim by name). Complex
data must be projected to a real component first:

```python
from sigmondsamplings import CallableXModel, ParamSetSpec, ParamSpec, SamplingFit

series = matrix.diagonal(component="real").sel(operator=matrix.operators[0].coordinate)
model = CallableXModel(lambda t, theta: theta[0] * np.exp(-theta[1] * t), ("A", "E"))
param_set = ParamSetSpec(values={"A": ParamSpec(initial=1.0), "E": ParamSpec(initial=0.3)}).resolve()
fit = SamplingFit.from_data(series, model, param_set)   # fit.x holds the times
result = fit.fit_resampled()
```

From the command line, `ss query corr` shows the same index:

```bash
ss query corr corrs.hdf5                          # stored series: sink, source, key form, times
ss query corr corrs.hdf5 --operators              # numbered operators (also --channels, --vevs)
ss query corr corrs.hdf5 --sink 0 --source 1      # one correlator over time
ss query corr corrs.hdf5 --matrix 0 --time 8      # a channel's matrix at t=8 (Hermitian by default)
ss query corr bins.hdf5 --matrix 0 --subtract-vev --sampling jackknife
```

Operators and channels are picked by their number in the listings or by label
(a short operator string, or a unique substring of the channel). Complex values
are shown as `re ± im i`; bins need `--sampling jackknife` or
`--sampling bootstrap:N[:SEED]` to subtract VEVs.

The benchmark in `benchmarks/correlator_matrix.py` times loading, the index
build, a one-time request and full assembly on a synthetic file.

### Write HDF5 Output

Only HDF5 output is written by this package. Use `SigmondWriter` to write a
subset, round-trip converted data, or newly constructed observables:

```python
from sigmondsamplings import SigmondWriter

writer = SigmondWriter(create_backups=False)
writer.write_hdf5("subset.hdf5", observables[:5], group="samplings", overwrite=True)
```

### Create Synthetic Data

Synthetic samplings are useful for tests, examples, and checking analysis code
without touching file IO:

```python
from sigmondsamplings import ObservableInfo, SamplingInfo, create_gaussian_sampling

sampling_info = SamplingInfo("bootstrap", 1000, seed=1234)
observable_info = ObservableInfo("synthetic", index=0, op_type="n", re_im="re")

synthetic = create_gaussian_sampling(
    mean=1.0,
    std=0.1,
    sampling_info=sampling_info,
    observable_info=observable_info,
)

print(synthetic.pdg_format())
```

See the `examples/` directory for complete scripts using the committed test
fixtures.

## Sampling Compatibility

Samplings are compatible for arithmetic operations if they have:
1. **Same `sampling_info`** (method, number of resamplings, seed)
2. **Same data length**
3. **Different `ensemble_info` is allowed** (enables multi-ensemble analysis)

### Outputs

- Arithmetic operations will create names for the output observables, then the user can edit these if desired

```python
result = observables[0] + observables[1]
result.observable_info.name = 'some_new_samp'
```

- To copy an observable such that its state isn't tied to the original, use `.copy()`

## Multi-Ensemble Analysis

SigmondSamplings automatically handles covariance between different ensembles:
- **Same ensemble**: Normal covariance calculation
- **Different ensembles**: Covariance = 0.0 (automatic)
- **Diagonal elements**: Always return variance (error²)


## Work in Progress

The project is still a work in progress, and will not support all sigmond observables. However, the hope
is the code is structured such that adding new observables is not too painful. Nonetheless, for analyzing spectrum data,
it is unlikely you'll run into any hurdles.

Some features exist as code but not have been fully tested/used. I would not waste your time. These include
- `fit.py`
- `fit_plotter.py`

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_loader_writer.py -v

# Run with coverage
python -m pytest tests/ --cov=sigmondsamplings --cov-report=html
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
