# Proposed SigmondData shape

Status: proposed, 2026-10-16. Examples describe the target API, not the current
implementation. Companion to [the key/info proposal](key_info_redesign_proposal.md).

## The object

`SigmondData` is a labeled array of observables with an internal Monte Carlo
sample axis. A scalar observable is the zero-dimensional case. Use one concrete
public class for bins, jackknife and bootstrap, whether eager or lazy.

Keep `SigmondBins(...)` and `SigmondSampling(...)`, if desirable, as convenience
constructors returning this class. They do not require another implementation
hierarchy. There are no distinct array or lazy subclasses. Statistical behavior
dispatches through the validated sample context, as the current analyzer strategy
already largely does.

Conceptual implementation state:

```python
@dataclass(frozen=True, slots=True, eq=False)
class SigmondData:
    _buffer: SampleBuffer
    _context: SampleContext
    _layout: DataLayout
    _observations: ObservationMap
    _display: ContainerDisplay

    __hash__ = None
```

This is a design sketch; validation, caching and accessors are omitted. Its
fields have separate responsibilities:

| Field | Owns |
| --- | --- |
| `_buffer` | Numeric storage/deferred reads, dtype, full buffer shape |
| `_context` | Statistical interpretation, bin/ensemble specifications, resampling and alignment |
| `_layout` | Observable dimensions, index/auxiliary coordinates, retained scalar coordinates and semantic bindings |
| `_observations` | Aligned infos, estimate refs, optional stored origins, validity and shared provenance records |
| `_display` | Container title/default display; no competing physical identity |

There is one authoritative source for each fact. Keys are projected from infos;
ensemble/sampling metadata come from context; numeric dtype comes from buffer.
No independent `is_complex`, parent observable info, or duplicated physical
array attributes can drift out of sync.

## Shape contract

For observable shape `S`, the physical buffer has shape `S + (N,)`.
`shape`, `ndim` and `dims` describe `S`, excluding the sample axis.

| Data | `shape` | `dims` | Sample buffer shape |
| --- | --- | --- | --- |
| One energy | `()` | `()` | `(R + 1,)` |
| One correlator over time | `(T,)` | `("time",)` | `(T, R + 1)` |
| Correlator matrix over time | `(T, M, M)` | `("time", "sink", "source")` | `(T, M, M, R + 1)` |
| Energy estimates from different fits | `(F,)` | `("fit",)` | `(F, R + 1)` |
| Correlator matrix containing bins | `(T, M, M)` | `("time", "sink", "source")` | `(T, M, M, B)` |

For jackknife/bootstrap, slot 0 is the full-sample value and subsequent slots
are resamples. For bins, all slots are bins. The context validates the length:
jackknife count derives from bins; bootstrap count comes from the resolved plan.
The sample axis is not an ordinary selectable observable dimension.

`dtype` describes numeric storage. `info.key.part` describes scientific component
meaning. Promoting a Re-only record's buffer to complex does not recover its
missing imaginary measurement.

## Public access

| Access | Meaning |
| --- | --- |
| `.shape`, `.ndim`, `.dims`, `.coords` | Observable layout, no sample axis |
| `.dtype`, `.num_samples` | Numeric metadata, available without reading lazy values |
| `.context` | Immutable sample context |
| `.ensemble_info`, `.sampling_info` | Convenience projections where unambiguous; bins have no resampling plan |
| `.samples` | Read-only NumPy samples of the selected data; materializes if necessary |
| `.info`, `.key`, `.estimate` | Single-cell values; require `ndim == 0` |
| `.infos`, `.keys`, `.estimates` | Explicit inspection over observable shape |
| `.obs.psq`, `.obs.kind`, etc. | Registered aligned metadata projections |
| `.origins`, `.provenance` | Stored component locations and producer/input references |
| `.display` | Container presentation |
| `.full_sample_value`, `.mean`, `.error` | Statistics following the context |

An array's `.info` raises even if all its infos happen to compare equal. There
is no scan-dependent change in the meaning or return type of this accessor.
A caller selects a cell or uses an aligned inspection API.

The private observation map can contain constant, dense or coordinate-bound
fields. Public `.infos`, `.keys` and `.estimates` materialize detached xarray
inspection arrays on explicit access. Normal operations use the private mapping
without materializing that object grid. Editing an inspection DataArray cannot
change the source. Users can select data first to inspect a small subset.

Scalar `.obs` facets can return scalar values; array `.obs` facets return
DataArrays over the relevant observable layout. A common sector, where useful,
is an explicit computed summary. It is never stored as a second authoritative
physical sector on the container.

## Selection preserves identity

Target example, with a sampled matrix `C`:

```python
C.shape                         # (48, 12, 12)
C.dims                          # ("time", "sink", "source")
C.samples.shape                 # (48, 12, 12, 1001)

series = C.sel(sink=op, source=op)
series.shape                    # (48,)
series.samples.shape            # (48, 1001)
series.ts.key                   # family key with time unbound

point = series.sel(time=8)
point.shape                     # ()
point.samples.shape             # (1001,)
point.key == series.ts.key.at(time=8)  # True
point.info.sector_info
point.estimate
```

`C` here is the shape-generic data object (`matrix.values` when using the
domain wrapper). Operator-coordinate normalization is supplied by that producer.

Indexing selects samples, observation mappings, validity, origins and estimate
refs together. The layout retains scalar coordinates after dimension removal,
while the evaluated semantic point is part of the selected key. Selection does
not construct an unrelated placeholder observable or allocate a new estimate.
Basic indexing may share sample memory; advanced indexing may require a copy.
Both preserve identity.

`.ts.key` requires one established temporal family after other choices are
selected. It is not inferred from labels. The binding identifies physical time
even if its layout dimension is renamed to `t`. Display offsets belong to the
definition/axis convention; they never replace source-time coordinates.

## Statistics and array operations

Statistics reduce only the internal sample axis. Scalar statistics return
Python/NumPy scalar values; array statistics return xarray DataArrays over
observable dimensions and coordinates. Those are numerical summaries, not
resampled `SigmondData` objects.

```python
C.error                         # DataArray(time, sink, source)
point.error                     # scalar
C.central("median")             # DataArray using supported context convention
```

Unsupported metrics raise based on context. For example, a percentile-bootstrap
interval does not silently become a jackknife interval.

Supported NumPy ufuncs execute over sample buffers. The Python layer validates
sample alignment and observable-coordinate alignment, then applies the matching
metadata producer policy. Named coordinates align exactly; mismatched labels do
not silently drop data. Compatible broadcasting does not manufacture observations.

```python
ratio = numerator / denominator
real = C.component("real")
trials = SigmondData.stack(energies, dim="fit", coords=fit_ids)
eff = series.component("real").ts.effective_energy(step=1)
```

Generic arithmetic creates a derived definition/output ref. Domain helpers can
declare a more specific quantity. Component projection creates a canonical
component key. Stack preserves input keys and refs, including repeated semantic
keys from different fit outputs. Bin-wise nonlinear operations retain explicit
statistical meaning and cannot impersonate a function of ensemble means.

Implement NumPy functions only where their dimension/sample and metadata
semantics are defined. Do not silently reuse old scientific metadata after an
unsupported reduction or transform. `.samples`/`np.asarray(data)` provide explicit
raw-array access when users want ordinary NumPy semantics without SS metadata.

## Construction and validation

Provide short scalar constructors and raw-buffer constructors, while loaders,
matrix builders, fits and domain operations normally supply metadata themselves:

```python
energy = SigmondSampling(samples, info=energy_info, context=sample_context)
bins = SigmondBins(bin_values, info=simple_info, context=bin_context)

array = SigmondData.from_samples(
    values,
    dims=("time",),
    coords={"time": times},
    infos=point_infos,
    context=sample_context,
)
```

`from_samples` can accept a validated bound metadata specification internally;
ordinary callers can supply explicit infos. It must validate rank/dim uniqueness,
coordinates and bindings, metadata shape, native capability constraints,
numeric component meaning and sample-axis length/context. Factories normalize
the supported forms into the same stored state.

One array has one declared sample space. Different physical definitions and
estimates can coexist inside it. Different sample contexts require grouping or
an explicit supported joint context. A multi-ensemble collection is valid without
forcing it into one dense array. Joint contexts retain ensemble membership and
relationships; scalar convenience access never pretends they contain one ensemble.

## Ownership, laziness and editing

The public object is immutable and unhashable. Numerical dataset comparison uses
`.equals()`; semantic comparison uses keys, and result comparison uses refs.
Public access cannot rebind infos/context or mutate owned samples/coordinates.
Default construction detaches external writable buffers; trusted internal views
share library-owned frozen storage.

The buffer backend can be an owned ndarray, a selection of existing storage,
or a deferred selected read from a source. Metadata, dtype, shape and selection
are available before materialization. Reading `.samples` materializes only the
selected block. `.materialize()` returns an eager handle with the same scientific
identity; a backend may cache reads internally. This does not require lazy
subclasses or a public general-purpose lazy expression engine.

Native kernels consume selected arrays and explicit settings. Their outputs are
wrapped by the same Python metadata/context machinery. The public data object is
not a proxy for an `MCObsHandler` registry entry.

Metadata-only replacements share samples:

```python
pretty = energy.with_display(latex=r"E_0")
tagged = energy.with_annotations(study="window_scan")
```

They preserve semantic keys and estimate refs. Scientific reinterpretation or
replacement of samples creates a new typed output linked to its source. It
does not mutate results already held by an index or collection.

An explicit editor owns writable candidate state for batched work. Freeze validates
the complete result and detaches it from future editor edits. A `.copy()` can
copy sample storage while sharing immutable metadata and preserving estimate refs;
physical copying alone does not create a new measurement.

## Collection and domain boundaries

`.to_collection()` yields scalar observations with the same keys, sample context,
estimate refs, presentation and origins. Missing cells remain absent. A retained
layout manifest allows reconstruction, including layout-only fit/trial dimensions
whose quantities may repeat. `.from_collection(shape_by=...)` without a manifest
is an explicit validated pivot with collision/missingness rules.

`CorrelatorMatrix` remains a small domain wrapper around `values: SigmondData`,
plus validated basis/form, optional VEV data, coverage and transformation artifacts.
Its selection and export consume the values' same authoritative metadata. Pivots
and fit results are similarly domain results containing or producing ordinary
`SigmondData`, not additional numeric container hierarchies.

The invariant is that every selected observable has one coherent set of samples,
context, meaning and result identity, regardless of whether it came from a
scalar constructor, an array selection, a collection, native code or a file.

This document specifies the design only. It does not change runtime code.
