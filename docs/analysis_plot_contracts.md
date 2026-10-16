# Shared diagnostic and spectrum contracts

`sigmondsamplings.plot_contracts` contains strict, lightweight schemas and
`PlotDefinition` metadata. Importing it does not load data or rendering engines.
`get_plot_definition(name, family="sampling"|"diagnostic")` supplies options,
style, supported backends, data validation and a factory returning `PlotResult`.
The SS CLI and KBfit use that factory. Sampling methods support matplotlib and
plotly except corner (matplotlib plus the optional `corner` package). Optimagic
history diagnostics require plotly and optimization history or a trace file.
Synthetic fit parameter samplings remain usable and are labeled as synthetic.

```python
from sigmondsamplings.plot_contracts import get_plot_definition
result = get_plot_definition("histogram").make(
    collection, options={"bins": 20}, backend="plotly")
try:
    result.figure.save("histogram.html")
finally:
    result.figure.close()
```

`UncertaintyOptions` resolves ambient choices at the factory boundary. Spectrum
centers use full-sample values; bootstrap/bin interval bounds stay exact even
when they exclude the center. Jackknife data rejects percentile intervals.
Other sampling diagnostics retain their documented mean-based centers.

Summary metadata uses `statistics.panels`, ordered like the drawn panels.
Each histogram records its `observable_index`, `method`, central metric and
error/interval choice (bootstrap defaults to confidence level 0.68). Errorbar
panels record mean/error; correlation and effective sample size panels record
only their panel name.

```python
from sigmondsamplings.plotting import prepare_spectrum, draw_spectrum
prepared = prepare_spectrum(levels, options={"percentiles": (10, 90)},
                            overlays=ordinary_reference_markers)
# Query range-derived overlays, such as NI levels, using prepared.energy_range.
axes = draw_spectrum(prepared, overlays=range_derived_markers, backend="plotly")
```

Preparation computes uncertainty and hover statistics once without constructing
a figure. Drawing uses the stored style and payload, with stacking based on the
final axes limits and backend pixel metrics. Ordinary overlays can expand an
automatic range during preparation; explicit limits win. Additional overlays
passed to drawing never expand it. Prepared styles cannot be replaced at draw
time. No quantization or K-matrix logic belongs to SS.

Prepared marker values, hover, styles and metadata are immutable snapshots.
Editing source markers, styles or sampling buffers cannot change a prepared
drawing. `HMarker.from_sampling` also snapshots its drawing values and hover
at construction and is immutable; construct a new marker for different choices.
Plain `HMarker` instances remain editable.

`slat.plotting.export_capabilities(backend)` separates accepted formats, default
formats and advisory optional dependency availability. An installed engine can
still fail to export; callers must report that failure without substituting
formats. HTML can use CDN resources and is not guaranteed self-contained.
`slat.plotting.figure_scope()` closes allocated figures on failure; successful
factories retain the returned figure and transfer closing responsibility to the
caller. Nested retained allocations transfer to the parent scope, so a later
outer failure closes them. Caller-supplied axes are never adopted by a factory's
failure cleanup.
