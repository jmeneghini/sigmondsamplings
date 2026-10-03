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

`slat.plotting.export_capabilities(backend)` separates accepted formats, default
formats and advisory optional dependency availability. An installed engine can
still fail to export; callers must report that failure without substituting
formats. HTML can use CDN resources and is not guaranteed self-contained.
`slat.plotting.figure_scope()` closes allocated figures on failure; successful
factories retain the returned figure and transfer closing responsibility to the
caller.
