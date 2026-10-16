# Plotting API design: matplotlib + plotly backends

Status: phases 1 and 2 implemented; phases 3 and 4 outstanding.
Scope: `sigmondsamplings`, `slat`, and `kbfit`.

## 1. Why

Today every plot in both packages is written directly against `matplotlib.axes.Axes`:

- `sigmondsamplings/plotter.py` — `SamplingPlotter` (histogram, errorbar, correlation
  matrix, eff. sample size, bootstrap intervals, summary grid, corner)
- `sigmondsamplings/spectrum_plotter.py` — `SpectrumPlotter` / `SectorSpectrumPlotter`
- `sigmondsamplings/fitting/fit_plotter.py` — `plot_fit_result`, `plot_chi2_{1,2}d`,
  `plot_chi2_function_{1,2}d`
- `kbfit/plotting/box_quantization/` — the `omega` / `intersection` / `lanes` /
  `eigenvalues` panel plugins, orchestrated by `kbfit/quantization_plotter.py`

`fit_plotter._plot_chi2_2d_plotly` is already a one-off plotly escape, and kbfit already
lists `plotly` + `dash` in its `extra` optional deps. The two backends serve different
jobs, and that split is what the design is organised around:

| backend | job |
| --- | --- |
| matplotlib | publication figures — PDF/PGF, `text.usetex`, exact sizing |
| plotly | exploration and HTML reports — hover labels, zoom, `kbfit.report` embedding |

The point is *not* backend parity. It is that the ~15 plots worth having in both places
get written once.

## 2. Non-goals

Stated up front so the abstraction stays small:

- **No pixel-identical output.** A plotly spectrum plot will not match the mpl one.
- **No general rcParams ↔ template bridge.** Only the handful of keys in `PlotStyle`.
- **Not abstracted at all:** corner plots, `LineCollection` colour gradients, 3D
  surfaces, `clabel` placement, `constrained_layout`/`tight_layout`, `savefig`
  kwargs, `patches.*`, and the ipywidgets/ipympl interactive path.
- **No auto colour cycling.** Frontends already pass explicit colours from
  `slat.colors.IndexedCycle`; that stays a hard rule, because mpl cycles per-Axes and
  plotly cycles per-trace and the two will never agree.

## 3. Layout

```
slat/plotting/
    __init__.py      # figure(), wrap(), use(), backend(), current()
    base.py          # Axes / Figure protocols, UnsupportedFeature
    style.py         # PlotStyle, palette + marker cycles (wraps slat.colors)
    text.py          # ensure_math() / to_html() / to_plotly() label handling
    _mpl.py          # MatplotlibBackend
    _plotly/         # PlotlyBackend (units, conversion, axes, figure layout, post_script.js)
    recording.py     # RecordingBackend, for tests

sigmondsamplings/plotting/   # domain frontends, re-exports the slat core
kbfit/plotting/              # unchanged location; panels retargeted onto slat.plotting
```

Core lives in `slat` because it has no heavy deps of its own and both packages already
depend on it (kbfit via `sigmondsamplings`, plus its own `slatmeta`→`slat` import in
`constants.py`). matplotlib and plotly each stay an optional extra.

Note: `sigmondsamplings/__init__.pyi:13` already declares a `plotting` submodule that
does not exist — accessing `sg.plotting` currently fails. This fills that slot.

### Naming

The drawing surface is called **`Axes`**, not `Panel`. kbfit already uses `Panel` for its
plugin descriptor (`kbfit/plotting/box_quantization/_panel.py`), and `RowSpec` stays
readable as `tuple[Axes, int | None]` — identical to today, just a different `Axes`.

## 4. Three tiers

1. **Portable** — the mark vocabulary in §5 plus axes configuration. Written once.
2. **Backend-specific** — code catches `UnsupportedFeature` (raised by the optional
   `contour`), or reaches through `ax.native` to do backend-specific work.
3. **Single-backend functions** — e.g. `plot_corner` (mpl), `plot_chi2_surface`
   (plotly). Live in the same namespace, documented as such, raise a clear error when
   the active backend is wrong.

## 5. The mark vocabulary

Twelve primitives, plus one optional. Everything else is composed in numpy in the frontend.

The public methods below are concrete on `base.Axes`: they own the signatures,
defaults and docstrings, and each forwards every argument by keyword to a protected
hook of the same name (`line` -> `_line`). A backend implements only the hooks.
Line widths, marker sizes and font sizes are in points; `None` takes matplotlib's
default on every backend. Every mark also takes `native=` (see §4).

```python
class Axes(ABC):
    backend: str
    figure: Figure
    native: Any          # plt.Axes  |  (go.Figure, row, col)

    # marks
    def line(self, x, y, *, color=None, linewidth=None, linestyle="-", marker=None,
             markersize=None, label=None, alpha=None, zorder=None, hover=None,
             fill=True) -> None: ...
    def points(self, x, y, *, color=None, markersize=None, marker="o",
               label=None, alpha=None, zorder=None, hover=None, fill=True) -> None: ...
    def errorbar(self, x, y, *, yerr=None, xerr=None, color=None, marker="o",
                 markersize=None, capsize=None, linewidth=None, linestyle=None,
                 label=None, alpha=None, zorder=None, hover=None, fill=True) -> None: ...
    def band(self, x, lo, hi, *, color=None, alpha=0.25, label=None, zorder=None) -> None: ...
    def bar(self, x, height, *, width=0.8, color=None, label=None, alpha=None,
            hover=None) -> None: ...                 # width: bar width, data units
    def polygon(self, x, y, *, facecolor=None, edgecolor=None, linewidth=None,
                alpha=None, label=None, zorder=None) -> None: ...
    def hlines(self, y, xmin, xmax, *, color=None, linestyle="-", linewidth=None,
               alpha=None, label=None, zorder=None, hover=None) -> None: ...
    def vlines(self, x, ymin, ymax, *, color=None, linestyle="-", linewidth=None,
               alpha=None, label=None, zorder=None) -> None: ...
    def axhline(self, y, *, color=None, linestyle="-", linewidth=None, alpha=None,
                label=None, zorder=None, hover=None) -> None: ...      # full span
    def axvline(self, x, *, ymin=0.0, ymax=1.0, ...same as axhline) -> None: ...
    def heatmap(self, values, *, x=None, y=None, cmap="viridis", vmin=None, vmax=None,
                origin="lower", colorbar=True, colorbar_label=None, hover=None) -> None: ...
    def text(self, x, y, s, *, coords="data", ha="left", va="bottom",
             fontsize=None, color=None, alpha=None) -> None: ...
    def contour(self, x, y, z, *, levels, labels=False, colors=None,
                filled=False, linewidth=None) -> None: ...   # optional; may raise UnsupportedFeature

    # configuration
    def set(self, *, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
            xscale=None, yscale=None, label_fontsize=None) -> None: ...
    def ticks(self, axis, positions, labels=None, *, rotation=0, fontsize=None) -> None: ...
    def grid(self, show=True, *, axis="both", alpha=None) -> None: ...
    def legend(self, show=True, *, loc=None, ncol=1, frameon=False, fontsize=None) -> None: ...
    def legend_group(self): ...  # context manager for one composed trace item

    # sizing
    def point_size(self, points, axis="y", *, units="data") -> float: ...
    def marker_extent(self, markersize: float) -> float: ...   # y data units
```

The pre-rename keywords (`width=` and `style=` for line width and style, `points(size=)`,
`contour(linewidths=)`) were removed outright; passing one raises `TypeError`.

`slat.plotting.figure(...)` resolves the style (with any `figsize` applied) into a frozen
`base.FigureSpec` and hands that to the backend's `make_figure(spec)`.

Use `with axes.legend_group():` around trace marks that describe one item,
labelling the first visible mark. Plotly gives the scope a fresh, subplot-local
identity and one legend entry; a click toggles every trace in that scope. Nested
scopes describe separate items. Axis-line shapes keep independent entries and
toggle their hover traces with them. Matplotlib and recording retain their usual
label behavior. Interval boxes, filled/hollow series and fit bands with their
central/model lines use this convention.

Deliberately *derived* rather than primitive:

- **histogram** → `np.histogram` + `bar` in the frontend. Guarantees identical binning
  across backends, and drops `ax.hist` vs `go.Histogram` semantics from the surface.
- **confidence ellipse** (`fit_plotter._plot_noisy_x_points`) → parametrise in numpy,
  emit `polygon`. Kills the `matplotlib.patches.Ellipse` dependency.
- **stacked positions** → `stacked_positions` keeps its pure-numpy core; the
  marker-size correction moves behind `Axes.marker_extent` (see §7).

### Backend mapping

| facade | matplotlib | plotly |
| --- | --- | --- |
| `line` | `ax.plot` | `go.Scatter(mode="lines")` |
| `points` | `ax.scatter` | `go.Scatter(mode="markers")` |
| `errorbar` | `ax.errorbar` | `go.Scatter(error_y=dict(array=...))` |
| `band` | `ax.fill_between` | two `Scatter`, `fill="tonexty"` |
| `bar` | `ax.bar` | `go.Bar` |
| `polygon` | `ax.fill` | `Scatter(fill="toself")` |
| `hlines`/`vlines` | `ax.hlines`/`ax.vlines` | `Scatter` with `None`-separated segments |
| `axhline`/`axvline` | `ax.axhline`/`ax.axvline` | `fig.add_hline(row=, col=)` |
| `heatmap` | `pcolormesh` + `fig.colorbar` | `go.Heatmap(colorbar=...)` |
| `text` | `ax.text(transform=)` | annotation with `xref="x"`/`"paper"` |
| `contour` | `ax.contour` + `clabel` | `go.Contour(contours.showlabels=)` |
| `ticks` | `set_xticks`/`set_xticklabels` | `update_xaxes(tickvals=, ticktext=, tickangle=)` |
| `legend` | `ax.legend` | one plotly legend per subplot (`legend`, `legend2`, …), hidden until shown, placed in the subplot |
| `grid` | `ax.grid` | `update_xaxes(showgrid=)` |
| `set(yscale=)` | `set_yscale` | `update_yaxes(type="log")` |

Silently ignored on plotly, documented per-field: `markevery`. `zorder` maps
to plotly's integer trace `zorder` (×10); an axis line is a shape, so it can only go under
every trace (`zorder < 1.5`, like matplotlib's gridlines) or over every trace.

Sizes given in points (marker sizes, line widths, cap sizes, font sizes) are converted
to plotly pixels at a fixed 100 px per inch, and a size left as `None` takes
matplotlib's default value, so a plotly figure is a scaled copy of the matplotlib one.
`style.dpi` sets only the raster export resolution on both backends.

## 6. Figure, layout, and output

```python
class Figure(Protocol):
    backend: str
    native: Any
    def axes(self, row=0, col=0) -> Axes: ...
    def suptitle(self, text, *, fontsize=None) -> None: ...
    def save(self, path, *, dpi=None) -> None: ...   # png/pdf/svg both; html plotly-only
    def embed_html(self) -> str: ...                 # <img data:...> (mpl) | <div> (plotly)
    def show(self) -> None: ...
    def close(self) -> None: ...

def figure(rows=1, cols=1, *, figsize=None, height_ratios=None, width_ratios=None,
           sharex=False, sharey=False, style=None, backend=None) -> Figure
```

`figure(rows=n, height_ratios=[...], sharex=True)` maps cleanly onto
`fig.add_gridspec(...)` and `make_subplots(rows=n, cols=1, row_heights=...,
shared_xaxes=True)`. This is exactly what `kbfit.quantization_plotter._create_layout`
builds today, and what `SamplingPlotter.plot_stats_summary` needs.

For created Plotly grids, `save`, `embed_html` and `show` preserve cells that
contain only titles, text or shapes by appending null-coordinate traces at
output time. Existing trace indices stay unchanged; hidden cells receive no
placeholder and nulls contribute no autorange values. Direct exports through
`figure.native` bypass this step. Wrapped native layouts retain their own
export behavior.

`embed_html()` is the piece that makes `kbfit/report/base.py` backend-agnostic:
`PendingFigure.write` currently hardcodes `fig.savefig(...)`. With `embed_html`, an HTML
report gets interactive plotly figures and a markdown/LaTeX report gets a saved image,
from the same panel code.

### Backend selection

```python
import slat.plotting as slp

slp.use("plotly")                     # process default
with slp.backend("plotly"):  ...      # scoped
fig = slp.figure(backend="plotly")    # per call
```

Every plot function grows `backend: str | None = None`, resolved from context.

## 7. The genuinely leaky bits

**Pixel metrics.** `utils.stacked_positions` used to convert marker points to y data
units through the mpl transform (`markersize=..., ax=ax`, now removed) — that needs a
renderer and a fixed figure size. `Axes.marker_extent(markersize)` becomes the capability: mpl implements it
via the transform, plotly estimates from the axis range and a nominal panel height.
The result is that plotly stacking is approximate. That is acceptable; it is the one
place the abstraction is visibly leaky and it should be documented as such.

**LaTeX in labels.** mpl uses mathtext (or `usetex`); plotly uses MathJax, which works in
titles and axis titles but is unreliable in *tick labels and legend entries* — precisely
where irrep names live in spectrum plots. `slat.plotting.text` provides `ensure_math(s)`
plus a `latex_to_unicode(s)` fallback covering the symbols actually used (Greek,
sub/superscripts, `\pi`, `\Delta`, `\bar`). Plotly tick labels and legends route through
the HTML fallback (`to_html`). Titles, axis labels and annotations go through
`to_plotly`, which merges a mixed string such as `"$68\%$ CI"` into one MathJax span
(MathJax drops plain text around a span) and falls back to HTML for multi-line text. `slat/labels.py` already owns
`get_irrep_latex_str`, so this belongs next to it.

**Style.** One `PlotStyle` dataclass (palette, marker cycle, font sizes, grid alpha,
default figsize). mpl translates it to an `rc_context`; plotly to a template. The
existing `sg.rc["plot.*"]` and `kbfit.rcparams` keys feed into it and keep working.

## 8. Plot-by-plot disposition

| plot | tier | note |
| --- | --- | --- |
| `plot_sampling_histogram` | portable | `np.histogram` + `bar`, `axvline` |
| `plot_sampling_errorbar` | portable | |
| `plot_correlation_matrix` | portable | `heatmap` |
| `plot_effective_sample_size` | portable | `bar` |
| `plot_bootstrap_intervals` | portable | `polygon` per interval |
| `plot_stats_summary` | portable | `figure(rows, cols)` |
| `plot_corner` | **mpl-only** | third-party `corner` package |
| `SpectrumPlotter` / `SectorSpectrumPlotter` | portable | needs `marker_extent`; plotly gains hover labels per level — the biggest single win |
| `plot_fit_result` | portable | ellipses → `polygon`, cloud → `points`, band, `text` |
| `plot_chi2_1d` | portable | except the `LineCollection` gradient variant → **mpl-only** |
| `plot_chi2_2d` | portable | `heatmap` + `contour` |
| `plot_chi2_function_2d(plot_type="surface")` | **plotly-only** | already is |
| kbfit `omega`, `intersection`, `eigenvalues` | portable | `line`/`points`/`axvline`/`axhline` |
| kbfit `lanes` | portable | `vlines`/`errorbar`/`points` + `marker_extent` |
| kbfit `quantization_plotter` layout | portable | rows + height ratios + sharex |
| kbfit `interactive()` | **mpl-only** | ipywidgets + ipympl; a plotly `FigureWidget`/dash path is a separate feature, not an abstraction |

kbfit's `_apply_axes_settings` currently forwards arbitrary keys via
`getattr(ax, f"set_{key}")`. That becomes `ax.set(**known_keys)`, with unknown keys
warned and dropped — the generic forwarding cannot survive two backends.
`_auto_ylim` is pure numpy on data and is unaffected.

## 9. Testing

`slat.plotting.recording.RecordingBackend` is a third backend that appends every mark
call to a list instead of drawing. Tests assert on structure without rendering:

```python
with slp.backend("recording") as rec:
    plot_fit_result(x, y, model)
assert [m.kind for m in rec.marks] == ["band", "line", "line", "errorbar"]
assert rec.marks[0].kwargs["alpha"] == 0.22
```

This is fast, needs neither matplotlib nor plotly installed, and catches the class of
bug where a frontend silently stops emitting an element. Smoke tests per real backend
then only need to check that nothing raises.

## 10. Migration

Existing signatures take `ax: plt.Axes | None` and return `plt.Axes` or
`(plt.Figure, plt.Axes)`. Plan:

- Add `axes: Axes | None = None`. (`ax=` was kept as a deprecated alias for a while and
  has since been removed; `axes=` itself accepts a raw `plt.Axes` via `slp.wrap`.)
- Return `Axes`. It exposes `.native` and a `.mpl` property that returns the underlying
  `plt.Axes` (raising on non-mpl backends) for callers that need it —
  `cli/plot.py::_finish_plot` becomes `fig.save(output)` / `fig.show()`.

Phases:

1. `slat/plotting/` — protocols, `_mpl.py`, `style.py`, `text.py`, `recording.py`, tests.
   Port `SamplingPlotter` and `SpectrumPlotter`. No user-visible change; mpl only.
2. `_plotly.py` + `Figure.embed_html` + spectrum hover labels. Validate both backends on
   the §8 portable set.
3. Port `fit_plotter`; split the mpl-only chi2 extras (gradient line, corner) into
   clearly-named single-backend functions.
4. kbfit: `RowSpec` → `tuple[slat.plotting.Axes, int | None]`, port the four panels,
   route `report/base.py` through `embed_html`. `interactive()` stays mpl-only.

Phases 1–2 are independently useful; kbfit does not have to move for sigmondsamplings to
benefit.

### Completed correctness contracts (2026-10-05)

Wrapped Plotly figures retain their native layout and expose primary Cartesian
cells. Native grid holes and non-Cartesian cells have no portable Axes; they
remain exportable. Without native grid metadata, cells are ordered by domains,
top to bottom and left to right. Ambiguous domain layouts raise an explicit
error; secondary/overlay axes remain accessible through `native`. Wrapper
outputs preserve trace-free Cartesian cells without reordering existing traces.

Limits are reported in data units and survive linear/log scale changes.
Matched Plotly axes share explicit range and scale updates. `point_size` uses
inner subplot dimensions; nonlinear data offsets are local midpoint estimates,
and their sign follows the supplied points even on inverted axes. Automatic
browser margins can still make pre-render size estimates approximate.

Series plots preserve explicit caller limits, and their automatic ranges include
kept data, marker bands and drawn fit curves. Filtered data can still display
reference or fit overlays. `SeriesStyle.marker_cycle=None` uses the figure's
`PlotStyle.markers` palette; set an explicit cycle to choose other symbols.

Contours require finite, strictly increasing levels. Filled contours shade
adjacent level intervals and leave values outside the level range unfilled;
labels apply to contour lines. Both rendering backends follow these semantics.
