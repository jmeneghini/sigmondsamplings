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
    text.py          # mathtext() / latex_to_unicode() label handling
    _mpl.py          # MatplotlibBackend
    _plotly.py       # PlotlyBackend
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
2. **Capability-gated** — code calls `ax.supports("contour")` / catches
   `UnsupportedFeature`, or reaches through `ax.native` to do backend-specific work.
3. **Single-backend functions** — e.g. `plot_corner` (mpl), `plot_chi2_surface`
   (plotly). Live in the same namespace, documented as such, raise a clear error when
   the active backend is wrong.

## 5. The mark vocabulary

Ten primitives, plus one optional. Everything else is composed in numpy in the frontend.

```python
class Axes(Protocol):
    backend: str
    figure: Figure
    native: Any          # plt.Axes  |  (go.Figure, row, col)

    # marks
    def line(self, x, y, *, color=None, width=None, style="-", marker=None,
             markersize=None, label=None, alpha=None, zorder=None) -> None: ...
    def points(self, x, y, *, color=None, size=None, marker="o",
               label=None, alpha=None, zorder=None) -> None: ...
    def errorbar(self, x, y, *, yerr=None, xerr=None, color=None, marker="o",
                 markersize=None, capsize=None, width=None,
                 label=None, alpha=None, zorder=None) -> None: ...
    def band(self, x, lo, hi, *, color=None, alpha=0.25, label=None, zorder=None) -> None: ...
    def bar(self, x, height, *, width=0.8, color=None, label=None, alpha=None) -> None: ...
    def polygon(self, x, y, *, facecolor=None, edgecolor=None, linewidth=None,
                alpha=None, label=None, zorder=None) -> None: ...
    def hlines(self, y, xmin, xmax, *, color=None, style="-", width=None,
               alpha=None, label=None, zorder=None) -> None: ...
    def vlines(self, x, ymin, ymax, *, ...) -> None: ...
    def axhline(self, y, **kw) -> None: ...      # full span
    def axvline(self, x, **kw) -> None: ...
    def heatmap(self, values, *, x=None, y=None, cmap="RdBu", vmin=None, vmax=None,
                colorbar_label=None) -> None: ...
    def text(self, x, y, s, *, coords="data", ha="left", va="bottom",
             fontsize=None, color=None, alpha=None) -> None: ...
    def contour(self, x, y, z, *, levels, labels=False, colors=None,
                filled=False) -> None: ...        # optional; may raise UnsupportedFeature

    # configuration
    def set(self, *, xlabel=None, ylabel=None, title=None,
            xlim=None, ylim=None, xscale=None, yscale=None) -> None: ...
    def ticks(self, axis, positions, labels=None, *, rotation=0, fontsize=None) -> None: ...
    def grid(self, show=True, *, axis="both", alpha=None) -> None: ...
    def legend(self, show=True, *, loc=None, ncol=1, frameon=False, fontsize=None) -> None: ...

    # capability / escape hatch
    def supports(self, feature: str) -> bool: ...
    def point_size(self, points, axis="y", *, units="data") -> float: ...
    def marker_extent(self, markersize: float) -> float: ...   # y data units
```

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
| `legend` | `ax.legend` | `update_layout(showlegend=, legend=)` |
| `grid` | `ax.grid` | `update_xaxes(showgrid=)` |
| `set(yscale=)` | `set_yscale` | `update_yaxes(type="log")` |

Silently ignored on plotly, documented per-field: `capsize`, `capthick`, `markevery`,
`zorder` (plotly orders by trace insertion — frontends must not rely on it).

## 6. Figure, layout, and output

```python
class Figure(Protocol):
    backend: str
    native: Any
    def axes(self, row=0, col=0) -> Axes: ...
    def suptitle(self, text) -> None: ...
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

**Pixel metrics.** `utils.stacked_positions(markersize=..., ax=ax)` converts marker
points to y data units through the mpl transform — it needs a renderer and a fixed
figure size. `Axes.marker_extent(markersize)` becomes the capability: mpl implements it
via the transform, plotly estimates from the axis range and a nominal panel height.
The result is that plotly stacking is approximate. That is acceptable; it is the one
place the abstraction is visibly leaky and it should be documented as such.

**LaTeX in labels.** mpl uses mathtext (or `usetex`); plotly uses MathJax, which works in
titles and axis titles but is unreliable in *tick labels and legend entries* — precisely
where irrep names live in spectrum plots. `slat.plotting.text` provides `mathtext(s)`
plus a `latex_to_unicode(s)` fallback covering the symbols actually used (Greek,
sub/superscripts, `\pi`, `\Delta`, `\bar`). Plotly tick labels and legends route through
the fallback unless `rc["plot.plotly_mathjax"]` is set. `slat/labels.py` already owns
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

- Add `axes: Axes | None = None`; keep `ax=` as a deprecated alias that goes through
  `slp.wrap(mpl_axes)`, so `plot_fit_result(..., ax=my_ax)` keeps working.
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
