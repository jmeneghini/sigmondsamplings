# Plotting API refactor plan

Handoff for an implementing agent. Written 2026-10-05 from a review of the code
as it stands in the working tree: `slat/plotting/*`, `sigmondsamplings/plotting/*`,
`sigmondsamplings/plot_contracts.py`, and kbfit's callers under
`kbfit/src/kbfit/project/{plots,analysis}`.

The work is split into tiers ordered so that each tier makes the later ones
smaller or simpler, and never redoes an earlier one:

1. Make **local, behaviour-neutral fixes** first, so later moves carry corrected code.
2. Make **every behaviour change inside the plotly backend before it is split**.
   The split (Tier 6) can then be a pure move, reviewable as a rename.
3. **Add new slat API before deleting old API**, so callers have somewhere to go.
4. **Delete dead public API before structural rewrites**, so nothing dead gets
   ported into a new structure.
5. **Leave the one breaking, all-touching change (Tier 7) for last.** It is
   approved but deferred. By the time it is done, the code it rewrites is at
   its smallest.

**Original implementation scope (historical): Tiers 1–6.** Tiers 7–8 were
subsequently completed as recorded below. The full review at the end of this
plan audits Tier 9 and adds Tiers 10–11. All plotting tiers are now completed
using the audit's revised remedies; see the implementation records below.
The unrelated lint cleanup in 9.8 remains outside scope.

---

## 0. Ground rules

- **Python.** Use the `lqcd-dev` env binaries directly. `conda activate` fails
  silently in this tool environment.
  - `/opt/homebrew/Caskroom/miniconda/base/envs/lqcd-dev/bin/python`
  - `/opt/homebrew/Caskroom/miniconda/base/envs/lqcd-dev/bin/ruff`
- **Baseline.** Before Tier 1, record the failing-test *set* in sigmondsamplings:
  ```bash
  /opt/homebrew/Caskroom/miniconda/base/envs/lqcd-dev/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)" | sort > <scratchpad>/baseline.txt
  ```
  About 76 failures and 6 errors are expected and unrelated. After each task,
  compare the *set* of failures, not the count.
- **kbfit.** For any task that touches kbfit or an API kbfit uses
  (`prepare_spectrum`, `draw_spectrum`, `HMarker`, `PlotDefinition`,
  `export_capabilities`, `slp.as_figure`), run, from `../kbfit`:
  ```bash
  /opt/homebrew/Caskroom/miniconda/base/envs/lqcd-dev/bin/python -m pytest tests/python -q -p no:cacheprovider
  ```
- **Visual check.** For any task touching plotly output (Tiers 2, 6, 7),
  render each frontend to PNG on both backends and compare at full resolution.
  `docs/plotly_mpl_parity_findings.md` → *How to check* has the procedure. kaleido is installed.
- **Uncommitted work.** Both repos have large amounts of uncommitted work from
  earlier sessions.
  - Do not commit unless the user asks.
  - Do not revert or reformat code outside the task.
  - The subsequent user instruction to finish all remaining tiers authorizes
    continuing between tier milestones; report progress without pausing.
- Run `ruff check` on the touched files before finishing each task.
- Match the surrounding style: comment density, docstring tone, naming.

## Decisions (made by the user, 2026-10-05)

| ID | Question | Decision | Where it applies |
|---|---|---|---|
| **D1** | Should frontends force `axes.grid(True)`, or follow `PlotStyle.grid` (default `False`)? | **Follow the style.** No frontend forces a grid on. | 3.4, 4.4 |
| **D2** | Public removals: hard-remove, or deprecate for a release first? | **Hard-remove.** No deprecation warnings or shims, including style fields that TOML files may set. | Tier 4; Tier 7 renames (decided when Tier 7 was done) |
| **D3** | Do Tier 7 (template-method `Axes` plus parameter rename)? | **Yes, eventually.** Deferred; not part of the current pass. | Tier 7 |

## Summary

| Tier | Theme | Behaviour change | Touches kbfit |
|---|---|---|---|
| 1 | Local cleanups in frontends and kbfit glue | none, except 1.1 under a non-default matplotlib font size | yes (1.7, 1.8) |
| 2 | Plotly backend internals, before the split | 2.4, 2.5 (small, intended) | no |
| 3 | New slat API, then adopt it; grid follows the style | 3.3 changes the error type and timing of `use()`; 3.4 removes forced grids | yes |
| 4 | Remove dead public API (hard removal) | API and style-field removals | yes (tests and style TOML) |
| 5 | Structural: spectrum draw path, plot catalog, NI hover | none | yes (keep the listed surface) |
| 6 | Split `_plotly.py` into a package (pure move) | none | no |
| 7 | *(done)* `Axes` template method and parameter names | public API rename, hard (no shim) | no direct calls |
| 8 | *(done)* Post-pass review follow-ups | 8.1 (figure style), 8.2 (NI hover text) | yes (8.2) |
| 9 | *(done)* Second review follow-ups, using the audited remedies below | legend toggles and empty-cell preservation | API validation |
| 10 | *(done)* Correctness follow-ups from the full review | wrapping, ranges, ownership, metadata, legends, preparation | yes |
| 11 | *(done)* Measured improvements and integration coverage | contours and the series marker palette | validation |

---

## Tier 1: local cleanups, no behaviour change

These are independent and can be done in any order.

### 1.1 One font-size fallback in the spectrum plotter
- **Why.** `SpectrumPlotter` resolves "specific size, else `style.fontsize`" in
  three places:
  - `_draw_xticks`
  - `_draw_outer_labels`
  - `_draw_markers`

  The marker-label copy uses `or` (so a size of 0 counts as unset). The other
  two use `is None`.

  `_draw_outer_labels` then falls back to a literal `fs = 10`
  (`spectrum.py:447`). This is now redundant: `None` already means 10 pt on
  both backends, since plotly reproduces matplotlib's defaults. On matplotlib it
  is also wrong, because it overrides the user's `rcParams["font.size"]`.
- **Do.**
  - Add `def _fontsize(self, specific): return specific if specific is not None else self.style.fontsize`.
  - Use it in all three places, and delete `fs = 10`.
  - If Tier 5.1 lands first, put the helper where the draw functions end up.
- **Behaviour.** Identical, unless a matplotlib user has changed `font.size`.
  Their outer labels then follow it, as every other label already does.
- **Verify.** `tests/test_spectrum_plotter.py`. Add a recording-backend test
  showing that `outer_label_fontsize=None, fontsize=None` passes `fontsize=None`
  to `text`.

### 1.2 *(folded into 3.4)*
This was "drop the literal `alpha=0.3` from the sampling plots' grid calls".
With D1 decided, 3.4 deletes those calls outright, so editing them here first
would touch the same lines twice. Skip it.

### 1.3 `_matrix_figsize(n)` in `correlators.py`
- **Why.** `side = min(12.0, 3.0 + 0.45 * n); figsize = (side + 1.5, side)` is
  duplicated in `plot_coverage` (~line 322) and `plot_heatmap` (~line 626),
  each wrapped in the same "no axes, no ax, no figsize" guard.
- **Do.** Add a module-level `_matrix_figsize(n) -> tuple[float, float]` and
  call it from both places.

### 1.4 Collapse the histogram interval branches (`sampling.py:161`)
- **Why.** The bootstrap and jackknife branches each draw two `axvline`s with
  the same colour, style and alpha. Only the bounds, legend label and hover
  differ, and the bootstrap branch also sets `xlim`.
- **Do.**
  - In the branch, compute `lower, upper, label, hover`, and set `xlim` (bootstrap only).
  - After the branch, draw the two `axvline`s once.
  - Keep the mark order and kwargs exactly as now; the recording tests check them.

### 1.5 Small `SamplingPlotter` cleanups
- `plot_bootstrap_intervals(confidence_levels=[0.68, 0.95])` is a mutable
  default (`sampling.py:601`). Make it `(0.68, 0.95)`.
- `plot_stats_summary` dispatches through a nested `_draw` if-chain. Replace it
  with a dict from panel name to bound method. Histogram is the only panel that
  takes an index, so keep that special case visible.
- The tick-label limit is hard-coded as `n_samplings <= 20` in
  `plot_sampling_errorbar` and `plot_bootstrap_intervals`, while
  `plot_effective_sample_size` has a `max_labels=20` parameter. Add a
  module-level `_MAX_TICK_LABELS = 20` and use it in all three. Keep the
  `max_labels` parameter, defaulting to that constant.

### 1.6 `HMarker.from_sampling`: resolve the error metric once
- **Why.** It works out the effective metric by hand twice
  (`markers.py:108` and `:114`, `(error_metric or rc["plot.error_metric"]) == ...`).
  `UncertaintyOptions.resolved()` already does this, and
  `full_sample_and_bounds` calls it too.
- **Do.**
  ```python
  resolved = UncertaintyOptions(error_metric=error_metric,
                                confidence_level=confidence_level,
                                percentiles=percentiles).resolved()
  interval = resolved["error_metric"] == "confidence_interval"
  ```
  - Use `err = float(sampling.error)` when not `interval`; otherwise use the
    signed offsets.
  - Use the same `interval` for the hover arguments.
  - Equivalence: the validator forbids `error_metric="error"` together with
    `percentiles` or `confidence_level`, so the resolved metric is `"error"`
    exactly when the old expression was. A test should state this.
- **Verify.** `tests/test_spectrum_plotter.py` (the marker hover tests) and
  kbfit's `test_spectrum_plotting.py`.

### 1.7 kbfit: statistics round-trip and the unreachable `interval_options` fallback
- **Why.**
  - In `kbfit/project/plots/spectrum.py`, `draw_spectra` builds
    `statistics = {k: v for k, v in prepared.statistics.items() if k != "central"}`.
    It passes that on, then records `{"central": "full_sample", **statistics}`.
    That is exactly `prepared.statistics`.
  - `_ni_markers(..., interval=None)` falls back to `interval_options(options)`,
    but its only caller always passes `interval`.
- **Do.**
  - Record `dict(prepared.statistics)` directly.
  - Make `interval` a required keyword of `_ni_markers`.
  - Delete `interval_options`, and update its one test
    (`kbfit/tests/python/test_spectrum_plotting.py:26`, `:135`) to test
    `ss_options` instead.

### 1.8 kbfit: drop the lazy `as_figure` wrapper and the unreachable `None` path
- **Why.**
  - `kbfit/project/plots/base.py::as_figure` is now just a call to
    `slat.plotting.as_figure`. Importing `slat.plotting` loads neither
    matplotlib nor plotly, so the lazy wrapper isn't needed.
  - `plotting_backend(None)` has a branch for `None`, but its only caller
    (`figures.run_spectrum`) passes a resolved export backend, which is never
    `None` for figure tasks (see `invocation.resolve_export`).
- **Do.**
  - Have callers import `as_figure` from `slat.plotting`. Keep the kbfit name
    re-exported from `plots/__init__.py` if anything outside the package uses it
    (grep first).
  - Leave `plotting_backend` itself for 3.3.

---

## Tier 2: plotly backend internals, before the split

Every change to plotly behaviour happens here, so that Tier 6 moves finished
code. Do **not** reorder module constants in this tier; Tier 6 does that.

### 2.1 Axis-line legend fields reuse `_entry`; long positional signatures become keyword-only
- **Why.**
  - `_axline` re-spells the legend fields (`name`, `showlegend`, `legend`,
    `legendrank`) that `_entry` builds for traces. Only `zorder` differs: shapes
    use `layer` instead.
  - `_axline` (10 parameters) and `_segments` (11 parameters) are called
    positionally from `axhline`, `axvline`, `hlines` and `vlines`. One swapped
    argument would go unnoticed.
- **Do.**
  - Split `_entry` into `_legend_entry(label)` (the four legend fields) and the
    `zorder`. Traces use both; shapes use only `_legend_entry`.
  - Make every parameter of `_axline` and `_segments` after the first
    keyword-only, and update their four callers.
- **Verify.** `tests/test_plotting_backends.py` (legend and axis-line tests).
  Plotly output should be byte-identical: compare `fig.to_json()` before and
  after for one spectrum and one histogram.

### 2.2 `PlotlyAxes.set`: loop over the two axes
- **Why.** The x and y update dicts are built with copy-pasted code: title,
  range, type, then title font.
- **Do.** Loop over
  `(("x", xlabel, xlim, xscale, self._fig.update_xaxes), ("y", ylabel, ylim, yscale, self._fig.update_yaxes))`.
  Keep the title handling where it is.

### 2.3 `hlines`: build segments in one pass
- **Why.** `hover is not None` is tested four times. `labels` is only bound
  inside one branch but read in another. `xs`, `vals`, `hovertext` and
  `segments` are built in separate loops over the same data.

  There are two hover implementations on purpose:
  - **Python** adds a midpoint so a figure has an interior hover target with no
    JavaScript (`fig.show()`, notebooks).
  - **The post-script** rebuilds dense targets on zoom in saved HTML.

  Keep both, but say so in the code.
- **Do.**
  - Normalise `hover` to `labels: list[str] | None` once, with the existing
    length check.
  - Build `xs`, `vals`, `hovertext` and `segments` in a single loop.
  - Add a two-line comment naming the JS (`slat_hover_segments`) as the dense
    path and the Python midpoint as the no-JS fallback.
- **Verify.** `tests/test_plotly_hover_targets.py`, and `fig.to_json()` byte
  equality.

### 2.4 `"none"` line style on segment and axis-line marks *(intended behaviour fix)*
- **Why.** `line()` and `errorbar()` check `_NO_LINE`. `_segments` (`hlines`,
  `vlines`) and `_axline` (`axhline`, `axvline`) go through `_dash`, where
  `_DASHES["none"] = "solid"`. So `hlines(..., style="none")` draws a solid line
  on plotly and nothing on matplotlib.
- **Do.**
  - Remove `"none"` from `_DASHES`.
  - In `_segments` and `_axline`, when `style in _NO_LINE`, set the line width
    to 0. Don't skip the mark: matplotlib still creates the legend entry and
    hover target for a line it doesn't draw.
- **Verify.** New test: `hlines(..., style="none")` gives line width 0 on
  plotly; `axhline(..., style="none")` gives a shape with width 0.

### 2.5 Convert hover text in the backend, not in frontends *(small visual change)*
- **Why.**
  - Frontends strip LaTeX from hover text themselves: `sampling.py::_plain` and
    `series.py:381` (`plain = slp.to_display(...)`). Other hover text, such as
    spectrum and HMarker hovers and heatmap cells, isn't stripped.
  - Plotly legend labels are already converted inside the backend (`_label`
    calls `to_html`). Hover labels accept `<sub>`, `<sup>` and `<br>`, so the
    same conversion works there.
- **Do.**
  - Add `_hover_text(s)` to the plotly backend. It splits on `"<br>"`, runs
    `to_html` on each part that `has_math`, and rejoins.
  - Apply it in `PlotlyAxes._hover`, in heatmap `text`, in `_axline_hover`, and
    to the labels stored in `slat_hover_segments` (2.3) so the JS path gets
    converted text too.
  - Delete `_plain` and its two uses in sampling.py, and the `plain = ...` line
    in series.py. Pass the raw labels; matplotlib ignores hover.
- **Watch out.** `to_html` converts `_x`/`^x` anywhere in a string that
  contains a `$...$` span, not only inside the span. That is why to apply it per
  `<br>` part and only to parts with math. Plain-text parts such as
  `"Ecm/M_N"` must stay unchanged; add a test for exactly that string.
- **Behaviour.** Hover subscripts render as HTML instead of Unicode
  approximations. Nothing changes on matplotlib.
- **Unblocks.** 4.2 can then delete `to_display`.

---

## Tier 3: add slat API, then adopt it

### 3.1 `Figure.set_size(width, height)`, in inches
- **Why.** `catalog.make_plot` imports the private
  `slat.plotting._plotly.PX_PER_INCH` (`catalog.py:95`) and branches on
  `figure.backend` to resize native figures (corner on matplotlib, optimagic
  diagnostics on plotly).
- **Do.**
  - Add an abstract `set_size` to `base.Figure`. Only the three in-tree backends
    subclass it (checked), so making it abstract breaks nothing.
  - matplotlib: `self._fig.set_size_inches(w, h)`.
  - recording: `self.style = self.style.replace(figsize=(w, h))`.
  - plotly:
    - Replace `self._style` with the new figsize.
    - `update_layout(width=round(w * PX_PER_INCH), height=round(h * PX_PER_INCH))`.
    - Call `self._apply_margins()`, which re-runs `_layout_rows` and re-places
      legends and colorbars.
    - Document that colorbar thickness and the width taken from the subplot are
      fixed when the heatmap is drawn, and are not recomputed.
  - In catalog, replace the backend branch with `figure.set_size(*geometry.figsize)`,
    and delete the `PX_PER_INCH` import and its comment.
- **Verify.**
  - `tests/test_plot_contracts.py`.
  - Add one test per backend: size before and after.

### 3.2 `slp.show(figures)`
- **Why.** kbfit's `show_figures` branches on `figures[0].backend == "matplotlib"`:
  `plt.show()` shows every open figure, so it must be called once.
- **Do.**
  - Add `show(figures)` to `slat/plotting/__init__.py`. Normalise each argument
    with `as_figure`. If any figure is matplotlib, call `show()` on the first
    matplotlib one once; call `show()` on each of the others.
  - Have kbfit's `show_figures` delegate to it. **Keep the kbfit name**:
    `kbfit/tests/python/test_analysis_publication.py:431` monkeypatches
    `kbfit.project.plots.base.show_figures`.
  - Mixed-backend lists aren't handled correctly today (the first figure's
    backend decides). Handle them, and add a test.

### 3.3 `slp.use()` checks availability *(changes when the error is raised)*
- **Why.** `use("plotly")` succeeds without plotly installed; the error only
  appears at the first `figure()`. kbfit's `plotting_backend` exists to add
  this check.
- **Do.**
  - Factor a `_require(name)` out of `_module()`: the unknown-name check plus
    the `find_spec` check, raising the existing messages. Call it from both
    `use()` and `_module()`.
    - `_module()` must keep its own check: `figure(backend=...)` and
      `wrap(backend=...)` reach it without going through `use()`.
  - In kbfit, replace `plotting_backend(backend)` with `slp.backend(backend)` in
    `figures.run_spectrum`, and delete `plotting_backend` and its
    `plots/__init__` export (after grepping for other users).
- **Behaviour.**
  - `use("plotly")` without plotly now raises `ImportError`.
  - kbfit previously raised `ValueError("unknown plotting backend ...")`. Grep
    kbfit tests and CLI error handling for that message and type before
    changing it.
  - Export resolution validates backend names earlier
    (`invocation.resolve_export`), so CLI users should see no difference.

### 3.4 Grid follows the style (D1) *(behaviour change)*
- **Why.** `PlotStyle.grid` (default `False`) is applied by both backends'
  `make_figure`, but every frontend then forces a grid on anyway:
  - `sampling.py`: four `axes.grid(True, alpha=0.3)` calls (histogram, errorbar,
    effective sample size, bootstrap intervals)
  - `series.py:297`: `axes.grid(True, alpha=self.style.grid_alpha)`
  - `spectrum.py:609`: `axes.grid(True, alpha=prepared.style.grid_alpha, axis="y")`

  So the style setting only reaches the heatmaps.
- **Do.**
  - Delete the four sampling calls and the series call.
  - Spectrum: delete the forced y-grid. When the active style turns the grid
    on, the spectrum should still show horizontal lines only, because vertical
    lines between irrep columns mean nothing. Do this with
    `axes.grid(False, axis="x")` after drawing, guarded by
    `slp.get_style().grid`.
    - Use the style the figure was made with, not the global one, if that is
      reachable. If it isn't, use `get_style()` and note it in a comment.
    - *Assumption made in writing this plan, not confirmed by the user.* If
      they want the x-grid as well, drop this step.
  - `SeriesStyle.grid_alpha` and `SpectrumStyle.grid_alpha` are now unused.
    Remove them in 4.4, not here, so this task stays a behaviour change only.
  - Update the `PlotStyle` docstring: frontends never force a grid; set
    `PlotStyle(grid=True)` (or `style_context(grid=True)`) to get one.
- **Behaviour.** No plot draws a grid by default. With `style_context(grid=True)`,
  every plot gets one (y only on spectrum) at `PlotStyle.grid_alpha`.
- **Verify.**
  - Update the recording tests that expect a `grid` mark.
  - Add one test that a spectrum under `style_context(grid=True)` turns the
    x-grid off and leaves the y-grid on.
  - PNG check of a sampling summary and a spectrum on both backends, with
    `grid` off and on.

---

## Tier 4: remove dead public API

Do this after Tier 2 (2.5 frees `to_display`) and Tier 3 (3.4 frees the
`grid_alpha` style fields), and before Tiers 5–7, so nothing dead is moved or
rewritten.

**Hard removal (D2):**
- Delete the code, its `__all__` entries, its tests and its doc mentions.
- No `DeprecationWarning`, no compatibility aliases.
- Removed `StrictModel` style fields make any TOML that still sets them fail
  validation. That is intended. Update in-repo fixtures, tests and docs that set
  them.

### 4.1 The capability system
- **What.**
  - `FEATURES`, `Axes.supported`, `Axes.supports()`.
  - The backend-level `_SUPPORTED` constants.
  - Exports in `slat.plotting.__all__` and `sigmondsamplings.plotting.__all__`.
  - Docstring references in `base.py` (`contour`, `point_size`).
- **Why.** No production code calls `supports()`. Plotly's set also claims
  `"mathtext"`, which its own module docstring contradicts.
- **Tests.** `test_plotting_backends.py:639, 741, 808–815` use `supports()`.
  - 639 and 741: delete the `supports` assertions; the surrounding tests already
    check the behaviour.
  - 808–815: delete the test.
- **Keep.** `UnsupportedFeature`. The base `contour()` raises it, and the
  catalog uses it.

### 4.2 Unused style and text API
- **Delete:**
  - `PlotStyle.mathtext`, which is never read.
  - `PlotStyle.pixel_size`, now unused since plotly lays out at `PX_PER_INCH`;
    `test_plotting_backends.py:185` is its only user.
  - `PlotStyle.marker_cycle()`, which has no callers.
  - `strip_math`, which only tests use.
  - `to_display`, which has no callers after 2.5.
- Remove each from `__all__` and delete their tests (`test_plotting_backends.py:185–212`).
- Update the `PlotStyle` docstring (`markers` mentions `marker_cycle()`).

### 4.3 The `ax=` keyword
- **What to change:**
  - The `ax` parameter of `resolve_axes`, and the `ax` parameter of every public
    plot method: SamplingPlotter ×5, SeriesPlotter, CorrelatorPlotter ×2,
    `CorrelatorMatrixPlotter.plot_diagonal`/`plot_heatmap`, `plot_coverage`,
    `SpectrumPlotter.plot`, `SectorSpectrumPlotter.plot`, `draw_spectrum`.
  - `correlators._plot_tail`.
  - `SectorSpectrumPlotter.plot`, which re-lists twelve parameters; it becomes
    `def plot(self, **kwargs): return super().plot(group_by=("psq", "irrep"), column_label_fn=..., outer_label_fn=..., **kwargs)`.
  - The "legacy `ax=`" wording in the `slat.plotting.wrap` docstring.
- **Keep.** `slp.wrap`. `axes=` still accepts a raw matplotlib Axes through it.
- **Tests.** `test_plotting_backends.py:1125–1148` (the legacy `ax` tests): replace
  them with one test that a raw matplotlib Axes passed as `axes=` is wrapped.
- **Watch out.** `spectrum.plot` reads `existing = axes if axes is not None else ax`;
  simplify it. `draw_spectrum`'s `created = axes is None and ax is None` loses its
  second clause.

### 4.4 Remaining leftovers
- `SamplingPlotter(default_style=...)`: stored, never read. Remove the parameter
  and attribute.
- `stacked_positions(markersize=, ax=)` and `utils.markersize_to_ydata`: used
  only by `test_plotting_backends.py:853`. Remove them, and turn that test into
  a `marker_extent=` test. kbfit's `lanes.py` call passes neither.
- Style fields that nothing reads. Remove these from the models:
  - `SpectrumStyle.capthick` (`plot_contracts.py:115`), which was never read.
  - `SpectrumStyle.grid_alpha` (`plot_contracts.py:125`), unused after 3.4.
  - `SeriesStyle.grid_alpha` (`series.py:71`), unused after 3.4. Remove its
    docstring line too.
- Then update everything that names those fields:
  - kbfit `tests/python/test_analysis_invocations.py:245` and `:254` write
    `grid_alpha = 0` into a `[spectrum]` style and assert on it. Switch the test
    to another real `SpectrumStyle` field, e.g. `ypad_frac`.
  - `docs/plotting_api_design.md:157` lists `capthick` as ignored on plotly;
    remove it from that list.
  - `docs/correlator_views_design.md:346` shows `SeriesStyle.grid_alpha`;
    remove it.
  - kbfit's `SpectrumStyle` inherits from the sigmondsamplings one, and
    `default_style_document()` scaffolds every field. Regenerate or check any
    committed default style TOML in kbfit (grep for `capthick` and
    `grid_alpha` under kbfit, including test fixtures and docs).
- Leave the `grid_alpha` in `fitting/fit_plotter.py` alone. It is raw
  matplotlib and out of scope.

---

## Tier 5: structural changes in sigmondsamplings

### 5.1 Spectrum: separate drawing from preparation
- **Why.**
  - `draw_spectrum` builds a plotter with `object.__new__(SpectrumPlotter)`
    (`spectrum.py:590`) just to reach the `_draw_*` methods.
  - `PreparedSpectrum.columns` still carries the energy collections, which
    drawing ignores.
  - `_marker_range` runs twice per marker: once in `prepare()` to filter, once
    in `_draw_markers`.
- **Do.**
  - Move `_draw_columns`, `_draw_xticks`, `_draw_outer_labels`,
    `_draw_group_separators`, `_draw_markers`, `_marker_range` and `_color_cycle`
    to module-level private functions taking `(axes, prepared, ...)`. Read style
    from `prepared.style`.
  - `SpectrumPlotter` keeps input handling, grouping, `prepare()`, and `plot()`
    (which calls `draw_spectrum`).
  - `columns` becomes `tuple[tuple[outer, col, x], ...]`. `_ylabel` and the
    level loop in `prepare()` read the collections before they are dropped.
  - Store `markers` as `(HMarker, (x0, x1))` pairs. `draw_spectrum` computes
    spans only for `overlays` and skips any it can't place, as now.
  - In `draw_spectrum`, `axes.set(xlim=..., ylim=...)` is called before drawing
    (needed, since `point_size` depends on limits) and again at the end with the
    y label. Pass only `ylabel` to the second call, then check the recording and
    plotly tests. If a backend's autoscale moves the limits, keep the second
    call and add a comment saying why.
  - Type `column_label_fn` and `outer_label_fn` as `LabelFn | None` instead of `Any`.
- **Keep stable.** kbfit and the tests use only `prepared.energy_range`,
  `.levels` and `.statistics`, plus `draw_spectrum(prepared, *, overlays=, axes=, figsize=, backend=)`
  and `prepare_spectrum(...)`.
- **Verify.** `tests/test_spectrum_plotter.py`, `tests/test_plot_contracts.py`,
  kbfit `test_spectrum_plotting.py`, and a PNG comparison of a sector spectrum
  on both backends.

### 5.2 One table for the plot catalog (needs 3.1)
- **Why.** The plot catalog is split across two modules:
  - `plot_contracts._METHODS` maps names to options models.
  - `catalog._METHODS` maps the same names to `SamplingPlotter` method names.
  - `PlotDefinition.validate_input` and `.make` are `partial`s over module
    functions. `_make` then looks the definition up again by name.
  - `make_plot` special-cases `"corner"` (twice), the `histogram` and
    `bootstrap-intervals` statistics, and the diagnostic family. Its statistics
    block ends by resetting the dict to `{}` for three names.
- **Do.**
  - Extend `PlotDefinition` (kept in `plot_contracts`, still free of heavy
    imports) with:
    - `method: str | None` (the `SamplingPlotter` method name; `None` for diagnostics)
    - `takes_figsize: bool` (false for corner and diagnostics, which are resized after drawing)
    - `statistics: Callable[[options, target], dict]`
  - Define the nine definitions as explicit literals instead of a generator
    with nested conditionals.
  - Make `validate_input` and `make` methods that lazily import catalog
    functions and pass `self`.
  - `make_plot` then has no name checks. It calls the method, resizes when
    `not takes_figsize` and a figsize is set (using 3.1), and calls
    `definition.statistics(...)`.
- **Keep stable.** kbfit `project/analysis/ss.py` uses `name`, `family`,
  `options_model`, `style_model`, `backends`, `make`, `get_plot_definition` and
  `plot_definitions`. The CLI uses `.make` (`sigmondsamplings/cli/plot.py`).
- **Verify.** `tests/test_plot_contracts.py`, `tests/test_cli_corr_plot.py`,
  and kbfit's analysis tests.

### 5.3 NI-marker hover through the shared formatter (kbfit and `_util`)
- **Why.** In `kbfit/project/plots/spectrum.py::_ni_markers`, the fallback branch
  (no sampling) hand-builds `"NI: …<br>irrep, d²=…<br>Ecm = …"`. That copies the
  format of `_util.energy_level_hover_text`, which needs a `SigmondData`.
- **Do.**
  - Split `energy_level_hover_text` into a context-row builder
    (`_hover_context(irrep, psq, level, ni_pair, ni_rank, include_level)`) and
    the value rows.
  - Add a public `ni_level_hover_text(ni_pair, irrep, psq, ecm, coupling)` for
    levels without a sampling, and use it in kbfit.
  - Leave the hard-coded psq→momentum table in `_ni_pair_energies` unless a
    shared helper already exists in `sigmondsamplings.kinematics`; grep first,
    and don't invent one.

---

## Tier 6: split `slat/plotting/_plotly.py` into a package (pure move)

- **Why.** At 1,567 lines the file holds:
  - unit and default-size tables
  - colour, marker and dash conversion
  - `PlotlyAxes`
  - a figure-layout engine (margins, row gaps, frames, colorbars)
  - about 100 lines of JavaScript in a Python string, whose name
    `_ZOOM_MARKERS_SCRIPT` is out of date (it also rebuilds hlines hover targets)

  Constants such as `_NO_LINE` and `_MARGIN` are defined after the class that
  uses them.
- **Layout.**
  ```
  slat/plotting/_plotly/
      __init__.py     # re-exports, see below
      _units.py       # PX_PER_INCH, _DEFAULT_PT, _LEADING, _FONT_FAMILY, _GRID_COLOR, _Z_*, _px, _size, _z
      _convert.py     # _MARKERS, _DASHES, _NO_LINE, _COLORSCALES, _NAMED_RGB, _LEGEND_LOC,
                      # _to_rgb, _color, _opacity, _marker, _dash, _colorscale, _clean, _err,
                      # _label, _hover_text, _marker_spec
      _axes.py        # PlotlyAxes
      _figure.py      # _MARGIN, PlotlyFigure, _row_gap, _column_spacing, make_figure,
                      # _frame_name, _frame, _axis_look, wrap
      post_script.js  # was _ZOOM_MARKERS_SCRIPT
  ```
- **Rules.**
  - **No logic changes.** Moves, renames of the JS constant, and import edits
    only. Diff each function against the original.
  - `_figure.py` imports `PlotlyAxes`. `_axes.py` refers to `PlotlyFigure` only
    in annotations (under `TYPE_CHECKING`, with `from __future__ import annotations`).
  - `__init__.py` re-exports `PX_PER_INCH`, `PlotlyAxes`, `PlotlyFigure`,
    `make_figure` and `wrap` (the backend registry imports the `._plotly` module
    by name), plus the private names tests import:
    - `_err`, `_color`, `_to_rgb`, `_marker`, `_dash` (`test_plotting_backends.py`)
    - `_ZOOM_MARKERS_SCRIPT` (`test_plotly_hover_targets.py`), aliased to the
      loaded script text

    Alternatively, update those test imports to the new modules; either is
    fine, but pick one.
  - Load the script with
    `importlib.resources.files(__package__).joinpath("post_script.js").read_text()`.
    Plotly substitutes `{plot_id}` in the text, so keep that placeholder exactly.
  - Add the `.js` file to `[tool.setuptools.package-data]` in `pyproject.toml`:
    `slat = ["py.typed", "plotting/_plotly/*.js"]`.
- **Verify.**
  - Full sigmondsamplings suite (failure set unchanged).
  - Build a wheel (`python -m build --wheel` or `pip wheel . --no-deps`) and
    check that `post_script.js` is inside it. Editable installs hide this
    failure.
  - Save one figure to HTML and confirm the script is present.
  - `fig.to_json()` byte-equal to the pre-split output for one spectrum and one
    correlator grid.

---

## Tier 7 *(done 2026-10-05; see status at the end)*: template-method `Axes` and consistent parameter names

Done at the user's request after Tiers 1–6. The design below is kept as written;
the status note at the end of this tier records what landed and where it differs.

This is last because it touches every mark in four implementations and every
frontend call site. After Tiers 1–6 that code is as small as it will get.

**Shims and D2.** D2 (hard removal) covers Tier 4's dead API. Tier 7 renames
*live* parameters that users pass. Keep the one-release deprecation shim below
unless the user says otherwise when Tier 7 starts.

- **Why.**
  - Every mark's full signature, with types, defaults and docstring, is written
    four times: in `base.py`, `_mpl.py`, plotly `_axes.py` and `recording.py`.
    `make_figure`'s signature is written four times too.
  - Defaults can drift between copies. The recording backend takes `**kwargs`,
    so recorded marks show what was passed, not the defaults that apply.
  - Parameter names disagree:
    - line width is `width` (line, errorbar, segments, axis lines), `linewidth`
      (polygon) or `linewidths` (contour)
    - `width` on `bar` means bar width
    - line style is `style`
    - marker size is `size` on `points` and `markersize` elsewhere
- **Design.**
  - **Public marks in `base.Axes` become concrete methods.** They carry the full
    typed signature, defaults and docstring. Each calls a protected hook with
    every argument resolved, by keyword, e.g.
    `self._line(x, y, color=..., linewidth=..., linestyle=..., ...)`.
  - **Backends implement only the hooks** (`_line`, `_points`, ...): keyword-only,
    no defaults, no docstrings.
  - **`RecordingAxes`** implements each hook as a one-line call to `_record`. It
    then records resolved defaults; update the four tests that compare kwargs
    exactly.
  - **New names:**
    - `width` → `linewidth` (line, errorbar, hlines, vlines, axhline, axvline)
    - `style` → `linestyle`
    - `points(size=)` → `markersize`
    - `contour(linewidths=)` → `linewidth`
    - `bar(width=)` is unchanged: it is a bar width
  - **Deprecation shim.** One base helper,
    `_renamed(legacy: dict, mapping: dict) -> dict`, is called from each public
    mark that accepts `**legacy`. It warns with `DeprecationWarning` and maps old
    names to new ones. Because of the template method, the shim lives in one
    place. Remove it after one release.
  - **`make_figure`:** pass a frozen `FigureSpec` dataclass from `figure()` to
    each backend's `make_figure(spec)`. This also stops matplotlib-only options
    such as `constrained_layout` from leaking into the other backends.
- **Churn.** About 19 renamed-argument call sites in `sigmondsamplings/plotting/*`,
  about 35 direct mark calls in tests, and `HMarker`'s field names are
  unaffected (`linestyle` and `linewidth` already). kbfit makes no direct mark
  calls (checked).
- **Verify.**
  - Full suites in both repos.
  - PNG parity check on both backends.
  - A test that the old names still work and warn.
- **API added in Tiers 1–6 that Tier 7 must carry** (noted in the post-pass review):
  - `hover=` on `bar`, `axhline` and `axvline`; `label_fontsize=` on `set`;
    `suptitle(fontsize=)` (was `**kwargs`); abstract `Figure.set_size`.
  - `vlines` takes no `hover` while `hlines` does. Decide whether to add it
    (the plotly `_segments` path already supports it) when the hooks are written.

### Tier 7 status: done (2026-10-05)

What landed:
- **Template methods.** Every public method in `base.Axes` is now concrete:
  the 13 marks and `set`, `ticks`, `grid`, `legend`. Each forwards all its
  arguments by keyword to an abstract `_hook`.
  - `_contour` and `_limits` keep their defaults: one raises
    `UnsupportedFeature`, the other returns None.
  - `point_size` stays abstract. It returns a value and has one signature, so a
    hook would add nothing.
- **Backends** implement only the hooks.
  - In matplotlib, `hlines` and `vlines` share one
    `_collection_kwargs` helper.
  - The plotly-internal `_segments`, `_axline` and `_line_spec` use the new
    names too.
- **`RecordingAxes`** implements each hook as one `_record` call, so recorded
  marks now include resolved defaults.
  - Tests that looked for `"ylim" in m.kwargs` / `"ylabel" in m.kwargs` now use
    `.get(...) is not None`, since every `set` mark carries every key.
- **`FigureSpec`** (in `base`) is built once in `slp.figure()`, with `style`
  resolved and any `figsize` applied to `style.figsize`. All three
  `make_figure(spec)` read it.
  - This also fixes matplotlib, whose figure's `_style.figsize` used to stay at
    the global size when `figsize=` was passed.
- **Renames** as listed above, with every frontend call site migrated. Only
  four tests still used the old names; the "about 35" estimate counted every
  direct mark call. The old names are rejected (`TypeError`), with no shim.

Deviations from the design above:
- **No deprecation shim.** The user chose hard removal (as D2) when Tier 7
  was done. An earlier draft had a `_renamed` decorator; it was dropped.
- **Hook-name clash.** Plotly's `PlotlyAxes._legend` attribute (the layout key
  `"legend2"` etc.) clashed with the `_legend` hook, so it is renamed
  `_legend_name`. Hook names are now reserved on `Axes` subclasses.
- **`vlines(hover=)`: not added.** Nothing needs it, and plotly's vertical
  segments have no midpoint hover path. Add it when a frontend needs it.

Verified:
- **sigmondsamplings.** The failure set is unchanged. New tests cover
  resolved defaults in recordings and figsize reaching the style.
- **kbfit.** 602 passed.
- **Rendering.** PNG renders of a sampling summary, a sector spectrum with a
  marker, and bootstrap intervals agree across both backends.

8.1 (`Axes.style` / `Figure.style`) was done afterwards, in Tier 8.

---

## Tier 8 *(done 2026-10-05; see status at the end)*: follow-ups found after Tiers 1–6

Found by a review of the Tiers 1–6 changes. The quick fixes from that review
were applied directly (see *Review log* below); these are what was left because
each one changes behaviour, crosses into kbfit, or needs a visual check.
Independent of each other. 8.1 adds public `Axes`/`Figure` API, so do it
before Tier 7 or as part of it.

### 8.1 Frontends read the figure's style, not the global one *(bug fix)*
- **Why.** `PlotStyle` can be per-figure (`slp.figure(style=...)`, `wrap(style=...)`),
  but frontends drawing on a given `Axes` read the process-wide `slp.get_style()`:
  - `spectrum.draw_spectrum`: `if slp.get_style().grid: axes.grid(False, axis="x")`.
    Its own comment admits the figure's style "is not reachable from its axes".
  - `spectrum._draw_columns`: the color palette comes from `resolve_colors(..., slp.get_style())`.
  - `_util.cycle_colors` (series and correlator-matrix colors) does the same.

  So `fig = slp.figure(style=PlotStyle(grid=True))` followed by
  `draw_spectrum(prepared, axes=fig.axes())` keeps the x-grid (the y-only rule
  is skipped), and colors come from the global palette, not the figure's.
- **Do.**
  - Add a read-only `style` property to `base.Figure` and `base.Axes`
    (`Axes.style` returns `self.figure.style`). Matplotlib and plotly already
    keep `_style`.
  - `RecordingFigure.style` is a public, reassigned attribute today
    (`set_size` replaces it). Store it as `_style`, expose the property, and
    update tests that read `fig.style`.
  - Have frontends take the style from `axes.style`: the spectrum grid guard,
    `_draw_columns`, and `cycle_colors(palette, n, style)`. Delete the comment
    in `draw_spectrum` that explains the workaround.
- **Verify.** One test per call site: a figure made with a non-default style,
  under a different global style, draws with the figure's.

### 8.2 One hover format for NI levels (sigmondsamplings + kbfit)
- **Why.** NI markers get their hover text in two ways, and they put the
  coupling rank in different places:
  - With a sampling, `HMarker.from_sampling(..., ni_rank=r)` goes through
    `energy_level_hover_text` → `_hover_context`. The rank ends up in the
    context row: `"A1g, d²=1, rank 2"`.
  - Without one, kbfit calls `ni_level_hover_text(name, irrep, psq, ecm, coupling)`
    with `coupling="rank 2"` or `"coupling unclassified"`. That puts it in the
    value row: `"Ecm = 1.23 (rank 2)"`.
- **Do.**
  - Change it to `ni_level_hover_text(ni_pair, irrep, psq, ecm, *, ni_rank: int | None)`
    and pass `ni_rank` to `_hover_context`.
  - Show an unclassified level (`ni_rank=None`) as `"coupling unclassified"` in
    the same context row. This needs a small flag on `_hover_context`, because
    for sampling-backed markers `None` currently just means "don't show".
  - In kbfit `_ni_markers`, pass `ni_rank=rank` and delete the `coupling` string.
  - Grep kbfit tests for `"Ecm = "` and `"(rank "` before changing.

### 8.3 `to_plotly`: plain text inside `\text{}` *(needs a visual check)*
- **Why.** A mixed math/plain string becomes one MathJax span, with each plain
  part wrapped as-is in `\text{...}`. Some plain text breaks that:
  - `<br>` (`"$E_0$<br>x"` → `$E_0\text{<br>x}$`)
  - unbalanced `{`/`}`
  - a backslash
  - possibly `%`, `#` and `&`

  Frontends pass `\n`, not `<br>`, so nothing in-tree hits this today. But
  `axes.set(title=...)` and `axes.text(...)` are public.
- **Do.**
  - Treat `<br>` like `\n`: split on both before merging spans.
  - When a plain part contains `{`, `}` or `\`, fall back to `to_html` for the
    whole string rather than escaping. MathJax's text mode prints most escapes
    literally (as the existing `\_` handling shows).
  - Check `%`, `#` and `&` inside `\text{}` by rendering with kaleido before
    deciding whether they need the same fallback.
- **Verify.** Unit tests for each case, plus one PNG render per case.

### 8.4 `heatmap(origin="upper")` followed by `set(ylim=...)` *(document)*
- **Why.**
  - On matplotlib's `pcolormesh` path, `origin="upper"` now inverts the y axis.
    A later `set(ylim=(lo, hi))` with `lo < hi` un-inverts it.
  - On plotly, an explicit `range` overrides `autorange="reversed"` in the
    same way.

  The two backends agree, but a caller who sets limits loses the flip without
  warning.
- **Do.** Say so in the `base.Axes.heatmap` docstring: pass `ylim` as
  `(hi, lo)` to keep row 0 at the top. Making `set` preserve the inversion is
  not worth the hidden state.

### 8.5 Small structural leftovers *(optional; only if the code is touched anyway)*
- **`central_lo_hi` signals symmetry by identity.** It returns the same array
  object as `lo` and `hi` for a symmetric error, and `SeriesPlotter._prepare`
  tests `symmetric = lo is hi`. Any copy in between would silently switch
  series hover and error bars to the asymmetric path. Return an explicit
  `symmetric` flag instead.
- **`catalog.validate_input` still dispatches on plot name** for three checks:
  - corner's optional `corner` package
  - histogram's `sampling` index option, where others use `obs_index`
  - bootstrap-intervals needing bootstrap samplings

  After 5.2 these are the only name checks left. Moving them into
  `PlotDefinition` fields (`requires: str | None`, `bootstrap_only: bool`) is
  only worth it if the catalog grows; three checks don't justify two fields.
  Don't rename the `sampling` option: it is live TOML/CLI API.

### Review log (quick fixes already applied, 2026-10-05)
- `_util.full_sample_and_bounds`: back to an explicit jackknife check. The
  `try/except ValueError` around `confidence_interval` relabelled *any*
  `ValueError` (for example from the analyzer) as "unavailable for jackknife
  data".
- `_mpl.heatmap`: `origin="upper"` inverts the y axis only if it isn't already
  inverted. A second upper-origin heatmap on the same axes used to flip it back.
- Plotly `_hover_text`: `\n` becomes `<br>`. Plotly hover labels ignore `\n`,
  and `to_html` used to flatten it to a space.
- `SeriesPlotter._draw_series`: hover texts are indexed per kept point
  (`series.data[i]`), instead of first materialising every element with
  `list(series.data)`.
- `plot_contracts._DEFINITIONS`: `method`, `takes_figsize` and `statistics` are
  passed by keyword. Eight positional arguments, including a bare
  `True`/`False`, made swapped arguments easy to miss.
- Plotly package: one `_units._suffix(index)` replaces five copies of
  `"" if index == 1 else str(index)`. The `_axline` span and `set` font-size
  lines are reflowed.
- Stale text:
  - the "legacy `ax=`" wording in `_mpl.wrap`
  - `_util.py` spacing
  - `docs/plotting_api_design.md` (`mathtext()`, `stacked_positions(markersize=, ax=)`,
    the `ax=` alias)
  - `docs/correlator_views_design.md` (`plot_summary` "on both plotters";
    the matrix one was removed)

### Tier 8 status: done (2026-10-05)

- **8.1.** `Figure.style` and `Axes.style` are read-only properties in `base`.
  - Backends keep the style as `Figure._style`.
  - Matplotlib and plotly axes no longer hold their own copy, so there is one
    source of truth; they read `self.style`.
  - `RecordingFigure.style` became `_style` plus the property.
  - Frontends read `axes.style`: the spectrum grid guard, `_draw_columns`'
    palette, and `SeriesPlotter` colors via `cycle_colors(palette, n, style)`.
  - **Exception:** `CorrelatorMatrixPlotter` fixes operator colors in
    `__init__`, before any figure exists, so every view of a matrix agrees. It
    still passes `slp.get_style()`, with a comment saying why.
  - Tests: one for the spectrum (grid and palette), one for series colors.
- **8.2.** `ni_level_hover_text(ni_pair, irrep, psq, ecm, *, ni_rank)`.
  - `ni_rank` is `int | "unclassified"` (the `NIRank` alias in `_util`).
    `None` still means "not shown", so no extra flag was needed.
  - Both NI paths put the coupling in the context row:
    `"T1g, d²=0, rank 1"` or `"…, coupling unclassified"`.
  - kbfit passes `ni_rank="unclassified"` for an unknown rank on both paths.
    The sampling-backed path used to show nothing in that case.
- **8.3.** Renders with kaleido showed:
  - `%`, `&`, `#`, backslashes and balanced braces print literally inside
    `\text{}`.
  - An unbalanced brace blanks the whole label.
  - `<br>` shows as `\lt br\gt`.

  So `to_plotly` now:
  - splits on `\n` and `<br>` alike
  - falls back to `to_html` only when a plain part contains a brace. The
    fallback reads the brace as TeX grouping and drops it, which is far better
    than a blank label.

  The PNG re-render is correct.
  - **Also fixed:** axes and figure titles reserved height by counting only
    `\n`, so a `<br>` title was clipped. A new `text.line_count` counts both.
- **8.4.** Documented in `base.Axes.heatmap`.
- **8.5.** `SeriesPlotter._prepare` takes symmetry from `central_and_error`'s
  shape, not `lo is hi`. The catalog name checks stay, as planned.

Found while verifying, left for later:
- **Empty plotly subplots collapse.** plotly.js drops axes that no trace
  references. A subplot with only a title or text, and no mark, has no axes of
  its own, so its frame and title land inside another subplot. This predates
  Tiers 7–8.
  - matplotlib draws an empty framed cell.
  - Frontends always draw into the cells they keep, and hide the rest.
  - A fix (an invisible placeholder trace per subplot in `make_figure`) shifts
    `fig.native.data` indices that many tests use. It needs its own change.

## Tier 9: follow-ups from the second review (completed 2026-10-05)

The proposals below are historical. The later audit revises 9.1, 9.2, 9.4 and
9.5 and drops the extraction in 9.3. The implementation record after that audit
describes the completed work.

A full read of `slat/plotting`, `sigmondsamplings/plotting` and
`plot_contracts` after Tiers 1–8. The suite failure set is unchanged
(all 149 are in `io`/`cli_query`/`scripts`, none in plotting) and kbfit passes
(602). Independent of each other; ordered by value.

### 9.1 `plot_bootstrap_intervals` emits one `band` trace per box *(plotly output change)*
- **Why.** Each (confidence level, observable) pair is its own `axes.band`, so
  20 observables × 2 levels is 40 plotly traces, 40 legend ranks and 40 hover
  skips. Only the first box of a level carries the label, so toggling the
  legend entry hides one box.
- **Do.** Add a `rects(x0, x1, y0, y1, ...)` mark (matplotlib: `PatchCollection`
  or `fill_between` per box; plotly: one `fill="toself"` trace with `None` gaps),
  or give `band` a NaN-separated form. One mark per level then. Needs Tier-7
  style hooks, so write the base method and the three hooks together.

### 9.2 Resolve the error metric once for the spectrum and markers
- **Why.** `full_sample_and_bounds` builds `UncertaintyOptions(...).resolved()`;
  `HMarker.from_sampling` builds the same object again to learn whether the
  metric is an interval; `SpectrumPlotter.prepare` does it a third time
  (`options.resolved()`) and re-derives "is interval" in two more places.
- **Do.** Return a small `Bounds(central, lo, hi, interval: bool)` from
  `full_sample_and_bounds` (or accept an already-resolved dict), and let callers
  read `.interval`. Then delete the second `resolved()` in `markers.py` and the
  `statistics["error_metric"] == ...` tests in `spectrum.prepare`.

### 9.3 Split `SpectrumPlotter.prepare` (about 70 lines)
- **Why.** It validates options, lays out columns, filters markers, builds hover
  text per level, derives the y range and assembles the result.
- **Do.** Extract `_prepare_levels(columns, statistics, hover_confidence)` and
  `_auto_ylim(values, placed)` as module functions. Move the
  `hover_confidence_level` validation to the top of `prepare` (it currently runs
  after the layout work, so a bad option costs a full layout first).

### 9.4 Series hover builds one `SigmondData` slice and a CI per point
- **Why.** `SeriesPlotter._draw_series` calls `series.data[i]` and then
  `estimate_hover_text`, which calls `confidence_interval` (a percentile over
  all resamples) per point whenever `rc["plot.error_metric"]` is an interval.
  The matrix grid does this for every cell of N² series. `_prepare` already has
  `y`, `lo`, `hi` as arrays.
- **Do.** Build hover text from the prepared arrays, formatting the value and
  error with `uncertainties` directly (or add a vectorised
  `SigmondData.pdg_format_array`). The interval row comes from `y - lo`, `y + hi`
  with no second percentile call. Measure on a 12×12 grid first; do it only if
  it shows up.

### 9.5 Empty plotly subplots collapse *(carried over from Tier 8)*
Unchanged from the note at the end of Tier 8. Do it with a placeholder trace
per subplot added in `make_figure`, and fix the `fig.native.data` indices in
the tests in the same change.

### 9.6 *(done in full review)* Fold `_plot_tail` away
- **Why.** `correlators._plot_tail(title, axes, figsize, backend)` returns a
  dict only to be `**`-splatted into one call, which hides which keywords the
  next call receives. It exists because `ax=` was removed (4.3); the four
  names are now just ordinary keywords.
- **Do.** Inline it at its five call sites.

### 9.7 Hook signatures are still written three times
- **Observation, no action planned.** Public method (typed, with defaults),
  abstract hook in `base`, and each backend's hook repeat the parameter list.
  Tier 7 removed the *defaults* and *docstrings* duplication, which was the part
  that drifted. A generic forwarder (`self._call("line", locals())`) would
  remove the rest, at the cost of signatures that no type checker or IDE can
  follow. Not worth it. Revisit only if a mark is added more than once a quarter.

### 9.8 Pre-existing lint in files outside the plotting work
`ruff check` reports unsorted imports in `slat/labels.py`,
`sigmondsamplings/__init__.pyi`, `fitting/minimizer.py` (plus an unused
`import json`), `io/spec.py` and six test files. Run `ruff check --fix` on those
in a commit of its own; they are not part of this refactor.

### Tier 9 review log (quick fixes applied)
- `plot_contracts._histogram_statistics` read `target.stats[0]`, but the
  histogram plots `stats[options.sampling or 0]`. With mixed bootstrap/jackknife
  inputs the recorded statistics described the wrong observable. Now indexed.
- `_plotly._convert._label` and `_hover_text` were the same conversion except
  that the legend one split on `\n` only. `_label` now calls `_hover_text`.
- `_mpl`: `_merge(..., None)` used as a bare None-filter in `_legend` and
  `suptitle`; replaced by the filter itself.
- `correlators`: `plot_coverage` and `plot_heatmap` guarded
  `figsize = _matrix_figsize(n)` with `axes is None and figsize is None`;
  `resolve_axes` already ignores `figsize` when `axes` is given, so the guard
  was dead. `plot_grid` computed the normalisation time twice.
- `catalog.make_plot`: the three-way `origin` if-chain is a dict lookup.
- `spectrum.py`: `__all__` reflowed; a stray blank line removed.
- `_plotly/_axes.py`: two blank lines before the class.
- `tests/test_plot_contracts.py`: unused `recorder` binding.

---

## Review of Tiers 1–9 (2026-10-05)

This review treats Tiers 1–8 and the earlier review-log fixes as completed.
Tier 9 was a proposal, not an approved implementation prescription. The work
below was deferred at review time and subsequently completed as recorded here.
Priority labels describe impact, not execution order.

Reviewed the current implementations of every frontend, all three backends,
text/style conversion, figure ownership/export, the catalog/schema/CLI glue,
and kbfit's spectrum/analysis callers. Checked the original plan's removals
and replacements against the current tree. No further backend framework or
catalog registry rewrite is justified.

### Quick fixes applied in this review

- **Plotly hidden marks:** `line(linestyle="none", marker=None)` used to select
  `mode="markers"`, producing default circles. `errorbar(marker=None)` did the
  same, even when only error bars were requested. Modes now depend on both the
  line and marker choices. Kaleido SVG confirms `mode="none"` retains error
  bars without point glyphs; PNGs agree with matplotlib.
- **One-sided heatmap bounds:** Plotly ignored `vmin=0` when `vmax=None`
  because `zauto` stayed enabled. In the significance frontend, values near
  55–57 filled the entire color scale instead of using a zero baseline.
  Disable `zauto` when either bound is explicit. Confirmed through Plotly's
  resolved figure and PNGs: the scale is now 0–56.668 for the review fixture.
- **Gray colormap direction:** matplotlib `gray` runs black to white; Plotly
  `Greys` runs white to black. Map `gray`/`grey` to `Greys_r`.
- **Tick rotation reset:** Plotly now writes `tickangle=0` when requested,
  rather than keeping an earlier rotation.
- **Array limits:** `_range` checks `lim is None`, avoiding the ambiguous truth
  value error for a two-element NumPy array.
- **Figure style after resize:** matplotlib now updates `style.figsize`, as
  Plotly and recording already do. `Axes.style` sees the resized figure.
- **Suptitle style:** both rendering backends now use the figure's
  `title_fontsize` when no explicit size is given; an explicit size wins.
- **Invalid figure geometry:** `FigureStyle` rejects nonfinite dimensions and
  dpi, before a backend can fail at layout/export or produce unusable output.
- **Tier 9.3, validation only:** the Gaussian-only hover option check happens
  before grouping/layout. No helper extraction was needed.
- **Tier 9.6:** removed `_plot_tail` and spelled out the four forwarding
  keywords at its call sites. Its small dict abstraction hid more than it saved.

- **Reusable series palettes:** `SeriesStyle` snapshots color/marker iterables
  as tuples. A generator previously worked on the first view, then raised
  `IndexedCycle requires a non-empty iterable` when the same plotter drew again.

`tests/test_plotting_review_regressions.py` adds 18 regression cases. The fixes
are local; no kbfit source was changed.

### Tier 9 audit: keep, revise, or drop

| Proposal | Review decision |
|---|---|
| 9.1 interval boxes | **Keep the legend bug; revise the remedy.** One trace per box is real, but 40 traces is not by itself evidence of a performance problem. A new `rects` mark expands three backends and the public vocabulary. First solve logical legend grouping (10.5); batch rectangles only if measurement justifies it. |
| 9.2 repeated uncertainty resolution | **Optional, low priority.** The repeated Pydantic resolution exists, but the scalar helper is also a useful validation boundary. If touched, use a private helper accepting resolved choices; keep the tuple-returning helper for existing callers. A new public `Bounds` object solely to remove two tiny validations is unnecessary. |
| 9.3 split `prepare` | **Drop extraction based on line count alone.** Early option validation is now fixed. The remaining straight-line preparation code is readable; extract only when a second consumer or a focused test needs a helper. |
| 9.4 hover work | **Measured; revise before implementing.** See 11.1. Prepared errors are clipped offsets, so `y-lo`, `y+hi` do not necessarily reproduce the true CI. PDG formatting also uses Gaussian error even when interval offsets are plotted. Reusing those offsets as the PDG error changes scientific meaning. |
| 9.5 empty subplots | **Confirmed; revise the remedy.** Kaleido's resolved figure drops `xaxis2`/`yaxis2` for a title-only second row. Preserve cells with minimal empty traces, but check autorange, hidden cells, HTML and trace-index stability rather than assuming eager placeholders are the only solution. See 10.1. |
| 9.6 `_plot_tail` | **Done** in this review. |
| 9.7 hook signatures | **Keep the no-action decision.** Explicit signatures beat a `locals()`/dynamic forwarding mechanism. |
| 9.8 unrelated lint | **Still deferred and outside scope.** Do not mix it into plotting correctness work. |

The earlier Tier 9 failure-count claim is not the current baseline. This review
observed **76 failed + 6 errors**, all in loader/writer, query and script tests;
compare identities, not an old count.

### Tier 9 implementation record (2026-10-05)

- **9.1:** added `Axes.legend_group()` scopes for composed trace items. Plotly
  assigns fresh subplot-local groups and one visible entry per scope. Every
  box of an interval level, both filled/hollow subsets of a series, and a fit's
  band plus central/model lines toggle together. Static/recording legends keep
  their existing labels. Axis-line shapes retain independent entries; their
  hover traces now use the same subplot legend as their shape. Explicit
  `tracegroupgap=0` for composed trace groups preserves the existing legend
  spacing; other legends keep their native gap. No rectangles mark
  or batching API was needed. This also completes the trace-group work in 10.5.
- **9.2:** a private scalar-bounds helper accepts validated uncertainty choices.
  Spectrum preparation and markers resolve once; the public tuple-returning
  helper retains its validation and return contract. Preparation carries one
  interval flag for hover/bounds decisions. No public bounds object was added.
- **9.3, 9.6–9.8:** early validation and `_plot_tail` removal were already done.
  Keep preparation and hook signatures explicit, as the audit recommends.
  Unrelated lint remains outside this plotting task.
- **9.4:** series preparation retains the original absolute CI bounds alongside
  clipped error-bar offsets. Hover formats prepared central values with batched
  Gaussian errors, preserving scalar PDG formatting and signed negative values
  on log axes. There are no per-point data slices or repeated percentile calls.
  Matplotlib skips hover formatting; recording retains it. This completes 11.1
  without extending the public data API.
- **9.5:** created Plotly grids append minimal null traces for visible cells
  lacking a visible trace reference, at `save`, `embed_html` or `show`. Existing
  trace indices remain unchanged, hidden cells receive no placeholder, and
  repeated outputs add no duplicates. Renderer checks cover title-only,
  text-only, shape-only, all-NaN and all-filtered cells, including shared axes.
  Native exports bypass the wrapper's output step; native wrapping remains
  the separate 10.1 follow-up.

Measured on a 12×12 grid with eight times, median of three runs after warmup:

| Bootstrap resamples | Gaussian recording, before → after | CI recording, before → after |
|---|---|---|
| 40 | 0.3530 → 0.1037 s | 0.4385 → 0.1436 s |
| 400 | 0.3577 → 0.1056 s | 0.4474 → 0.1625 s |
| 2000 | 0.3709 → 0.1141 s | 0.5696 → 0.2459 s |

These measure complete recording-grid construction, including the scalar-slice
overhead omitted by the earlier empty-`_hover` experiment. Matplotlib grid
construction also improved (approximately 18–35% in these runs).

New regression coverage is in `tests/test_plotting_tier9.py` and
`tests/test_plotting_tier9_rendering.py`. Browser checks perform actual mouse
clicks in exported HTML for interval levels, split series with duplicate labels
in separate subplots, fit overlays and axis-line hover targets. Rendering checks
skip explicitly when the optional headless browser is unavailable.

Validation:
- Baseline: 1189 passed, 8 skipped, 76 failed, 6 errors.
- Final: 1230 passed, 8 skipped, 76 failed, 6 errors. The exact failure/error
  set is unchanged; all 41 added regression cases pass, including browser and
  renderer checks.
- kbfit: 602 passed, 6 skipped, 2 xfailed; no kbfit source changes.
- Ruff passes on all touched Python files.
- All 16 frontend/view variants were rendered on both backends (32 PNGs).
  Full-resolution pixel comparisons against the pre-change fixtures are
  identical after preserving the original legend spacing.

Validation and images are under `/tmp/tier9`; no commits were made.

## Tier 10: correctness follow-ups from the full review *(done)*

### 10.1 Preserve empty and wrapped Plotly drawing surfaces [P1]

**Tier 9 follow-up:** created-grid preservation is now implemented and tested
at wrapper output time. Native wrapping/drawing is now completed as recorded below.

Two related failures live in `_plotly/_figure.wrap`, `PlotlyAxes._add` and the
row/column-based configuration calls:

- `slp.wrap(go.Figure()).axes().line(...)` and `.set(ylim=...)` raise
  "you must first use ... make_subplots". Wrapping accepts the object, but
  drawing/configuring it assumes `_grid_ref` metadata that a plain figure lacks.
  Exporting such a wrapped figure works, which is why optimagic's existing
  catalog tests do not expose this.
- A grid cell with only text/title/shapes has no trace referencing its axes.
  Plotly.js removes its axes and the annotation/frame can land in another cell.
  This also occurs with all-filtered/all-invalid series, so it is not confined
  to callers deliberately leaving a cell empty.

**Do:** support a plain wrapped figure using explicit axis references for marks
and layout, without inventing a subplot grid or discarding its native layout.
For created grids, use minimal empty traces referencing each necessary pair of
axes. Try appending missing-cell placeholders at export/show time before
choosing eager allocation; preserve user traces' relative order. Explicitly
state the supported wrapping behavior for native multi-subplot figures rather
than silently representing them as `(1, 1)`.

**Verify:** plain and `make_subplots` native figures; title-only, text-only,
shape-only and all-NaN cells; hidden cells; shared axes; PNG and HTML; no
unwanted legend entries, hover targets or autorange expansion. Check kbfit's
native diagnostic export path.

### 10.2 Make axis range/scale state coherent [P1]

`PlotlyAxes._set`, `_range`, `_limits` and `point_size` each read the local
layout axis, but the displayed range can depend on scale or `matches`:

- `set(ylim=(1,100)); set(yscale="log")` retains Plotly range `(1,100)` in
  exponent units. `get_limits("y")` then returns `(10,1e100)` instead of
  `(1,100)`. The reverse scale change has the analogous problem.
- In a shared-x two-row figure, setting the bottom range to `(2,8)` leaves
  the top's `get_limits("x", explicit_only=True)` as `None`, even though it
  matches `x2` and displays the same range. Spectrum preparation may therefore
  overwrite a caller's shared limits; size estimates can also return zero.
- `point_size` uses the entire figure width/height, including margins, instead
  of the inner plotting area. With 200 px of vertical margins in a 600 px
  figure, the estimated data extent is one third too small. This affects lane
  spacing, caps and outer labels. On inverted matplotlib axes the current
  transform gives a negative result for positive points, despite the base
  docstring's sign-of-points guarantee; settle that contract explicitly.

**Do:** keep ranges in data units across scale changes, validating positivity
when entering log scale. Resolve shared-axis ranges through `matches` with a
small cycle-safe helper. Use inner paper dimensions for size conversion and
document the local approximation on nonlinear axes; do not pretend exponent
spans are additive data offsets. Avoid a second private cache of axis state.

**Verify:** both call orders for limits/log scale, log→linear, reversed ranges,
shared x/y, resize and title margins, spectrum on caller-supplied shared axes,
and `point_size(units="axes")` against the rendered subplot extent.

### 10.3 Close owned figures consistently on failure [P2]

`draw_spectrum` and the catalog handle failure, but the other direct frontends
allocate figures before validating/calculating everything. Examples:
`plot_stats_summary(layout=(1,1))` with several slots fails after allocating;
`plot_bootstrap_intervals` with no bootstrap inputs also allocates first;
series/summary/matrix calculations can raise after creation. Spectrum and
correlator CLI renderers additionally lack the generic renderer's close-on-
export-failure handling.

Nested `figure_scope` contexts do not actually transfer retained figures to
the parent: a successful inner scope retains a figure, the outer tracks zero
figures, and an outer failure cannot close it. Confirmed with a recording
figure and a close spy. This contradicts the scope's transfer wording.

**Do:** validate inexpensive inputs before allocation, and apply one consistent
ownership scope at public factories and CLI render entry points. Close figures
allocated by the operation on any failure; never close caller-supplied axes.
Transfer inner retained allocations to the active parent on success. Handle
native corner/optimagic ownership too, including failures before adoption.
Do not duplicate bespoke try/except cleanup in every draw helper.

**Verify:** direct frontend failures, invalid summary geometry, failing fit
projection, export/show failures, nested scopes, and caller-owned figures.
No double close. Run CLI and kbfit factory tests.

### 10.4 Record the statistics actually drawn by summary panels [P2]

`plot_contracts._DEFINITIONS` assigns `_mean_and_error` to `summary`, while
its default bootstrap histogram panels draw 68% confidence intervals.
The returned metadata nevertheless says only `error_metric="error"`.
Mixed bootstrap/jackknife summaries have different uncertainty rules per
observable; a single global error declaration cannot describe them.

**Do:** record per-panel statistics, including the selected histogram index,
method and confidence level. Correlation and effective sample size panels
should not claim mean/error marks. Share the histogram description helper;
keep plotting and metadata defaults in agreement. Check downstream consumers
before changing the metadata shape.

**Verify:** default bootstrap summary, jackknife summary, mixed methods,
`obs_index`, custom panels, and kbfit's recorded analysis products.

### 10.5 Toggle all marks belonging to a logical legend item [P2]

**Completed with Tier 9** for interval boxes, split series, fit overlays and
axis-line hover targets. The original findings and prescription follow.

Tier 9.1 correctly identifies that a CI legend entry hides only the first box.
`SeriesPlotter._draw_series` has the same failure: filled and hollow subsets
are separate traces, only the first is labelled, and toggling it leaves the
other subset visible. A fit's labelled band and its unlabelled central line
are another composed legend item to consider.

**Do:** introduce the smallest shared legend-group convention that works for
composed marks, with one visible entry per group and a subplot-scoped identity.
Apply it to interval levels and split series; decide deliberately whether fit
lines should toggle with their band. Use stable groups rather than identifying
items by label text (labels can repeat). Reuse it for axis-line hover traces.
Keep matplotlib legends unchanged. Only then decide whether batching boxes
is worth a new mark; do not add both a rectangles API and a grouping framework
merely to fix one frontend.

**Verify:** real browser toggles in exported HTML, duplicate labels in different
subplots, all-hollow series, multiple confidence levels, and fit overlays.
JSON structure and stubbed JS tests alone cannot verify Plotly legend behavior.

### 10.6 Make prepared spectrum payloads and marker hover consistent [P2]

`PreparedSpectrum` is frozen only at the dataclass attribute level. It retains
mutable `HMarker` objects, dictionaries and a mutable `SpectrumStyle` model.
For example, prepare with `HMarker(y=.5)`, then change the original marker to
`y=50`: drawing uses 50 while `energy_range` remains around .5. This violates
the prepared-range/drawing consistency expected by report generation.
`HMarker.from_sampling` also caches automatic hover while its public
`ni_pair`, `ni_rank`, `group`, `column`, `percentiles` and sampling remain
mutable; changing them can leave the hover describing the old marker.

**Do:** snapshot marker drawing values and hover during preparation, as levels
already do. Keep source markers editable if useful, but do not retain their
live state inside the prepared payload. Choose and document whether a
sampling-backed marker is immutable after construction or recomputes its
hover; avoid silently cached mutable descriptors. Do not deep-copy all sampling
buffers to achieve this. Copy only drawing metadata/value decisions.

**Verify:** mutate source markers/styles after preparation and draw twice on
both backends; metadata, spans, values, hover and range must stay consistent.
Test edits to an unprepared sampling-backed marker according to the chosen
contract. Run kbfit spectrum tests.

### 10.7 Define series limits and overlay behavior consistently [P2]

`SeriesPlotter._seed_limits` unconditionally replaces explicit caller axes
limits, unlike the spectrum frontend. Automatic ranges include marker centres
but exclude their bands and fit predictions. A marker centred inside the
range with a wide band is clipped; a fit or marker with no kept data receives
no useful automatic range (`xs.size == 0` returns immediately).

**Do:** preserve caller-set limits unless explicit plot arguments override
them; expose x limits only if callers need an override. Include finite marker
band endpoints and drawn fit values in automatic limits, or explicitly define
fits as non-range-expanding overlays and document it. Handle overlay-only
plots deliberately. Prepare range inputs once rather than evaluating fit
predictions again during drawing. Coordinate with 10.2 for shared axes.

**Verify:** explicit/native/shared axes, wide marker bands, fits outside data
bounds, log scale, all-filtered data with overlays, and matrix grid ranges
(the grid intentionally supplies a shared fixed ylim).

## Tier 11: measured improvements and verification gaps *(done)*

### 11.1 Reduce hover overhead without changing statistics [P3; revised 9.4]

**Completed with Tier 9.** The implementation record above supersedes the
empty-`_hover` estimate below with full-path measurements and exact bounds.

On this machine, a 12×12 recording grid with eight times and 40 bootstrap
resamples, averaged over three runs, took:

| Error metric | Normal hover | `_hover` replaced by an empty string |
|---|---:|---:|
| Gaussian | 0.416 s | 0.355 s |
| Confidence interval | 0.480 s | 0.404 s |

Hover accounts for about 15–16% here; real rendering and larger sample counts
may change the fraction. The empty-hover experiment is an upper bound on the
possible saving, not an implementation benchmark. The proposed
`SigmondData.pdg_format_array` broadens the data API for a plotting concern and
is not justified yet.

**Do if needed:** use a private formatter fed central values, Gaussian errors
and **original absolute CI bounds**, computed in batches. Retain unclipped
bounds separately from errorbar offsets. Match the scalar PDG formatter and
all central metrics, and preserve signed values in hover for hollow negative
points on log axes. Avoid preparing hover on backends that ignore it, but keep
recording hover observable for tests. Benchmark larger resample counts before
committing to extra cached arrays or a public vectorized formatter.

### 11.2 Add integration coverage where smoke tests do not render [P3]

**Tier 9 follow-up:** renderer-backed empty-cell and browser legend-toggle
checks are now included. Minimum-version, contour-semantics and packaging
coverage below is also completed.

Most `test_every_portable_mark_renders`/contour smoke checks only construct
traces; the Plotly save smoke test exports HTML, which does not run Plotly.js.
Consequently empty cells, one-sided bounds and browser legend behavior can
pass the entire plotting suite.

**Do:** keep a small meaningful set of renderer-backed integration cases,
with an explicit skip when Kaleido's browser is unavailable. Inspect resolved
ranges/scales and SVG elements rather than adding brittle image snapshots.
Use a browser interaction check for legend/zoom callbacks. Exercise minimum
supported Plotly as well as the current installed version (6.8.0 in this
review); verify `zorder` support for every trace type used. Test contours with
`filled=True`, irregular levels and labels for actual numerical/rendered
semantics, not merely constructor success. A wheel install/import check should
ensure `post_script.js` remains packaged.

### 11.3 Keep small abstractions small [P3; optional]

- Preserve the explicit mark hooks and nine-entry plot catalog. Three catalog
  name checks do not justify plugin-style registration or new capability flags.
- Keep the spectrum preparation pipeline in one place until a helper has a
  distinct reusable responsibility. Avoid turning every ten lines into a class.
- `PlotStyle.markers` remains a configured public field, but no production
  frontend reads it: `SeriesStyle.marker_cycle=None` always selects `"o"`.
  Decide whether it is intentional user-facing palette metadata; either wire
  it into a documented default or remove it under D2 with affected callers
  reviewed. Do not keep a setting that appears to control the plots but does
  nothing.
- Avoid a new public bounds dataclass solely for a boolean. Resolve uncertainty
  choices once where it simplifies the path, retaining scalar input validation.

### Validation record for this review

- Baseline full suite: 1171 passed, 8 skipped, 76 failed, 6 errors.
- Final full suite: 1189 passed, 8 skipped, 76 failed, 6 errors. The exact
  failure/error set matches the baseline (no added or removed failure entries).
- kbfit: 602 passed, 6 skipped, 2 xfailed. No kbfit source changes.
- 16 frontend/view variants rendered to PNG on each backend (32 images),
  covering sampling plots, correlator/effective-energy/summary, matrix views,
  coverage and spectrum. Contact comparisons and focused full-size checks
  exposed the significance color-scale bug. Additional PNG/SVG checks verify
  marker suppression and gray direction. No claim of pixel-identical parity.
- Built a wheel and confirmed `slat/plotting/_plotly/post_script.js` is included.
- Ruff passes for the plotting packages, plot contracts and added regressions.
- Scratch results/images are under `/tmp/plot-review-*`; no commits were made.

---

### Tiers 10–11 implementation record (2026-10-05)

- **10.1:** Plotly marks and axis configuration use explicit native references.
  Plain figures, native grids with holes/secondary axes, and domain-based
  Cartesian layouts support portable drawing without replacing their layout.
  Trace references without explicit layout entries are supported too. Native
  grids expose primary Cartesian cells; secondary/overlay axes remain native,
  non-Cartesian cells remain exportable, and ambiguous domain layouts fail
  explicitly. Output preservation now includes wrapped empty Cartesian cells.
- **10.2:** scale changes preserve data-unit limits in either call order.
  Native `matches` components resolve ranges in both directions and tolerate
  cycles; range and scale updates propagate across them. Point conversion
  uses inner dimensions and subplot domains, native export defaults, resize
  and title margins. Nonlinear conversion is a local midpoint estimate;
  positive points yield positive extents even on inverted matplotlib axes.
- **10.3:** one ownership decorator covers public frontend factories and all
  CLI renderers. Nested successful allocations transfer to their parent scope;
  failures close owned figures once and leave caller-owned axes open. Cheap
  summary/bootstrap validation precedes allocation. Corner's native surface
  is owned before third-party drawing; catalog native results are owned before
  adoption. Save/show and adoption failures share the same cleanup path.
- **10.4:** summary statistics record one description per drawn panel.
  Histograms include their observable index, sampling method and actual
  interval/error choices; correlation and ESS panels claim no mean/error mark.
  A shared histogram confidence default keeps metadata and drawing aligned.
  kbfit's existing analysis record/publication consumers accept the new shape.
- **10.5:** completed with Tier 9; existing browser legend checks still pass.
- **10.6:** preparation snapshots marker geometry/hover, styles, native kwargs,
  ranges and statistics. Source edits cannot change a prepared drawing.
  Sampling-backed HMarkers are immutable snapshots; plain markers stay
  editable. Mutable interval inputs are copied, sampling buffers are retained
  only for inspection, and prepared styles remain serializable.
- **10.7:** series respect explicit/native/shared caller limits. Automatic
  ranges include finite marker band endpoints and fit curves, including
  filtered/reference-only and fit-only plots. Fit predictions are prepared
  once before allocation and reused for range selection and drawing.
- **11.1:** completed with Tier 9; batched hover semantics remain unchanged.
- **11.2:** numerical SVG/browser checks exposed and fixed filled contours:
  they now fill only the bands between adjacent, possibly irregular levels.
  Plotly constraint contours shade the invalid region, so the outside-allowed
  operation (`][`) is deliberately used to shade the inside band. Line labels
  and boundaries are verified against a known numerical field. Levels receive
  shared finite/increasing validation. Real browser drag-zoom verifies marker
  scaling/capping/reset, dense hover targets and stable groups/trace indices.
  Current Plotly 6.8.0 and minimum 5.21.0 both render Scatter, Bar, Heatmap and
  Contour with zorder. Minimum-version browser checks run independently of
  Kaleido 1's newer Plotly requirement. A built wheel was installed in scratch
  space; isolated imports confirm lightweight schemas and packaged JS.
- **11.3:** SeriesStyle's default now uses the drawing figure's marker palette;
  explicit marker cycles override it. No catalog registry, public bounds type,
  new mark vocabulary or generic hook framework was introduced.

Regression coverage lives in `tests/test_plotting_final_tiers.py` and
`tests/test_plotting_final_rendering.py`. Optional renderer/browser dependencies
skip explicitly when unavailable; started renderers must pass their checks.

Validation:
- Full baseline: 1230 passed, 8 skipped, 76 failed, 6 errors.
- Full final: 1287 passed, 8 skipped, 76 failed, 6 errors. The exact failure/error
  set is unchanged. All 57 added regression cases pass, including real browser
  and SVG checks.
- Plotly 5.21: 393 plotting cases passed, 1 skipped; the separate browser
  contour/zorder/zoom integration case also passed.
- kbfit: 602 passed, 6 skipped, 2 xfailed; no source changes required.
- Ruff passes on all touched Python files.
- The final built wheel contains `post_script.js`; an isolated scratch install
  imports the lightweight schemas and exports HTML with the packaged callback.

Scratch logs, wheel, diff and all 32 frontend PNGs are under
`/tmp/plotting-final`. Pixel comparisons
against Tier 9 are unchanged for 23 images; the nine changed images were
reviewed at full resolution and reflect marker palettes, preserved summary
ranges, and Plotly spectrum spacing from inner-dimension point conversion.
No commits or kbfit source changes were made.

## Out of scope

- Porting `sigmondsamplings/fitting/fit_plotter.py` (1,324 lines of raw
  matplotlib) and kbfit's `BoxQuantizationPlotter` to slat. This is a separate
  project. Its glue (`as_figure`, type sniffing in `wrap()`, `QCStyle.native`)
  stays until then.
- The remaining plotly-vs-matplotlib differences listed in
  `docs/plotly_mpl_parity_findings.md` → *Known remaining differences*.
