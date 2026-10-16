# matplotlib / plotly parity: findings and fixes

Found by rendering every sigmondsamplings frontend on both backends
(`SamplingPlotter`, `CorrelatorPlotter`, `CorrelatorMatrixPlotter`, `SectorSpectrumPlotter`)
and comparing the PNGs. matplotlib was correct throughout; every issue was plotly-only
unless noted. All items below are fixed (2026-10-05).

## Bugs

1. **Axis lines drawn before any trace vanished.** `add_hline`/`add_vline` skip a
   subplot with no traces unless `exclude_empty_subplots=False`.
2. **`\_` in labels became a subscript** in `to_html` (`\text{eta(0)\_ref}` →
   `eta(0)\<sub>r</sub>ef`). Escaped `\_ \% \& \#` are now held out of the script regex.
3. **`alpha` was dropped when no color was given** (`_color(None, a)` is None). It now
   becomes the trace opacity.
4. **dpi was counted twice on export.** Superseded by the layout-scale change below.
5. **`suptitle` overwrote the axes title of a 1×1 figure**, since both used
   `layout.title`. Axes titles are now always annotations.
6. **A legend appeared without `legend()` being called**, and showed every subplot's
   entries in one figure-level list (op0..op2 twice in the correlator summary; nine
   entries in the matrix grid), placed outside the plot over any colorbar.
7. **Template defaults**: `plotly_white` grids on both axes (so `grid(axis="y")` still
   showed an x grid), white bar outlines, SI tick labels (`100µ`), shapes at 0.3 opacity.
8. **Mixed math strings lost text.** MathJax renders a string only as one math span and
   drops plain text around it: `"$68\%$ CI"` lost `" CI"`, `"Mixed $x$ label"` lost
   `"Mixed "`, and `\n` was ignored.
9. **matplotlib** (`heatmap` with `x`/`y` and `origin="upper"`): the values were flipped
   but not the coordinates, so `values[0]` was drawn at `y[-1]`.
10. **plotly.js stops drawing axis lines once a subplot's traces have more than one
    `zorder`.** The frame is therefore drawn as a rectangle shape per subplot
    (`slat-frame{n}`), and `hide()` removes it.

## Design changes

- **Layout scale.** Plotly figures are laid out at a fixed `PX_PER_INCH = 100`, so HTML
  sizes are unchanged from before. `style.dpi` affects only raster export
  (`scale = dpi / 100`), as `savefig(dpi=)` does on matplotlib.
- **Points to pixels.** Every size given in points (marker size, line width, cap size,
  font size) is converted at 100/72 px per point, and `point_size` uses the same factor.
  A size of `None` takes matplotlib's default value (lines 1.5, markers 6, fonts 10,
  titles 12, no error-bar caps). This replaces the spectrum plotter's `_plotly_default`.
- **Per-axes legends.** Each subplot owns `legend{n}`, hidden until `Axes.legend()`
  shows it inside that subplot. Labelled axis lines are legend entries (shapes support
  them natively) instead of floating annotations.
- **zorder.** This maps to plotly's integer trace `zorder` (×10), with matplotlib's
  defaults (images 0, patches 1, lines 2). Axis-line shapes go under every trace when
  `zorder < 1.5`, otherwise over. Requires plotly ≥ 5.21.
- **Text routing.** Titles, axis labels and annotations go through `to_plotly`, which
  merges plain and math parts into one MathJax span and uses HTML for multi-line text.
  Ticks and legend entries use HTML (`to_html`).
- **Titles and spacing.** The figure title reserves top margin above the axes titles.
  Row gaps are set in pixels from the x axis plus that row's title, so two-line titles
  fit and shared-x rows sit close. Legends and colorbars follow their subplot when the
  rows move.
- **Colorbars** take their room from their own subplot, as `fig.colorbar(ax=)` does, and
  span only that subplot's height, with a vertical label.
- **Look.** A black frame, outward ticks, no grid unless `PlotStyle.grid` (now
  `False` by default, and honoured by both backends), no zero line, and powers of ten
  on log axes.
- **API.** `Figure.suptitle(text, *, fontsize=None)` replaces `**kwargs`, which a
  matplotlib-only option broke on plotly.
- **Frontend.** `SamplingPlotter` passes explicit colors from module constants in
  `sampling.py`.

## Known remaining differences

- `legend(loc="best")` is upper right on plotly; matplotlib picks the least-overlapping
  corner.
- MathJax glyphs render somewhat smaller than plotly text at the same font size.
- A log axis spanning only a few decades also labels the 2 and 5 ticks between powers
  of ten; matplotlib labels the decades only.
- Marker extents for spectrum stacking are still estimated on plotly, from the axis
  range and subplot height, rather than measured.

## How to check

- Run in the `lqcd-dev` conda env:
  `/opt/homebrew/Caskroom/miniconda/base/envs/lqcd-dev/bin/python`.
- kaleido is installed, so `figure.save("x.png")` works on plotly. Compare against the
  matplotlib PNG at full resolution: 1 px frame lines vanish in a downscaled preview.
- The test suite has 76 failures and 6 errors at baseline. Compare the set of failing
  tests before and after a change, not the count.
