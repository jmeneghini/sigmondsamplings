# Correlator views: derived quantities, diagnostics and plots

Status: all five phases (§7) implemented. Revised to follow the conventions of the
existing frontends (`SamplingPlotter`, `SpectrumPlotter`, `fit_plotter`).
Scope: `sigmondsamplings/correlators/` (new analysis layer, with private
NumPy kernels in `correlators/_kernels.py`),
`sigmondsamplings/plotting/{series,correlators,markers}.py` (new plots), two
small additions to `slat.plotting`, and `ss query corr --plot`.
Builds on `operators_correlators_api_design.md` and `plotting_api_design.md`.

## 1. Why

The correlator API reads labeled series and matrices, but nothing downstream
turns them into what people look at. Today that means going through Sigmond
`DoPlot` tasks (via PyCALQ) or writing numpy by hand. The rho E250 `pSq3 Ep`
file showed the gaps quickly:

- **Coverage is invisible.** Three of 66 cells were never measured (VI_0/VI_1).
  Only an error message tells you, and the available sub-basis is found by a
  CLI-private greedy helper (`cli/corr.py::_subset_hint`).
- **No effective energies.** It is the first plot anyone makes of a correlator,
  and it has subtle conventions (Sigmond keys the plain kinds at `t + step/2`
  but the added-constant kinds at `t`, and solves the cosh kinds numerically).
- **Nothing to judge a basis with.** Before a pivot you want the normalized
  matrix (which operators are nearly the same) and the significance of each
  cell (which are noise).

The split follows the existing packages. **Numbers first**: every plotted
quantity is a `SigmondData` or `CorrelatorMatrix` method or function you can
also fit, write or print. **Plots are thin**: they draw those objects through
`slat.plotting`, on either backend, using the same conventions as the sampling
and spectrum plotters.

## 2. Non-goals

- The single pivot itself (`single_pivot_design.md`). The
  plots take any `CorrelatorMatrix`, so rotated matrices will plot unchanged
  once the pivot exists.
- Eigenvalue diagnostics: per-time GEVP eigenvalues, condition numbers versus
  `t0`, and their plot. Their only job is choosing the pivot's parameters, and
  their open questions (metric per sample or full sample, level matching across
  samples) are the pivot's own, so they moved to
  `single_pivot_design.md` §4.7.
- Fits. Plots overlay a finished `FitResult`; they never run one.
- Sigmond's plot-file formats (grace/xmgrace). The output is `slat.plotting` figures.
- Porting `fit_plotter` (phase 3 of the plotting doc). Nothing here depends on it;
  see §4.6.

## 3. Analysis layer (not plots)

### 3.0 Where the numbers live

| layer | location | contents |
| --- | --- | --- |
| kernels | `correlators/_kernels.py` (private) | pure NumPy on raw buffers with the sample axis last, `(..., time, N)`: the four effective-energy kinds (the pivot adds its eigensolver here later). No SS types, no xarray. |
| calculators | `correlators/effective.py`; methods on `CorrelatorMatrix` (§3.3); `correlators/rotation.py` for the pivot later | take `SigmondData` / `CorrelatorMatrix`, call a kernel once on the whole buffer, rebuild dims, coords and infos around the result |
| plots | `plotting/{series,correlators,markers}.py` (§4) | draw calculator outputs only; no math |

Not in `slat`: it holds labels, constants and presentation, no physics math,
and nothing outside SS would reuse these (kbfit depends on SS; PyCALQ calls
`sigmond.getEffectiveEnergy`). Not in `utils.py`/`stats.py`, which are already
grab-bags. Not a `SigmondData` method: the generic container should not know
what a correlator is. `CorrelatorMatrix` may offer convenience methods.

The kernel/wrapper split is what makes the calculators fast: one kernel call
over `(time, operator, sample)` instead of a Python loop over cells or samples.
It also fixes the seam for a compiled kernel later (§3.4).

### 3.1 Effective energy — `correlators/effective.py`

```python
def effective_energy(
    corr: SigmondData,            # real samplings, has a "time" dim; other dims broadcast
    *,
    kind: str = "forward",        # forward | symmetric | forward_const | symmetric_const
    step: int = 1,
    time_extent: int | None = None,   # default: corr.ensemble_info.temporal_extent
    subtract_const: float = 0.0,  # Sigmond's subtract_const, removed before the kernel
    on_fail: str = "drop",        # drop | nan | raise
) -> SigmondData                  # dim "time", integer coords: the source time t
```

**The four kinds**, from `EffectiveEnergyCalculator` in
`sigmond/cpp/tasks/task_utils.cc`. `b = e^{-m}`, `s = step`, `K = T - 2t`:

| kind (Sigmond type) | model | solved from | plotted at |
| --- | --- | --- | --- |
| `forward` (0) | `A e^{-mt}` | closed form: `m = -ln(C(t+s)/C(t)) / s`, fails if the ratio `<= 0` | `t + s/2` |
| `symmetric` (1) | `A (e^{-mt} + e^{-m(T-t)})` | root of `(1 + b^K) r - b^s - b^{K-s} = 0`, `r = C(t+s)/C(t)` | `t + s/2` |
| `forward_const` (2) | `A e^{-mt} + B0` | closed form on `r = (C(t+s) - C(t)) / (C(t) - C(t-s))` | `t` |
| `symmetric_const` (3) | `A (e^{-mt} + e^{-m(T-t)}) + B0` | root of `(1 - b^{K+s}) r - b^s + b^K = 0` on the same difference ratio | `t` |

Details that must match Sigmond exactly:

- **Arguments.** `T >= 8` and `1 <= step <= T/4`, else `ValueError` (Sigmond's
  constructor checks). The `_const` kinds need `C(t - step)`, so their first time
  is `step`.
- **Symmetric kinds solve numerically.** Sigmond uses `rtsafe` (Newton with a
  bisection safeguard) from the initial guess `b = r^{1/s}`, first walking the
  Newton step outward at most 30 times to bracket the root inside `(0, 1)`, then
  converging to `1e-10` in at most 200 iterations. Failing to bracket or to
  converge is a failure, the same as a non-positive log argument.
- **Validity ranges and mirroring.** The symmetric kinds fail unless `0 <= r < 1`,
  and unless `t < T - s` (kind 1) or `t < T - 2s` (kind 3). For `t >= T/2` both
  mirror onto the near side of the lattice, but not in the same way: kind 1 uses
  `t' = T - t - s` with the inverted ratio `C(t)/C(t+s)`; kind 3 uses `t' = T - t`
  with `(C(t-s) - C(t)) / (C(t) - C(t+s))`. Reproduce both as written.
- **Full sample included.** The kernel runs on every column of the buffer, the
  full-sample value at column 0 included. The central value is therefore the
  effective energy of the full-sample correlator, as in Sigmond's `getEstimate`,
  not a mean over resamples.
- **Failure policy.** If one sample fails at time `t`, Sigmond drops `t`
  entirely. `on_fail="drop"` does the same and emits one warning naming the
  dropped times (`SigmondData` has no `attrs`, so a warning is where they go;
  the times missing from the result's coord say the same thing).
  `"nan"` keeps the time with NaN in the failed samples, which is what plots use.
  `"raise"` raises and names the times.
- **Inputs.** Complex data raises and asks for a component, the same rule as
  `SamplingFit.from_data`, which covers Sigmond's `ComplexArg`. Bins raise and
  ask for samplings, since the map is nonlinear (`NonlinearBinsWarning` territory).
- **Not the acosh estimator.** `acosh((C(t+1) + C(t-1)) / 2C(t))` is also exact
  for a pure cosh, but it is a different three-point estimator and differs sample
  by sample on real data. If it is ever wanted, it gets its own `kind` name;
  `symmetric` always means Sigmond's type 1.
- **Aligned by position.** Build `C(t)`, `C(t+s)` and `C(t-s)` as positional
  slices of the raw buffer inside the kernel, then attach coords. Dividing two
  coord-labelled slices would make xarray align them by time label and give
  `C(t)/C(t) = 1`. This is the main correctness trap, so it gets a dedicated test.
- **Time coord (open question 1).** `SigmondData` holds one index coord per dim
  and no extra coords or attrs, so the draft's `source_time` coord and
  `plot_offset` attr cannot be stored. The proposal: the `time` coord is the
  integer source time `t`, so the result `sel`s and fits like the correlator,
  and the plot offset is a pure function of the arguments,
  `effective_time_offset(kind, step)` (`step/2` or `0`). The result's
  array-level `observable_info` records `kind` and `step`, and `SeriesPlotter`
  takes `x_offset=`, defaulting to the offset that info implies.
- Works on anything with `time`: one correlator, `matrix.diagonal("real")`
  with dims `(time, operator)`, principal correlators `(time, level)` once the
  pivot exists, or a fit's theory
  curve (§4.6) — which is what lets the plots overlay a model's effective energy
  without a second code path.
- **Oracle:** the `sigmond` Python bindings (`EffectiveEnergyCalculator` per
  sample). These tests are skipped when the bindings are absent, like the label
  tests. `getEffectiveEnergy` needs an `MCObsHandler`, so its drop rule and
  keying are tested from its source instead: a time is kept iff every sample
  succeeds, the result keyed at `t`.
- **As built:** the time coords may be sparse or unsorted; the wrapper scatters
  them onto a contiguous grid and times whose inputs are missing are simply
  absent from the result (not warned about, as in Sigmond). The drop warning is
  an `EffectiveEnergyWarning`, and it covers only times whose inputs were
  present. A non-finite result counts as a failure. Each cell is an `AtObsInfo` whose
  immutable `EffectiveEnergyInfo` element carries `kind`, `step`, `time_extent`,
  `subtract_const` and `time_offset`.

### 3.2 Coverage — `CorrelatorIndex`

Move the logic out of the CLI into the library:

```python
index.coverage(basis, *, hermitian=None, subtract_vev=None, key_form=None) -> xr.DataArray
    # bool (time, sink, source): cell available in the chosen form, Hermitian-completed
index.available_bases(basis, times=None, *, hermitian=None, subtract_vev=None,
                     key_form=None) -> list[tuple[Operator, ...]]
    # maximal available sub-bases, largest first (exact up to 16 operators; greedy above)
```

`basis` is a `Channel`, a `CorrelatorMatrixInfo` or a sequence of operators,
with the flags defaulting as in `matrix()`.

`coverage` reuses `_Request.cell_times`, so it gives the same answer that
`matrix()` would. The key-form ambiguity rules of `_choose_form` apply, which
fixes the "hint, not a guarantee" caveat of the old `_subset_hint`. A connected
request derived from raw data also needs both operators' VEVs. The CLI hint
(`matrix_records`) now calls `available_bases`. The operators ride along on
`attrs["operators"]` (with `key_form` and the flags) so a plot can label the
axes without the index.

As built: `coverage` spans every time at which any cell is stored. A
sub-basis is available when all its cells are available at every one of
`times`, and `times=None` means every time in the coverage, so `matrix()` of
the result keeps them all; pass `times` to ask about a shorter range. The exact
search is Bron–Kerbosch over the graph of operators whose diagonal is stored,
linked where both directions are available; it lists every maximal clique,
ties in size ordered by basis position.

### 3.3 Matrix-level views — `CorrelatorMatrix`

| method | returns | purpose |
| --- | --- | --- |
| `normalized(norm_time)` | `SigmondData (time, sink, source)` | `C_ij(t) / sqrt(C_ii(t0) C_jj(t0))`, i.e. Sigmond's `TemporalCorrelatorMatrixRescaled` |
| `significance(time)` | real `(sink, source)` `xr.DataArray` | `|C_ij| / err`, for spotting noise-dominated operators |
| `check_hermitian()` | exists | plotted as a residual heatmap |

**Normalization uses the full-sample estimate, as Sigmond does.** In
`TemporalCorrelatorMatrixRescaled` (`tasks/task_plot.cc`), the factor `f_i`
for row `i` is the full estimate of `Re C_ii(t0)` for a Hermitian matrix, and
the modulus of the full estimate of `C_ii(t0)` otherwise. Each cell's estimate
and error, real and imaginary parts alike, are multiplied by
`1 / sqrt(f_i f_j)`. That is one constant per cell, so the rescale is linear
and adds no bias; every sample is divided by the same number. There is no
per-sample option. A missing `C_ii(t0)` or a negative factor raises and names
the operator (Sigmond skips the plots for that row).

`normalized` returns a plain `SigmondData`, not a `CorrelatorMatrix`. It is
`D C D` with a real diagonal `D`, which is formally a basis change, but the
result is a display quantity and wrapping it would invite treating the rescaled
operators as a new basis. It is used by the grid (`what="normalized"`) and
the heatmap: `|C̃_ij(t0)|` close to 1 means operators `i` and `j` are nearly the
same, which is the first thing to check when choosing a basis.

Eigenvalues and condition numbers are not here; see §2 and the pivot plan.

As built: a zero factor raises like a negative one ("not positive"), and a
`norm_time` outside the matrix's times is the "missing `C_ii(t0)`" case.
`significance` uses the full-sample value and the error of the complex data
(both parts), with `inf` where the error is zero.

### 3.4 Kernel performance

A prototype of the `symmetric` root solve, T=96, 48 times, run on a loaded
machine (ratios matter, not absolute times):

| series × samples | `forward`, NumPy | `symmetric`, NumPy | `symmetric`, compiled (numba) |
| --- | --- | --- | --- |
| 1 × 201 | 0.06 ms | 1.7 ms | 1.1 ms |
| 12 × 201 | 0.6 ms | 31 ms | 15 ms |
| 144 × 201 | 11 ms | 630 ms | 180 ms |
| 144 × 2001 | 0.40 s | 21.7 s | 1.9 s |

The two agree to about `1e-15`. The closed-form kinds are one division and one
log per element and are already as fast as NumPy gets. The symmetric kinds are
where compiled code wins, and the gap grows with size (2× to 11×). The prototype
NumPy version is naive: it recomputes three powers per iteration and keeps
iterating over every element until the slowest one converges.

Plan:

1. **NumPy kernels with a strict raw-array interface** (arrays in, arrays plus a
   failure mask out). **Done and measured.** The kernel iterates only on the
   elements still bracketing or converging. Same machine, T=96, 48 times:

   | series × samples | `forward` | `symmetric` |
   | --- | --- | --- |
   | 12 × 201 | 2.6 ms | 43 ms |
   | 144 × 201 | 20 ms | 0.52 s |
   | 144 × 2001 | 0.21 s | 10 s |

   That is about 2× the naive prototype, not the hoped-for most of the gap: the
   cost is the handful of full-size Newton evaluations (Sigmond's `rtsafe`
   starts from the bracket midpoint, not the good initial guess), not the
   powers. Computing the powers as `exp(k ln b)` was tried and bought nothing,
   so the kernel keeps plain `pow` like Sigmond. Agreement with Sigmond's
   calculator: identical failure flags, values within the `1e-10` root tolerance.
2. **No Cython.** It makes SS a compiled package: wheels for every OS ×
   architecture × Python (cibuildwheel), or a compiler at install time; a
   different build backend; editable installs that need a manual rebuild. That
   cost is only worth paying if SS needs compiled code for something bigger, such
   as a port of Sigmond's pivot.
3. **Optional numba backend, only if profiling asks for it.** Thousands of
   samples, or effective energies of many rotated matrices once the pivot
   exists, are the likely cases.
   It would be an extra, `sigmondsamplings[fast]`, selected behind the same kernel
   interface, with the NumPy kernel as the fallback. SS stays a pure-Python wheel.
   Numba's costs are a heavy dependency (llvmlite), a JIT compile of about a
   second on first use per process, and upper bounds on the NumPy versions it
   supports, which is why it stays optional. Tests run both backends against the
   same oracle when numba is installed.

Realistic use, a diagonal of 12 operators with 201 samples, is well under
0.1 s either way. Reading that matrix from HDF5 takes about 0.8 s.

## 4. Plots

### 4.1 Conventions taken from the existing frontends

The first draft proposed free functions. The existing frontends settled a set
of conventions, and the correlator plots should follow them rather than start a
third style:

| convention | where it comes from | what it means here |
| --- | --- | --- |
| **A plotter bound to one data object**, with `plot_*` methods | `SamplingPlotter(stats)`, `SpectrumPlotter(collection)` | `CorrelatorPlotter(series)`, `CorrelatorMatrixPlotter(matrix)` |
| **One style dataclass** per plotter holding every visual setting, with a `*_kwargs` / `native` escape | `SpectrumStyle`, `FitPlotStyle` | `SeriesStyle` (§4.3) |
| **Uniform call tail** `axes=None, figsize=None, backend=None, ax=None`, through `_util.resolve_axes`; single plots return `Axes` | every `SamplingPlotter` method, `SpectrumPlotter.plot` | unchanged |
| **Composites return `Figure`**, drawing each panel by passing `axes=fig.axes(r, c)` to the single-plot method; `panels=[...]` selects them | `plot_stats_summary` | `CorrelatorPlotter.plot_summary(panels=...)` |
| **Colours are explicit**, drawn from a cycle that restarts per plot | `IndexedCycle`, `PlotStyle.color_cycle()` | cycle from `slp.get_style()` unless the style overrides it |
| **Hover text on every data mark** | spectrum `_hover_text`, `plot_sampling_errorbar` | `"{label}<br>t={t}: {value:.6g} ± {err:.2g}"` |
| **Horizontal reference lines are `HMarker`s** | `SpectrumPlotter.plot(markers=...)` | same class, shared (§4.5) |
| **Seed limits before size-dependent work** | `SpectrumPlotter._seed_axes_limits` | the grid's per-cell y ranges (§4.4) |
| **LaTeX through `ensure_math`**; plotly flattens tick and legend text itself | `_util.latex_label`, `slat.plotting.text` | operator `latex_str` works on both backends |

The first draft argued against a class because "`SpectrumPlotter`'s grouping
state has no analogue here". It does: the **operator basis**. Every view of one
matrix (the grid, the diagonal overlay, the heatmap ticks)
must agree on each operator's label, colour and order. Holding that on a plotter
object is what makes the views consistent without threading `labels=` and
`colors=` through every call.

### 4.2 Layout

```
sigmondsamplings/plotting/
    _util.py        + central_and_error(), labeled_heatmap(), summary_grid()
    markers.py      HMarker (moved from spectrum.py, re-exported there), FitOverlay
    series.py       SeriesStyle, SeriesPlotter     # any SigmondData over one dim
    correlators.py  CorrelatorPlotter, CorrelatorMatrixPlotter, plot_coverage
```

Three shared helpers move into `_util.py` so the new plots reuse rather than
copy:

- **`central_and_error(data, *, confidence_level=None)`** returns `(y, yerr)`
  honouring `rc["plot.central_metric"]` and `rc["plot.error_metric"]` (a `(2, N)`
  asymmetric `yerr` for `"confidence_interval"`, offsets clipped at 0). These rc
  keys were defined but read by no plotter. The correlator plots are their first
  users, and `SamplingPlotter`/`SpectrumPlotter` can adopt the helper later.
  A bootstrap-only metric (`bias_corrected_mean`, `confidence_interval`) on
  jackknife data raises and names the rc key; there is no silent fallback.
  Complex data raises and asks for a component.
- **`labeled_heatmap(axes, values, *, labels, cmap, vmin, vmax, colorbar_label,
  hover=None, annotate=None)`** — the `origin="upper"` + ticks recipe of
  `SamplingPlotter.plot_correlation_matrix`, which is rebuilt on it without
  changing behaviour.
- **`summary_grid(n)`** — today's `sampling._summary_grid`, moved unchanged.

### 4.3 `SeriesPlotter`: the shape-generic primitive

Correlators, effective energies and, later, principal correlators are all the
same kind of object: a `SigmondData` over one dim (usually `time`) with at most
one more dim whose entries are separate series. After the move to a
shape-generic `SigmondData`, that is the plot to write once. The correlator
plotters are thin layers over it.

```python
@dataclass
class SeriesStyle:
    markersize: float = 5.0
    capsize: float = 3.0
    linewidth: float = 1.2
    dodge: float = 0.2               # total x spread shared by the hue series, in x units
    color_cycle: Iterable[str] | None = None   # None -> slp.get_style().color_cycle()
    marker_cycle: Iterable[str] | None = None  # None -> one marker ("o")
    negative: str = "hollow"         # on a log axis: hollow | drop
    max_rel_error: float | None = None   # hide points with err/|y| above this
    band_alpha: float = 0.25
    errorbar_kwargs: dict[str, Any] = field(default_factory=dict)

class SeriesPlotter:
    def __init__(
        self,
        data: SigmondData | Sequence[SigmondData],
        *,
        x: str = "time",
        x_offset: float | None = None,  # added to x when drawing; default from the data (§3.1)
        hue: str | None = None,       # default: the one non-x dim, if any
        labels: Sequence[str] | None = None,   # default: each series' latex_str
        colors: Sequence[str] | None = None,   # pin colours (used by the matrix plotter)
        style: SeriesStyle | None = None,
    ): ...

    def plot(
        self, *,
        yscale: str = "linear",
        markers: Sequence[HMarker] | None = None,
        fits: Sequence[FitOverlay] | None = None,
        xlabel: str | None = None, ylabel: str | None = None, title: str | None = None,
        axes=None, figsize=None, backend=None, ax=None,
    ) -> slp.Axes: ...
```

Behaviour:

- **Series.** A list of 1-D inputs, or the `hue` dim of one 2-D input. More than
  two dims raises and asks for a `sel`. Series `k` of `n` is offset in x by
  `dodge * (k/(n-1) - 1/2)` so overlapping error bars stay readable; hover shows
  the true x.
- **Error bars** come from `central_and_error`. Complex data raises and asks for
  a component, the same rule as `SamplingFit.from_data`.
- **Log axes.** Where the central value is negative, `negative="hollow"` draws
  `|y|` with open markers (the new `fill=False` in §4.7). `"drop"` omits the point.
- **`max_rel_error`** is Sigmond's cut for effective-energy plots at large `t`.
  It also drives autoscaling: limits are seeded from the kept points only, so one
  noise-dominated late time does not flatten the plateau.
- **`markers` and `fits`** are drawn after the data. They are the same objects on
  every series plot (§4.5, §4.6).

`SamplingPlotter.plot_sampling_errorbar` is a special case of this (one dim of
observables). Rebasing it is a follow-up, not part of this work.

### 4.4 Correlator plotters

```python
class CorrelatorPlotter:
    """One correlator or a stack of them: SigmondData (time,) or (time, operator)."""

    def __init__(self, corr, *, component="real", labels=None, colors=None, style=None): ...

    def plot(self, *, yscale="log", markers=None, fits=None, **tail) -> Axes
    def plot_effective_energy(self, *, kind="forward", step=1, markers=None,
                              fits=None, **tail) -> Axes
    def plot_summary(self, *, panels=("correlator", "effective_energy"),
                     kind="forward", step=1, markers=None, fits=None,
                     figsize=None, backend=None) -> Figure
```

`plot_effective_energy` calls `effective_energy(..., on_fail="nan")` and hands
the result to a `SeriesPlotter` with the same labels and colours, so a correlator
and its effective energy always look like the same series. `plot_summary` is a
`figure(rows=2, height_ratios=[1, 2], sharex=True)` with `C(t)` above `E_eff(t)`.

```python
class CorrelatorMatrixPlotter:
    def __init__(
        self,
        matrix: CorrelatorMatrix,
        *,
        labels: str | Callable[[Operator], str] | Sequence[str] = "index",
        colors: Sequence[str] | None = None,   # default: the style cycle, resolved here once
        component: str = "real",
        style: SeriesStyle | None = None,
    ): ...

    operator_labels: tuple[str, ...]     # resolved once; used for ticks, legends, cell titles
    operator_colors: tuple[str, ...]     # one per operator, fixed for every view

    def diagonal(self) -> CorrelatorPlotter          # hand-off: the diagonals, same labels/colours
    def plot_diagonal(self, *, effective=False, **kw) -> Axes
    def plot_grid(self, *, what="correlator", norm_time=None,
                  diag_range=None, offdiag_range=None, **kw) -> Figure
    def plot_heatmap(self, time, *, value="normalized", norm_time=None, **tail) -> Axes
```

| method | draws | built from |
| --- | --- | --- |
| `plot_diagonal` | all `C_ii(t)` (or their effective energies) overlaid, coloured by operator | `diagonal().plot` / `.plot_effective_energy` |
| `plot_grid(what=)` | N×N small multiples of `C_ij`, `C̃_ij` (`"normalized"`) or `E_eff` of the diagonals (`"effective_energy"`, off-diagonals hidden) | one `SeriesPlotter` per cell on `fig.axes(i, j)` |
| `plot_heatmap(value=)` | `"normalized"` (RdBu, ±1), `"significance"` (viridis, from 0), `"hermiticity"` (Reds, `vmax = 2·tol`) at one time | `labeled_heatmap` |

Grid details, which carry the Sigmond conventions:

- **Two y ranges.** Sigmond scales diagonals and off-diagonals separately.
  `sharey` is all-or-nothing, so the plotter computes the two ranges in numpy
  (from the kept points, §4.3) and seeds each cell's `ylim` before drawing, the
  way `SpectrumPlotter` seeds its limits. `diag_range=` and `offdiag_range=`
  fix them the way Sigmond's `DiagonalVerticalRange` and
  `OffDiagonalVerticalRange` do: diagonals span `[-0.1, diag_range]`,
  off-diagonals `[-offdiag_range, offdiag_range]`. Each is automatic when unset.
- **`stored` shows as fill.** A cell point is hollow where `matrix.stored` is
  false, i.e. the value came from the flipped direction. Transformed or
  hand-built matrices have `stored=None` and draw everything filled.
- **Labels on the edges only**: row labels as the first column's `ylabel`,
  column labels as the first row's `title`. With `labels="index"` these stay short.
- **Size guard.** Above 12 operators a grid raises and points to `plot_heatmap`
  or `select()`. A 144-subplot plotly figure is too slow to be useful.
- **As built:** diagonal cells take the operator's colour and off-diagonal ones
  a neutral grey; each cell's hover names both operators in full. The hollow
  points use a new `SeriesPlotter(hollow=[mask, ...])` option, and the seeded
  ranges a new `plot(ylim=...)`; both are general, not grid-only.
  `what="normalized"` defaults `norm_time` to the first time. `plot_heatmap`
  normalizes at the shown time by default (so the diagonal is 1) and shows the
  chosen component; its hover gives the full complex value.

**Operator labels** are the one real design problem: a ππ short form is about
70 characters. `labels="index"` (numbers from `--operators`, the default) keeps
grids and heatmaps legible, `"short"` uses a compact derived form such as
`ππ[(-1,0,-1),(2,1,0)] CG1`, `"latex"` uses `op.latex_str`, and a callable does
anything else. The compact form belongs on `BasicLapHOperator` (e.g. `compact_str`),
not in the plotting code.

As built: `compact_str` on both operator kinds. A single hadron is
`π(0,0,0) SD5` (flavour symbol, momentum, spatial type and id); several are
flavour symbols, the momenta, then any non-`SS_0` spatial parts and the
`CG`/`isoCG` ids when non-zero; a GI operator is `ROT 3`. It is not
guaranteed unique, which is why hover keeps the full form. **Hover always carries the full `str(op)`**, whatever
the tick label is — on the heatmap this needs `hover=` (§4.7).

### 4.5 Coverage

```python
def plot_coverage(coverage: xr.DataArray, *, labels="index", title=None, axes=None,
                  figsize=None, backend=None, ax=None) -> Axes
```

The `labels` options (§4.4) are resolved by `operator_labels(operators, labels)`
in `plotting/correlators.py`, which also accepts a sequence (the CLI passes the
`--operators` numbers). Until `compact_str` exists, `"short"` is the full Sigmond
short string.

This one is a function, not a plotter method. Its input is the output of
`index.coverage()` (§3.2), reads no samples and has no `CorrelatorMatrix`
(the matrix is usually what cannot be built). It draws, via `labeled_heatmap`,
the fraction of times each cell is available, with missing cells NaN (empty on
both backends) plus an `×` from `axes.text`. The hover shows the times each cell
has. `CorrelatorIndex` stays free of plotting imports.

### 4.6 Reference levels and fit overlays

**Reference levels reuse `HMarker`.** It moves from `spectrum.py` to
`markers.py` (re-exported from both modules and from `plotting`) and gains one
optional field:

```python
@dataclass
class HMarker:
    y: float
    label: str | None = None
    group: Hashable | None = None     # spectrum only; ignored by series plots
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.0
    alpha: float = 0.7
    err: float | tuple[float, float] | None = None  # new: band y ± err, or (lower, upper) offsets
    band_alpha: float = 0.2           # new: band transparency

    @classmethod
    def from_sampling(cls, sampling: SigmondSampling, **kw) -> HMarker: ...
```

`from_sampling` takes the central value and error from `central_and_error`, so
a marker follows the same rc metrics as the plotted data. That is why `err` may
be a `(lower, upper)` pair: under `plot.error_metric = "confidence_interval"`
the band is asymmetric.

A threshold, a `SingleEnsembleEnergyCollection` level or a lattice-irrep energy
is one `HMarker` on either a spectrum plot or an effective-energy plot.
`SpectrumPlotter` also draws the new band, so the extension is not
correlator-only.

**Fits overlay as `FitOverlay`**, a descriptor like `HMarker`:

```python
@dataclass
class FitOverlay:
    result: FitResult                 # carries x / x_name (open question 4, decided)
    energy: str | None = None         # parameter drawn as a flat band on E_eff plots
    label: str | None = None
    color: str | None = None          # None -> the colour of `series`
    series: int | str | None = None   # which plotted series the fit belongs to (label or position)
    band_alpha: float = 0.25
```

`FitResult` gains `x` and `x_name`, copied from `SamplingFit`, and
`theory_series()`, the theory as a 1-D `SigmondData` over `x_name`. Since
`FitResult` has no single HDF5 form, `x` rides along as the `X`/`XName`
attributes of the theory group (`write_theory`, `from_hdf5(theory_file=)`),
which is where an overlay needs it. A result without `x` (from `from_model`)
raises when used as an overlay. A plot is an effective-energy plot when its
data's cells have `EffectiveEnergyInfo` elements; the model's effective energy then uses
that element's `kind`, `step`, `time_extent` and `subtract_const`.

- On a **correlator** plot: `result.theory` (already the model at the fitted
  data points) as a `band` plus a central `line` over `x`.
- On an **effective-energy** plot: the `energy` parameter as a flat `band` over
  `[min(x), max(x)]`, **and** `effective_energy(theory)`, the model's own
  effective energy, as a line. The second one shows the excited-state curvature a
  two-exponential fit claims. This is the numbers-first rule at work: no new
  plotting code, just §3.1 applied to the theory.

Both use only portable marks, so this does not wait for the `fit_plotter` port.
When that port happens, `fit_plotter._plot_model_band` and this overlay should
share one "model band" helper in `_util`.

### 4.7 Two small additions to `slat.plotting`

The mark vocabulary is closed on purpose, so these are options on existing
marks, not new marks. Each also has a use outside the correlator plots.

| addition | matplotlib | plotly | also used by |
| --- | --- | --- | --- |
| `fill: bool = True` on `points`, `errorbar`, `line` | `markerfacecolor="none"` / `facecolors="none"` | symbol `"<name>-open"` | optionally, spectrum excluded levels (today only greyed) |
| `hover=` on `heatmap` (2-D array of strings) | ignored (matplotlib has no hover) | `text=` + `hoverinfo="text"` | `plot_correlation_matrix` (observable pair and value) |

Also document the existing behaviour as a contract, with a recording and a smoke
test: **NaN heatmap cells render empty on both backends**.

`RecordingBackend` records the new kwargs for free.

Found while building phase 2: plotly takes a log axis's `range` in log10 units,
and the plotly `set()` passed limits through raw. It now converts them, using
the scale set in the same call or the axis's current one.

## 5. CLI — `ss query corr --plot`

There is no `ss plot` command. Plotting hangs off queries (`ss query --plot
METHOD`, `--plot-spectrum`). The correlator plots follow suit and reuse the
query's selectors (`--sink/--source`, `--matrix`, `--ops`, `--time`, `--sampling`,
`--hermitian`, `--subtract-vev`), so a plot is always of exactly what the query
shows:

```
ss query corr FILE --sink 8 --source 8 --plot correlator
ss query corr FILE --sink 8 --source 8 --plot effective-energy --eff-kind symmetric --eff-step 2
ss query corr FILE --matrix 0 --ops 0-8 --plot grid --norm-time 3
ss query corr FILE --matrix 0 --plot heatmap --time 5 --plot-value significance
ss query corr FILE --matrix 0 --plot coverage
ss query corr FILE --matrix 0 --ops 0-8 --plot heatmap --time 8
```

The shared output flags are the same as `ss query`: `--plot-output`, `--no-gui`,
`--plot-backend`, `--latex`, `--plot-panels`. `cli/plot.py` gains
`CORRELATOR_PLOT_METHODS` and `render_correlator_plot(...)` beside
`render_generic_plot`/`render_spectrum_plot`, going through the same
`_backend_context`, `_latex_context` and `_finish_plot`. The per-method flags are
`--eff-kind`, `--eff-step`, `--norm-time` and `--plot-value`.
Bins require `--sampling`, the same rule as queries. `--plot coverage` is the
one method that also works when `--matrix` fails, which is the point of it.

As built:

| mode | `--plot` methods |
| --- | --- |
| `--sink/--source` | `correlator`, `effective-energy`, `summary` (panels `correlator,effective-energy`) |
| `--matrix` | `diagonal`, `effective-energy` (of the diagonal), `grid`, `heatmap`, `coverage` |

- Matrix plots take every complete time (the matrix query shows one time);
  `--time` picks the heatmap's time, and is refused for the other matrix plots.
  `--plot grid` shows `C_ij`, or `C~_ij` when `--norm-time` is given.
- Operators are labelled with their `--operators` numbers.
- `--time-extent T` was added: an effective energy needs `T`, and a file's
  ensemble info may not carry it. The library error names the flag.
- Every plot of bins except `coverage` needs `--sampling`, since errors and
  effective energies come from samplings. (Queries need it only with
  `--subtract-vev`.)
- `--plot-panels` accepts hyphens (`effective-energy`) for every plot command.
- The missing-cell `--ops` hint now also covers matrix plots, whose request is
  over all times.

## 6. Testing

- §3 against the `sigmond` bindings where they exist, and against closed forms
  otherwise (a pure `A e^{-mt}` series must give exactly `m`, the symmetric
  kind on a cosh, and so on).
- Effective energy, each of the four kinds:
  - exact `m` on its own model with `B0 != 0` for the `_const` kinds, for
    `step` 1 and 2, on both sides of `T/2` (the mirrored branches);
  - failures at the validity limits (`t = T - s`, `t = T - 2s`, `r >= 1`), and
    the drop / nan / raise policies, where one bad resample drops the time;
  - the plot offset: `step/2` for the plain kinds, `0` for the `_const` kinds;
  - per-sample agreement with `EffectiveEnergyCalculator`, and the same set of
    kept times as `getEffectiveEnergy`, when the bindings are installed;
  - NumPy and numba kernels agree, when numba is installed.
- Alignment regression: an effective energy that is not identically 0 when
  `step=1`.
- `coverage`/`available_bases` on the `herm_file` fixture from `test_cli_corr.py`,
  plus a synthetic 11-operator copy of the E250 hole pattern.
- Plots through `slp.record()`, asserting on structure, in the style of
  `tests/test_plotting_backends.py`:
  - `SeriesPlotter`: one `errorbar` per series, dodged x, colours in cycle order,
    `fill=False` exactly at negative points on a log axis, and nothing past
    `max_rel_error`.
  - Grid: N² cells, `fill=False` exactly where `stored` is false, two distinct
    `ylim`s (diagonal vs off-diagonal).
  - Consistency: an operator has the same colour in `plot_diagonal` and
    `plot_grid` of the same plotter, and the same tick label in `plot_heatmap`.
  - `normalized`: equals `C / sqrt(f_i f_j)` with the full-sample factors on every
    sample, uses the modulus for a non-Hermitian complex diagonal, and raises on
    a missing or negative `C_ii(t0)`.
  - `HMarker(err=...)` emits a `band` then `hlines` on both the spectrum and the
    series plotters. `FitOverlay` emits `band` + `line`.
  - `plot_correlation_matrix` records the same marks before and after the move
    onto `labeled_heatmap`.
- `slat`: `fill` and heatmap `hover` translation tests next to the existing
  `TestPlotlyTranslation`, and NaN-cell heatmaps on both backends.
- One smoke test per real backend per public method.

## 7. Phases

1. **Shared pieces. Done.** The `slat` `fill`/heatmap-`hover` additions, `_util` helpers
   (`central_and_error`, `labeled_heatmap`, `summary_grid`), `HMarker` → `markers.py`
   with `err`. `plot_correlation_matrix` moves onto `labeled_heatmap` and gains
   hover. `SpectrumPlotter` reads its default colours from `slp.get_style()`
   instead of building `IndexedCycle(COLORS)` (open question 5, decided). No new
   user-facing plots yet, but every later phase stands on these.
2. **Done.** `correlators/_kernels.py` (the four kinds, NumPy, with the convergence
   improvements of §3.4) and `effective_energy`, then `SeriesPlotter`,
   `CorrelatorPlotter` (`plot`, `plot_effective_energy`, `plot_summary`),
   `FitResult.x`/`x_name` with their HDF5 round trip, and `FitOverlay`. Useful on
   its own and exercises the alignment trap early.
3. **Done.** `coverage`/`available_bases`, with the CLI hint rewired onto them; `plot_coverage`.
4. **Done.** `normalized`, `significance`, `CorrelatorMatrixPlotter` (`plot_diagonal`,
   `plot_grid`, `plot_heatmap`).
5. **Done.** `ss query corr --plot` covering phases 2–4.

Eigenvalue diagnostics are part of the pivot plan, not a phase here (§2).

## 8. Open questions

1. **Effective-energy time coord. Decided:** the integer source time `t`, with
   the offset from `effective_time_offset(kind, step)` recorded through the
   result's `observable_info` and applied by `SeriesPlotter(x_offset=)` (§3.1).
2. **Normalization estimator. Resolved by Sigmond's source:** the full-sample
   estimate (§3.3).
3. **Eigenvalue sorting. Moved** to the pivot plan with the eigenvalue
   diagnostics.
4. **`FitResult` carries `x`/`x_name`. Decided:** yes, including HDF5 (§4.6).
5. **`SpectrumPlotter` colour source. Decided:** fixed in phase 1, so
   `style_context(colors=...)` reaches it like the new plotters.
6. **Metric for eigenvalue diagnostics. Moved** to the pivot plan.
