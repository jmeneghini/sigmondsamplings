# The ObsInfo redesign, in slides

Concepts first, for readers familiar with the old ObservableInfo API. All new API examples are proposed.

## 1. The proposed ObsInfo redesign

A guide for users of ObservableInfo
and its subclasses

**What each observable needs, and where that data lives**

## 2. The ObservableInfo you know

One object carries the file name and index, the ensemble,
and the physical meaning of the energy.

```python
info = EnergyObsInfo(
    name="fit_E2", index=0, ensemble_info=ens,
    irrep="A1g", psq=1, energy_type="elab",
    level_index=2,
)
```

**The redesign gives those fields separate homes.**

## 3. Where the data goes

| Data | Proposed home |
| --- | --- |
| Samples and resampling plan | SigmondData.data and sampling_info |
| Ensemble and lattice context | SigmondData.ensemble_info |
| Physical meaning | data.quantity |
| This particular result | data.estimate |
| Stored name, index and component | data.address |
| Inputs and calculation settings | data.provenance |
| Array axes and cell labels | data.dims, coords and dim_roles |

**A quantity contains no sample values or ensemble identity.**

## 4. What happens to the subclasses

| Familiar class | Meaning in the proposed API |
| --- | --- |
| EnergyObsInfo | Energy key |
| SHEnergyObsInfo | Energy with a single-hadron descriptor |
| CorrelatorObsInfo | Correlator, possibly evaluated at time t |
| EffectiveEnergyObsInfo | EffectiveEnergy of a source correlator |
| VEVObsInfo | VEV of an operator |
| OverlapObsInfo | Overlap with operator and bound pivot mode |
| ArrayObsInfo | Array layout plus quantities for its cells |

**Old classes remain as adapters while the new API takes over.**

## 5. An energy and its metadata

| Energy-related data | New home |
| --- | --- |
| irrep and psq | Energy.channel |
| level_index | Energy.state: an assigned LevelKey |
| energy_type="elab" or "ecm" | Energy.frame |
| name and index | Address |
| ensemble_info | Data container |
| Fit window and model settings | Provenance |
| One fit output | Estimate reference |

**Two fit windows can produce the same Energy key.**

## 6. Single hadrons, shifts and ratios

Single hadron energy
The key records the measured hadron and its momentum.

Energy shift
The key names the noninteracting reference being subtracted.

E / m_pi
The key records normalization by the pion mass.

**Provenance identifies the reference samples used in the calculation.**

## 7. A correlator matrix

| Needed data | Where it lives |
| --- | --- |
| Sink and source operators | Correlator quantity |
| Connected or unsubtracted meaning | Correlator quantity |
| Source times | Time coordinate |
| Matrix positions | sink and source coordinates |
| Samples at each (t, i, j) | Data buffer |
| Rotation coefficients and source basis | Pivot artifact and provenance |

**Coordinates identify each cell. A template builds its quantity.**

## 8. An effective energy

| Needed data | Where it lives |
| --- | --- |
| Source correlator | EffectiveEnergy.of |
| Kind and time step | EffectiveEnergy key |
| Constant subtraction and temporal model | Quantity context and provenance |
| Input samples | Referenced input estimates |
| Output samples | New data buffer |
| Source time t | Time coordinate |
| Plotting offset, such as 0.5 | TimeAxis, derived from the convention |

**Its energy target comes from the source correlator.**

## 9. A vacuum expectation value

| Needed data | Where it lives |
| --- | --- |
| Operator O_i | VEV quantity |
| VEV samples | Data buffer |
| Ensemble and resampling plan | Data container |
| Operator grid for several VEVs | Operator coordinate |
| Stored real or imaginary component | Address and Component key |

**The operator identifies the quantity. Samples belong to the data.**

## 10. An overlap

| Needed data | Where it lives |
| --- | --- |
| Basis operator O_i | Overlap quantity |
| Rotated mode n and its pivot | Bound level operator in the key |
| Normalization convention | Overlap quantity |
| Rotation and fitted amplitudes | Provenance input references |
| Overlap samples | Data buffer |
| Operator and mode grid | Coordinates |

**The mode binding follows the operator wherever it appears.**

## 11. A series and a value at one time

```python
eff.quantity
# EffectiveEnergy(...) for the whole series

eff.sel(time=8).quantity
# TimePoint(EffectiveEnergy(...), time=8)

eff.ts.at(8).quantity
# The same key
```

**Selecting time fixes the argument of the series.**

## 12. The familiar analysis workflow

```python
D = pivot.rotate(C).diagonal(component="real")
c2 = D.isel(operator=2)

eff = c2.ts.effective_energy()
fit = c2.ts.fit(single_exp, window=(6, 18),
                **fit_options)
E2 = fit.param("E")
eff.ts.plot(overlay=fit)
```

**The standard model supplies the meaning of its energy parameter.**

## 13. Same quantity, different estimates

```python
a = fit_6_18.param("E")
b = fit_8_20.param("E")

a.quantity == b.quantity       # True
a.estimate == b.estimate       # False
a.provenance.settings          # Window 6 to 18
b.provenance.settings          # Window 8 to 20
```

**Collections preserve both results by estimate identity.**

## 14. The labels on an array

D.quantities gives keys over operator.
A fit-trial array gives repeated keys over fit.
The .estimates array distinguishes its individual results.

| Axis role | Meaning | Example |
| --- | --- | --- |
| Semantic | Changes what is measured | operator or assigned level |
| Parameter | Changes where a function is evaluated | time |
| Replicate | Changes which estimate is shown | fit window trial |

**Producers declare roles. Axis names alone do not decide them.**

## 15. Meaning follows the operation

| Operation | Quantity after the operation |
| --- | --- |
| Select or transpose | Same meaning, with selected coordinates |
| Take the real part | Component(of=quantity, part="re") |
| Compute C / C | Derived ratio quantity |
| Normalize E by m_pi | Energy with a normalization reference |
| Rotate the basis | Correlators with new bound operators |

**Arithmetic produces new estimates and records its inputs.**

## 16. What changes for an ObsInfo user

Inspect physical meaning with data.quantity.
Inspect the stored name through data.address.
Use data.estimate to distinguish fit results.
Read data.provenance to see how a result was produced.
Use .ts for time-series operations.
Keep existing loaders, filters and sample statistics.

**The first implementation proves one complete pivot workflow.**

Detailed contracts and agent implementation gates: [quantity_redesign_plan.md](quantity_redesign_plan.md).
