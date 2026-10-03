"""Effective energies: the kernels and ``effective_energy``.

Checked against closed forms (each kind is exact on its own model) and, when
the ``sigmond`` bindings are installed, sample by sample against Sigmond's own
``EffectiveEnergyCalculator``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sigmondsamplings.correlators import _kernels
from sigmondsamplings.correlators.effective import (
    KINDS,
    EffectiveEnergyObsInfo,
    EffectiveEnergyWarning,
    effective_energy,
    effective_time_offset,
)
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, SamplingInfo

T = 32
BOOT = SamplingInfo("bootstrap", 20, seed=1)
ENS = EnsembleInfo("ens", 100, temporal_extent=T)
M = 0.37


def _data(values, times, *, info=BOOT, ens=ENS, dims=("time",), coords=None) -> SigmondData:
    return SigmondData.from_data(
        np.asarray(values, dtype=float),
        "C",
        info,
        dim_names=dims,
        coords=coords or {"time": list(times)},
        ensemble_info=ens,
    )


def _model(kind: str, t, amplitude, m=M, const=0.25):
    t = np.asarray(t, dtype=float)
    if kind.startswith("symmetric"):
        c = np.exp(-m * t) + np.exp(-m * (T - t))
    else:
        c = np.exp(-m * t)
    c = amplitude * c
    return c + const if kind.endswith("_const") else c


def _samples(kind: str, times, n=BOOT.num_resamplings + 1, **kw):
    """``(time, n)``: each sample a different amplitude of the same model, so every sample gives M."""
    amplitudes = np.linspace(0.8, 1.2, n)
    return np.stack([_model(kind, times, a, **kw) for a in amplitudes], axis=-1)


def _quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EffectiveEnergyWarning)
        return effective_energy(**kwargs)


# ---------------------------------------------------------------------------
# Exact on each kind's own model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("step", [1, 2])
def test_each_kind_is_exact_on_its_model(kind, step):
    times = np.arange(T)
    corr = _data(_samples(kind, times), times)
    eff = _quiet(corr=corr, kind=kind, step=step)
    kept = np.asarray(eff.coords["time"])
    np.testing.assert_allclose(eff.data, M, rtol=1e-8)
    # both sides of T/2, so the mirrored branches of the symmetric kinds are exercised
    assert kept.min() < T // 2 - step and kept.max() > T // 2 + step


def test_symmetric_kinds_are_not_exact_on_the_wrong_model():
    times = np.arange(T)
    corr = _data(_samples("symmetric", times), times)
    eff = _quiet(corr=corr, kind="forward")
    late = eff.sel(time=[t for t in eff.coords["time"] if t > T // 2 - 4])
    assert np.max(np.abs(late.data - M)) > 0.01


def test_subtract_const_is_removed_before_the_kernel():
    times = np.arange(20)
    corr = _data(_samples("forward_const", times, const=0.4), times)
    eff = effective_energy(corr, subtract_const=0.4)
    np.testing.assert_allclose(eff.data, M, rtol=1e-10)


# ---------------------------------------------------------------------------
# Keying, offsets and the alignment trap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "step", "offset"),
    [
        ("forward", 1, 0.5),
        ("symmetric", 2, 1.0),
        ("forward_const", 2, 0.0),
        ("symmetric_const", 1, 0.0),
    ],
)
def test_plot_offset(kind, step, offset):
    assert effective_time_offset(kind, step) == offset
    times = np.arange(T)
    eff = _quiet(corr=_data(_samples(kind, times), times), kind=kind, step=step)
    info = eff.observable_info
    assert isinstance(info, EffectiveEnergyObsInfo)
    assert (info.kind, info.step, info.time_extent, info.time_offset) == (kind, step, T, offset)


def test_times_are_the_source_times():
    times = np.arange(10)
    eff = effective_energy(_data(_samples("forward", times), times), step=2)
    assert list(eff.coords["time"]) == list(range(8))  # t + 2 must exist
    const = effective_energy(_data(_samples("forward_const", times), times), kind="forward_const")
    assert list(const.coords["time"]) == list(range(1, 9))  # needs t - 1 and t + 1


def test_neighbours_are_found_by_time_not_by_position():
    """Regression: C(t+s) must be the value at time t+s, not the next stored row."""
    times = [2, 3, 4, 6, 7]  # 5 is missing
    values = _samples("forward", times)
    eff = effective_energy(_data(values, times))
    assert list(eff.coords["time"]) == [2, 3, 6]
    np.testing.assert_allclose(eff.data, M, rtol=1e-12)
    assert not np.allclose(eff.data, 0.0)


def test_unsorted_time_coords():
    times = [5, 2, 4, 3]
    eff = effective_energy(_data(_samples("forward", times), times))
    assert list(eff.coords["time"]) == [2, 3, 4]
    np.testing.assert_allclose(eff.data, M, rtol=1e-12)


def test_other_dims_broadcast_and_keep_their_place():
    times = np.arange(12)
    a, b = _samples("forward", times, m=0.3), _samples("forward", times, m=0.6)
    stacked = np.stack([a, b], axis=0)  # (operator, time, N): time is not first
    corr = _data(
        stacked,
        times,
        dims=("operator", "time"),
        coords={"operator": ["a", "b"], "time": list(times)},
    )
    eff = effective_energy(corr)
    assert eff.dims == ("operator", "time") and list(eff.coords["operator"]) == ["a", "b"]
    np.testing.assert_allclose(eff.sel(operator="a").data, 0.3, rtol=1e-12)
    np.testing.assert_allclose(eff.sel(operator="b").data, 0.6, rtol=1e-12)


def test_central_value_is_the_full_sample_effective_energy():
    rng = np.random.default_rng(0)
    times = np.arange(12)
    values = _samples("forward", times) * (1 + 0.05 * rng.normal(size=(12, 21)))
    eff = effective_energy(_data(values, times), on_fail="nan")
    expected = -np.log(values[1:, 0] / values[:-1, 0])
    np.testing.assert_allclose(np.asarray(eff.full_sample_value), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Failures and the drop / nan / raise policies
# ---------------------------------------------------------------------------


def _with_one_bad_resample():
    """A forward correlator whose resample 5 fails at t=3 and t=4 (C(4) < 0)."""
    times = np.arange(10)
    values = _samples("forward", times)
    values[4, 5] = -values[4, 5]  # C(4) < 0: the ratios at t=3 and t=4 fail for sample 5
    return _data(values, times)


def test_drop_removes_the_time_and_warns():
    with pytest.warns(EffectiveEnergyWarning, match=r"dropped times \[3, 4\]"):
        eff = effective_energy(_with_one_bad_resample())
    assert 3 not in eff.coords["time"] and 4 not in eff.coords["time"]
    assert len(eff.coords["time"]) == 7


def test_nan_keeps_the_time_with_nan_in_the_failed_sample():
    eff = effective_energy(_with_one_bad_resample(), on_fail="nan")
    assert list(eff.coords["time"]) == list(range(9))
    at3 = eff.sel(time=3).data
    assert np.isnan(at3[5]) and np.isfinite(np.delete(at3, 5)).all()


def test_raise_names_the_times():
    with pytest.raises(ValueError, match=r"times \[3, 4\]"):
        effective_energy(_with_one_bad_resample(), on_fail="raise")


def test_symmetric_fails_where_the_ratio_is_not_below_one():
    times = np.arange(10)
    rising = _samples("forward", times, m=-0.2)  # C grows on the near side: r > 1
    with pytest.raises(ValueError, match="no time has a symmetric"):
        _quiet(corr=_data(rising, times), kind="symmetric")


def test_symmetric_validity_limit_t_below_T_minus_step():
    times = np.arange(T + 1)  # periodic copy at T, so C(t+s) exists at t = T - s
    corr = _data(_samples("symmetric", times), times)
    eff = _quiet(corr=corr, kind="symmetric", step=1, on_fail="nan")
    assert np.isnan(eff.sel(time=T - 1).data).all()  # t = T - s fails
    np.testing.assert_allclose(eff.sel(time=T - 2).data, M, rtol=1e-8)


def test_symmetric_const_validity_limit_t_below_T_minus_two_steps():
    times = np.arange(T)
    corr = _data(_samples("symmetric_const", times), times)
    eff = _quiet(corr=corr, kind="symmetric_const", step=2, on_fail="nan")
    assert np.isnan(eff.sel(time=T - 4).data).all()  # t = T - 2s fails
    np.testing.assert_allclose(eff.sel(time=T - 5).data, M, rtol=1e-8)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestInputs:
    times = np.arange(12)

    def test_bins_are_rejected(self):
        corr = _data(_samples("forward", self.times, n=30), self.times, info=None)
        with pytest.raises(TypeError, match="resample"):
            effective_energy(corr)

    def test_complex_data_asks_for_a_component(self):
        corr = _data(_samples("forward", self.times), self.times).to_complex()
        with pytest.raises(ValueError, match="real"):
            effective_energy(corr)

    def test_time_extent_is_required_without_the_ensemble_one(self):
        corr = _data(_samples("forward", self.times), self.times, ens=EnsembleInfo("e", 10))
        with pytest.raises(ValueError, match="time_extent"):
            effective_energy(corr, kind="symmetric")
        assert effective_energy(corr, time_extent=T).observable_info.time_extent == T

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"kind": "symmetric", "time_extent": 6}, "time_extent"),
            ({"step": 0}, "step"),
            ({"kind": "symmetric", "step": T // 4 + 1}, "step"),
            ({"kind": "cosh"}, "kind"),
            ({"on_fail": "skip"}, "on_fail"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            effective_energy(_data(_samples("forward", self.times), self.times), **kwargs)

    def test_needs_integer_time_coords(self):
        corr = _data(_samples("forward", self.times), self.times + 0.5)
        with pytest.raises(ValueError, match="integers"):
            effective_energy(corr)

    def test_needs_a_time_dim(self):
        corr = _data(
            _samples("forward", self.times), self.times, dims=("t",), coords={"t": list(self.times)}
        )
        with pytest.raises(ValueError, match="'time' dim"):
            effective_energy(corr)


# ---------------------------------------------------------------------------
# Sigmond oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("step", [1, 2])
def test_kernel_matches_sigmond_per_sample(kind, step):
    sigmond = pytest.importorskip("sigmond")
    try:
        calculator = sigmond.EffectiveEnergyCalculator
    except ImportError:
        pytest.skip("Sigmond EffectiveEnergyCalculator bindings unavailable")
    code = KINDS[kind]
    rng = np.random.default_rng(code * 10 + step)
    Tk, n = 24, 40
    times = np.arange(Tk)
    m = rng.uniform(0.15, 0.7, size=(2, 1, 1))
    clean = np.cosh(m * (times[None, :, None] - Tk / 2)) + (0.3 if code >= 2 else 0.0)
    buf = clean * (1 + 0.03 * rng.normal(size=(2, Tk, n)))
    energy, failed = _kernels.effective_energy(buf, times, kind=code, step=step, time_extent=Tk)

    calc = calculator(step, Tk, code)
    compared = 0
    for a in range(2):
        for i, t in enumerate(times):
            if i + step >= Tk or (kind.endswith("_const") and i - step < 0):
                assert failed[a, i].all()  # a missing input
                continue
            for j in range(n):
                back = buf[a, i - step, j] if i >= step else 0.0
                ref = calc.calculate(int(t), buf[a, i, j], buf[a, i + step, j], back)
                ref_failed = ref == -1.0 or not np.isfinite(ref)
                assert failed[a, i, j] == ref_failed, (a, t, j)
                if not ref_failed:
                    # rtsafe stops at |db| < 1e-10, so values agree to about that
                    assert energy[a, i, j] == pytest.approx(ref, rel=1e-8, abs=1e-10)
                    compared += 1
    assert compared > 0


def test_drop_rule_keeps_exactly_the_times_where_every_sample_succeeds():
    """Sigmond's getEffectiveEnergy drops a time if any sample fails there."""
    rng = np.random.default_rng(5)
    times = np.arange(T)
    values = _samples("symmetric", times) * (1 + 0.05 * rng.normal(size=(T, 21)))
    corr = _data(values, times)
    energy, failed = _kernels.effective_energy(
        values, times, kind=KINDS["symmetric"], step=1, time_extent=T
    )
    expected = [int(t) for t, bad in zip(times, failed.any(axis=-1)) if not bad]
    eff = _quiet(corr=corr, kind="symmetric")
    assert list(eff.coords["time"]) == expected
