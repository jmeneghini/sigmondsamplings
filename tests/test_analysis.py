"""
Tests for the shared sample-axis analysis (sigmondsamplings.analysis): error
formulas against sigmond's jack_analyze/boot_analyze, and the
NonlinearBinsWarning policy for raw bins.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sigmondsamplings.analysis import NonlinearBinsWarning, analyzer_for
from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.fitting.fit import _error_factor
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats

RNG = np.random.default_rng(7)
ENS = EnsembleInfo("ens", 100)


def _obs(name: str) -> ObservableInfo:
    return ObservableInfo(name=name, index=0, ensemble_info=ENS)


def _sampling(method: str, n: int, name: str = "x") -> SigmondSampling:
    return SigmondSampling(
        1.0 + 0.1 * RNG.standard_normal(n + 1), _obs(name), SamplingInfo(method, n)
    )


def _bins(n: int = 40, name: str = "b") -> SigmondBins:
    return SigmondBins(2.0 + 0.2 * RNG.standard_normal(n), _obs(name))


# ──────────────────────────────────────────────────────────────────────────────
# Formulas, written out as in sigmond's MCObsHandler
# ──────────────────────────────────────────────────────────────────────────────


def _sigmond_jack_error(samp: np.ndarray) -> float:
    vals = samp[1:]
    n = len(vals)
    avg = vals.sum() / n
    var = ((vals - avg) ** 2).sum() * (1.0 - 1.0 / n)
    return float(np.sqrt(var))


def _sigmond_boot_error(samp: np.ndarray) -> float:
    vals = samp[1:]
    nb = len(vals)
    avg = vals.sum() / nb
    return float(np.sqrt(((vals - avg) ** 2).sum() / (nb - 1)))


class TestFormulas:
    def test_jackknife_error_matches_sigmond(self):
        s = _sampling("jackknife", 30)
        assert s.error == pytest.approx(_sigmond_jack_error(s.data), rel=1e-12)

    def test_bootstrap_error_matches_sigmond(self):
        s = _sampling("bootstrap", 50)
        assert s.error == pytest.approx(_sigmond_boot_error(s.data), rel=1e-12)

    def test_bins_error_is_standard_error(self):
        b = _bins(40)
        assert b.error == pytest.approx(np.std(b.data, ddof=1) / np.sqrt(40), rel=1e-12)

    def test_full_and_mean(self):
        s = _sampling("jackknife", 10)
        assert s.full_sample_value == s.data[0]
        assert s.mean == pytest.approx(s.data[1:].mean())
        b = _bins(10)
        assert b.full_sample_value == pytest.approx(b.data.mean())
        assert b.mean == b.full_sample_value

    def test_analyzer_reduces_last_axis(self):
        data = RNG.standard_normal((3, 4, 21))
        analyzer = analyzer_for(SamplingInfo("jackknife", 20))
        err = analyzer.error(data)
        assert err.shape == (3, 4)
        assert err[1, 2] == pytest.approx(_sigmond_jack_error(data[1, 2]))

    def test_bootstrap_only_quantities(self):
        jk = _sampling("jackknife", 10)
        with pytest.raises(ValueError, match="bootstrap"):
            jk.confidence_interval()
        with pytest.raises(ValueError, match="bootstrap"):
            jk.bootstrap_bias
        bs = _sampling("bootstrap", 100)
        lo, hi = bs.confidence_interval(0.68)
        assert lo < hi
        assert bs.bootstrap_bias == pytest.approx(bs.mean - bs.full_sample_value)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown sampling method"):
            analyzer_for(SamplingInfo("spawn", 10))


class TestConsumersShareFormula:
    @pytest.mark.parametrize("method", ["jackknife", "bootstrap"])
    def test_cov_diagonal_matches_error_squared(self, method):
        samps = [_sampling(method, 25, name=f"y{i}") for i in range(3)]
        stats = SamplingStats(samps)
        for i, s in enumerate(samps):
            assert stats.cov_matrix[i, i] == pytest.approx(s.error**2, rel=1e-12)
            assert stats.cov(i, i) == pytest.approx(s.error**2, rel=1e-12)

    @pytest.mark.parametrize("method", ["jackknife", "bootstrap"])
    def test_fit_error_factor_matches_error(self, method):
        s = _sampling(method, 25)
        factor = _error_factor(s.sampling_info, 25)
        assert np.std(s.data[1:], ddof=1) * factor == pytest.approx(s.error, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# NonlinearBinsWarning
# ──────────────────────────────────────────────────────────────────────────────


def _warned(func) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func()
    return any(issubclass(w.category, NonlinearBinsWarning) for w in caught)


class TestNonlinearBinsWarning:
    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: -a,
            lambda a, b: 2.0 * a,
            lambda a, b: a * 2.0,
            lambda a, b: a / 2.0,
            lambda a, b: np.conjugate(a),
        ],
    )
    def test_linear_is_silent(self, op):
        a, b = _bins(), _bins(name="c")
        assert not _warned(lambda: op(a, b))

    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a * b,
            lambda a, b: a / b,
            lambda a, b: 1.0 / a,
            lambda a, b: a**2,
            lambda a, b: np.log(a),
            lambda a, b: abs(a),
        ],
    )
    def test_nonlinear_warns(self, op):
        a, b = _bins(), _bins(name="c")
        assert _warned(lambda: op(a, b))

    def test_warning_points_at_caller(self):
        a = _bins()
        with pytest.warns(NonlinearBinsWarning) as record:
            np.log(a)
        assert record[0].filename == __file__

    def test_filter_can_escalate_to_error(self):
        a = _bins()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NonlinearBinsWarning)
            with pytest.raises(NonlinearBinsWarning):
                a * a

    def test_per_bin_is_silent(self):
        a = _bins()
        assert not _warned(lambda: a.per_bin(np.log))
        np.testing.assert_allclose(a.per_bin(np.log).data, np.log(a.data))

    def test_samplings_never_warn(self):
        s = _sampling("bootstrap", 20)
        assert not _warned(lambda: np.log(s * s))
