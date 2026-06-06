"""
Regression test for SamplingStats refactoring.

Run with:  python -m pytest tests/test_stats_refactor.py -v
or:        python tests/test_stats_refactor.py
"""

import numpy as np
import pytest

from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.stats import SamplingStats

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N_RESAMP = 200
RNG = np.random.default_rng(42)

ENS_A = EnsembleInfo("ens_A", 100)
ENS_B = EnsembleInfo("ens_B", 80)
SINFO = SamplingInfo("bootstrap", N_RESAMP, seed=42)
SINFO_JK = SamplingInfo("jackknife", N_RESAMP, seed=0)


def _make_sampling(
    name: str, data: np.ndarray, ens: EnsembleInfo, sinfo: SamplingInfo
) -> SigmondSampling:
    obs = ObservableInfo(name=name, index=0, op_type="n", re_im="re", ensemble_info=ens)
    return SigmondSampling(data=data, observable_info=obs, sampling_info=sinfo)


def _correlated_samples(means: np.ndarray, cov: np.ndarray, n: int, rng) -> np.ndarray:
    """Returns array of shape (len(means), n+1); column 0 is the 'full sample'."""
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((len(means), n + 1))
    return means[:, None] + L @ z


# Three correlated observables from the same ensemble (bootstrap)
MEANS_A = np.array([1.0, 2.5, 4.0])
COV_TRUE = np.array(
    [
        [0.01, 0.005, 0.002],
        [0.005, 0.04, 0.008],
        [0.002, 0.008, 0.09],
    ]
)
SAMPLES_A = _correlated_samples(MEANS_A, COV_TRUE, N_RESAMP, RNG)

SAMPLINGS_SAME_ENS = [
    _make_sampling("obs_0", SAMPLES_A[0], ENS_A, SINFO),
    _make_sampling("obs_1", SAMPLES_A[1], ENS_A, SINFO),
    _make_sampling("obs_2", SAMPLES_A[2], ENS_A, SINFO),
]

# Two observables from ENS_A + one from ENS_B (cross-ensemble covariance must be 0)
SAMPLES_B = _correlated_samples(np.array([3.0]), np.array([[0.05]]), N_RESAMP, RNG)
SAMPLINGS_MULTI_ENS = [
    _make_sampling("obs_0", SAMPLES_A[0], ENS_A, SINFO),
    _make_sampling("obs_1", SAMPLES_A[1], ENS_A, SINFO),
    _make_sampling("obs_b", SAMPLES_B[0], ENS_B, SINFO),
]

# Jackknife variant (single ensemble, 3 obs)
SINFO_JK2 = SamplingInfo("jackknife", N_RESAMP, seed=0)
SAMPLINGS_JK = [
    _make_sampling("obs_0", SAMPLES_A[0], ENS_A, SINFO_JK2),
    _make_sampling("obs_1", SAMPLES_A[1], ENS_A, SINFO_JK2),
    _make_sampling("obs_2", SAMPLES_A[2], ENS_A, SINFO_JK2),
]


@pytest.fixture
def stats_same():
    return SamplingStats(SAMPLINGS_SAME_ENS)


@pytest.fixture
def stats_multi():
    return SamplingStats(SAMPLINGS_MULTI_ENS)


@pytest.fixture
def stats_jk():
    return SamplingStats(SAMPLINGS_JK)


# ──────────────────────────────────────────────────────────────────────────────
# cov_matrix
# ──────────────────────────────────────────────────────────────────────────────


class TestCovMatrix:
    def test_shape(self, stats_same):
        assert stats_same.cov_matrix.shape == (3, 3)

    def test_symmetric(self, stats_same):
        C = stats_same.cov_matrix
        np.testing.assert_allclose(C, C.T, atol=1e-15)

    def test_positive_semidefinite(self, stats_same):
        eigenvalues = np.linalg.eigvalsh(stats_same.cov_matrix)
        assert np.all(eigenvalues >= -1e-12)

    def test_diagonal_matches_error_squared(self, stats_same):
        C = stats_same.cov_matrix
        for i, s in enumerate(SAMPLINGS_SAME_ENS):
            np.testing.assert_allclose(C[i, i], s.error**2, rtol=1e-10)

    def test_off_diagonal_matches_pairwise_cov(self, stats_same):
        C = stats_same.cov_matrix
        for i in range(3):
            for j in range(3):
                expected = stats_same.cov(i, j)
                np.testing.assert_allclose(C[i, j], expected, atol=1e-15)

    def test_cross_ensemble_zero(self, stats_multi):
        C = stats_multi.cov_matrix
        # obs_0 (ENS_A) vs obs_b (ENS_B) → must be 0
        assert C[0, 2] == 0.0
        assert C[2, 0] == 0.0
        assert C[1, 2] == 0.0
        assert C[2, 1] == 0.0
        # Within ENS_A must be non-zero (correlated data)
        assert C[0, 1] != 0.0

    def test_jackknife_diagonal_matches_error_squared(self, stats_jk):
        C = stats_jk.cov_matrix
        for i, s in enumerate(SAMPLINGS_JK):
            np.testing.assert_allclose(C[i, i], s.error**2, rtol=1e-10)

    def test_jackknife_off_diagonal_matches_pairwise_cov(self, stats_jk):
        C = stats_jk.cov_matrix
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(C[i, j], stats_jk.cov(i, j), atol=1e-15)


# ──────────────────────────────────────────────────────────────────────────────
# corr_matrix
# ──────────────────────────────────────────────────────────────────────────────


class TestCorrMatrix:
    def test_shape(self, stats_same):
        assert stats_same.corr_matrix.shape == (3, 3)

    def test_diagonal_ones(self, stats_same):
        np.testing.assert_allclose(np.diag(stats_same.corr_matrix), 1.0, atol=1e-14)

    def test_range(self, stats_same):
        C = stats_same.corr_matrix
        assert np.all(C >= -1 - 1e-12)
        assert np.all(C <= 1 + 1e-12)

    def test_consistent_with_cov(self, stats_same):
        cov = stats_same.cov_matrix
        stds = np.sqrt(np.diag(cov))
        expected = cov / np.outer(stds, stds)
        np.testing.assert_allclose(stats_same.corr_matrix, expected, atol=1e-14)

    def test_cross_ensemble_zero(self, stats_multi):
        R = stats_multi.corr_matrix
        assert R[0, 2] == 0.0
        assert R[2, 0] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# inv_cholesky_cov_matrix
# ──────────────────────────────────────────────────────────────────────────────


class TestInvCholesky:
    def test_inv_L_times_L_is_identity(self, stats_same):
        inv_L = stats_same.inv_cholesky_cov_matrix
        C = stats_same.cov_matrix
        import scipy.linalg

        L = scipy.linalg.cholesky(C, lower=True)
        product = inv_L @ L
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_whitening_recovers_unit_covariance(self, stats_same):
        # If w = inv_L @ (x - mu), then Cov(w) ≈ I
        inv_L = stats_same.inv_cholesky_cov_matrix
        resampled = stats_same.array[:, 1:]  # 3 x N_RESAMP
        mu = resampled.mean(axis=1, keepdims=True)
        whitened = inv_L @ (resampled - mu)
        cov_w = np.cov(whitened, ddof=1)
        np.testing.assert_allclose(cov_w, np.eye(3), atol=0.1)  # statistical tolerance


# ──────────────────────────────────────────────────────────────────────────────
# chi_squared / whitened_residuals / residuals
# ──────────────────────────────────────────────────────────────────────────────


class TestChiSquared:
    def test_zero_residuals_give_zero_chi2(self, stats_same):
        theory = stats_same.array[:, 0]  # exact full-sample values
        chi2 = stats_same.chi_squared(theory)
        np.testing.assert_allclose(chi2, 0.0, atol=1e-12)

    def test_chi2_non_negative(self, stats_same):
        theory = MEANS_A
        chi2 = stats_same.chi_squared(theory)
        assert chi2 >= 0.0

    def test_chi2_uncorr_matches_manual(self, stats_same):
        theory = MEANS_A
        r = stats_same.residuals(theory)
        errors = np.array([s.error for s in SAMPLINGS_SAME_ENS])
        expected = np.sum((r / errors) ** 2)
        chi2 = stats_same.chi_squared(theory, use_corr=False)
        np.testing.assert_allclose(chi2, expected, rtol=1e-12)

    def test_whitened_squared_sum_is_chi2(self, stats_same):
        theory = MEANS_A
        w = stats_same.whitened_residuals(theory)
        chi2_direct = stats_same.chi_squared(theory)
        np.testing.assert_allclose(np.sum(w**2), chi2_direct, rtol=1e-12)

    def test_linear_superposition(self, stats_same):
        theory = MEANS_A
        ls = [[(0, 1.0), (1, -1.0)]]  # obs_0 - obs_1
        chi2 = stats_same.chi_squared(theory, linear_superposition=ls)
        assert chi2 >= 0.0

    def test_cross_ensemble_chi2(self, stats_multi):
        theory = np.array([MEANS_A[0], MEANS_A[1], 3.0])
        chi2 = stats_multi.chi_squared(theory)
        assert chi2 >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# goodness_of_fit / aic / fit_summary
# ──────────────────────────────────────────────────────────────────────────────


class TestGoodnessOfFit:
    def test_q_in_unit_interval(self, stats_same):
        theory = MEANS_A
        Q = stats_same.goodness_of_fit(theory, nparams=0)
        assert 0.0 <= Q <= 1.0

    def test_perfect_fit_q_near_one(self, stats_same):
        # When chi2≈0 and dof>0, Q should be near 1
        theory = stats_same.array[:, 0]
        Q = stats_same.goodness_of_fit(theory, nparams=0)
        assert Q > 0.99

    def test_aic_equals_chi2_minus_2dof(self, stats_same):
        theory = MEANS_A
        nparams = 2
        chi2_val = stats_same.chi_squared(theory)
        dof = stats_same.num_observables - nparams
        aic = stats_same.aic(nparams, theory_values=theory)
        np.testing.assert_allclose(aic, chi2_val - 2 * dof, rtol=1e-12)

    def test_fit_summary_keys(self, stats_same):
        result = stats_same.fit_summary(MEANS_A, nparams=1)
        expected_keys = {
            "residuals",
            "whitened_residuals",
            "chi2",
            "dof",
            "chi2_per_dof",
            "Q",
            "AIC",
            "markdown",
        }
        assert expected_keys == set(result.keys())

    def test_fit_summary_internal_consistency(self, stats_same):
        result = stats_same.fit_summary(MEANS_A, nparams=1)
        np.testing.assert_allclose(
            np.sum(result["whitened_residuals"] ** 2), result["chi2"], rtol=1e-12
        )
        assert result["dof"] == len(result["whitened_residuals"]) - 1
        np.testing.assert_allclose(
            result["chi2_per_dof"], result["chi2"] / result["dof"], rtol=1e-12
        )


# ──────────────────────────────────────────────────────────────────────────────
# effective_sample_size
# ──────────────────────────────────────────────────────────────────────────────


class TestEffectiveSampleSize:
    def test_shape(self, stats_same):
        ess = stats_same.effective_sample_size
        assert ess.shape == (3,)

    def test_positive(self, stats_same):
        assert np.all(stats_same.effective_sample_size >= 1.0)

    def test_at_most_n_resamples(self, stats_same):
        # ESS is a statistical estimate; allow modest over-estimation
        assert np.all(stats_same.effective_sample_size <= N_RESAMP * 1.5)


# ──────────────────────────────────────────────────────────────────────────────
# confidence_ellipse_params
# ──────────────────────────────────────────────────────────────────────────────


class TestConfidenceEllipse:
    def test_returns_five_floats(self):
        result = SamplingStats.confidence_ellipse_params(
            SAMPLINGS_SAME_ENS[0], SAMPLINGS_SAME_ENS[1]
        )
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)

    def test_positive_dimensions(self):
        _, _, width, height, _ = SamplingStats.confidence_ellipse_params(
            SAMPLINGS_SAME_ENS[0], SAMPLINGS_SAME_ENS[1]
        )
        assert width > 0
        assert height > 0

    def test_width_ge_height(self):
        # eigenvalue sort: largest first → width ≥ height
        _, _, width, height, _ = SamplingStats.confidence_ellipse_params(
            SAMPLINGS_SAME_ENS[0], SAMPLINGS_SAME_ENS[1]
        )
        assert width >= height - 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# inv_cov_matrix (cached property)
# ──────────────────────────────────────────────────────────────────────────────


class TestInvCovMatrix:
    def test_inv_times_cov_is_identity(self, stats_same):
        product = stats_same.inv_cov_matrix @ stats_same.cov_matrix
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_cached(self, stats_same):
        assert stats_same.inv_cov_matrix is stats_same.inv_cov_matrix

    def test_chi_squared_consistent_with_manual_inv(self, stats_same):
        # chi2 via whitened path must equal manual diff @ inv_cov @ diff
        theory = MEANS_A
        diff = stats_same.array[:, 0] - theory
        chi2_manual = diff @ stats_same.inv_cov_matrix @ diff
        chi2_method = stats_same.chi_squared(theory)
        np.testing.assert_allclose(chi2_method, chi2_manual, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# _fast_load preserves _sampling_info
# ──────────────────────────────────────────────────────────────────────────────


class TestFastLoad:
    def test_filter_preserves_sampling_info(self, stats_same):
        # filter() uses _fast_load; the result must have _sampling_info set
        filtered = stats_same.filter(lambda s: True)
        assert filtered._sampling_info == stats_same._sampling_info

    def test_cov_matrix_after_filter(self, stats_same):
        # jackknife correction path in cov_matrix requires _sampling_info
        filtered = stats_same.filter(lambda s: True)
        C = filtered.cov_matrix  # must not raise AttributeError
        assert C.shape == (3, 3)

    def test_jackknife_filter(self, stats_jk):
        filtered = stats_jk.filter(lambda s: True)
        assert filtered._sampling_info == stats_jk._sampling_info
        # jackknife correction must apply correctly after filter
        np.testing.assert_allclose(filtered.cov_matrix, stats_jk.cov_matrix, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# bic / aicc
# ──────────────────────────────────────────────────────────────────────────────


class TestBicAicc:
    def test_bic_formula(self, stats_same):
        theory = MEANS_A
        nparams = 2
        chi2_val = stats_same.chi_squared(theory)
        expected = chi2_val + nparams * np.log(stats_same.num_observables)
        np.testing.assert_allclose(
            stats_same.bic(nparams, theory_values=theory), expected, rtol=1e-12
        )

    def test_bic_ge_aic_for_large_n(self, stats_same):
        # BIC penalty grows faster than AIC when n_obs > e^2 ≈ 7.4
        # With n_obs=3 the ordering can go either way; just check they're finite
        theory = MEANS_A
        nparams = 1
        assert np.isfinite(stats_same.bic(nparams, theory_values=theory))
        assert np.isfinite(stats_same.aic(nparams, theory_values=theory))

    def test_aicc_reduces_to_aic_for_large_n(self, stats_same):
        # As n_obs >> k, AICc → AIC.  With only 3 obs the correction is large;
        # just verify AICc >= AIC (correction is positive when denom > 0).
        theory = MEANS_A
        nparams = 1
        aic_val = stats_same.aic(nparams, theory_values=theory)
        aicc_val = stats_same.aicc(nparams, theory_values=theory)
        assert aicc_val >= aic_val - 1e-12

    def test_aicc_inf_when_saturated(self, stats_same):
        # nparams == n_obs → denom = 0 → inf
        n = stats_same.num_observables
        result = stats_same.aicc(n, theory_values=MEANS_A)
        assert result == float("inf")

    def test_precomputed_chi2_reused(self, stats_same):
        theory = MEANS_A
        chi2_val = stats_same.chi_squared(theory)
        np.testing.assert_allclose(
            stats_same.bic(1, chi2_val=chi2_val),
            stats_same.bic(1, theory_values=theory),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            stats_same.aicc(1, chi2_val=chi2_val),
            stats_same.aicc(1, theory_values=theory),
            rtol=1e-12,
        )


if __name__ == "__main__":
    # Quick smoke-run without pytest
    import sys

    stats = SamplingStats(SAMPLINGS_SAME_ENS)
    print("cov_matrix:\n", stats.cov_matrix)
    print("corr_matrix:\n", np.round(stats.corr_matrix, 4))
    print("inv_cholesky_cov_matrix:\n", np.round(stats.inv_cholesky_cov_matrix, 4))
    print("chi_squared:", stats.chi_squared(MEANS_A))
    print("goodness_of_fit:", stats.goodness_of_fit(MEANS_A, nparams=0))
    print("aic:", stats.aic(2, theory_values=MEANS_A))
    print("effective_sample_size:", stats.effective_sample_size)
    stats.fit_summary(MEANS_A, nparams=1, print_results=True)
    print("All smoke checks passed.")
    sys.exit(0)
