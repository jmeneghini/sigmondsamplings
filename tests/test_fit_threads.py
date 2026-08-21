"""Chi-squared scan + landscape-plotting smoke tests for the fitting package.

The resampling-orchestration and minimizer-adapter tests that used to live here
moved with the retired ``sigmondsamplings.fit`` / ``sigmondsamplings.minimizers``
modules; the optimagic-native driver is covered by ``test_fitting_driver.py``.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sigmondsamplings.fitting import (
    evaluate_chi2_function_scan,
    plot_chi2_function_1d,
    plot_chi2_function_2d,
)


def test_chi2_function_scan_validates_parameter_indices():
    def chi2(params):
        return float(params[0])

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_chi2_function_scan(chi2, [-1], np.asarray([[1.0]]))
    with pytest.raises(ValueError, match="unique"):
        evaluate_chi2_function_scan(chi2, [0, 0], np.asarray([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="overlap"):
        evaluate_chi2_function_scan(
            chi2,
            [0],
            np.asarray([[1.0]]),
            fixed_params={0: 2.0},
        )


def test_evaluate_chi2_function_scan_thread_backend_preserves_order():
    def chi2(params):
        return params[0] + 2 * params[1]

    scan = evaluate_chi2_function_scan(
        chi2,
        [0],
        np.asarray([[1.0], [2.0], [3.0]]),
        fixed_params={1: 5.0},
        backend="thread",
        num_workers=2,
    )

    np.testing.assert_allclose(scan.param_stack, [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    np.testing.assert_allclose(scan.chi2_values, [11.0, 12.0, 13.0])


def test_plot_chi2_function_wrappers_smoke():
    def chi2(params):
        return float((params[0] - 1.0) ** 2 + (params[1] + 2.0) ** 2)

    fig_1d, ax_1d = plot_chi2_function_1d(
        chi2,
        0,
        (0.0, 2.0),
        n_points=5,
        fixed_params={1: -2.0},
        n_total_params=2,
    )
    fig_2d, ax_2d = plot_chi2_function_2d(
        chi2,
        (0, 1),
        ((0.0, 2.0), (-3.0, -1.0)),
        n_points=(4, 4),
    )

    assert ax_1d.figure is fig_1d
    assert ax_2d.figure is fig_2d
