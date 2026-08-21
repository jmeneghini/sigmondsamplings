"""Tests for the optimagic MinimizerConfig in sigmondsamplings.fitting."""

import pytest

from sigmondsamplings.fitting import (
    MinimizerConfig,
    algorithm_capabilities,
    algorithm_settings,
    available_algorithms,
)
from slat import StrictModel

om = pytest.importorskip("optimagic")


def test_is_strict_model_with_defaults():
    cfg = MinimizerConfig()
    assert issubclass(MinimizerConfig, StrictModel)
    assert cfg.algorithm == "scipy_lbfgsb"
    assert cfg.options == {}


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError):
        MinimizerConfig(algorithm="scipy_lbfgsb", backend="nevergrad")


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="unknown optimagic algorithm"):
        MinimizerConfig(algorithm="not_a_real_algo")


def test_unknown_option_key_rejected():
    with pytest.raises(ValueError, match="invalid options"):
        MinimizerConfig(algorithm="scipy_lbfgsb", options={"nope": 1})


def test_build_returns_configured_algorithm():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb", options={"stopping_maxiter": 250})
    algo = cfg.build()
    assert algo.stopping_maxiter == 250


def test_resolved_settings_merges_defaults_and_options():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb", options={"stopping_maxiter": 250})
    settings = cfg.resolved_settings()
    assert settings["stopping_maxiter"] == 250  # override
    assert "convergence_ftol_rel" in settings  # default still present


def test_capabilities_enum_cleaned():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb")
    caps = cfg.capabilities()
    assert caps["name"] == "scipy_lbfgsb"
    assert caps["solver_type"] == "scalar"  # AggregationLevel enum -> .value
    assert caps["supports_bounds"] is True


def test_module_introspection_helpers():
    names = available_algorithms()
    assert "scipy_lbfgsb" in names
    assert names == sorted(names)
    assert "stopping_maxiter" in algorithm_settings("scipy_lbfgsb")
    assert algorithm_capabilities("scipy_lbfgsb")["name"] == "scipy_lbfgsb"


def test_drives_optimagic_minimize():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb")
    res = om.minimize(
        lambda p: (p["x"] - 2.0) ** 2,
        params={"x": 0.0},
        algorithm=cfg.build(),
    )
    assert res.params["x"] == pytest.approx(2.0, abs=1e-5)


def test_toml_round_trip():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb", options={"stopping_maxiter": 500})
    restored = MinimizerConfig.from_toml(cfg.to_toml())
    assert restored == cfg


def test_resolved_writes_all_settings_to_toml():
    cfg = MinimizerConfig(algorithm="scipy_lbfgsb", options={"stopping_maxiter": 500})
    resolved = cfg.resolved()
    # options now contain every setting for the algorithm, including defaults
    assert resolved.options == algorithm_settings("scipy_lbfgsb") | {"stopping_maxiter": 500}
    assert resolved.resolved_settings() == resolved.options
    # round-trips and behaves identically to the sparse config
    assert MinimizerConfig.from_toml(resolved.to_toml()) == resolved
    assert resolved.build().stopping_maxiter == 500
    # original is unchanged
    assert cfg.options == {"stopping_maxiter": 500}
