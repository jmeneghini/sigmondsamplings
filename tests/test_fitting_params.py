"""Tests for the optimagic-native parameter layer in sigmondsamplings.fitting.

The layer follows the Spec/Resolved config pattern: authoring specs
(:class:`ParamSpec`/:class:`ConstraintSpec`/:class:`ParamSetSpec`) resolve into the
canonical, optimagic-facing :class:`ParamSetResolved`.
"""

import pytest

from sigmondsamplings.fitting import (
    ConstraintSpec,
    ParamSetSpec,
    ParamSpec,
)
from slat import StrictModel

om = pytest.importorskip("optimagic")


# --- ParamSpec authoring ------------------------------------------------------


def test_param_spec_defaults():
    spec = ParamSpec()
    assert spec.initial is None
    assert (spec.lower, spec.upper) == (None, None)
    assert spec.fixed is False


def test_param_spec_bounds_pair_shorthand_expands():
    spec = ParamSpec(bounds=[0.0, 1.0], soft_bounds=[0.1, 0.9])
    assert (spec.lower, spec.upper) == (0.0, 1.0)
    assert (spec.soft_lower, spec.soft_upper) == (0.1, 0.9)


def test_param_spec_bounds_pair_conflict_rejected():
    with pytest.raises(ValueError, match="either 'bounds' or 'lower'"):
        ParamSpec(bounds=[0.0, 1.0], lower=0.0)


def test_param_spec_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        ParamSpec(lower=1.0, upper=0.0)


def test_param_spec_is_strict():
    assert issubclass(ParamSpec, StrictModel)
    with pytest.raises(ValueError):
        ParamSpec(typo=3.0)


# --- resolve ------------------------------------------------------------------


def test_resolve_requires_initial():
    spec = ParamSetSpec(values={"A": ParamSpec(initial=1.0), "m": ParamSpec()})
    with pytest.raises(ValueError, match="initial value is missing for parameter 'm'"):
        spec.resolve()


def test_resolved_param_builds_observable_info():
    resolved = ParamSetSpec(
        values={"A": ParamSpec(initial=1.0, latex=r"A_0")}
    ).resolve()
    info = resolved.values["A"].observable_info("A")
    assert info.name == "A"
    assert info.index == 0
    assert info.op_type == "n"
    assert info.re_im == "re"
    assert info.latex_str == r"A_0"


def test_resolve_canonicalizes_constraint_to_param_groups():
    resolved = ParamSetSpec(
        values={"m1": ParamSpec(initial=0.5), "m2": ParamSpec(initial=0.5)},
        constraints=[ConstraintSpec(kind="increasing", params=["m1", "m2"])],
    ).resolve()
    assert resolved.constraints[0].param_groups == [["m1", "m2"]]


# --- ParamSetSpec validation --------------------------------------------------


def test_parameter_names_are_mapping_keys():
    spec = ParamSetSpec(values={"A": ParamSpec(initial=1.0)})
    assert spec.names == ["A"]


def test_param_set_rejects_unknown_constraint_param():
    with pytest.raises(ValueError, match="unknown parameter"):
        ParamSetSpec(
            values={"m1": ParamSpec(), "m2": ParamSpec()},
            constraints=[ConstraintSpec(kind="increasing", params=["m1", "m3"])],
        )


# --- start_values / bounds (resolved) -----------------------------------------


def test_start_values():
    ps = ParamSetSpec(
        values={"A": ParamSpec(initial=1.0), "m": ParamSpec(initial=0.5)}
    ).resolve()
    assert ps.start_values() == {"A": 1.0, "m": 0.5}


def test_bounds_only_sets_declared_channels():
    ps = ParamSetSpec(
        values={
            "A": ParamSpec(initial=1.0),
            "m": ParamSpec(initial=0.5, lower=0.0),
            "s": ParamSpec(initial=1.0, soft_upper=10.0),
        }
    ).resolve()
    bounds = ps.bounds()
    assert bounds.lower == {"m": 0.0}
    assert bounds.soft_upper == {"s": 10.0}
    assert bounds.upper is None and bounds.soft_lower is None


def test_bounds_none_when_unbounded():
    ps = ParamSetSpec(values={"A": ParamSpec(initial=1.0)}).resolve()
    assert ps.bounds() is None


# --- constraints_for_optimagic (resolved) -------------------------------------


def test_fixed_flag_produces_fixed_constraint():
    ps = ParamSetSpec(
        values={"A": ParamSpec(initial=1.0), "c": ParamSpec(initial=5.0, fixed=True)}
    ).resolve()
    cons = ps.constraints_for_optimagic()
    assert len(cons) == 1
    assert isinstance(cons[0], om.FixedConstraint)
    # selector picks the fixed parameter from a dict PyTree
    assert cons[0].selector({"A": 1.0, "c": 5.0}).tolist() == [5.0]


def test_linear_constraint_forwards_extra_fields():
    ps = ParamSetSpec(
        values={"a": ParamSpec(initial=1.0), "b": ParamSpec(initial=1.0)},
        constraints=[
            ConstraintSpec(kind="linear", params=["a", "b"], weights=1.0, value=2.0),
        ],
    ).resolve()
    (constraint,) = ps.constraints_for_optimagic()
    assert isinstance(constraint, om.LinearConstraint)
    assert constraint.value == 2.0


def test_pairwise_equality_uses_param_groups():
    ps = ParamSetSpec(
        values={name: ParamSpec(initial=1.0) for name in ("a1", "a2", "b1", "b2")},
        constraints=[
            ConstraintSpec(kind="pairwise_equality", param_groups=[["a1", "a2"], ["b1", "b2"]]),
        ],
    ).resolve()
    (constraint,) = ps.constraints_for_optimagic()
    assert isinstance(constraint, om.PairwiseEqualityConstraint)
    assert len(constraint.selectors) == 2


def test_add_constraint_validates_names():
    ps = ParamSetSpec(values={"m1": ParamSpec(initial=0.5), "m2": ParamSpec(initial=0.5)})
    ps.add_constraint("increasing", ["m1", "m2"])
    assert ps.constraints[0].kind == "increasing"
    with pytest.raises(ValueError, match="unknown parameter"):
        ps.add_constraint("increasing", ["m1", "nope"])


# --- end-to-end through optimagic.minimize ------------------------------------


def test_end_to_end_minimize_respects_fixed_and_increasing():
    ps = ParamSetSpec(
        values={
            "A": ParamSpec(initial=0.0),
            "m1": ParamSpec(initial=0.5),
            "m2": ParamSpec(initial=0.5),
            "c": ParamSpec(initial=5.0, fixed=True),
        },
        constraints=[ConstraintSpec(kind="increasing", params=["m1", "m2"])],
    ).resolve()

    def crit(p):
        # unconstrained optimum wants m1=2 > m2=1, so 'increasing' forces m1<=m2
        return (p["A"] - 3) ** 2 + (p["m1"] - 2) ** 2 + (p["m2"] - 1) ** 2 + (p["c"] - 0) ** 2

    res = om.minimize(
        crit,
        params=ps.start_values(),
        algorithm="scipy_lbfgsb",
        bounds=ps.bounds(),
        constraints=ps.constraints_for_optimagic(),
    )
    assert res.params["c"] == 5.0  # fixed, untouched
    assert res.params["m1"] <= res.params["m2"] + 1e-6  # increasing held
    assert res.params["A"] == pytest.approx(3.0, abs=1e-4)


# --- TOML round-trip ----------------------------------------------------------


def test_toml_round_trip():
    ps = ParamSetSpec(
        values={
            "A": ParamSpec(initial=1.0, latex=r"$A$"),
            "m": ParamSpec(initial=0.5, lower=0.0),
        },
        constraints=[ConstraintSpec(kind="fixed", params=["A"])],
    )
    text = ps.to_toml()
    assert "[params.values.A]" in text
    assert "[params.values.m]" in text
    assert "name =" not in text
    restored = ParamSetSpec.from_toml(text)
    assert restored == ps


def test_resolved_round_trips_through_to_spec():
    spec = ParamSetSpec(
        values={
            "A": ParamSpec(initial=1.0, lower=0.0),
            "m": ParamSpec(initial=0.5, fixed=True),
        },
        constraints=[ConstraintSpec(kind="increasing", params=["A", "m"])],
    )
    resolved = spec.resolve()
    assert resolved.to_spec().resolve() == resolved


def test_toml_unknown_key_rejected():
    ps = ParamSetSpec(values={"A": ParamSpec(initial=1.0)})
    bad = ps.to_toml().replace("initial", "intial")
    with pytest.raises(ValueError):
        ParamSetSpec.from_toml(bad)
