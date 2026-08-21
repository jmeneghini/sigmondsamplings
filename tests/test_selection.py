"""The shared selection language.

``ss query`` and ``ss edit`` resolve ``--where``/``--contains``/``--regex``/``--spec``
through the same code, so this covers both. It also stands in for
``tests/test_cli_query.py`` on a fresh clone, where that file's gitignored HDF5
fixtures are absent: the helpers it imports from ``cli.query`` are exercised here
against a synthetic file.
"""

from __future__ import annotations

import pytest

from sigmondsamplings.cli.query import (
    QuerySpec,
    apply_query,
    attr_records,
    collection_dataframe,
    load_collection,
    parse_where_specs,
    unique_records,
)
from sigmondsamplings.selection import filter_collection, resolve_where


@pytest.fixture
def levels(samplings_file, tmp_path):
    """The synthetic file, tagged with energy attrs, as an energy-level collection."""
    from pathlib import Path

    from sigmondsamplings.edit_spec import EditSpec
    from sigmondsamplings.io.edit import edit_file

    spec = EditSpec.from_toml('[[edit]]\nop = "tag-energy"\n').resolve(base_dir=Path.cwd())
    tagged = edit_file(str(samplings_file), str(tmp_path / "tagged.hdf5"), spec)
    return load_collection(tagged, energy=True)


def names(collection) -> set[str]:
    return {s.observable_info.name for s in collection}


# ---------------------------------------------------------------------------
# where parsing
# ---------------------------------------------------------------------------


def test_parses_scalars_and_membership_lists():
    assert parse_where_specs(["psq=0", "irrep=A1g,E"]) == {"psq": 0, "irrep": ["A1g", "E"]}


def test_parses_the_literal_scalars():
    parsed = parse_where_specs(["a=none", "b=true", "c=false", "d=1.5", "e=text"])
    assert parsed == {"a": None, "b": True, "c": False, "d": 1.5, "e": "text"}


def test_repeated_specs_for_one_attr_merge_into_membership():
    assert parse_where_specs(["psq=0", "psq=1"]) == {"psq": [0, 1]}


def test_rejects_a_spec_without_an_equals():
    with pytest.raises(ValueError, match="expected attr=value"):
        parse_where_specs(["psq"])


def test_rejects_an_unknown_attr_with_a_suggestion(levels):
    with pytest.raises(ValueError, match="Did you mean 'psq'"):
        parse_where_specs(["psqq=0"], levels)


def test_parses_a_tuple_valued_attr(levels):
    assert parse_where_specs(["sector=0:A1g"], levels) == {"sector": (0, "A1g")}


def test_a_mapping_where_needs_no_parsing(levels):
    assert resolve_where({"psq": 0}, levels) == {"psq": 0}


def test_a_mapping_where_is_validated_too(levels):
    with pytest.raises(ValueError, match="Did you mean 'psq'"):
        resolve_where({"psqq": 0}, levels)


# ---------------------------------------------------------------------------
# clause resolution
# ---------------------------------------------------------------------------


def test_where_filters(levels):
    assert names(filter_collection(levels, where={"psq": 0})) == {
        "PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1", "PSQ0_N", "PSQ0_pi",
    }


def test_contains_filters(levels):
    assert names(filter_collection(levels, contains="PSQ1")) == {"PSQ1_E_elab_0"}


def test_regex_filters(levels):
    assert names(filter_collection(levels, regex=r"_elab_\d$")) == {
        "PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1", "PSQ1_E_elab_0",
    }


def test_spec_filters(levels, tmp_path):
    spectrum = tmp_path / "spectrum.toml"
    spectrum.write_text('[[spectrum]]\npsq = 0\nirrep = "A1g"\nlevels = [1]\n')
    assert names(filter_collection(levels, spec=str(spectrum))) == {"PSQ0_A1g_elab_1"}


def test_clauses_combine_with_and(levels):
    assert names(filter_collection(levels, where={"psq": 0}, contains="elab")) == {
        "PSQ0_A1g_elab_0", "PSQ0_A1g_elab_1",
    }


def test_no_clauses_matches_everything(levels):
    assert names(filter_collection(levels)) == names(levels)


def test_filtering_returns_a_view_not_a_copy(levels):
    """Relied on by the edit executor: mutating a scope reaches the parent."""
    filter_collection(levels, contains="PSQ1").obs.update(index=9)
    assert next(s.observable_info.index for s in levels if s.observable_info.psq == 1) == 9


def test_spec_clause_needs_an_energy_collection(samplings_file, tmp_path):
    spectrum = tmp_path / "spectrum.toml"
    spectrum.write_text('[[spectrum]]\npsq = 0\nirrep = "A1g"\nlevels = [0]\n')
    plain = load_collection(samplings_file, energy=False)
    with pytest.raises(ValueError, match="requires an energy-level collection"):
        filter_collection(plain, spec=str(spectrum))


# ---------------------------------------------------------------------------
# the query front-end still runs on the extracted code
# ---------------------------------------------------------------------------


def test_apply_query_applies_where_sort_and_limit(levels):
    result = apply_query(levels, QuerySpec(where=("psq=0",), sort="level_index", limit=2))
    assert len(result) == 2


def test_apply_query_reverses(levels):
    spec = QuerySpec(contains="A1g", sort="level_index", reverse=True)
    assert [s.observable_info.level_index for s in apply_query(levels, spec)] == [1, 0]


def test_unique_records(levels):
    assert {r["value"] for r in unique_records(levels, "irrep")} >= {"A1g", "E"}


def test_collection_dataframe_selects_columns(levels):
    df = collection_dataframe(levels, ("name", "psq", "irrep"))
    assert {"name", "psq", "irrep"} <= set(df.columns)
    assert len(df) == len(levels)


def test_attr_records_counts_occurrences(levels):
    records = {r["attribute"]: r for r in attr_records(levels, ("name",))}
    assert "psq" in records


# ---------------------------------------------------------------------------
# obs_kind and the boolean facets
# ---------------------------------------------------------------------------


def test_facets_exist_on_every_observable(samplings_file):
    plain = load_collection(samplings_file, energy=False)
    for sampling in plain:
        obs = sampling.observable_info
        assert (obs.obs_kind, obs.is_energy, obs.is_single_hadron) == ("observable", False, False)


def test_is_energy_matches_single_and_multi_hadron(levels):
    assert names(filter_collection(levels, where=["is_energy=true"])) == {
        "PSQ0_A1g_elab_0",
        "PSQ0_A1g_elab_1",
        "PSQ1_E_elab_0",
        "PSQ0_N",
        "PSQ0_pi",
    }


def test_is_single_hadron_selects_only_the_hadrons(levels):
    assert names(filter_collection(levels, where=["is_single_hadron=true"])) == {
        "PSQ0_N",
        "PSQ0_pi",
    }


def test_obs_kind_names_one_exact_class(levels):
    # Deliberately non-hierarchical: single hadrons carry their own tag.
    assert names(filter_collection(levels, where=["obs_kind=energy"])) == {
        "PSQ0_A1g_elab_0",
        "PSQ0_A1g_elab_1",
        "PSQ1_E_elab_0",
    }


def test_obs_kind_accepts_the_short_alias(levels):
    short = filter_collection(levels, where=["obs_kind=energy_sh"])
    long = filter_collection(levels, where=["obs_kind=energy_single_hadron"])
    assert names(short) == names(long) == {"PSQ0_N", "PSQ0_pi"}


def test_obs_kind_energy_warns_about_the_hierarchical_form(levels, caplog):
    with caplog.at_level("WARNING"):
        filter_collection(levels, where=["obs_kind=energy"])
    assert "is_energy=true" in caplog.text


def test_unknown_attr_suggests_the_facet(levels):
    with pytest.raises(ValueError, match="Did you mean 'is_single_hadron'"):
        filter_collection(levels, where=["single_hadron=true"])


def test_energy_view_default_sort_groups_by_kind(levels):
    from sigmondsamplings.cli.query import ENERGY_DEFAULT_SORT

    ordered = apply_query(levels, QuerySpec(default_sort=ENERGY_DEFAULT_SORT))
    flags = [s.observable_info.is_single_hadron for s in ordered]
    assert flags == sorted(flags)
