from __future__ import annotations

from pathlib import Path

from sigmondsamplings.cli.query import (
    QuerySpec,
    apply_query,
    collection_dataframe,
    column_records,
    parse_where_specs,
    unique_records,
)
from sigmondsamplings.loader import SigmondLoader


DATA_DIR = Path(__file__).parent / "data"
ENERGY_HDF5 = DATA_DIR / "energy_levels_samplings.hdf5"


def _energy_levels():
    return SigmondLoader(str(ENERGY_HDF5), lazy=True).energy_observables()


def test_energy_sector_is_tuple_and_collection_helpers_use_it():
    levels = _energy_levels()
    sectors = [sector for sector in levels.sectors if sector is not None]

    assert sectors
    assert all(isinstance(sector, tuple) for sector in sectors)
    assert levels.psq_irrep_pairs == levels.sectors
    assert levels.group_by_sector().keys() == levels.group_by("sector").keys()


def test_collection_filter_accepts_sector_tuple_as_scalar():
    levels = _energy_levels()
    sector = next(sector for sector in levels.sectors if sector is not None)

    filtered = levels.filter(sector=sector)

    assert len(filtered) > 0
    assert all(sampling.observable_info.sector == sector for sampling in filtered)


def test_cli_where_normalizes_tuple_valued_attrs_from_collection():
    levels = _energy_levels()
    sector = next(sector for sector in levels.sectors if sector is not None)
    spec = f"sector={sector[0]},{sector[1]}"

    filters = parse_where_specs([spec], levels)
    filtered = apply_query(levels, QuerySpec(where=(spec,)))

    assert filters == {"sector": sector}
    assert len(filtered) == len(levels.filter(sector=sector))


def test_to_dataframe_excludes_default_observable_attrs():
    levels = _energy_levels()

    df = levels.to_dataframe()

    assert "_latex_str" not in df.columns
    assert "ensemble_info" not in df.columns
    assert "sampling_info" not in df.columns
    assert "name" in df.columns


def test_to_dataframe_merges_default_and_requested_excluded_attrs():
    levels = _energy_levels()

    df = levels.to_dataframe(excluded_attrs=["particles"])

    assert "_latex_str" not in df.columns
    assert "ensemble_info" not in df.columns
    assert "particles" not in df.columns


def test_cli_dataframe_can_recover_explicit_default_excluded_column():
    levels = _energy_levels()

    auto_df = collection_dataframe(levels)
    explicit_df = collection_dataframe(levels, columns=["_latex_str"], excluded_attrs=None)

    assert "_latex_str" not in auto_df.columns
    assert "_latex_str" in explicit_df.columns


def test_column_records_lists_ordered_available_dataframe_columns():
    levels = _energy_levels()

    records = column_records(levels, ["name", "sector", "mean"])
    columns = [record["column"] for record in records]

    assert columns[:3] == ["name", "sector", "mean"]
    assert "_latex_str" not in columns
    assert "ensemble_info" not in columns
    assert "energy_type" in columns


def test_unique_records_handles_numeric_unique_values_as_lists():
    levels = _energy_levels()
    levels.return_type = "list"

    records = unique_records(levels, "psq")

    assert records
    assert all(set(record) == {"value"} for record in records)
