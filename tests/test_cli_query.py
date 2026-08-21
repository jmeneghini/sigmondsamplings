from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sigmondsamplings.cli.main import app
from sigmondsamplings.cli.query import (
    QuerySpec,
    apply_query,
    attr_records,
    collection_dataframe,
    parse_where_specs,
    unique_records,
)
from sigmondsamplings.io.loader import SigmondLoader


DATA_DIR = Path(__file__).parent / "data"
ENERGY_HDF5 = DATA_DIR / "energy_levels_samplings.hdf5"
runner = CliRunner()


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
    spec = f"sector={sector[0]}:{sector[1]}"

    filters = parse_where_specs([spec], levels)
    filtered = apply_query(levels, QuerySpec(where=(spec,)))

    assert filters == {"sector": sector}
    assert len(filtered) == len(levels.filter(sector=sector))


def test_cli_where_parses_comma_separated_tuple_values():
    levels = _energy_levels()
    sector = next(sector for sector in levels.sectors if sector is not None)
    spec = f"sector={sector[0]}:{sector[1]},999:Nope"

    filters = parse_where_specs([spec], levels)
    filtered = apply_query(levels, QuerySpec(where=(spec,)))

    assert filters == {"sector": [sector, (999, "Nope")]}
    assert len(filtered) == len(levels.filter(sector=sector))


def test_cli_where_rejects_comma_as_tuple_separator():
    levels = _energy_levels()

    try:
        parse_where_specs(["sector=0,A1gm"], levels)
    except ValueError as exc:
        assert "must use ':'" in str(exc)
    else:
        raise AssertionError("Expected comma tuple syntax to be rejected")


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


def test_attr_records_lists_ordered_attributes_with_occurrence_counts():
    levels = _energy_levels()

    records = attr_records(levels, ["name", "sector", "mean"])
    columns = [record["attribute"] for record in records]

    assert columns[:3] == ["name", "sector", "mean"]
    assert "_latex_str" not in columns
    assert "ensemble_info" not in columns
    assert "energy_type" in columns
    # every record carries a non-negative occurrence count
    assert all(isinstance(record["count"], int) and record["count"] >= 0 for record in records)
    # "name" is present on every observable
    name_record = next(record for record in records if record["attribute"] == "name")
    assert name_record["count"] == len(list(levels))


def test_unique_records_handles_numeric_unique_values_as_lists():
    levels = _energy_levels()
    levels.return_type = "list"

    records = unique_records(levels, "psq")

    assert records
    assert all(set(record) == {"value"} for record in records)


def test_cli_generic_plot_writes_queried_collection(tmp_path):
    out = tmp_path / "errorbar.png"

    result = runner.invoke(
        app,
        [
            "query",
            "obs",
            str(ENERGY_HDF5),
            "--where",
            "index=0",
            "--limit",
            "2",
            "--plot",
            "errorbar",
            "--plot-output",
            str(out),
            "--no-gui",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


def test_cli_energy_spectrum_plot_writes_queried_collection(tmp_path):
    out = tmp_path / "spectrum.png"

    result = runner.invoke(
        app,
        [
            "query",
            "energy",
            str(ENERGY_HDF5),
            "--where",
            "sector=0:A1gm",
            "--plot-spectrum",
            "--plot-output",
            str(out),
            "--no-gui",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0


def test_cli_rejects_plot_with_non_collection_output_mode():
    result = runner.invoke(
        app,
        [
            "query",
            "energy",
            str(ENERGY_HDF5),
            "--unique",
            "psq",
            "--plot-spectrum",
            "--no-gui",
        ],
    )

    assert result.exit_code != 0
    assert "Use only one of" in result.output


def test_cli_reports_old_comma_tuple_syntax_as_parameter_error():
    result = runner.invoke(
        app,
        [
            "query",
            "energy",
            str(ENERGY_HDF5),
            "-w",
            "sector=0,A1gm",
            "--unique",
            "sector",
        ],
    )

    assert result.exit_code != 0
    assert "must use ':'" in result.output


def test_cli_reports_unknown_where_attr_with_suggestion():
    result = runner.invoke(
        app,
        [
            "query",
            "energy",
            str(ENERGY_HDF5),
            "-w",
            "obs_type=energy_single_hadron",
            "--plot-spectrum",
            "--no-gui",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown --where attribute 'obs_type'" in result.output
    assert "obs_kind" in result.output


def test_cli_reports_empty_collection_before_plotting():
    result = runner.invoke(
        app,
        [
            "query",
            "energy",
            str(ENERGY_HDF5),
            "-w",
            "obs_kind=does_not_exist",
            "--plot-spectrum",
            "--no-gui",
        ],
    )

    assert result.exit_code != 0
    assert "Query matched no observables" in result.output
