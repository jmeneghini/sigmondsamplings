from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from correlator_data import OPS, ckey, observable, series, vev
from sigmondsamplings.cli.main import app
from sigmondsamplings.io.writer import SigmondWriter

runner = CliRunner()
TIMES = (0, 1, 2)


@pytest.fixture(scope="module")
def bins_file(tmp_path_factory):
    """A complex bins file: a 2x2 matrix over three times plus both VEVs."""
    rng = np.random.default_rng(3)
    items = [
        observable(ckey(i, j, t), series(rng, None, loc=1.0), None)
        for i in range(2)
        for j in range(2)
        for t in TIMES
    ]
    items += [vev(OPS[i], series(rng, None), None) for i in range(2)]
    path = tmp_path_factory.mktemp("corr") / "corr_bins.hdf5"
    SigmondWriter().write_bins_hdf5(str(path), items)
    return path


def _json(*args):
    result = runner.invoke(app, ["query", *args, "--format", "json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _error(*args) -> str:
    result = runner.invoke(app, ["query", *args], env={"COLUMNS": "400"})
    assert result.exit_code != 0
    return " ".join(result.output.split())


def test_obs_lists_complex_bins(bins_file):
    rows = _json("obs", str(bins_file))

    assert len(rows) == 4 * len(TIMES) + 2
    assert all(row["data_str"].endswith("i") for row in rows)
    assert all(row["CI_lower"] is None for row in rows)


def test_corr_default_lists_stored_series(bins_file):
    rows = _json("corr", str(bins_file))

    assert len(rows) == 4
    assert {row["times"] for row in rows} == {"0..2"}
    assert rows[0]["sink"].startswith("0: ")


def test_corr_listings(bins_file):
    ops = _json("corr", str(bins_file), "--operators")
    channels = _json("corr", str(bins_file), "--channels")
    vevs = _json("corr", str(bins_file), "--vevs")

    assert [row["operator"] for row in ops] == [str(OPS[0]), str(OPS[1])]
    assert all(row["vev"] for row in ops)
    assert channels == [{"#": 0, "channel": str(OPS[0].channel), "n_operators": 2}]
    assert len(vevs) == 2


def test_corr_pair_by_number_and_string(bins_file):
    by_number = _json("corr", str(bins_file), "--sink", "0", "--source", "1")
    by_string = _json("corr", str(bins_file), "--sink", str(OPS[0]), "--source", str(OPS[1]))
    subset = _json("corr", str(bins_file), "--sink", "0", "--source", "1", "--times", "1,2")

    assert [row["time"] for row in by_number] == list(TIMES)
    assert by_number == by_string
    assert subset == by_number[1:]


def test_corr_matrix_is_hermitian_by_default(bins_file):
    rows = _json("corr", str(bins_file), "--matrix", "0", "--time", "1")

    assert [row["sink"] for row in rows] == [str(OPS[0]), str(OPS[1])]
    re01, sign01, im01 = rows[0]["1"].split(" ")
    re10, sign10, im10 = rows[1]["0"].split(" ")
    assert (re10, im10) == (re01, im01)
    assert {sign01, sign10} == {"+", "-"}


def test_corr_subtract_vev_needs_sampling_for_bins(bins_file):
    assert "sampling_info" in _error("corr", str(bins_file), "--sink", "0", "--source", "1", "--subtract-vev")

    rows = _json(
        "corr", str(bins_file), "--sink", "0", "--source", "1", "--subtract-vev", "--sampling", "jackknife"
    )
    boot = _json("corr", str(bins_file), "--matrix", "0", "--subtract-vev", "--sampling", "bootstrap:20:4")

    assert len(rows) == len(TIMES)
    assert len(boot) == 2


@pytest.fixture(scope="module")
def herm_file(tmp_path_factory):
    """HermMat bins over three operators: upper triangle only, with (2, 2) unmeasured."""
    rng = np.random.default_rng(5)
    cells = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    items = [
        observable(ckey(i, j, t, herm=True), series(rng, None, loc=1.0), None)
        for i, j in cells
        for t in TIMES
    ]
    path = tmp_path_factory.mktemp("corr") / "herm_bins.hdf5"
    SigmondWriter().write_bins_hdf5(str(path), items)
    return path


def _numbers(path) -> list[str]:
    """``--operators`` numbers of ``OPS``, as strings."""
    rows = _json("corr", str(path), "--operators")
    number = {row["operator"]: row["#"] for row in rows}
    return [str(number[str(op)]) for op in OPS]


def test_corr_pair_follows_hermmat_file(herm_file):
    n = _numbers(herm_file)
    rows = _json("corr", str(herm_file), "--sink", n[0], "--source", n[2])
    flipped = _json("corr", str(herm_file), "--sink", n[2], "--source", n[0])

    assert [row["time"] for row in rows] == list(TIMES)
    assert [complex(row["full_sample_value"]) for row in flipped] == [
        complex(row["full_sample_value"]).conjugate() for row in rows
    ]
    message = _error("corr", str(herm_file), "--sink", n[2], "--source", n[0], "--no-hermitian")
    assert "stored forms are 'HermMat' (use --hermitian/--subtract-vev to match)" in message


def test_corr_matrix_operator_subset(herm_file):
    n = _numbers(herm_file)
    message = _error("corr", str(herm_file), "--matrix", "0", "--time", "1")
    rows = _json("corr", str(herm_file), "--matrix", "0", "--time", "1", "--ops", f"{n[0]},{n[1]}")
    reordered = _json("corr", str(herm_file), "--matrix", "0", "--time", "1", "--ops", f"{n[1]},{n[0]}")

    assert "not every requested cell" in message
    assert f"--ops {','.join(sorted(n[:2]))}" in message
    assert [row["sink"] for row in rows] == [str(OPS[0]), str(OPS[1])]
    assert [row["sink"] for row in reordered] == [str(OPS[1]), str(OPS[0])]
    assert reordered[0]["0"] == rows[1]["1"]


@pytest.mark.parametrize(
    "args, message",
    [
        (["--sink", "0"], "--sink and --source must be given together"),
        (["--sink", "0", "--source", "9"], "operator #9 out of range"),
        (["--operators", "--channels"], "Use only one of"),
        (["--time", "1"], "--time requires --matrix"),
        (["--hermitian"], "require --sink/--source or --matrix"),
        (["--matrix", "nope"], "no channels match"),
        (["--matrix", "0", "--sampling", "bootstrap"], "--sampling must be"),
        (["--ops", "0"], "--ops requires --matrix"),
        (["--matrix", "0", "--ops", "0,x"], "--ops takes operator numbers"),
        (["--matrix", "0", "--ops", "0,0"], "--ops repeats an operator"),
        (["--matrix", "0", "--ops", "0-5"], "operator #2 out of range"),
    ],
)
def test_corr_rejects_bad_options(bins_file, args, message):
    assert message in _error("corr", str(bins_file), *args)
