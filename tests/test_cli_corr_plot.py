"""``ss query corr --plot``: end to end, from a written file to a saved figure.

Invoked through the real CLI so a keyword dropped anywhere between the typer
option and the plotter fails here.
"""

from __future__ import annotations

import numpy as np
import pytest
from correlator_data import BL, JACK, ckey, observable, series
from typer.testing import CliRunner

import slat.plotting as slp
from sigmondsamplings.cli.main import app
from sigmondsamplings.io.writer import SigmondWriter

runner = CliRunner()
OPS3 = tuple(BL(f"pion P=(0,0,0) A1um_1 SD_{k}") for k in range(3))
TIMES = tuple(range(12))
T = 32
BACKENDS = [b for b in slp.available() if b != "recording"]


@pytest.fixture(autouse=True)
def _agg():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def samplings_file(tmp_path_factory):
    """Jackknife HermMat samplings of a 3x3 matrix, upper triangle only, (2, 2) missing at t=11."""
    rng = np.random.default_rng(4)
    energies = np.array([0.3, 0.5, 0.8])
    z = rng.normal(size=(3, 3)) + 2 * np.eye(3)
    n = JACK.num_resamplings + 1
    items = []
    for t in TIMES:
        c = (z * np.exp(-energies * t)) @ z.T
        for i in range(3):
            for j in range(i, 3):
                if (i, j, t) == (2, 2, 11):
                    continue
                data = c[i, j] * (1 + 0.01 * rng.normal(size=n)) + 0j
                data[0] = c[i, j]
                items.append(observable(ckey(i, j, t, True, ops=OPS3), data, JACK))
    path = tmp_path_factory.mktemp("corrplot") / "matrix.hdf5"
    SigmondWriter().write_hdf5(str(path), items)
    return path


@pytest.fixture(scope="module")
def bins_file(tmp_path_factory):
    rng = np.random.default_rng(3)
    items = [
        observable(ckey(i, j, t), series(rng, None, loc=1.0), None)
        for i in range(2)
        for j in range(2)
        for t in (0, 1, 2)
    ]
    path = tmp_path_factory.mktemp("corrplot") / "bins.hdf5"
    SigmondWriter().write_bins_hdf5(str(path), items)
    return path


def _numbers(path) -> list[str]:
    result = runner.invoke(app, ["query", "corr", str(path), "--operators", "--format", "json"])
    import json

    rows = json.loads(result.output)
    number = {row["operator"]: row["#"] for row in rows}
    return [str(number[str(op)]) for op in OPS3]


def _plot(path, out, *args):
    result = runner.invoke(
        app,
        ["query", "corr", str(path), *args, "--no-gui", "--plot-output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 0
    return result


def _error(path, *args) -> str:
    result = runner.invoke(app, ["query", "corr", str(path), *args], env={"COLUMNS": "400"})
    assert result.exit_code != 0
    return " ".join(result.output.split())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "args",
    [
        ["--plot", "correlator"],
        ["--plot", "effective-energy", "--eff-kind", "symmetric", "--eff-step", "2"],
        ["--plot", "summary"],
    ],
)
def test_single_correlator_plots(samplings_file, tmp_path, backend, args):
    n = _numbers(samplings_file)
    out = tmp_path / ("p.html" if backend == "plotly" else "p.png")
    _plot(
        samplings_file,
        out,
        "--sink",
        n[1],
        "--source",
        n[1],
        *args,
        "--time-extent",
        str(T),
        "--plot-backend",
        backend,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "args",
    [
        ["--plot", "diagonal"],
        ["--plot", "effective-energy"],
        ["--plot", "grid"],
        ["--plot", "grid", "--norm-time", "3"],
        ["--plot", "heatmap", "--time", "5", "--plot-value", "significance"],
        ["--plot", "heatmap", "--norm-time", "2"],
        ["--plot", "coverage"],
    ],
)
def test_matrix_plots(samplings_file, tmp_path, backend, args):
    out = tmp_path / ("m.html" if backend == "plotly" else "m.png")
    _plot(
        samplings_file,
        out,
        "--matrix",
        "0",
        *args,
        "--time-extent",
        str(T),
        "--plot-backend",
        backend,
    )


def test_matrix_plots_use_the_complete_times_and_ops(samplings_file, tmp_path, monkeypatch):
    """The plotted matrix is what the query selects: --ops, every complete time."""
    import sigmondsamplings.cli.plot as plot_module

    seen = {}
    real = plot_module.render_correlator_plot

    def spy(data, **kwargs):
        seen["data"], seen["kwargs"] = data, kwargs
        return real(data, **kwargs)

    monkeypatch.setattr(plot_module, "render_correlator_plot", spy)
    n = _numbers(samplings_file)
    with pytest.warns(UserWarning, match=r"times \[11\]"):
        _plot(
            samplings_file,
            tmp_path / "d.png",
            "--matrix",
            "0",
            "--ops",
            f"{n[2]},{n[0]}",
            "--plot",
            "diagonal",
        )
    matrix = seen["data"]
    assert matrix.operators == (OPS3[2], OPS3[0])
    assert matrix.times == TIMES[:-1]  # 11 lacks (2, 2)
    assert seen["kwargs"]["labels"] == [n[2], n[0]]


def test_coverage_works_where_the_matrix_fails(samplings_file, tmp_path):
    # at t=11 the full matrix is incomplete, but coverage needs no matrix
    message = _error(samplings_file, "--matrix", "0", "--time", "11")
    assert "not every requested cell" in message and "--ops" in message
    _plot(samplings_file, tmp_path / "c.png", "--matrix", "0", "--plot", "coverage")


def test_bins_need_sampling_except_for_coverage(bins_file, tmp_path):
    assert "--sampling" in _error(
        bins_file, "--sink", "0", "--source", "0", "--plot", "correlator", "--no-gui"
    )
    _plot(bins_file, tmp_path / "c.png", "--matrix", "0", "--plot", "coverage")
    _plot(
        bins_file,
        tmp_path / "b.png",
        "--matrix",
        "0",
        "--plot",
        "diagonal",
        "--sampling",
        "jackknife",
    )


def test_missing_time_extent_names_the_flag(samplings_file):
    message = _error(samplings_file, "--matrix", "0", "--plot", "effective-energy", "--eff-kind", "symmetric", "--no-gui")
    assert "pass --time-extent" in message


@pytest.mark.parametrize(
    "args, message",
    [
        (["--plot-output", "x.png"], "Plot options require --plot"),
        (["--eff-kind", "symmetric"], "Plot options require --plot"),
        (["--plot", "grid"], "--plot requires --sink/--source or --matrix"),
        (["--plot", "sparkline", "--matrix", "0"], "Unknown --plot 'sparkline'"),
        (["--plot", "grid", "--sink", "0", "--source", "0"], "needs --matrix"),
        (["--plot", "correlator", "--matrix", "0"], "not a matrix plot"),
        (
            ["--plot", "grid", "--matrix", "0", "--plot-value", "significance"],
            "--plot-value applies",
        ),
        (["--plot", "grid", "--matrix", "0", "--time", "3"], "--time applies"),
        (["--plot", "heatmap", "--matrix", "0", "--plot-value", "phase"], "--plot-value must be"),
        (
            ["--plot", "effective-energy", "--matrix", "0", "--eff-kind", "cosh"],
            "--eff-kind must be",
        ),
    ],
)
def test_rejects_bad_plot_options(samplings_file, args, message):
    assert message in _error(samplings_file, *args, "--no-gui", "--time-extent", str(T))
