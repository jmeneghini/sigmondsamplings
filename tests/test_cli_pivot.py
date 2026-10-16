import json

from pivot_data import spectrum
from typer.testing import CliRunner

from sigmondsamplings.cli.main import app
from sigmondsamplings.correlators import SinglePivot
from sigmondsamplings.io.loader import SigmondLoader

TIMES = ['--norm-time', '1', '--metric-time', '1', '--diagonalize-time', '3']


def test_cli_pivot_export_reuse_and_overwrite(tmp_path):
    matrix, _ = spectrum(noise=.0001)
    source, output, pivot = [tmp_path / f for f in ('source.hdf5', 'rotated.hdf5', 'pivot.hdf5')]
    matrix.write(source)
    args = ['pivot', str(source), '--matrix', '0', *TIMES, '--min-inv-condition', '1e-8',
            '--format', 'json', '--write-rotated', str(output), '--write-pivot', str(pivot),
            '--pivot-group', 'channel', '--diagonal-only-before-time', '3']
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    records = json.loads(result.output)
    assert records[0]['pivot']['metric_rank'] == 3
    assert SinglePivot.read(pivot, group='channel').gevp.rank == 3
    assert len(SigmondLoader(output).observables) == 3 * 3 + 4 * 9

    rerun = CliRunner().invoke(app, args)
    assert rerun.exit_code != 0 and 'already exists' in rerun.output
    rerun = CliRunner().invoke(app, [*args, '--overwrite'])
    assert rerun.exit_code == 0, rerun.output


def test_cli_pivot_requires_explicit_times(tmp_path):
    result = CliRunner().invoke(app, ['pivot', str(tmp_path / 'no.hdf5'), '--matrix', '0'])
    assert result.exit_code != 0
    assert '--norm-time' in result.output


def test_cli_pivot_option_dependencies(tmp_path):
    base = ['pivot', str(tmp_path / 'no.hdf5'), '--matrix', '0', *TIMES]
    result = CliRunner().invoke(app, [*base, '--upper-triangle-only'])
    assert result.exit_code != 0 and '--write-rotated' in result.output
    result = CliRunner().invoke(app, [*base, '--plot', 'coverage'])
    assert result.exit_code != 0 and 'Unknown --plot' in result.output
    result = CliRunner().invoke(app, ['query', 'corr', str(tmp_path / 'no.hdf5'), '--pivot'])
    assert result.exit_code != 0
