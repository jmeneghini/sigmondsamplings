from pathlib import Path

import numpy as np
import pytest
from pivot_data import REFERENCE_CASES, connected_case, reference_fixture, spectrum
from sigmond_oracle import read_pivot, rotate, transformation_from_log

from sigmondsamplings.correlators import SinglePivot
from sigmondsamplings.io.loader import SigmondLoader


def compare_reference(matrix, pivot, reference, output):
    source = reference['source_labels']
    transformation = pivot.source_transformation
    labels = pivot.operators if transformation is None else transformation.source_operators
    rows = [source.index(op) for op in labels]
    R = reference['transformation'][rows]
    ss_R = pivot.transformation if transformation is None else transformation.coefficients @ pivot.transformation
    Z = reference['zmatrix'][[reference['pivot_labels'].index(op) for op in pivot.operators]]
    inner = np.sum(R.conj() * ss_R, axis=0)
    phases = inner / abs(inner)
    # Conditioning of BOTH stages enters this comparison; cap the tolerance.
    W = pivot.gevp.metric_whitening
    conditioning = np.linalg.cond(W) * max(1, 1 / pivot.gevp.inverse_condition)
    tolerance = min(1e-7, max(2e-11, conditioning * 2e-13))
    assert conditioning * 2e-13 < 1e-7, 'ill-conditioned reference fixture'
    np.testing.assert_allclose(ss_R, R * phases, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(pivot.zmatrix, Z * phases, rtol=tolerance, atol=tolerance)
    index = SigmondLoader(output, group='rotated').correlators
    ss = pivot.rotate(matrix)
    # Sigmond only writes diagonals before tD; compare complete shared times.
    flags = 'HermMat SubVEV' if matrix.info.subtract_vev else 'HermMat'
    native = index.matrix(pivot.rotated_operators, times=list(range(pivot.settings.diagonalize_time, 7)),
                          subtract_vev=matrix.info.subtract_vev, key_form=flags)
    expected = native.values.data * phases.conj()[None, :, None, None] * phases[None, None, :, None]
    np.testing.assert_allclose(ss.sel_times(native.times).values.data, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.sigmond_oracle
@pytest.mark.parametrize('complex_values', [False, True])
def test_live_cn_build_and_rotation(tmp_path, sigmond_binary, complex_values):
    matrix, _ = spectrum(complex_values=complex_values, noise=.0001)
    root, path, output = rotate(matrix, tmp_path)
    reference = read_pivot(path)
    assert reference['numeric_type'] == 'complex'  # a genuine CN oracle
    p = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                min_inv_condition=1e-8)
    printed = transformation_from_log(root, reference['source_labels'])
    np.testing.assert_allclose(printed, reference['transformation'], rtol=1e-11, atol=1e-11)
    compare_reference(matrix, p, reference, output)
    compare_build_logs(root, matrix, p)


@pytest.mark.sigmond_oracle
@pytest.mark.parametrize('case', [case for case in REFERENCE_CASES if case.startswith('rn_')])
def test_live_rn_build_and_rotation(tmp_path, case):
    matrix, comparison, options, task = reference_fixture(case)
    root, path, output = rotate(matrix, tmp_path, norm_time=options['norm_time'],
                               metric_time=options['metric_time'],
                               diagonalize_time=options['diagonalize_time'],
                               cutoff=options['min_inv_condition'], **task)
    reference = read_pivot(path)
    assert reference['numeric_type'] == 'real'
    p = SinglePivot.from_matrix(matrix, **options)
    compare_reference(comparison, p, reference, output)
    compare_build_logs(root, matrix, p)


@pytest.mark.sigmond_oracle
def test_upstream_mode_s_unsubtracted_empty_write(tmp_path, sigmond_binary):
    matrix, _ = spectrum(noise=.0001)
    rotate(matrix, tmp_path, rotate_mode='samplings', expect_empty=True)
    p = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                min_inv_condition=1e-8)
    assert len(p.rotate(matrix).to_collection()) > 0


def test_existing_log_parser():
    from sigmond_oracle import parse_log
    for path in (Path(__file__).parents[1] / 'test_data/2rotate_corrs/logs').glob('sigmond_rotation_log-*.xml'):
        root, errors = parse_log(path)
        assert not errors
        assert root.find('.//AnalyzeMetric/MetricRank') is not None
        assert len(root.findall('.//CorrelatorRotation')) > 0


@pytest.mark.sigmond_oracle
@pytest.mark.parametrize('mode', ['samplings', 'samplings_unsubt', 'samplings_all', 'bins'])
def test_live_connected_modes(tmp_path, sigmond_binary, mode):
    connected, raw = connected_case(mode)
    root, path, output = rotate(connected, tmp_path, rotate_mode=mode, input_matrix=raw)
    p = SinglePivot.from_matrix(connected, norm_time=1, metric_time=1, diagonalize_time=3,
                                min_inv_condition=1e-8)
    compare_reference(connected if mode == 'samplings' else raw, p, read_pivot(path), output)
    check = p.off_diagonal_check(p.rotate(connected if mode == 'samplings' else raw))
    for node in root.findall('.//CorrelatorRotation'):
        time = int(node.findtext('TimeValue'))
        native = node.find('OffDiagonalChecks')
        if native is None:
            continue
        observed = check.by_time.sel(time=time)
        assert observed.max_ratio.item() == pytest.approx(float(native.findtext('.//RelativeToError')), rel=2e-8, abs=1e-7)
        for sigma, word in enumerate(('One', 'Two', 'Three', 'Four'), start=1):
            assert observed[f'beyond_{sigma}'].item() == pytest.approx(float(native.findtext(f'.//GreaterThan{word}Sigma')), abs=1e-9)



@pytest.mark.sigmond_oracle
def test_live_overlap_samplings_and_energy_reorder(tmp_path, sigmond_binary, sigmond_bindings):
    import subprocess
    import sys

    from sigmondsamplings.info import ObservableInfo
    from sigmondsamplings.io.writer import SigmondWriter
    from sigmondsamplings.sampling import SigmondSampling
    matrix, _ = spectrum(complex_values=True, noise=.0001)
    rotate(matrix, tmp_path)
    p = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                min_inv_condition=1e-8)
    rng = np.random.default_rng(42)
    amps, energies = {}, {}
    for i, op in enumerate(p.rotated_operators):
        amps[op] = SigmondSampling(-1 - .1 * i + rng.normal(0, .01, 13),
                                  ObservableInfo('oracle_amplitude', i, ensemble_info=matrix.values.ensemble_info), matrix.sampling_info)
        energies[op] = SigmondSampling(np.full(13, [0.5, .2, .2][i]),
                                      ObservableInfo('oracle_energy', i, ensemble_info=matrix.values.ensemble_info), matrix.sampling_info)
    SigmondWriter().write_hdf5(str(tmp_path / 'fit.hdf5'), [*amps.values(), *energies.values()],
                              group='fit', overwrite=True)
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name('pivot_overlap_worker.py')), str(tmp_path)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    values = list(SigmondLoader(tmp_path / 'overlaps.hdf5', group='overlaps').observables)
    reference = read_pivot(tmp_path / 'pivot.hdf5')
    row_order = [reference['pivot_labels'].index(op) for op in matrix.operators]
    buffer = np.empty((3, 3, 13))
    for value in values:
        level = int(value.observable_info.name.removeprefix('Z2oracle'))
        buffer[value.observable_info.index, level] = value.data
    expected = p.overlaps(amps, energies=energies)
    np.testing.assert_allclose(expected.data, buffer[row_order], rtol=1e-9, atol=1e-10)
    import xml.etree.ElementTree as ET

    from sigmondsamplings.correlators import operator_from_xml
    table = ET.parse(tmp_path / 'overlap_table.xml').getroot()
    normalized = p.overlaps(amps, energies=energies, normalize=True)
    for node in table.findall('.//OperatorZMagnitudeSquares'):
        op = operator_from_xml(next(e for e in node if e.tag.endswith('OperatorString')))
        for cell in node.findall('ZMagSquare'):
            level = int(cell.findtext('Level'))
            ss = normalized.sel(operator=op.coordinate).isel(level=level)
            assert ss.full_sample_value == pytest.approx(float(cell.findtext('.//FullEstimate')), rel=2e-10, abs=1e-12)
            assert ss.error == pytest.approx(float(cell.findtext('.//SymmetricError')), rel=2e-10, abs=1e-12)



@pytest.mark.sigmond_oracle
@pytest.mark.parametrize('case', ['reverse', 'mask', 'metric_cut', 'improved'])
def test_live_basis_masks_cuts_and_reverse(tmp_path, sigmond_binary, case):
    import warnings

    from sigmondsamplings.correlators import GenIrrepOperator, PivotWarning
    m, _ = spectrum(complex_values=True, noise=.0001)
    options = dict(norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-8)
    task = {}
    source = None
    if case == 'reverse':
        options.update(metric_time=3, diagonalize_time=1)
    elif case == 'mask':
        options.update(set_imaginary_parts_zero=True, zero_elements=[(m.operators[0], m.operators[1])])
        task.update(set_imaginary_parts_zero=True, zero_elements=options['zero_elements'])
    elif case == 'metric_cut':
        options['min_inv_condition'] = .02
    else:
        source = m
        op = GenIrrepOperator.from_channel(m.info.channel, name='IMP', index=0)
        m = m.improve([m.operators[0], op], {op: {m.operators[1]: 1, m.operators[2]: .3j}})
        task['input_matrix'] = source
    root, path, output = rotate(m, tmp_path, metric_time=options['metric_time'],
                               diagonalize_time=options['diagonalize_time'], cutoff=options['min_inv_condition'], **task)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PivotWarning)
        p = SinglePivot.from_matrix(m, **options)
    reference = read_pivot(path)
    if source is None:
        compare_reference(m, p, reference, output)
    else:
        R = reference['transformation'][[reference['source_labels'].index(op) for op in source.operators]]
        ss = p.source_transformation.coefficients @ p.transformation
        phase = np.sum(R.conj() * ss, axis=0)
        phase /= abs(phase)
        np.testing.assert_allclose(ss, R * phase, rtol=1e-9, atol=1e-10)
        Z = reference['zmatrix'][[reference['pivot_labels'].index(op) for op in m.operators]]
        np.testing.assert_allclose(p.zmatrix, Z * phase, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('case', REFERENCE_CASES)
def test_tracked_pivot_goldens(case):
    import hashlib
    import json
    directory = Path(__file__).with_name('fixtures') / 'pivot' / case
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['sigmond_commit'] == '926e10ee52a2becef96c2f8ec96a729487089d55'
    for name, digest in metadata['artifacts'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    matrix, comparison, options, _ = reference_fixture(metadata['case'])
    import warnings

    from sigmondsamplings.correlators import PivotWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PivotWarning)
        p = SinglePivot.from_matrix(matrix, **options)
    reference = read_pivot(directory / 'pivot.hdf5')
    compare_reference(comparison, p, reference, directory / 'rotated.hdf5')


def compare_build_logs(root, matrix, pivot):
    from sigmondsamplings.correlators import operator_from_xml
    for node in root.findall('.//DiagonalCorrelatorFractionalErrors/DiagonalCorrelator'):
        op = operator_from_xml(node[0])
        i = pivot.operators.index(op)
        for row, tag in enumerate(('MetricTimeFractionalError', 'MatrixTimeFractionalError')):
            assert pivot.diagonal_fractional_errors.values[row, i] == pytest.approx(float(node.findtext(tag)), rel=2e-9, abs=1e-12)
    samples = pivot.metric_eigenvalue_samples
    for node in root.findall('.//RescaledMetricConditionVariation/MetricEigenvalue'):
        i = int(node.findtext('Level'))
        value = samples.isel(eigenvalue=i)
        assert value.full_sample_value == pytest.approx(float(node.findtext('.//FullEstimate')), rel=2e-9, abs=1e-12)
        assert value.error == pytest.approx(float(node.findtext('.//SymmetricError')), rel=2e-7, abs=1e-12)


@pytest.mark.sigmond_oracle
def test_fullrank_upstream_null_flag_is_vacuous_in_ss(tmp_path, sigmond_binary):
    import warnings
    matrix, _ = spectrum(noise=.0001)
    root, path, output = rotate(matrix, tmp_path, check_null=True)
    with warnings.catch_warnings(record=True) as emitted:
        p = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                    min_inv_condition=1e-8)
    assert not emitted and p.gevp.null_space is None
    # Native setMatrix returns -5 here, which its task accepts and does not log.
    compare_reference(matrix, p, read_pivot(path), output)


@pytest.mark.sigmond_oracle
def test_native_consumer_reads_ss_rotated_cn_file(tmp_path, sigmond_bindings):
    import xml.etree.ElementTree as ET

    from sigmondsamplings.io.writer import SigmondWriter
    matrix, _ = spectrum()  # real buffer must still write CN Im keys
    p = SinglePivot.from_matrix(matrix, norm_time=1, metric_time=1, diagonalize_time=3,
                                min_inv_condition=1e-8)
    rotated = p.rotate(matrix)
    path = tmp_path / 'ss_rotated.hdf5'
    rotated.write(path, diagonal_only_before_time=3, upper_triangle_only=True)
    root = ET.Element('SigMonD')
    init = ET.SubElement(root, 'Initialize')
    ET.SubElement(init, 'ProjectName').text = 'ss_downstream'
    ET.SubElement(init, 'LogFile').text = str(tmp_path / 'read.xml')
    header = ET.fromstring(SigmondWriter()._generate_header_xml(matrix.values.ensemble_info, matrix.sampling_info))
    for child in header:
        init.append(child)
    files = ET.SubElement(ET.SubElement(init, 'MCObservables'), 'SamplingData')
    ET.SubElement(files, 'FileName').text = f'{path}[samplings]'
    sig = sigmond_bindings
    xml = sig.XMLHandler()
    xml.set_from_string(ET.tostring(root, encoding='unicode'))
    handler = sig.TaskHandler(xml)
    for i, op in enumerate(rotated.operators):
        operator = sig.OperatorInfo(sig.GenIrrepOperatorInfo(str(op)))
        for t in rotated.times:
            key = sig.MCObsInfo(operator, operator, t, True, sig.ComplexArg.RealPart, False)
            data = handler.getMCObsHandler().getFullAndSamplingValues(key, sig.SamplingMode.Jackknife)
            np.testing.assert_array_equal(np.asarray(data.array()), rotated.values.data[t, i, i])
