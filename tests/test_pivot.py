from copy import deepcopy

import numpy as np
import pytest
from pivot_data import spectrum

from sigmondsamplings.correlators import (
    GenIrrepOperator,
    PivotError,
    PivotWarning,
    SinglePivot,
    solve_gevp,
)
from sigmondsamplings.info import ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling


def _canonical(matrix):
    """Sigmond's HermMat reading of ``matrix``: upper triangle and real diagonal."""
    return matrix._transform(np.eye(len(matrix)), matrix.operators, canonical_hermitian=True)


def build(matrix, **kwargs):
    defaults = dict(norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-8)
    defaults.update(kwargs)
    return SinglePivot.from_matrix(matrix, **defaults)


@pytest.mark.parametrize('numeric_type', ['real', 'complex'])
@pytest.mark.parametrize('rank', [2, 3])
def test_solver_identities(numeric_type, rank):
    B = np.diag([.0 if rank == 2 else .4, 1., 2.])
    A = np.diag([0. if rank == 2 else .2, .7, .3])
    result = solve_gevp(A, B, min_inv_condition=.001, numeric_type=numeric_type)
    Y, Z = result.eigenvectors, result.zmatrix
    np.testing.assert_allclose(Y.T @ B @ Y, np.eye(rank), atol=1e-14)
    np.testing.assert_allclose(Y.T @ A @ Y, np.diag(result.eigenvalues), atol=1e-14)
    np.testing.assert_allclose(Z, B @ Y)
    assert result.reconstruction_residual < 1e-14
    assert (result.null_space is None) == (rank == 3)
    if rank == 2:
        assert result.null_space.ok
        np.testing.assert_allclose(result.null_space.projections, [1])
    assert not result.eigenvectors.flags.writeable
    with pytest.raises(ValueError):
        result.eigenvectors[0, 0] = 1


def test_two_cuts_and_reconstruction_residual():
    result = solve_gevp(np.diag([.01, 1]), np.eye(2), min_inv_condition=.1)
    assert result.metric_rank == 2 and result.rank == 1
    assert result.reconstruction_residual > 0
    with pytest.warns(PivotWarning, match='null space'):
        result = solve_gevp(np.eye(2), np.diag([.01, 1]), min_inv_condition=.1)
    assert not result.null_space.ok


def test_cn_rn_floor_and_cut_equality():
    cn = solve_gevp(np.diag([.001, .01]), np.diag([.01, .02]), min_inv_condition=.1)
    assert cn.metric_rank == 2
    with pytest.raises(PivotError, match='metric: all'):
        solve_gevp(np.diag([.001, .01]), np.diag([.01, .02]), min_inv_condition=.1, numeric_type='real')
    result = solve_gevp(np.diag([.5, 1]), np.diag([.5, 1]), min_inv_condition=.5)
    assert result.metric_rank == 2 and result.rank == 2


@pytest.mark.parametrize('A,B,cut,alarm,message', [
    (np.eye(2), np.diag([0., 1]), 0, None, 'retained zero'),
    (np.eye(2), np.diag([-.01, 1]), .001, None, 'metric:'),
    (np.diag([-.01, 1]), np.eye(2), .001, None, 'matrix:'),
    (np.eye(2), np.eye(2), 2, None, 'all'),
    (np.eye(2), np.eye(2), -1, None, 'nonnegative'),
    (np.eye(2), np.eye(2), 0, .1, 'nonpositive'),
    (np.eye(2), np.ones((2, 3)), 0, None, 'square'),
    (np.eye(2), np.eye(3), 0, None, 'equal shapes'),
    (np.array([[1., 2], [0, 1]]), np.eye(2), 0, None, 'Hermitian'),
    (np.full((2, 2), np.nan), np.eye(2), 0, None, 'finite'),
])
def test_solver_errors(A, B, cut, alarm, message):
    with pytest.raises(PivotError, match=message):
        solve_gevp(A, B, min_inv_condition=cut, negative_eigenvalue_alarm=alarm)


def test_zero_g_and_lapack_failure(monkeypatch):
    with pytest.warns(PivotWarning, match='maximum is zero'):
        result = solve_gevp(np.zeros((2, 2)), np.eye(2), min_inv_condition=0)
    assert np.isnan(result.inverse_condition) and result.rank == 2
    def fail(*args):
        raise np.linalg.LinAlgError('injected')
    monkeypatch.setattr(np.linalg, 'eigh', fail)
    with pytest.raises(PivotError, match='injected'):
        solve_gevp(np.eye(2), np.eye(2), min_inv_condition=0)


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_exact_spectrum_rotation_and_overlaps(complex_values, reverse):
    m, energies = spectrum(complex_values=complex_values)
    t0, td = (3, 1) if reverse else (1, 3)
    p = build(m, metric_time=t0, diagonalize_time=td)
    rot = p.rotate(m)
    np.testing.assert_allclose(rot.at(t0).data[..., 0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rot.at(td).data[..., 0], np.diag(p.gevp.eigenvalues), atol=1e-12)
    ordered_energies = energies[::-1] if reverse else energies
    expected = np.exp(-np.outer(m.times, ordered_energies) + ordered_energies * t0)
    np.testing.assert_allclose(rot.diagonal().data[..., 0], expected, atol=1e-12)
    def samplings(vals):
        return {op: SigmondSampling(np.full(13, value), ObservableInfo(op.coordinate, ensemble_info=m.values.ensemble_info), m.sampling_info)
                for op, value in zip(p.rotated_operators, vals)}
    amps = samplings(np.exp(ordered_energies * t0))
    z = p.overlaps(amps, energies=samplings(ordered_energies))
    np.testing.assert_allclose(z.data[..., 0], abs(p.zmatrix[:, ::-1] if reverse else p.zmatrix)**2 * np.exp(energies * t0))
    normalized = p.overlaps(amps, normalize=True)
    np.testing.assert_allclose(normalized.data.sum(axis=1), 1)
    assert len(normalized.to_collection()) == 9


def test_rotation_commutes_with_resampling():
    from sigmondsamplings.bins import _block_resample
    m, _ = spectrum(bins=True, noise=.002)
    p = build(m)
    raw_rot = p.rotate(m)
    sampled = m._replace(values=m.values.resample(SamplingInfo.jackknife_for(m.values.ensemble_info)))
    np.testing.assert_allclose(p.rotate(sampled).values.data,
                               _block_resample(raw_rot.values.data, m.values.ensemble_info, sampled.sampling_info, ['mean'])[0], atol=1e-12)
    assert p.metric_eigenvalue_samples is None


@pytest.mark.parametrize('complex_values', [False, True])
def test_connected_workflows(complex_values):
    connected, _ = spectrum(connected=True, vevs=True, complex_values=complex_values)
    p = build(connected)
    raw = connected._replace(info=type(connected.info)(connected.operators, True, False),
                             values=connected.values + connected.vevs.data[None, :, None, :] * connected.vevs.data[None, None, :, :].conj())
    s, u = p.rotate(connected), p.rotate(raw)
    assert s.vevs is None and not u.vevs.is_complex
    w = p.transformation.conj().T @ connected.vevs.data
    np.testing.assert_allclose(w[:, 0].imag, 0, atol=1e-13)
    assert np.all(w[:, 0].real >= 0)
    np.testing.assert_allclose(u.values.data - u.vevs.data[None, :, None, :] * u.vevs.data[None, None, :, :],
                               s.values.data + w[None, :, None, :] * w[None, None, :, :].conj() - w.real[None, :, None, :] * w.real[None, None, :, :], atol=1e-12)
    p.off_diagonal_check(u)
    raw_pivot = build(raw)
    assert raw_pivot.rotate(raw).vevs is None


def test_improved_and_source_basis_equivalence_and_averaging():
    m, _ = spectrum(complex_values=True, averaged=True)
    new = GenIrrepOperator.from_channel(m.info.channel, name='IMP', index=0)
    improved = m.improve([m.operators[0], new], {new: {m.operators[1]: 1, m.operators[2]: .3j}})
    p = build(improved)
    np.testing.assert_allclose(p.rotate(improved).values.data, p.rotate(m).values.data, atol=1e-12)
    repeated = improved.transform(np.array([[1., .2], [0., 1.]]))
    pp = build(repeated)
    np.testing.assert_allclose(pp.rotate(m).values.data, pp.rotate(repeated).values.data, atol=1e-12)
    # A GI source label may also name a newly defined GI operator.
    m.transform(np.ones((3, 1)), [m.operators[0]])


def test_masks_canonicalization_and_ownership(tmp_path):
    m, _ = spectrum(complex_values=True, noise=.0001)
    original = m.values.data.copy()
    p = build(m, set_imaginary_parts_zero=True, zero_elements=[(m.operators[0], m.operators[1])])
    np.testing.assert_array_equal(m.values.data, original)
    assert p.summary()['set_imaginary_parts_zero']
    np.testing.assert_allclose(p.rotate(m).values.data, m.transform(p.transformation, p.rotated_operators).values.data)
    report = p.summary()
    report['operators'].clear()
    assert len(p.summary()['operators']) == 3
    meta = p.ensemble_info
    meta.name = 'changed'
    assert p.ensemble_info.name != 'changed'
    metric = p.metric_eigenvalue_samples
    metric.data[:] = 0
    assert np.any(p.metric_eigenvalue_samples.data)
    p.write(tmp_path / 'pivot.hdf5')
    with pytest.raises(FileExistsError):
        p.write(tmp_path / 'pivot.hdf5')
    p.write(tmp_path / 'pivot.hdf5', group='channel/two')
    q = SinglePivot.read(tmp_path / 'pivot.hdf5')
    assert q.summary() == p.summary()
    np.testing.assert_array_equal(q.transformation, p.transformation)
    np.testing.assert_array_equal(q.rotate(m).values.data, p.rotate(m).values.data)
    m.values.data[:, 0, 0] += 4j
    m.values.data[:, 1, 0] = 99
    canonical = _canonical(m)
    assert np.all(canonical.values.data[:, 0, 0].imag == 0)
    np.testing.assert_array_equal(canonical.values.data[:, 1, 0], original[:, 0, 1].conj())
    expected = _canonical(canonical.transform(p.transformation, p.rotated_operators))
    rotated = p.rotate(m)
    np.testing.assert_allclose(rotated.values.data, expected.values.data, rtol=1e-12, atol=1e-12)
    assert np.all(rotated.values.data[:, np.arange(p.gevp.rank), np.arange(p.gevp.rank)].imag == 0)


def test_rotation_validation_and_connected_bins():
    m, _ = spectrum()
    p = build(m)
    with pytest.raises(PivotError, match='complete'):
        p.rotate(m.select(m.operators[:2]))
    ens = deepcopy(m.values.ensemble_info)
    ens.temporal_extent += 1
    values = deepcopy(m.values.observable_info)
    values.ensemble_info = ens
    original = m.values.observable_info
    m.values.observable_info = values
    with pytest.raises(PivotError, match='ensemble'):
        p.rotate(m)
    m.values.observable_info = original
    bins, _ = spectrum(bins=True, connected=True)
    with pytest.raises(PivotError, match='connected bins'):
        build(bins)
    with pytest.raises(PivotError, match='distinct'):
        build(m, diagonalize_time=1)
    with pytest.raises(PivotError, match='unavailable'):
        build(m, diagonalize_time=7)


def test_metric_resamples_and_invalid_diagnostics():
    m, _ = spectrum(noise=.001)
    m.values.data[1, 0, 0, 2] = 0
    with pytest.warns(PivotWarning, match='invalid samplings'):
        p = build(m)
    assert np.isnan(p.metric_eigenvalue_samples.data[:, 2]).all()
    assert p.summary()['invalid_metric_samples'] == [2]


def test_overlaps_validation_and_stable_ties():
    m, _ = spectrum()
    p = build(m)
    values = {op: SigmondSampling(np.arange(13) + 1., ObservableInfo('A', ensemble_info=m.values.ensemble_info), m.sampling_info)
              for op in p.rotated_operators}
    duplicate = dict(values)
    duplicate[p.rotated_operators[0].coordinate] = next(iter(values.values()))
    with pytest.raises(PivotError, match='duplicate'):
        p.overlaps(duplicate)
    with pytest.raises(PivotError, match='missing'):
        p.overlaps({})
    z = p.overlaps(values, energies={op: v * 0 + 1 for op, v in values.items()}, normalize=True)
    assert list(z.coords['level']) == [op.coordinate for op in p.rotated_operators]
    np.testing.assert_allclose(z.data[..., 1], z.data[..., 0] * 2)
    for v in values.values():
        v.data[:] = 0
    with pytest.raises(PivotError, match='row'):
        p.overlaps(values, normalize=True)


def test_offdiagonal_empty_and_single_level():
    m, _ = spectrum(n=1)
    p = build(m)
    result = p.off_diagonal_check(p.rotate(m))
    assert result.summary()['n_entries'] == 0
    assert np.isnan(result.max_ratio)
    assert np.isnan(result.percent_beyond[1])
    assert len(p.off_diagonal_check(p.rotate(m), times=[]).by_time.time) == 0


def test_export_filter():
    m, _ = spectrum()
    p = build(m)
    rot = p.rotate(m)
    assert len(rot.to_collection(remove_off_diagonal=True)) == 7 * 3
    assert len(rot.to_collection(diagonal_only_before_time=3)) == 3 * 3 + 4 * 9


def test_overlap_writer_roundtrip(tmp_path):
    from sigmondsamplings.correlators.observables import OverlapObsInfo
    from sigmondsamplings.io.loader import SigmondLoader
    from sigmondsamplings.io.writer import SigmondWriter
    m, _ = spectrum()
    p = build(m)
    amps = {op: SigmondSampling(np.ones(13), ObservableInfo('A', ensemble_info=m.values.ensemble_info), m.sampling_info)
            for op in p.rotated_operators}
    overlaps = p.overlaps(amps, normalize=True)
    path = tmp_path / 'overlaps.hdf5'
    SigmondWriter().write_hdf5(str(path), list(overlaps.to_collection()), overwrite=True)
    loaded = list(SigmondLoader(path).observables)
    assert len(loaded) == 9
    for value in loaded:
        info = value.observable_info
        assert isinstance(info, OverlapObsInfo)
        assert info.normalize and info.norm_time == 1
        original = overlaps.sel(operator=info.operator.coordinate, level=info.level.coordinate)
        np.testing.assert_array_equal(value.data, original.data)
        assert info.name == f'Z2N_{p.identifier}_{info.level.index}'
        assert info.index == p.operators.index(info.operator)
        assert 3 <= len(info.name) <= 64 and ' ' not in info.name  # Sigmond's name rules
    assert len({(v.observable_info.name, v.observable_info.index) for v in loaded}) == 9
    # Another basis in the same channel reuses the ROT labels but not the names.
    other = build(m.select(m.operators[:2]))
    assert other.identifier != p.identifier
    assert SinglePivot.read(_written(p, tmp_path)).identifier == p.identifier


def _written(pivot, tmp_path):
    path = tmp_path / 'pivot.hdf5'
    pivot.write(path, overwrite=True)
    return path


def test_offdiagonal_components_edges_invalid_and_zero_error():
    m, _ = spectrum(n=2)
    p = build(m)
    rot = p.rotate(m)
    rot.values.data[:] = 0
    rot.values.data[:, 0, 1] = rot.values.data[:, 1, 0] = 1
    result = p.off_diagonal_check(rot, times=[3])
    assert result.by_time.n_entries.item() == 2  # CN, including the zero Im part
    assert np.isinf(result.max_ratio)
    assert result.percent_beyond[4] == 50
    rot.values.data[3, 0, 1, 1] = np.nan
    result = p.off_diagonal_check(rot, times=[3])
    assert result.by_time.n_invalid.item() == 1
    assert np.isnan(result.percent_beyond[1])


def test_plain_selected_labels_and_nonhermitian_transform():
    from correlator_data import OPS

    from sigmondsamplings.correlators.matrix import _matrix_infos
    m, _ = spectrum(complex_values=True)
    info = type(m.info)(OPS)
    original = m.values
    from sigmondsamplings.data import _kind_cls
    values = _kind_cls(original.sampling_info)._from_parts(original.data.copy(), original.observable_info,
                                                        original.sampling_info, True, original.dims,
                                                        {'time': list(m.times), 'sink': list(info.coordinates), 'source': list(info.coordinates)},
                                                        _matrix_infos(OPS, m.times, True, False, m.values.ensemble_info))
    m = m._replace(info=info, values=values)
    selected = m.transform(np.eye(3)[:, :1], [OPS[0]])
    assert selected.operators == (OPS[0],)
    with pytest.raises(ValueError, match='unit selection'):
        m.transform(np.ones((3, 1)), [OPS[0]])
    m.values.data[:, 1, 0] += 3j
    m = m._replace(info=type(info)(OPS, False))
    T = np.array([[1, .5j], [0, 1], [1j, 0]])
    result = m.transform(T)
    expected = np.stack([T.conj().T @ m.values.data[..., sample] @ T for sample in range(13)], axis=-1)
    np.testing.assert_allclose(result.values.data, expected)


def test_persistence_overwrite_schema_and_inconsistent_fields(tmp_path):
    import h5py
    m, _ = spectrum()
    p = build(m)
    path = tmp_path / 'pivot.hdf5'
    p.write(path)
    p.write(path, overwrite=True)
    with h5py.File(path, 'a') as file:
        assert list(file) == ['pivot']
        del file['pivot/column_phases']
        file['pivot/column_phases'] = np.ones(1)
    with pytest.raises(PivotError, match='column_phases'):
        SinglePivot.read(path)
    with h5py.File(path, 'a') as file:
        file['pivot'].attrs['ss_pivot_schema'] = 900
    with pytest.raises(PivotError, match='schema'):
        SinglePivot.read(path)
    with pytest.raises(PivotError, match='invalid SS pivot file'):
        SinglePivot.read(path, group='missing')


def test_phase_zero_and_real_sign():
    m, _ = spectrum(connected=True, vevs=True)
    m.vevs.data[:] = 0
    with pytest.warns(PivotWarning, match='zero rotated VEV'):
        p = build(m)
    assert p.vev_rephased
    np.testing.assert_array_equal(p.column_phases, 1)
    p = build(m, numeric_type='real')  # RN's sign rule does not warn
    np.testing.assert_array_equal(p.column_phases, 1)
    m.vevs.data[:] = -.1
    p = build(m, numeric_type='real')
    assert np.all((p.transformation.T @ m.vevs.data)[:, 0] >= 0)


def test_negative_normalization_and_zero_scale():
    m, _ = spectrum()
    m.values.data[0, 0, 0] *= -1
    with pytest.warns(PivotWarning, match='negative normalization'):
        p = build(m, norm_time=0)
    assert p.norm_diagonal[0] < 0
    m.values.data[0, 0, 0] = 0
    with pytest.raises(PivotError, match='nonzero'):
        build(m, norm_time=0)


def test_raw_bin_connected_check_joint_resampling():
    from pivot_data import connected_case
    connected, raw = connected_case('bins')
    p = build(connected)
    result = p.off_diagonal_check(p.rotate(raw))
    assert result.summary()['n_entries'] == 4 * 6


def test_sigma_bucket_equality_and_rn_component_count():
    from sigmondsamplings.analysis import analyzer_for
    m, _ = spectrum(n=2)
    p = build(m, numeric_type='real')
    rot = p.rotate(m)
    resamples = np.tile([-1., 1.], 6)
    error = analyzer_for(m.sampling_info).error(np.r_[0., resamples])
    for t, sigma in zip(range(3, 7), range(1, 5)):
        rot.values.data[t, 0, 1] = rot.values.data[t, 1, 0] = np.r_[sigma * error, resamples]
    result = p.off_diagonal_check(rot)
    assert result.summary()['n_entries'] == 4
    for sigma in range(1, 5):
        assert result.percent_beyond[sigma] == (4 - sigma) * 25


def test_degenerate_projector_and_basis_permutation():
    rng = np.random.default_rng(311)
    basis = rng.normal(size=(4, 4))
    B = basis @ basis.T + np.eye(4)
    A = .5 * B
    result = solve_gevp(A, B, min_inv_condition=.0001)
    permutation = [3, 1, 0, 2]
    other = solve_gevp(A[np.ix_(permutation, permutation)], B[np.ix_(permutation, permutation)],
                       min_inv_condition=.0001)
    y = other.eigenvectors[np.argsort(permutation)]
    np.testing.assert_allclose(result.eigenvectors @ result.eigenvectors.T, y @ y.T, atol=1e-13)
    np.testing.assert_allclose(result.eigenvalues, other.eigenvalues, atol=1e-14)


def test_cn_real_buffer_export_explicit_imaginary_keys(tmp_path):
    import h5py

    from sigmondsamplings.io.loader import SigmondLoader
    m, _ = spectrum()
    p = build(m)
    rot = p.rotate(m)
    assert not np.iscomplexobj(rot.values.data)  # no numerical dtype inference
    path = tmp_path / 'rotated.hdf5'
    rot.write(path, upper_triangle_only=True, diagonal_only_before_time=3)
    with h5py.File(path) as file:
        assert len(file['samplings/Values']) == 2 * (3 * 3 + 4 * 6)
    loaded = SigmondLoader(path).correlators.matrix(p.rotated_operators, times=[3, 4, 5, 6])
    np.testing.assert_array_equal(loaded.values.data, rot.sel_times([3, 4, 5, 6]).values.data)


def test_solver_type_validation():
    with pytest.raises(PivotError, match='nonnegative'):
        solve_gevp(np.eye(2), np.eye(2), min_inv_condition='invalid')
    with pytest.raises(PivotError, match='numeric'):
        solve_gevp(np.full((2, 2), 'invalid'), np.eye(2), min_inv_condition=0)


@pytest.mark.parametrize('complex_values', [False, True])
def test_rotation_kernel_rectangular_nonhermitian_strided_samples(complex_values):
    from sigmondsamplings.correlators._kernels import rotate_matrix
    rng = np.random.default_rng(731)
    buffer = rng.normal(size=(8, 12, 12, 4002))
    coefficients = rng.normal(size=(12, 9))
    if complex_values:
        buffer = buffer + 1j * rng.normal(size=buffer.shape)
        coefficients = coefficients + 1j * rng.normal(size=coefficients.shape)
    buffer = buffer[..., ::2]
    before = buffer.copy()
    actual = rotate_matrix(buffer, coefficients)
    expected = np.einsum('ia,tijs,jb->tabs', coefficients.conj(), buffer, coefficients,
                         optimize=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(buffer, before)
    assert actual.flags.c_contiguous


def test_overlaps_from_fit_results():
    pytest.importorskip('optimagic')
    from sigmondsamplings.fitting import (
        CallableXModel,
        MinimizerConfig,
        ParamSetSpec,
        ParamSpec,
        SamplingFit,
    )
    m, _ = spectrum(noise=.0001)
    p = build(m)
    rot = p.rotate(m).sel_times([3, 4, 5, 6])
    model = CallableXModel(lambda t, theta: theta[0] * np.exp(-theta[1] * t), ('A', 'E'))
    params = ParamSetSpec(values={'A': ParamSpec(initial=1.0), 'E': ParamSpec(initial=0.3)}).resolve()
    fits = {op: SamplingFit.from_data(rot.correlator(op, op), model, params,
                                      minimizer=MinimizerConfig(algorithm='scipy_ls_trf'))
            .fit_resampled(progress=False)
            for op in p.rotated_operators}
    expected = p.overlaps({op: f.param('A') for op, f in fits.items()},
                          energies={op: f.param('E') for op, f in fits.items()}, normalize=True)
    actual = p.overlaps(fits, energy='E', normalize=True)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert list(actual.coords['level']) == list(expected.coords['level'])
    np.testing.assert_array_equal(p.overlaps(fits).data,
                                  p.overlaps({op: f.param('A') for op, f in fits.items()}).data)
    one = next(iter(fits))
    with pytest.raises(PivotError, match='mix'):
        p.overlaps({**fits, one: fits[one].param('A')})
    with pytest.raises(PivotError, match='fit parameter'):
        p.overlaps({op: f.param('A') for op, f in fits.items()}, energy='E')
    with pytest.raises(PivotError, match='not both'):
        p.overlaps(fits, energy='E', energies={op: f.param('E') for op, f in fits.items()})
    with pytest.raises(PivotError, match="no parameter 'B'"):
        p.overlaps(fits, amplitude='B')
