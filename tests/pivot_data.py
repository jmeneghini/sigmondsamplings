"""Small exact/noisy spectra shared by unit and reference tests."""
import numpy as np

from sigmondsamplings.correlators import CorrelatorMatrix, CorrelatorMatrixInfo, GenIrrepOperator
from sigmondsamplings.correlators.matrix import _matrix_infos, _vev_infos
from sigmondsamplings.data import SigmondData
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.sampling_array import ArrayObsInfo


def spectrum(*, complex_values=False, bins=False, connected=False, vevs=False, noise=0,
             averaged=False, n=3, bootstrap=False):
    ens = EnsembleInfo("ss_pivot_test", 12, spatial_extent=8, temporal_extent=16, num_streams=1)
    plan = None if bins else (SamplingInfo.bootstrap(12, seed=271) if bootstrap else SamplingInfo.jackknife_for(ens))
    length = 12 if bins else 13
    prefix = "PSQ=0" if averaged else "P=(0,0,0)"
    ops = tuple(GenIrrepOperator.from_sigmond(f"isosinglet S=0 {prefix} A1g_1 TEST {i}") for i in range(n))
    times = list(range(7))
    rng = np.random.default_rng(271)
    z = rng.normal(size=(n, n)) + 2 * np.eye(n)
    if complex_values:
        z = z + .3j * rng.normal(size=(n, n))
    energies = .2 + .15 * np.arange(n)
    buf = np.einsum('in,tn,jn->tij', z, np.exp(-np.outer(times, energies)), z.conj())
    data = np.repeat(buf[..., None], length, axis=-1)
    perturb = rng.normal(size=data.shape)
    if complex_values:
        perturb = perturb + 1j * rng.normal(size=data.shape)
    perturb = (perturb + perturb.swapaxes(1, 2).conj()) / 2
    if not bins:
        perturb[..., 0] = 0
    data = data + noise * perturb
    v = rng.normal(size=(n, length)) * .03 + .2
    if complex_values:
        v = v + 1j * rng.normal(size=v.shape) * .01
    if not connected and vevs:
        data = data + v[None, :, None, :] * v[None, None, :, :].conj()
    info = CorrelatorMatrixInfo(ops, True, connected)
    coords = {'time': times, 'sink': list(info.coordinates), 'source': list(info.coordinates)}
    values = SigmondData.from_data(data, ArrayObsInfo('spectrum', ensemble_info=ens), plan,
                                  dim_names=('time', 'sink', 'source'), coords=coords,
                                  is_complex=complex_values,
                                  infos=_matrix_infos(ops, times, True, connected, ens))
    vev_data = None
    if vevs:
        vev_data = SigmondData.from_data(v, ArrayObsInfo('vevs', ensemble_info=ens), plan,
                                       dim_names=('operator',), coords={'operator': list(info.coordinates)},
                                       is_complex=complex_values, infos=_vev_infos(ops, ens))
    return CorrelatorMatrix(info, values, vev_data), energies


def connected_case(mode, *, complex_values=True):
    """Connected samplings (resampling bins jointly with their VEVs) and the raw input."""
    from sigmondsamplings.correlators.matrix import _joint_resample
    from sigmondsamplings.data import _kind_cls
    raw, _ = spectrum(complex_values=complex_values, vevs=True, noise=.0001, bins=mode == 'bins')
    if raw.is_bins:
        plan = SamplingInfo.jackknife_for(raw.values.ensemble_info)
        c, v = _joint_resample([raw.values.data, raw.vevs.data], raw.values.ensemble_info, plan)
        def wrap(original, buffer):
            return _kind_cls(plan)._from_parts(buffer, original.observable_info, plan,
                                              original.is_complex, original.dims, original.coords, original.infos)
        samplings = raw._replace(values=wrap(raw.values, c), vevs=wrap(raw.vevs, v))
    else:
        samplings = raw
    v = samplings.vevs.data
    connected = samplings._replace(info=type(raw.info)(raw.operators, True, True),
                                  values=samplings.values - v[None, :, None, :] * v[None, None, :, :].conj())
    return connected, raw


REFERENCE_CASES = ('cn_real', 'cn_complex', 'cn_connected_s', 'cn_connected_u',
                   'cn_connected_a', 'cn_connected_b', 'cn_reverse', 'cn_mask',
                   'cn_metric_cut', 'cn_improved', 'cn_bootstrap',
                   'rn_real', 'rn_reverse', 'rn_metric_cut', 'rn_improved', 'rn_bootstrap',
                   'rn_connected_s', 'rn_connected_u', 'rn_connected_a', 'rn_connected_b')


def reference_fixture(case):
    """Build the recorded deterministic fixture and its native task options."""
    options = dict(norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-8)
    real = case.startswith('rn_')
    task = {'numeric_type': 'real'} if real else {}
    if real:
        options['numeric_type'] = 'real'
    kind = case[3:]
    m, _ = spectrum(complex_values=not real and kind != 'real', noise=.0001,
                    bootstrap=kind == 'bootstrap')
    compare_matrix = m
    if kind.startswith('connected_'):
        mode = {'s': 'samplings', 'u': 'samplings_unsubt', 'a': 'samplings_all', 'b': 'bins'}[case[-1]]
        m, raw = connected_case(mode, complex_values=not real)
        task.update(rotate_mode=mode, input_matrix=raw)
        compare_matrix = m if mode == 'samplings' else raw
    elif kind == 'reverse':
        options.update(metric_time=3, diagonalize_time=1)
    elif kind == 'mask':
        options.update(set_imaginary_parts_zero=True, zero_elements=[(m.operators[0], m.operators[1])])
        task.update(set_imaginary_parts_zero=True, zero_elements=options['zero_elements'])
    elif kind == 'metric_cut':
        options['min_inv_condition'] = .02
    elif kind == 'improved':
        original = m
        op = GenIrrepOperator.from_channel(m.info.channel, name='IMP', index=0)
        m = m.improve([m.operators[0], op], {op: {m.operators[1]: 1, m.operators[2]: .3 if real else .3j}})
        compare_matrix = m
        task['input_matrix'] = original
    return m, compare_matrix, options, task
