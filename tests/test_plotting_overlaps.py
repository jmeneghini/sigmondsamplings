import numpy as np
import pytest
from pivot_data import spectrum

from sigmondsamplings.correlators import SinglePivot
from sigmondsamplings.info import ObservableInfo
from sigmondsamplings.sampling import SigmondSampling


def test_overlap_plots_preserve_conventions_and_labels():
    import slat.plotting as slp
    from sigmondsamplings.plotting import OverlapPlotter
    m, _ = spectrum()
    p = SinglePivot.from_matrix(m, norm_time=1, metric_time=1, diagonalize_time=3, min_inv_condition=1e-8)
    amps = {op: SigmondSampling(np.ones(13), ObservableInfo('A', ensemble_info=m.values.ensemble_info), m.sampling_info)
            for op in p.rotated_operators}
    z = p.overlaps(amps, normalize=True)
    plotter = OverlapPlotter(z)
    with slp.record() as recorder:
        plotter.plot_bars(m.operators[0])
        plotter.plot_heatmap()
    assert any(mark.kind == 'bar' for mark in recorder.marks)
    assert any(mark.kind == 'heatmap' for mark in recorder.marks)
    assert plotter.normalized
    with pytest.raises(ValueError, match='metadata'):
        from sigmondsamplings.data import SigmondData
        plain = SigmondData.from_data(z.data, "plain", z.sampling_info,
                                      dim_names=z.dims, coords=dict(z.coords))
        OverlapPlotter(plain)
