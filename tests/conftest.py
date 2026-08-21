"""Shared synthetic fixtures.

Built in memory rather than read from ``tests/data``, which is gitignored, so these
run on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigmondsamplings.ensemble_collection import SingleEnsembleCollection
from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling

# Three interacting levels across two sectors, two single hadrons to serve as
# reference particles, and one name that cannot be read as an energy level at all.
OBSERVABLE_NAMES = (
    "PSQ0_A1g_elab_0",
    "PSQ0_A1g_elab_1",
    "PSQ1_E_elab_0",
    "PSQ0_N",
    "PSQ0_pi",
    "junk_obs",
)


def make_collection(names: tuple[str, ...] = OBSERVABLE_NAMES) -> SingleEnsembleCollection:
    """A single-ensemble collection of plain (untagged) observables named *names*."""
    ensemble = EnsembleInfo("ens_A", 100)
    # bootstrap: the dataframe/display path reports confidence intervals, which
    # SigmondSampling only defines for bootstrap resampling.
    sampling_info = SamplingInfo("bootstrap", 20, seed=42)
    rng = np.random.default_rng(0)

    samplings = []
    for index, name in enumerate(names):
        obs_info = ObservableInfo(
            name=name, index=0, op_type="n", re_im="re", ensemble_info=ensemble
        )
        data = (1.0 + 0.3 * index) + 0.01 * rng.standard_normal(21)
        samplings.append(SigmondSampling(data, obs_info, sampling_info))
    return SingleEnsembleCollection(samplings)


@pytest.fixture
def observables() -> SingleEnsembleCollection:
    """A fresh untagged collection. Function-scoped: edit ops mutate in place."""
    return make_collection()


@pytest.fixture
def samplings_file(tmp_path, observables):
    """The same collection written to an HDF5 samplings file."""
    path = tmp_path / "samplings.hdf5"
    observables.to_hdf5(str(path), overwrite=True, mode="w")
    return path
