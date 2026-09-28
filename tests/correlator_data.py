"""Synthetic correlator/VEV observables for the correlator tests (built in memory)."""

from __future__ import annotations

import numpy as np

from sigmondsamplings.bins import SigmondBins
from sigmondsamplings.correlators import (
    BasicOperator,
    CorrelatorKey,
    CorrelatorPair,
    VEVKey,
)
from sigmondsamplings.correlators.observables import observable_info_for
from sigmondsamplings.info import EnsembleInfo, SamplingInfo
from sigmondsamplings.sampling import SigmondSampling

ENS = EnsembleInfo("ens_corr", 40)
JACK = SamplingInfo("jackknife", 40)
BOOT = SamplingInfo("bootstrap", 25, seed=7)
N_BINS = 40

BL = BasicOperator.from_sigmond
OPS = (
    BL("pion P=(0,0,0) A1um_1 SD_5"),
    BL("pion P=(0,0,0) A1um_1 SS_0"),
    BL("isotriplet_pion_pion A1um_1 [P=(0,0,1) A2m SS_0] [P=(0,0,-1) A2m SS_0]"),
)


def length(sampling_info: SamplingInfo | None) -> int:
    return N_BINS if sampling_info is None else sampling_info.num_resamplings + 1


def series(rng, sampling_info=JACK, complex_=True, loc=0.0) -> np.ndarray:
    n = length(sampling_info)
    data = loc + rng.normal(size=n)
    if complex_:
        data = data + 1j * rng.normal(size=n)
    return data


def ckey(i, j, t, herm=False, subvev=False, ops=OPS) -> CorrelatorKey:
    return CorrelatorKey(CorrelatorPair(ops[i], ops[j]), t, herm, subvev)


def observable(key, data, sampling_info=JACK, ens=ENS):
    """A stored observable named by ``key`` (complex if ``data`` is), as the loader builds it."""
    info = observable_info_for(key.to_sigmond(), "re", ens)
    is_complex = bool(np.iscomplexobj(data))
    if sampling_info is None:
        return SigmondBins(np.asarray(data), info, is_complex=is_complex)
    return SigmondSampling(np.asarray(data), info, sampling_info, is_complex=is_complex)


def vev(op, data, sampling_info=JACK, ens=ENS):
    return observable(VEVKey(op), data, sampling_info, ens)
