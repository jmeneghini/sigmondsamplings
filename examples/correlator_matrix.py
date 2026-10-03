"""Read a two-operator matrix from corrs.hdf5 (no fitting dependency)."""
from sigmondsamplings import SigmondLoader

loader = SigmondLoader("corrs.hdf5", lazy=True)
at_t8 = loader.observables.filter(obs_kind="correlator", time=8)
index = loader.correlators
ops = ["pion P=(0,0,0) A1um_1 SD_5 D4", "pion P=(0,0,0) A1um_1 SS_1"]
matrix = index.matrix(ops, times=range(4, 21))
c01 = matrix.correlator(ops[0], ops[1])
print(c01.sel(time=8).full_sample_value)
