import numpy as np
import pytest

from sigmondsamplings.energy_levels import EnergyObsInfo, Particle, SHEnergyObsInfo
from sigmondsamplings.info import INDEP_ENSEMBLE, ObservableInfo, SamplingInfo
from sigmondsamplings.kinematics import TwoParticleKinem
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.utils import create_gaussian_sampling

MREF_L = 20.0
M1 = 1.0
M2 = 3.5


@pytest.fixture
def sampling_info():
    return SamplingInfo("bootstrap", 100, seed=42)


def make_sampling(mean, std, sampling_info, name):
    obs_info = ObservableInfo(name, 0, "n", "re", INDEP_ENSEMBLE)
    return create_gaussian_sampling(mean, std, sampling_info, obs_info)


def make_sh_mass(mean, std, sampling_info, particle):
    obs_info = SHEnergyObsInfo(psq=0, particle=particle)
    return create_gaussian_sampling(mean, std, sampling_info, obs_info)


class TestDvec:
    def test_optional_in_constructor(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        assert kinem.dvec is None

    def test_get_set(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 0, 1])
        assert np.array_equal(kinem.dvec, [0, 0, 1])

        kinem.dvec = (1, -1, 2)
        assert np.array_equal(kinem.dvec, [1, -1, 2])

        kinem.dvec = None
        assert kinem.dvec is None

    def test_accepts_integer_valued_floats(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=np.array([1.0, 0.0, 2.0]))
        assert kinem.dvec.dtype.kind == "i"
        assert kinem.dsq == 5

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError, match="integer"):
            TwoParticleKinem(MREF_L, M1, M2, dvec=[0.5, 0, 0])

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="3-vector"):
            TwoParticleKinem(MREF_L, M1, M2, dvec=[1, 2])

    def test_dsq_requires_dvec(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        with pytest.raises(ValueError, match="dvec"):
            kinem.dsq
        with pytest.raises(ValueError, match="dvec"):
            kinem.momentum_sqr


class TestFloatInputs:
    def test_dsq(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[1, 2, -2])
        assert kinem.dsq == 9

    def test_momentum_factor_sqr(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        assert kinem.momentum_factor_sqr == pytest.approx((2 * np.pi / MREF_L) ** 2)

    def test_momentum_sqr(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 1, 1])
        assert kinem.momentum_sqr == pytest.approx(2 * (2 * np.pi / MREF_L) ** 2)

    def test_ni_energy_at_rest(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        assert kinem.ni_energy(0, 0) == pytest.approx(M1 + M2)

    def test_ni_energy_moving(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        pfac_sqr = (2 * np.pi / MREF_L) ** 2
        expected = np.sqrt(M1**2 + 4 * pfac_sqr) + np.sqrt(M2**2 + pfac_sqr)
        assert kinem.ni_energy(4, 1) == pytest.approx(expected)


class TestArrayInputs:
    def test_ni_energy(self):
        m1 = np.array([1.0, 1.1])
        m2 = np.array([3.5, 3.6])
        kinem = TwoParticleKinem(MREF_L, m1, m2)
        result = kinem.ni_energy(1, 1)
        pfac_sqr = (2 * np.pi / MREF_L) ** 2
        expected = np.sqrt(m1**2 + pfac_sqr) + np.sqrt(m2**2 + pfac_sqr)
        np.testing.assert_allclose(result, expected)


class TestSamplingInputs:
    def test_momentum_factor_sqr(self, sampling_info):
        mref_L = make_sampling(MREF_L, 0.1, sampling_info, "mref_L")
        kinem = TwoParticleKinem(mref_L, M1, M2)
        result = kinem.momentum_factor_sqr
        assert isinstance(result, SigmondSampling)
        np.testing.assert_allclose(result.data, (2 * np.pi / mref_L.data) ** 2)

    def test_ni_energy_propagates_resamplings(self, sampling_info):
        mref_L = make_sampling(MREF_L, 0.1, sampling_info, "mref_L")
        m1 = make_sampling(M1, 0.01, sampling_info, "m1")
        m2 = make_sampling(M2, 0.02, sampling_info, "m2")

        kinem = TwoParticleKinem(mref_L, m1, m2, dvec=[0, 0, 1])
        result = kinem.ni_energy(1, 1)

        assert isinstance(result, SigmondSampling)
        pfac_sqr = (2 * np.pi / mref_L.data) ** 2
        expected = np.sqrt(m1.data**2 + pfac_sqr) + np.sqrt(m2.data**2 + pfac_sqr)
        np.testing.assert_allclose(result.data, expected)
        assert result.error > 0

    def test_mixed_float_and_sampling(self, sampling_info):
        m1 = make_sampling(M1, 0.01, sampling_info, "m1")
        kinem = TwoParticleKinem(MREF_L, m1, M2)
        result = kinem.ni_energy(0, 0)
        assert isinstance(result, SigmondSampling)
        assert result.full_sample_value == pytest.approx(M1 + M2)


def brute_force_levels(mref_L, m1, m2, dvec, emin, emax, radius=8):
    """Reference enumeration over all d1 vectors in a cube of given radius."""
    pfac_sqr = (2 * np.pi / mref_L) ** 2
    levels = {}
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                d1_sqr = x**2 + y**2 + z**2
                d2_sqr = (dvec[0] - x) ** 2 + (dvec[1] - y) ** 2 + (dvec[2] - z) ** 2
                energy = np.sqrt(m1**2 + pfac_sqr * d1_sqr) + np.sqrt(m2**2 + pfac_sqr * d2_sqr)
                if emin <= energy <= emax:
                    levels[(d1_sqr, d2_sqr)] = energy
    return levels


class TestNIEnergiesInInterval:
    def test_requires_dvec(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2)
        with pytest.raises(ValueError, match="dvec"):
            kinem.ni_energies_in_interval(0.0, 5.0)

    def test_matches_brute_force_distinct_particles(self):
        dvec = [0, 1, 1]
        emin, emax = 4.5, 5.2
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=dvec)
        result = kinem.ni_energies_in_interval(emin, emax)

        expected = brute_force_levels(MREF_L, M1, M2, dvec, emin, emax)
        assert {(d1, d2) for d1, d2, _ in result} == set(expected)
        for d1_sqr, d2_sqr, energy in result:
            assert energy == pytest.approx(expected[(d1_sqr, d2_sqr)])

    def test_matches_brute_force_identical_particles(self):
        dvec = [0, 0, 1]
        emin, emax = 0.0, 2.5
        kinem = TwoParticleKinem(MREF_L, M1, M1, dvec=dvec)
        result = kinem.ni_energies_in_interval(emin, emax)

        expected = brute_force_levels(MREF_L, M1, M1, dvec, emin, emax)
        expected_pairs = {tuple(sorted(pair)) for pair in expected}
        assert {(d1, d2) for d1, d2, _ in result} == expected_pairs

    def test_identical_particles_dedupe_swapped_pairs(self):
        kinem = TwoParticleKinem(MREF_L, M1, M1, dvec=[0, 0, 1])
        result = kinem.ni_energies_in_interval(0.0, 2.5)
        pairs = [(d1, d2) for d1, d2, _ in result]
        assert len(pairs) == len(set(pairs))
        assert all(d1 <= d2 for d1, d2 in pairs)

    def test_distinct_particles_keep_swapped_pairs(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 0, 1])
        result = kinem.ni_energies_in_interval(4.5, 5.2)
        pairs = {(d1, d2) for d1, d2, _ in result}
        # e.g. (0, 1) and (1, 0) are distinct levels for distinct masses
        assert (0, 1) in pairs
        assert (1, 0) in pairs

    def test_sorted_by_energy(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 1, 1])
        result = kinem.ni_energies_in_interval(4.5, 5.5)
        energies = [e for _, _, e in result]
        assert energies == sorted(energies)

    def test_empty_interval(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 0, 0])
        assert kinem.ni_energies_in_interval(0.0, 1.0) == []

    def test_sampling_masses(self, sampling_info):
        m1 = make_sh_mass(M1, 0.01, sampling_info, "pi")
        m2 = make_sh_mass(M2, 0.02, sampling_info, "N")
        kinem = TwoParticleKinem(MREF_L, m1, m2, dvec=[0, 0, 1])
        result = kinem.ni_energies_in_interval(4.5, 5.2)

        expected = brute_force_levels(MREF_L, M1, M2, [0, 0, 1], 4.5, 5.2)
        assert {(d1, d2) for d1, d2, _ in result} == set(expected)
        for d1_sqr, d2_sqr, energy in result:
            assert isinstance(energy, SigmondSampling)
            assert energy.full_sample_value == pytest.approx(expected[(d1_sqr, d2_sqr)])
            assert energy.observable_info.particles == (
                Particle("pi", psq=d1_sqr),
                Particle("N", psq=d2_sqr),
            )


class TestNIObservableInfo:
    def test_sh_masses_tag_energy_obs_info(self, sampling_info):
        m1 = make_sh_mass(M1, 0.01, sampling_info, "pi")
        m2 = make_sh_mass(M2, 0.02, sampling_info, "N")
        kinem = TwoParticleKinem(MREF_L, m1, m2, dvec=[0, 0, 1])

        energy = kinem.ni_energy(1, 0)
        info = energy.observable_info

        assert isinstance(info, EnergyObsInfo)
        assert not isinstance(info, SHEnergyObsInfo)
        assert info.energy_type == "elab"
        assert info.psq == 1
        assert info.particles == (Particle("pi", psq=1), Particle("N", psq=0))
        assert info.name == "PSQ1_elab_ni_pi(1)_N(0)"

    def test_no_dvec_omits_psq(self, sampling_info):
        m1 = make_sh_mass(M1, 0.01, sampling_info, "pi")
        m2 = make_sh_mass(M2, 0.02, sampling_info, "N")
        kinem = TwoParticleKinem(MREF_L, m1, m2)

        info = kinem.ni_energy(1, 1).observable_info
        assert info.psq is None
        assert info.particles == (Particle("pi", psq=1), Particle("N", psq=1))

    def test_ref_particle_propagates_when_shared(self, sampling_info):
        obs1 = SHEnergyObsInfo(psq=0, particle="pi", ref_particle="pi")
        obs2 = SHEnergyObsInfo(psq=0, particle="N", ref_particle="pi")
        m1 = create_gaussian_sampling(M1, 0.01, sampling_info, obs1)
        m2 = create_gaussian_sampling(M2, 0.02, sampling_info, obs2)
        kinem = TwoParticleKinem(MREF_L, m1, m2, dvec=[0, 0, 0])

        info = kinem.ni_energy(0, 0).observable_info
        assert info.ref_particle == "pi"
        assert info.name.endswith("_ref")

    def test_unresolvable_names_keep_default_info(self, sampling_info):
        m1 = make_sampling(M1, 0.01, sampling_info, "m1")
        m2 = make_sampling(M2, 0.02, sampling_info, "m2")
        kinem = TwoParticleKinem(MREF_L, m1, m2, dvec=[0, 0, 1])

        energy = kinem.ni_energy(1, 0)
        assert isinstance(energy, SigmondSampling)
        assert not isinstance(energy.observable_info, EnergyObsInfo)

    def test_float_inputs_return_plain_float(self):
        kinem = TwoParticleKinem(MREF_L, M1, M2, dvec=[0, 0, 1])
        energy = kinem.ni_energy(1, 0)
        assert not isinstance(energy, SigmondSampling)


def test_exported_from_package():
    import sigmondsamplings as ss

    assert "TwoParticleKinem" in ss.__all__
    assert ss.TwoParticleKinem is TwoParticleKinem
