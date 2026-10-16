from itertools import permutations, product

import numpy as np
import pytest

from sigmondsamplings.energy_levels import EnergyObsInfo, Particle, SHEnergyObsInfo
from sigmondsamplings.info import SamplingInfo
from sigmondsamplings.kinematics import MultiParticleKinem, TwoParticleKinem
from sigmondsamplings.sampling import SigmondSampling
from sigmondsamplings.utils import create_gaussian_sampling

MREF_L = 6.0
M_PI = 1.0
M_N = 3.5


@pytest.fixture
def sampling_info():
    return SamplingInfo("bootstrap", 100, seed=7)


def make_sh_mass(mean, std, sampling_info, particle):
    obs_info = SHEnergyObsInfo(psq=0, particle=particle)
    return create_gaussian_sampling(mean, std, sampling_info, obs_info)


def brute_force(mref_L, masses, dvec, emin, emax, radius=3):
    """All ordered vector configurations in a cube, keyed by the vector tuple."""
    pfac_sqr = (2 * np.pi / mref_L) ** 2
    cube = list(product(range(-radius, radius + 1), repeat=3))
    out = {}
    for head in product(cube, repeat=len(masses) - 1):
        last = tuple(int(d - sum(v[k] for v in head)) for k, d in enumerate(dvec))
        config = (*head, last)
        energy = sum(np.sqrt(m**2 + pfac_sqr * np.dot(v, v)) for m, v in zip(masses, config))
        if emin <= energy <= emax:
            out[config] = energy
    return out


def canonical(config, groups):
    config = list(config)
    for group in groups:
        for idx, vec in zip(group, sorted(config[j] for j in group)):
            config[idx] = vec
    return tuple(config)


class TestBasics:
    def test_requires_masses(self):
        with pytest.raises(ValueError, match="mass"):
            MultiParticleKinem(MREF_L, [])

    def test_ni_energy_from_squares_and_vectors_agree(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI, M_PI, M_N], dvec=[0, 0, 1])
        vectors = [[0, 0, 1], [1, 0, 0], [-1, 0, 0]]
        assert kinem.ni_energy(vectors) == pytest.approx(kinem.ni_energy([1, 1, 1]))

    def test_vectors_must_sum_to_dvec(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI, M_PI], dvec=[0, 0, 1])
        with pytest.raises(ValueError, match="sum"):
            kinem.ni_energy([[0, 0, 1], [0, 0, 1]])

    def test_wrong_count(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI, M_PI, M_PI])
        with pytest.raises(ValueError, match="3"):
            kinem.ni_energy([0, 0])

    def test_cm_energy(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI, M_PI], dvec=[0, 0, 1])
        e_lab = kinem.ni_energy([0, 1])
        expected = np.sqrt(e_lab**2 - (2 * np.pi / MREF_L) ** 2)
        assert kinem.cm_energy([[0, 0, 0], [0, 0, 1]]) == pytest.approx(expected)

    def test_two_particle_is_a_multi_particle(self):
        kinem = TwoParticleKinem(MREF_L, M_PI, M_N, dvec=[0, 0, 1])
        assert isinstance(kinem, MultiParticleKinem)
        assert kinem.masses == [M_PI, M_N]
        kinem.m2 = 2.0
        assert kinem.masses == [M_PI, 2.0]


class TestConfigurations:
    @pytest.mark.parametrize("dvec", [[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    def test_three_pions_match_brute_force(self, dvec):
        masses = [M_PI, M_PI, M_PI]
        emax = 5.2
        kinem = MultiParticleKinem(MREF_L, masses, dvec=dvec)
        result = kinem.configurations(emax)

        groups = [[0, 1, 2]]
        expected = {}
        for config, energy in brute_force(MREF_L, masses, dvec, 0.0, emax).items():
            expected.setdefault(canonical(config, groups), energy)
        got = {tuple(map(tuple, momenta.tolist())): energy for momenta, energy in result}
        assert set(got) == set(expected)
        for key, energy in got.items():
            assert energy == pytest.approx(expected[key])

    def test_mixed_groups_match_brute_force(self):
        # pi pi N: the two pions are identical, the nucleon is distinct.
        masses = [M_PI, M_N, M_PI]
        dvec = [0, 0, 1]
        emin, emax = 6.0, 7.2
        kinem = MultiParticleKinem(MREF_L, masses, dvec=dvec)
        result = kinem.configurations(emax, emin)

        groups = [[0, 2]]
        expected = {canonical(c, groups) for c in brute_force(MREF_L, masses, dvec, emin, emax)}
        assert {tuple(map(tuple, m.tolist())) for m, _ in result} == expected

    def test_identical_dedupes_permutations(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI] * 3, dvec=[0, 0, 1])
        reps = [tuple(map(tuple, m.tolist())) for m, _ in kinem.configurations(5.2)]
        orbits = {frozenset(permutations(rep)) for rep in reps}
        assert len(orbits) == len(reps)

    def test_explicit_distinct_keeps_permutations(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI] * 2, dvec=[0, 0, 1])
        distinct = kinem.configurations(3.0, identical=False)
        identical = kinem.configurations(3.0)
        assert len(distinct) > len(identical)

    def test_sorted_and_within_interval(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI, M_PI, M_N], dvec=[1, 1, 0])
        result = kinem.configurations(7.0, 5.5)
        energies = [e for _, e in result]
        assert energies == sorted(energies)
        assert all(5.5 <= e <= 7.0 for e in energies)

    def test_below_threshold_is_empty(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI] * 3, dvec=[0, 0, 0])
        assert kinem.configurations(2.9) == []

    def test_requires_dvec(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI] * 3)
        with pytest.raises(ValueError, match="dvec"):
            kinem.configurations(5.0)

    def test_bad_groups(self):
        kinem = MultiParticleKinem(MREF_L, [M_PI] * 3, dvec=[0, 0, 0])
        with pytest.raises(ValueError, match="disjoint"):
            kinem.configurations(5.0, identical=[[0, 1], [1, 2]])


class TestTwoParticleParity:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_explicit_pair_enumeration(self, seed):
        rng = np.random.default_rng(seed)
        m1, m2 = rng.uniform(0.5, 2.0, size=2)
        dvec = rng.integers(-1, 2, size=3)
        emax = m1 + m2 + rng.uniform(0.5, 2.5)
        emin = emax - 1.5
        kinem = TwoParticleKinem(MREF_L, m1, m2, dvec=dvec)
        got = {(d1, d2) for d1, d2, _ in kinem.ni_energies_in_interval(emin, emax)}
        expected = {
            (int(np.dot(a, a)), int(np.dot(b, b)))
            for a, b in brute_force(MREF_L, [m1, m2], dvec, emin, emax)
        }
        assert got == expected

    def test_multi_squares_view(self):
        multi = MultiParticleKinem(MREF_L, [M_PI, M_N], dvec=[0, 1, 1])
        two = TwoParticleKinem(MREF_L, M_PI, M_N, dvec=[0, 1, 1])
        a = [(dsqs, pytest.approx(e)) for dsqs, e in multi.ni_energies_in_interval(4.5, 6.0)]
        b = [((d1, d2), e) for d1, d2, e in two.ni_energies_in_interval(4.5, 6.0)]
        assert a == b


class TestSamplings:
    def test_three_particle_energy_propagates_and_tags(self, sampling_info):
        pi = make_sh_mass(M_PI, 0.01, sampling_info, "pi")
        nucleon = make_sh_mass(M_N, 0.02, sampling_info, "N")
        kinem = MultiParticleKinem(MREF_L, [pi, pi, nucleon], dvec=[0, 0, 1])
        levels = kinem.ni_energies_in_interval(5.0, 7.0)
        assert levels
        dsqs, energy = levels[0]
        assert isinstance(energy, SigmondSampling)
        assert energy.error > 0
        info = energy.observable_info
        assert isinstance(info, EnergyObsInfo)
        assert info.particles == tuple(Particle(n, psq=d) for n, d in zip(["pi", "pi", "N"], dsqs))
        assert info.name == "PSQ1_elab_ni_" + "_".join(
            f"{n}({d})" for n, d in zip(["pi", "pi", "N"], dsqs)
        )

    def test_identical_auto_detected_from_names(self, sampling_info):
        pi_a = make_sh_mass(M_PI, 0.01, sampling_info, "pi")
        pi_b = make_sh_mass(M_PI + 0.001, 0.01, sampling_info, "pi")
        kinem = MultiParticleKinem(MREF_L, [pi_a, pi_b], dvec=[0, 0, 1])
        assert kinem._identical_groups(None) == [[0, 1]]


def test_exported_from_package():
    import sigmondsamplings as ss

    assert "MultiParticleKinem" in ss.__all__
    assert ss.MultiParticleKinem is MultiParticleKinem
