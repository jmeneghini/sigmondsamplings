"""Case rules for particle-name resolution and parsing."""

import pytest

from slat import PARTICLE_LATEX_MAP, is_particle_name, resolve_particle_name
from sigmondsamplings.energy_levels import (
    Particle,
    SHEnergyObsInfo,
    create_energy_obs_info,
    parse_energy_attributes,
)
from sigmondsamplings.info import ObservableInfo


class TestResolveParticleName:
    @pytest.mark.parametrize("token", ["pi", "PI", "Pi", "pION"])
    def test_multichar_names_are_case_insensitive(self, token):
        assert resolve_particle_name(token) in {"pi", "pion"}

    @pytest.mark.parametrize(
        "token,expected", [("nucleon", "nucleon"), ("NUCLEON", "nucleon"), ("Kbar", "Kbar"), ("KBAR", "Kbar")]
    )
    def test_multichar_names_canonicalize(self, token, expected):
        assert resolve_particle_name(token) == expected

    @pytest.mark.parametrize("token", ["n", "s", "l", "x"])
    def test_lowercase_abbreviations_do_not_resolve(self, token):
        """A stray single-letter token must never be read as a particle."""
        assert resolve_particle_name(token) is None

    def test_kaon_and_antikaon_stay_distinct(self):
        assert resolve_particle_name("K") == "K"
        assert resolve_particle_name("k") == "k"
        assert PARTICLE_LATEX_MAP["K"] != PARTICLE_LATEX_MAP["k"]

    def test_unknown_token(self):
        assert resolve_particle_name("bogus") is None
        assert not is_particle_name("bogus")

    def test_every_key_resolves_to_itself(self):
        for name in PARTICLE_LATEX_MAP:
            assert resolve_particle_name(name) == name


class TestParticleCanonicalization:
    def test_constructor_canonicalizes(self):
        assert Particle("PI").name == "pi"
        assert Particle("Nucleon").name == "nucleon"

    def test_case_variants_are_equal_and_hash_alike(self):
        assert Particle("PI") == Particle("pi")
        assert hash(Particle("PI")) == hash(Particle("pi"))

    def test_string_comparison_resolves(self):
        assert Particle("pi") == "PI"

    def test_bad_name_still_raises(self):
        with pytest.raises(ValueError, match="Invalid particle name"):
            Particle("bogus")

    @pytest.mark.parametrize("text", ["pi", "pi+", "K-", "pi0", "rho0(2)", "Kbar(1)"])
    def test_from_string_round_trips_every_spelling(self, text):
        assert str(Particle.from_string(text)) == text

    def test_from_string_rejects_unknown(self):
        with pytest.raises(ValueError):
            Particle.from_string("bogus(1)")


class TestParsingCaseRules:
    @pytest.mark.parametrize("name", ["PSQ0_pi", "PSQ0_PI", "PSQ0_Pi", "psq0.pi"])
    def test_case_variants_parse_to_one_canonical_name(self, name):
        obs = create_energy_obs_info(ObservableInfo(name))
        assert isinstance(obs, SHEnergyObsInfo)
        assert obs.particle == "pi"
        assert obs.canonical_name == "PSQ0_pi"

    def test_full_name_case_insensitive_end_to_end(self):
        obs = create_energy_obs_info(ObservableInfo("psq0_a1g_ELAB_PION"))
        assert (obs.particle, obs.irrep, obs.energy_type) == ("pion", "A1g", "elab")

    @pytest.mark.parametrize("name,particle", [("PSQ0_K", "K"), ("PSQ0_k", "k"), ("PSQ0_Kbar", "Kbar")])
    def test_kaon_variants_stay_distinct(self, name, particle):
        assert create_energy_obs_info(ObservableInfo(name)).particle == particle

    def test_lowercase_abbreviation_is_not_a_particle(self):
        assert "particles" not in parse_energy_attributes("PSQ0_n")

    def test_wrong_case_abbreviation_gets_a_pointed_error(self):
        with pytest.raises(ValueError, match="did you mean 'N'"):
            create_energy_obs_info(ObservableInfo("PSQ0_n"))

    def test_unknown_particle_lists_the_vocabulary(self):
        with pytest.raises(ValueError, match="No recognized particle"):
            create_energy_obs_info(ObservableInfo("PSQ0_pian"))

    def test_ref_particle_is_canonicalized(self):
        obs = SHEnergyObsInfo(psq=0, particle="pi", ref_particle="NUCLEON")
        assert obs.ref_particle == "nucleon"

    def test_bad_ref_particle_still_raises(self):
        with pytest.raises(ValueError, match="Invalid ref_particle"):
            SHEnergyObsInfo(psq=0, particle="pi", ref_particle="bogus")
