"""Tests for the Spec/Resolved spectrum-selection config."""

import pytest

from sigmondsamplings.spectrum_spec import (
    SectorResolved,
    SectorSpec,
    SpectrumResolved,
    SpectrumSpec,
)
from slat import StrictModel


# --- SectorSpec authoring shorthand -------------------------------------------


def test_sector_n_levels_selects_leading_levels():
    resolved = SectorSpec(psq=0, irrep="A1g", n_levels=3).resolve()
    assert resolved.levels == [0, 1, 2]


def test_sector_list_is_sorted_and_deduplicated():
    resolved = SectorSpec(psq=1, irrep="E", levels=[2, 0, 0, 1]).resolve()
    assert resolved.levels == [0, 1, 2]


def test_sector_requires_exactly_one_level_source():
    with pytest.raises(ValueError, match="exactly one of 'levels' or 'n_levels'"):
        SectorSpec(psq=0, irrep="A1g")
    with pytest.raises(ValueError, match="exactly one of 'levels' or 'n_levels'"):
        SectorSpec(psq=0, irrep="A1g", levels=[0], n_levels=2)


def test_sector_is_strict():
    assert issubclass(SectorSpec, StrictModel)
    with pytest.raises(ValueError):
        SectorSpec(psq=0, irrep="A1g", levels=[0], typo=1)


def test_negative_psq_rejected():
    with pytest.raises(ValueError):
        SectorSpec(psq=-1, irrep="A1g", levels=[0])


# --- SectorResolved invariants ------------------------------------------------


def test_resolved_rejects_unsorted_levels():
    with pytest.raises(ValueError, match="sorted"):
        SectorResolved(psq=0, irrep="A1g", levels=[1, 0])


def test_resolved_rejects_duplicate_levels():
    with pytest.raises(ValueError, match="unique"):
        SectorResolved(psq=0, irrep="A1g", levels=[0, 0])


def test_resolved_rejects_empty_levels():
    with pytest.raises(ValueError):
        SectorResolved(psq=0, irrep="A1g", levels=[])


# --- SpectrumResolved ---------------------------------------------------------


def test_allowed_keys():
    resolved = SpectrumSpec(
        spectrum=[
            SectorSpec(psq=0, irrep="A1g", levels=[0, 1]),
            SectorSpec(psq=1, irrep="E", n_levels=1),
        ]
    ).resolve()
    assert resolved.allowed_keys() == {
        (0, "A1g", 0),
        (0, "A1g", 1),
        (1, "E", 0),
    }


def test_duplicate_sectors_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        SpectrumResolved(
            spectrum=[
                SectorResolved(psq=0, irrep="E", levels=[0]),
                SectorResolved(psq=0, irrep="E", levels=[1]),
            ]
        )


# --- TOML round-trip ----------------------------------------------------------


def test_toml_uses_flat_array_of_tables():
    spec = SpectrumSpec(
        spectrum=[
            SectorSpec(psq=0, irrep="A1g", levels=[0, 1, 2]),
            SectorSpec(psq=1, irrep="E", levels=[0]),
        ]
    )
    text = spec.to_toml()
    assert "[[spectrum]]" in text
    assert SpectrumSpec.from_toml(text) == spec


def test_resolved_round_trips_through_to_spec():
    resolved = SpectrumSpec(
        spectrum=[SectorSpec(psq=0, irrep="A1g", n_levels=3)]
    ).resolve()
    assert resolved.to_spec().resolve() == resolved
    assert SpectrumSpec.from_toml(resolved.to_spec().to_toml()).resolve() == resolved


def test_toml_unknown_key_rejected():
    spec = SpectrumSpec(spectrum=[SectorSpec(psq=0, irrep="A1g", levels=[0])])
    bad = spec.to_toml().replace("levels", "levls")
    with pytest.raises(ValueError):
        SpectrumSpec.from_toml(bad)
