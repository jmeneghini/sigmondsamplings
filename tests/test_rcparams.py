from pathlib import Path

import pytest

from sigmondsamplings import KnownEnsembles, rc, rc_context, rc_defaults


def _write_ensembles_xml(path: Path) -> None:
    path.write_text(
        """<KnownEnsembles>
  <Infos>
    <EnsembleInfo>
      <Id>test_ensemble</Id>
      <NMeas>64</NMeas>
      <NSpace>16</NSpace>
      <NTime>48</NTime>
    </EnsembleInfo>
  </Infos>
</KnownEnsembles>
"""
    )


def test_ensembles_xml_file_defaults_to_none():
    saved = dict(rc)
    try:
        rc_defaults()
        assert rc["ensembles.xml_file"] is None
    finally:
        rc.clear()
        rc.update(saved)


def test_known_ensembles_uses_rc_ensembles_xml_file(tmp_path):
    xml_file = tmp_path / "ensembles.xml"
    _write_ensembles_xml(xml_file)

    with rc_context({"ensembles.xml_file": str(xml_file)}):
        known = KnownEnsembles()

    assert known.list_ensembles() == ["test_ensemble"]
    ensemble = known.get("test_ensemble")
    assert ensemble.name == "test_ensemble"
    assert ensemble.num_measurements == 64
    assert ensemble.spatial_extent == 16
    assert ensemble.temporal_extent == 48


def test_known_ensembles_raises_for_missing_rc_ensembles_xml_file(tmp_path):
    missing = tmp_path / "missing.xml"

    with rc_context({"ensembles.xml_file": str(missing)}):
        with pytest.raises(FileNotFoundError, match="Previously configured ensemble file"):
            KnownEnsembles()
