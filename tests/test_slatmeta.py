from sigmondsamplings import COLORS, IndexedCycle
from slatmeta import (
    IRREP_LATEX_MAP,
    PARTICLE_LATEX_MAP,
    get_all_qctype_latex_mappings,
    get_energy_type_latex_str,
    get_irrep_latex_str,
    get_particle_latex_str,
    get_qc_matrix_latex_str,
    get_qc_type_latex_str,
    get_spectro_latex_str,
    get_spectro_str,
    get_twoJ_L_twoS_from_spectro,
    wrap_str_in_determinant,
)


def test_slatmeta_label_helpers_match_shared_maps():
    assert PARTICLE_LATEX_MAP["pi"] == r"\pi"
    assert IRREP_LATEX_MAP["A1g"] == r"A_{1g}"
    assert get_particle_latex_str("rho") == r"\rho"
    assert get_irrep_latex_str("A1g") == r"A_{1g}"
    assert get_energy_type_latex_str("ecm", 2) == r"E_{\mathrm{cm}}^{2}"
    assert get_particle_latex_str("unknown") == "unknown"
    assert get_irrep_latex_str("unknown") == "unknown"


def test_sigmondsamplings_reexports_slatmeta_colors():
    cycle = IndexedCycle(COLORS)

    assert next(cycle) == COLORS[0]
    assert cycle.get_current() == COLORS[1]


def test_slatmeta_extracts_kbfit_constants_helpers():
    assert get_spectro_str(4, 1, 2) == "3P2"
    assert get_twoJ_L_twoS_from_spectro("2D3/2") == (3, 2, 1)
    assert get_spectro_latex_str(3, 2, 1) == r"{}^{2}D_ {3/2}"
    assert get_qc_matrix_latex_str("ktilde") == r"\widetilde{K}"
    assert get_qc_type_latex_str("ktilde_b", show_det=False) == r"1 - \widetilde{K} B"
    assert get_qc_type_latex_str("ktilde_b") == r"\det\left(1 - \widetilde{K} B\right)"
    assert get_all_qctype_latex_mappings(show_det=False)["stilde_cb"] == r"1 + \widetilde{S} C_B"
    assert wrap_str_in_determinant("A") == r"\det\left(A\right)"
