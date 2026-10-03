from sigmondsamplings import COLORS, IndexedCycle
from slat import (
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


def test_slat_label_helpers_match_shared_maps():
    assert PARTICLE_LATEX_MAP["pi"] == r"\pi"
    assert IRREP_LATEX_MAP["A1g"] == r"A_{1g}"
    assert get_particle_latex_str("rho") == r"\rho"
    assert get_irrep_latex_str("A1g") == r"A_{1g}"
    assert get_energy_type_latex_str("ecm", 2) == r"E_{\mathrm{cm}}^{2}"
    assert get_particle_latex_str("unknown") == "unknown"
    assert get_irrep_latex_str("unknown") == "unknown"


def test_sigmondsamplings_reexports_slat_colors():
    cycle = IndexedCycle(COLORS)

    assert next(cycle) == COLORS[0]
    assert cycle.get_current() == COLORS[1]


def test_slat_extracts_kbfit_constants_helpers():
    assert get_spectro_str(4, 1, 2) == "3P2"
    assert get_twoJ_L_twoS_from_spectro("2D3/2") == (3, 2, 1)
    assert get_spectro_latex_str(3, 2, 1) == r"{}^{2}D_ {3/2}"
    assert get_qc_matrix_latex_str("ktilde") == r"\widetilde{K}"
    assert get_qc_type_latex_str("ktilde_b", show_det=False) == r"1 - \widetilde{K} B"
    assert get_qc_type_latex_str("ktilde_b") == r"\det\left(1 - \widetilde{K} B\right)"
    assert get_all_qctype_latex_mappings(show_det=False)["stilde_cb"] == r"1 + \widetilde{S} C_B"
    assert wrap_str_in_determinant("A") == r"\det\left(A\right)"


def test_every_sigmond_irrep_has_valid_latex():
    from matplotlib.mathtext import MathTextParser

    from slat.sigmond_names import LG_IRREPS

    parser = MathTextParser("path")
    assert set(LG_IRREPS) <= set(IRREP_LATEX_MAP)
    assert get_irrep_latex_str("A1um") == r"A_{1u}^{-}"
    assert get_irrep_latex_str("A1p") == r"A_1^{+}"
    for irrep in LG_IRREPS:
        parser.parse(f"${get_irrep_latex_str(irrep)}$")  # raises on malformed LaTeX


def test_sigmond_flavor_table():
    from slat.sigmond_names import HADRON_FLAVORS, TETRAQUARK_SHORT

    assert HADRON_FLAVORS["omega"].kind == "baryon" and HADRON_FLAVORS["omega"].strangeness == -3
    assert HADRON_FLAVORS["kbar"].strangeness == -1 and HADRON_FLAVORS["kaon"].isospin == "doublet"
    assert [HADRON_FLAVORS[t].code for t in TETRAQUARK_SHORT] == list(range(13, 25))
    assert HADRON_FLAVORS["isodoublet_kaon_pion"].strangeness == 1


def test_toml_models_load_lazily():
    import subprocess
    import sys

    code = "import sys, slat; assert 'pydantic' not in sys.modules; slat.StrictModel; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
