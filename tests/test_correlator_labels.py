"""
Immutable operator, correlator and key labels and their Sigmond short-form grammar.
Where the Sigmond Python bindings or ``sigmond_query`` are installed, the
canonical strings and SS-written keys are checked against Sigmond's own C++
parser.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sigmondsamplings.correlators import (
    BasicLapHOperator,
    CorrelatorAtTimeInfo,
    CorrelatorInfo,
    CorrelatorMatrixInfo,
    GenIrrepOperator,
    Hadron,
    SigmondParseError,
    VEVInfo,
    operator_from_xml,
    parse_key,
    parse_operator,
)
from sigmondsamplings.correlators import operators as ops_mod
from slat.sigmond_names import TETRAQUARK_SHORT

try:
    import sigmond as _sigmond
    _ = _sigmond.OperatorInfo, _sigmond.OpKind
except ImportError:  # pragma: no cover - depends on the environment
    _sigmond = None

needs_sigmond_module = pytest.mark.skipif(_sigmond is None, reason="sigmond module not installed")
needs_sigmond_query = pytest.mark.skipif(
    shutil.which("sigmond_query") is None, reason="sigmond_query not available"
)

BL = BasicLapHOperator.from_sigmond
GI = GenIrrepOperator.from_sigmond

BL_STRINGS = [
    "pion P=(0,0,0) A1um_1 SD_5",
    "pion P=(0,0,1) A2m_1 SS_0",
    "pion P=(0,0,0) A1um_1 SD_5 D4",
    "glueball P=(0,0,0) A1gp_1 TrEig",
    "nucleon P=(0,0,0) G1g_2 DDL_3",
    "kaon P=(0,1,1) A2_1 SD_3 D2",
    "isotriplet_pion_pion A1um_1 CG_1 [P=(0,0,1) A1p LSD_1] [P=(0,0,-1) A2m TSD_2]",
    "isotriplet_pion_pion A1um_1 [P=(0,0,1) A1p LSD_1] [P=(0,0,-1) A2m TSD_2 D2]",
    "isotriplet-2_pion_pion_pion A1um_1 [P=(0,0,1) A2m SS_0] [P=(0,0,-1) A2m SS_0] "
    "[P=(0,0,0) A1um SS_0]",
    "isodoublet_kaon_pion Eu_2 [P=(0,0,1) A2 SS_0] [P=(0,0,-1) A2m SS_0]",
    "tquuuu1p P=(0,0,0) A1um_1 QDX_1",
    "tquuuu1m P=(0,0,0) A1um_1 QDX_1 D2",
    "tqsudu2m P=(0,0,0) A1u_1 SS_1",
]
BWD_STRINGS = [
    "pion P=(0,0,0) A1um_1 SD_5 D4 bwd",
    "isotriplet_pion_pion A1um_1 CG_1 bwd [P=(0,0,1) A1p LSD_1] [P=(0,0,-1) A2m TSD_2]",
]
GI_STRINGS = [
    "isotriplet S=0 P=(0,0,0) A1um_1 ROT 2",
    "isotriplet S=-1 PSQ=1 A2m ROT 2",
    "isodoublet S=0 P=(0,0,0) G1g N[SS0] 0",
    "isosinglet S=-2 P=(1,1,1) A1 x 32767",
]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", BL_STRINGS + BWD_STRINGS)
def test_basic_laph_parse_print_parse(text):
    op = BL(text)
    expected = text.removesuffix(" D2") if text.startswith("tquuuu1m") else text
    assert str(op) == expected
    assert BL(str(op)) == op
    assert hash(BL(str(op))) == hash(op)
    assert op.coordinate == f"BL{{{expected}}}"


@pytest.mark.parametrize("text", GI_STRINGS)
def test_gen_irrep_parse_print_parse(text):
    op = GI(text)
    assert str(op) == text
    assert GI(str(op)) == op


def test_default_tokens_are_canonicalized():
    assert str(BL("pion P=(0,0,0) A1um_1 SD_5 D3")) == "pion P=(0,0,0) A1um_1 SD_5"
    assert BL("pion P=(0,0,0) A1um_1 SD_5 D3") == BL("pion P=(0,0,0) A1um_1 SD_5")
    assert str(BL("nucleon P=(0,0,0) G1g_1 SD_1 D2")) == "nucleon P=(0,0,0) G1g_1 SD_1"
    multi = BL("isotriplet_pion_pion A1um_1 CG_0 [P=(0,0,1) A1p SS_0] [P=(0,0,-1) A2m SS_0]")
    assert "CG_" not in str(multi)
    assert str(GI("triplet P=(0,0,0) A1um_1 ROT")) == "isotriplet S=0 P=(0,0,0) A1um_1 ROT 0"


def test_flags_in_either_order_single_hadron():
    assert BL("pion P=(0,0,0) A1um_1 SD_5 bwd D4") == BL("pion P=(0,0,0) A1um_1 SD_5 D4 bwd")


def test_vacuum_is_the_only_zero_hadron_label():
    vac = BasicLapHOperator.vacuum()
    assert vac.is_vacuum and str(vac) == "" and BL("") == vac
    with pytest.raises(ValueError):
        BasicLapHOperator((), irrep="A1gp", row=1)


def test_derived_quantum_numbers():
    multi = BL(BL_STRINGS[6])
    assert multi.n_hadrons == 2
    assert multi.momentum == (0, 0, 0) and multi.momentum_squared == 0
    assert multi.channel.isospin == "triplet"
    kp = BL("isodoublet_kaon_pion Eu_2 [P=(0,0,1) A2 SS_0] [P=(0,0,-1) A2m SS_0]")
    assert kp.strangeness == 1
    single = BL("kaon P=(0,1,1) A2_1 SD_3 D2")
    assert single.channel.isospin == "doublet" and single.momentum_squared == 2
    assert single.hadrons[0].displacement == 2


def test_gen_irrep_momentum_forms():
    psq0 = GI("isotriplet S=0 PSQ=0 A1um_1 ROT 0")
    assert psq0.momentum == (0, 0, 0) and str(psq0).count("P=(0,0,0)") == 1
    assert psq0 == GI("isotriplet S=0 P=(0,0,0) A1um_1 ROT 0")
    psq1 = GI("isotriplet S=0 PSQ=1 A2m ROT 0")
    assert not psq1.has_definite_momentum
    assert psq1 != GI("isotriplet S=0 P=(0,0,1) A2m ROT 0")
    assert psq1.channel != GI("isotriplet S=0 P=(0,0,1) A2m ROT 0").channel
    assert GI(GI_STRINGS[0]).with_index(5).index == 5
    with pytest.raises(ValueError):
        GenIrrepOperator("triplet", 0, "A1um", "X")
    with pytest.raises(ValueError):
        GenIrrepOperator("triplet", 0, "A1um", "X", momentum=(0, 0, 1), momentum_squared=2)


def test_tetraquark_color():
    p, m = BL("tquuuu1p P=(0,0,0) A1um_1 QDX_1"), BL("tquuuu1m P=(0,0,0) A1um_1 QDX_1")
    assert p.hadrons[0].flavor == "isosinglet_eta_eta" and p.hadrons[0].color == 1
    assert m.hadrons[0].color == -1 and p != m
    with pytest.raises(ValueError, match="color"):
        Hadron("isosinglet_eta_eta", (0, 0, 0), "A1um", "QDX", 1)
    with pytest.raises(ValueError, match="color"):
        Hadron("pion", (0, 0, 0), "A1um", "SS", color=1)


@pytest.mark.parametrize(
    "text, position, fragment",
    [
        ("pion P=(0,0,0) A1um_1 SD_5 D4 xyz", 30, "unexpected token"),
        ("pion P=(0,0) A1um_1 SD_5", 5, "momentum"),
        ("pion P=(0,0,0) A1q_1 SD_5", 15, "unknown irrep"),
        ("pion P=(0,0,0) A1um_1 XX_5", 22, "unknown spatial type"),
        ("pionn P=(0,0,0) A1um_1 SD_5", 0, "unknown flavor"),
        ("pion P=(0,0,0) A1um SD_5", 15, "irrep_row"),
        ("isotriplet_pion_pion A1um_1 [P=(0,0,1) A1p_1 SS_0] [P=(0,0,-1) A2m SS_0]", 39, "no row"),
        ("isotriplet_pion A1um_1 [P=(0,0,1) A1p SS_0]", 0, "at least two"),
        ("isotriplet_pion_pion A1um_1 [P=(0,0,1) A1p SS_0]", 48, "expected '['"),
        (
            "isotriplet_pion_pion A1um_1 [P=(0,0,1) A1p SS_0] [P=(0,0,-1) A2m SS_0] junk",
            71,
            "after",
        ),
    ],
)
def test_malformed_basic_laph_reports_position(text, position, fragment):
    with pytest.raises(SigmondParseError) as info:
        BL(text)
    assert fragment in info.value.reason
    assert info.value.position == position


def test_malformed_gen_irrep_reports_position():
    with pytest.raises(SigmondParseError) as info:
        GI("isotriplet P=(0,0,0) A1um_1 ROT 2 extra")
    assert info.value.position == 34 and "unexpected" in info.value.reason
    with pytest.raises(SigmondParseError):
        GI("isotriplet S=0 P=(0,0,0) A1um_1 ROT 2 extra")
    with pytest.raises(SigmondParseError):
        GI("isotriplet S=x P=(0,0,0) A1um_1 ROT")


def test_validation_is_not_limited_by_bitfields_until_export():
    big = BasicLapHOperator.single("pion", (200, 0, 0), "A2m", 20, "SD", 5000)
    assert big.row == 20
    with pytest.raises(ValueError, match="limit"):
        big.to_sigmond()
    long_name = GenIrrepOperator("triplet", 0, "A1um", "x" * 51, momentum=(0, 0, 0))
    with pytest.raises(ValueError, match="name length"):
        long_name.to_sigmond()
    assert ops_mod.MAX_GI_NAME == 50


def test_semantic_validation():
    with pytest.raises(ValueError, match="irrep"):
        BasicLapHOperator.single("nucleon", (0, 0, 0), "A1g", 1, "SS")
    with pytest.raises(ValueError, match="displacement"):
        BasicLapHOperator.single("pion", (0, 0, 0), "A1um", 1, "SS", 0, displacement=2)
    with pytest.raises(ValueError, match="row"):
        BasicLapHOperator.single("pion", (0, 0, 0), "A1um", 0, "SS")
    with pytest.raises(ValueError, match="isospin"):
        BasicLapHOperator(BL(BL_STRINGS[6]).hadrons, irrep="A1um", row=1)
    with pytest.raises(ValueError, match="backwards"):
        BL("glueball P=(0,0,0) A1gp_1 TrEig").with_backwards()


def test_parse_operator_dispatch():
    assert isinstance(parse_operator("pion P=(0,0,0) A1um_1 SS_0"), BasicLapHOperator)
    assert isinstance(parse_operator("isotriplet S=0 P=(0,0,0) A1um_1 ROT 0"), GenIrrepOperator)
    assert parse_operator("GI{isotriplet S=0 PSQ=1 A2m ROT 0}") == GI("isotriplet PSQ=1 A2m ROT")
    assert isinstance(parse_operator("BL{pion P=(0,0,0) A1um_1 SS_0}"), BasicLapHOperator)
    with pytest.raises(ValueError, match="unknown flavor"):
        parse_operator("hello world")


@pytest.mark.parametrize("text", BL_STRINGS + BWD_STRINGS + GI_STRINGS)
def test_short_xml_round_trip_and_pickle(text):
    op = parse_operator(text)
    assert operator_from_xml(op.to_xml()) == op
    assert operator_from_xml(ET.tostring(op.to_xml(), encoding="unicode")) == op
    clone = pickle.loads(pickle.dumps(op))
    assert clone == op and hash(clone) == hash(op)


def test_identity_is_the_canonical_form():
    a = BL("pion P=(0,0,0) A1um_1 SD_5 D3")
    b = BasicLapHOperator.single("pion", (0, 0, 0), "A1um", 1, "SD", 5)
    assert a == b and hash(a) == hash(b) and {a: 1}[b] == 1
    assert a != BL("pion P=(0,0,0) A1um_1 SD_5 D4")
    assert a != GI("isotriplet S=0 P=(0,0,0) A1um_1 ROT 2")
    assert sorted([BL(BL_STRINGS[1]), BL(BL_STRINGS[0])])[0].coordinate < BL(BL_STRINGS[1]).coordinate
    assert BL("pion P=(0,0,0) A1um_1 SS_0") is BL("pion P=(0,0,0) A1um_1 SS_0")  # memoized


def test_latex_and_sector():
    assert BL(BL_STRINGS[0]).latex_str == r"\pi(0,0,0)\,A_{1u}^{-}\,\mathrm{SD}_{5}"
    assert GI(GI_STRINGS[0]).latex_str == r"\mathrm{ROT}_{2}\,A_{1u}^{-}"
    assert r"\Omega" in BL("omega P=(0,0,0) G1g_1 SS_0").latex_str  # the baryon
    sector = BL(BL_STRINGS[9]).channel.to_sector()
    assert sector.twoI == 1 and sector.S == 1


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

SINK = BL("pion P=(0,0,1) A2m_1 SD_5 D4")
SOURCE = GI("isotriplet S=0 P=(0,0,1) A2m_1 ROT 3")


def test_correlator_key_short_form_sink_first_and_flag_order():
    key = CorrelatorAtTimeInfo(CorrelatorInfo(SINK, SOURCE), 25, hermitian=True, subtract_vev=True)
    text = key.to_sigmond()
    assert text == (
        "<CorrT>BL{pion P=(0,0,1) A2m_1 SD_5 D4} "
        "GI{isotriplet S=0 P=(0,0,1) A2m_1 ROT 3} time=25 HermMat SubVEV</CorrT>"
    )
    assert CorrelatorAtTimeInfo.from_sigmond(text) == key
    swapped = text.replace("HermMat SubVEV", "SubVEV HermMat")
    assert CorrelatorAtTimeInfo.from_sigmond(swapped) == key
    assert key.sink == SINK and key.source == SOURCE
    assert key.flipped().sink == SOURCE


def test_brace_aware_scanning_handles_spaces_and_brackets():
    multi = BL(BL_STRINGS[6])
    gi = GI("isodoublet S=0 P=(0,0,0) G1g N[SS0] 0")
    text = f"<CorrT>  BL{{{multi}}}   GI{{{gi}}} time=3 </CorrT>"
    key = CorrelatorAtTimeInfo.from_sigmond(text)
    assert key.sink == multi and key.source == gi and key.time == 3


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("BL{pion P=(0,0,0) A1um_1 SS_0} time=3", "expected an operator"),
        ("BL{pion P=(0,0,0) A1um_1 SS_0} BL{pion P=(0,0,0) A1um_1 SS_0}", "missing 'time"),
        ("BL{pion P=(0,0,0) A1um_1 SS_0} BL{pion P=(0,0,0) A1um_1 SS_0} time=3 Herm", "unexpected"),
        ("BL{pion P=(0,0,0) A1um_1 SS_0 BL{x}", "unterminated"),
        ("BL{pion P=(0,0,0) A1um_1 SS_7x} BL{pion P=(0,0,0) A1um_1 SS_0} time=1", "spatial id"),
    ],
)
def test_malformed_correlator_keys(body, fragment):
    with pytest.raises(SigmondParseError) as info:
        CorrelatorAtTimeInfo.from_sigmond(f"<CorrT>{body}</CorrT>")
    assert fragment in info.value.reason


def test_vacuum_cannot_be_a_correlator_end():
    with pytest.raises(ValueError):
        CorrelatorInfo(BasicLapHOperator.vacuum(), SINK)


def test_correlator_short_form():
    corr = CorrelatorInfo(SINK, SOURCE)
    assert CorrelatorInfo.from_sigmond(corr.to_sigmond()) == corr
    assert CorrelatorInfo.from_sigmond(str(corr)) == corr
    with pytest.raises(SigmondParseError, match="after the source"):
        CorrelatorInfo.from_sigmond(f"{corr} time=3")


def test_vev_keys():
    v = VEVInfo(SINK)
    assert (
        v.to_sigmond()
        == "<VEV><BLOperatorString>pion P=(0,0,1) A2m_1 SD_5 D4</BLOperatorString></VEV>"
    )
    assert VEVInfo.from_sigmond(v.to_sigmond()) == v
    assert parse_key(v.to_sigmond()) == v


def test_parse_key_memoizes_operator_parses():
    ops_mod._parse_cached.cache_clear()
    keys = [parse_key(f"<CorrT>BL{{{SINK}}} BL{{{SINK}}} time={t}</CorrT>") for t in range(10)]
    info = ops_mod._parse_cached.cache_info()
    assert info.misses == 1 and info.hits == 19
    assert keys[3].sink is keys[7].source


def test_xml_escaped_names_parse():
    gi = GenIrrepOperator("triplet", 0, "A1um", "a<b&c", momentum=(0, 0, 0))
    key = CorrelatorAtTimeInfo(CorrelatorInfo(gi, gi), 1)
    text = key.to_sigmond()
    assert "&lt;" in text and "&amp;" in text
    assert parse_key(text) == key


# ---------------------------------------------------------------------------
# Matrix info
# ---------------------------------------------------------------------------

P0 = BL("pion P=(0,0,0) A1um_1 SD_5")
P1 = BL("pion P=(0,0,0) A1um_1 SS_0")


def test_matrix_info_validation():
    matrix_info = CorrelatorMatrixInfo((P0, P1))
    assert matrix_info.hermitian and not matrix_info.subtract_vev and len(matrix_info) == 2
    assert matrix_info.coordinates == (P0.coordinate, P1.coordinate)
    with pytest.raises(ValueError, match="at least one"):
        CorrelatorMatrixInfo(())
    with pytest.raises(ValueError, match="duplicate"):
        CorrelatorMatrixInfo((P0, P0))
    with pytest.raises(ValueError, match="vacuum"):
        CorrelatorMatrixInfo((P0, BasicLapHOperator.vacuum()))
    with pytest.raises(ValueError, match="channels"):
        CorrelatorMatrixInfo((P0, BL("pion P=(0,0,1) A2m_1 SS_0")))
    with pytest.raises(ValueError, match="backward"):
        CorrelatorMatrixInfo((P0, P1.with_backwards()))
    rot = GI("isotriplet S=0 P=(0,0,0) A1um_1 ROT 0")
    assert CorrelatorMatrixInfo((P0, rot)).channel == P0.channel


def test_matrix_info_accepts_strings_and_writes_sigmond_xml():
    matrix_info = CorrelatorMatrixInfo(("pion P=(0,0,0) A1um_1 SD_5", P1))
    assert matrix_info.operators == (P0, P1)
    xml = matrix_info.to_sigmond_xml()
    root = ET.fromstring(xml)
    assert [operator_from_xml(c) for c in root if c.tag.endswith("String")] == [P0, P1]
    assert root.find("HermitianMatrix") is not None and root.find("SubtractVEV") is None


# ---------------------------------------------------------------------------
# Against Sigmond's C++ code
# ---------------------------------------------------------------------------


@needs_sigmond_module
@pytest.mark.parametrize("text", BL_STRINGS)
def test_basic_laph_strings_match_cpp(text):
    cpp = _sigmond.OperatorInfo(text, _sigmond.OpKind.BasicLapH)
    assert BL(text).to_sigmond() == cpp.op_str()


@needs_sigmond_module
@pytest.mark.parametrize("text", GI_STRINGS)
def test_gen_irrep_strings_match_cpp(text):
    cpp = _sigmond.OperatorInfo(text, _sigmond.OpKind.GenIrrep)
    assert str(GI(text)) == cpp.op_str()


@needs_sigmond_module
def test_tetraquark_short_output_matches_cpp():
    for stem in TETRAQUARK_SHORT.values():
        for color in "pm":
            for disp in ("", " D2", " D3", " D4"):
                text = f"{stem}{color} P=(0,0,0) A1um_1 QDX_1{disp}"
                cpp = _sigmond.OperatorInfo(text, _sigmond.OpKind.BasicLapH).op_str()
                assert BL(text).to_sigmond() == cpp


def _sigmond_query_keys(path: str, group: str) -> list[str]:
    out = subprocess.run(
        ["sigmond_query", "-k", f"{path}[{group}]"], capture_output=True, text=True, check=True
    ).stdout
    assert "Error" not in out, out
    root = ET.fromstring(f"<all>{out[out.index('<') :].replace('Record', '<!--R-->')}</all>")
    keys = []
    for obs in root.iter("MCObservable"):
        arg = obs.find("Arg")
        if arg is not None and arg.text.strip() != "Re":
            continue
        child = next(c for c in obs if c.tag != "Arg")
        text = ET.tostring(child, encoding="unicode").strip()
        keys.append(re.sub(r">\s+<", "><", text))
    return keys


@needs_sigmond_query
def test_ss_written_keys_parse_in_cpp(tmp_path):
    """Sigmond's own parser reads every SS-written key back to the same text."""
    from sigmondsamplings import SigmondSampling, SigmondWriter
    from sigmondsamplings.info import EnsembleInfo, ObservableInfo, SamplingInfo

    ens = EnsembleInfo("ens_cpp", 20)
    si = SamplingInfo("jackknife", 20)
    keys = [
        CorrelatorAtTimeInfo(CorrelatorInfo(SINK, SOURCE), 5, True, True),
        CorrelatorAtTimeInfo(CorrelatorInfo(BL(BL_STRINGS[7]), BL(BL_STRINGS[11])), 0, False, True),
        CorrelatorAtTimeInfo(CorrelatorInfo(GI(GI_STRINGS[2]), BL(BL_STRINGS[3])), 12, True, False),
        CorrelatorAtTimeInfo(CorrelatorInfo(BL(BL_STRINGS[8]), GI(GI_STRINGS[1])), 3),
        VEVInfo(BL("eta P=(0,0,0) A1gp_1 SS_0")),
        VEVInfo(GI(GI_STRINGS[0])),
    ]
    # Every record is complex (Re and Im datasets): the installed sigmond_query
    # was seen to reject some files mixing real-only and complex records.
    items = [
        SigmondSampling(
            np.arange(21.0) + 1j,
            ObservableInfo(k.to_sigmond(), 0, "n", "re", ens),
            si,
            is_complex=True,
        )
        for k in keys
    ]
    path = os.path.join(tmp_path, "keys.hdf5")
    SigmondWriter(create_backups=False).write_hdf5(path, items, group="g")
    parsed = _sigmond_query_keys(path, "g")
    assert sorted(parsed) == sorted(k.to_sigmond() for k in keys)
    assert {parse_key(text) for text in parsed} == set(keys)
