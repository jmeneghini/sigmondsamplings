"""Small physics label helpers, independent of plotting backends and identity."""
from slat import get_energy_type_latex_str, get_particle_latex_str


def coordinate_label(label: str, value="t") -> str:
    """Substitute literal coordinate tokens, leaving ordinary LaTeX braces alone."""
    if "{VAR}" in label or "{var}" in label:
        return label.replace("{VAR}", str(value)).replace("{var}", str(value))
    return label + f"({value})"


def energy_symbol(energy_type=None, *, level_index=None, hadron=None, psq=None):
    if hadron is not None and psq == 0:
        return f"M_{{{get_particle_latex_str(hadron)}}}"
    return get_energy_type_latex_str(energy_type, level_index) if energy_type else "E"


def normalized_energy_symbol(symbol, ref_particle=None):
    if ref_particle is None:
        return f"a_t {symbol}"
    denominator = r"M_{\text{ref}}" if ref_particle == "ref" else f"M_{{{get_particle_latex_str(ref_particle)}}}"
    return f"{symbol}/{denominator}"


def rotation_label(index):
    return rf"\mathrm{{ROT}}\ {index}"


def effective_energy_label(*, rot_idx=None, ref_particle=None, lattice=False):
    symbol = r"E_{\mathrm{eff}}"
    if lattice or ref_particle is not None:
        symbol = normalized_energy_symbol(symbol, ref_particle)
    if rot_idx is not None:
        symbol += rf"\,({rotation_label(rot_idx)})"
    return symbol


def correlator_label(correlator, time="t", *, rot_idx=None):
    label = coordinate_label(rf"C[{correlator.sink.latex_str};\,{correlator.source.latex_str}]", time)
    return label if rot_idx is None else label + rf"\,({rotation_label(rot_idx)})"
