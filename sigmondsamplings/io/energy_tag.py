"""
Add self-describing energy attributes to a Sigmond samplings file.

Loads a Sigmond samplings (or bins) file, converts each observable to its energy
level type where possible, optionally applies non-interacting (NI) pair
assignments from a PyCalQ YAML file, and writes a new HDF5 file. The written
datasets carry the energy metadata (irrep, psq, energy_type, level_index,
ref_particle) and NI pairs as attrs, so the result reads back deterministically
without relying on name-parsing heuristics.

Observables that cannot be interpreted as energy levels are copied through
unchanged (no attrs).

Library function backing the ``ss energy-tag`` command (see ``sigmondsamplings.cli``).
"""

import logging
from pathlib import Path

from ..energy_level_collection import SingleEnsembleEnergyCollection
from ..ensemble_collection import SingleEnsembleCollection
from .loader import DEFAULT_GROUP, SigmondLoader
from .writer import SigmondWriter

logger = logging.getLogger(__name__)


def add_energy_attrs(
    input_file: str,
    output_file: str,
    ni_yml: str | None = None,
    ref_particle: str | None = None,
    in_group: str | None = None,
    out_group: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Re-pack a Sigmond file with energy (and NI-pair) attrs where possible.

    Args:
        input_file: Input Sigmond file (.smp or .hdf5).
        output_file: Output path. A ``.h5``/``.hdf5`` suffix is preserved; if
            omitted, the input HDF5 suffix is used, with ``.hdf5`` as fallback.
        ni_yml: Optional PyCalQ YAML with non-interacting pair assignments.
        ref_particle: Optional reference particle name to assign to reference-mode
            energy levels (those with ``is_ref=True``).
        in_group: Root group to read from a multi-group HDF5 input (None = auto-detect).
        out_group: Root group for the output (default: input group or DEFAULT_GROUP).
        overwrite: Overwrite (and back up) an existing output file.

    Returns:
        Path to the written HDF5 file.
    """
    loader = SigmondLoader(filename=input_file, group=in_group)
    observables = list(loader.observables)
    if not observables:
        raise ValueError(f"No observables found in {input_file}")

    # Convert each observable to its energy type where possible; keep the
    # original (attr-less) object when it is not an energy level.
    non_energy_samps, energy_samps = [], []
    for samp in observables:
        try:
            energy = samp.as_energy_level()
            energy_samps.append(energy)
        except (ValueError, AttributeError) as e:
            logger.debug(f"""Failed to convert {samp.observable_info.name} to energy observable: {e}
                         \nKeeping original observable without energy attrs.""")
            non_energy_samps.append(samp)
            
    non_energy_coll = SingleEnsembleCollection(non_energy_samps)

    # Apply optional NI pairs / reference particle. The collection wraps the same
    # energy samplings, so these mutate the obs_info that will be written.
    if energy_samps:
        energy_coll = SingleEnsembleEnergyCollection(energy_samps)
        if ref_particle:
            energy_coll.set_ref(ref_particle)
        if ni_yml:
            energy_coll.set_shift_particles_from_pycalq_yml(ni_yml)
        out_coll: SingleEnsembleCollection = non_energy_coll + energy_coll
    else:
        if ni_yml or ref_particle:
            logger.warning("NI YAML / reference particle given but no energy levels found; ignoring.")
        out_coll: SingleEnsembleCollection = non_energy_coll
    
    group = out_group or loader.group or DEFAULT_GROUP
    out_path = SigmondWriter.hdf5_output_path(input_file, output_file)
    out_coll.to_hdf5(str(out_path), overwrite=overwrite, group=group)
    return out_path
