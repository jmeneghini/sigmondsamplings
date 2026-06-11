#!/usr/bin/env python3
"""
Script to add self-describing energy attributes to a Sigmond samplings file.

Loads a Sigmond samplings (or bins) file, converts each observable to its energy
level type where possible, optionally applies non-interacting (NI) pair
assignments from a PyCalQ YAML file, and writes a new HDF5 file. The written
datasets carry the energy metadata (irrep, psq, energy_type, level_index,
ref_particle) and NI pairs as attrs, so the result reads back deterministically
without relying on name-parsing heuristics.

Observables that cannot be interpreted as energy levels are copied through
unchanged (no attrs).

Uses SigmondLoader, the energy-level collection helpers, and SigmondWriter.
"""

import argparse
import logging
import sys
from pathlib import Path

from sigmondsamplings.ensemble_collection import SingleEnsembleCollection

logger = logging.getLogger(__name__)

try:
    from ..energy_level_collection import SingleEnsembleEnergyCollection
    from ..io.loader import DEFAULT_ROOT_PATH, SigmondLoader
    from ..io.writer import SigmondWriter
except ImportError:
    # Handle direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from sigmondsamplings.energy_level_collection import SingleEnsembleEnergyCollection
    from sigmondsamplings.io.loader import DEFAULT_ROOT_PATH, SigmondLoader
    from sigmondsamplings.io.writer import SigmondWriter


def add_energy_attrs(
    input_file: str,
    output_file: str,
    ni_yml: str | None = None,
    ref_particle: str | None = None,
    hdf5_path: str | None = None,
    hdf5_root_path: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Re-pack a Sigmond file with energy (and NI-pair) attrs where possible.

    Args:
        input_file: Input Sigmond file (.smp or .hdf5).
        output_file: Output path (``.hdf5`` enforced).
        ni_yml: Optional PyCalQ YAML with non-interacting pair assignments.
        ref_particle: Optional reference particle name to assign to reference-mode
            energy levels (those with ``is_ref=True``).
        hdf5_path: Path within a multi-path HDF5 input (None = auto-detect).
        hdf5_root_path: Root path for the output (default: input path or DEFAULT_ROOT_PATH).
        overwrite: Overwrite (and back up) an existing output file.

    Returns:
        Path to the written HDF5 file.
    """
    loader = SigmondLoader(filename=input_file, hdf5_path=hdf5_path)
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
    
    root_path = hdf5_root_path or loader.hdf5_path or DEFAULT_ROOT_PATH
    out_path = Path(output_file).with_suffix(".hdf5")
    out_coll.to_hdf5(str(out_path), overwrite=overwrite, root_path=root_path)
    return out_path


def main():
    """Main entry point for the energy-attr tagging script."""
    parser = argparse.ArgumentParser(
        description="Add energy and non-interacting-pair attributes to a Sigmond samplings file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tag energy observables with self-describing attrs
  ss-energy-obs input.hdf5 output.hdf5

  # Also attach non-interacting pairs from a PyCalQ YAML
  ss-energy-obs input.hdf5 output.hdf5 --ni-yml non_interacting.yml

  # Assign a reference particle to reference-mode levels (E/M_ref)
  ss-energy-obs input.hdf5 output.hdf5 --ref-particle L

  # Tag an fstream samplings file, writing HDF5
  ss-energy-obs input.smp output.hdf5

  # Select a specific input path and output root for a multi-path HDF5 file
  ss-energy-obs in.hdf5 out.hdf5 --hdf5-path /path/in/file --hdf5-root-path samplings
        """,
    )

    parser.add_argument("input_file", help="Input Sigmond samplings file (.smp or .hdf5)")
    parser.add_argument("output_file", help="Output HDF5 file")
    parser.add_argument(
        "--ni-yml",
        help="Optional PyCalQ YAML with non-interacting pair assignments",
    )
    parser.add_argument(
        "--ref-particle",
        help="Optional reference particle name to assign to reference-mode levels (E/M_ref)",
    )
    parser.add_argument(
        "--hdf5-path",
        help="Path within input HDF5 file (required for HDF5 inputs with multiple paths)",
    )
    parser.add_argument(
        "--hdf5-root-path",
        help=f"Root path for output HDF5 file (default: input path or {DEFAULT_ROOT_PATH})",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not Path(args.input_file).exists():
        logger.error(f"Input file {args.input_file} does not exist")
        sys.exit(1)

    if args.ni_yml and not Path(args.ni_yml).exists():
        logger.error(f"NI YAML file {args.ni_yml} does not exist")
        sys.exit(1)

    if Path(args.output_file).with_suffix(".hdf5").exists() and not args.force:
        logger.error(f"Output file {args.output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    try:
        add_energy_attrs(
            args.input_file,
            args.output_file,
            ni_yml=args.ni_yml,
            ref_particle=args.ref_particle,
            hdf5_path=args.hdf5_path,
            hdf5_root_path=args.hdf5_root_path,
            overwrite=True,
        )
    except Exception as e:
        logger.error(f"Failed to add energy attrs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
