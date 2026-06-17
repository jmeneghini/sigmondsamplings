"""
Combine multiple Sigmond files into a single HDF5 file.

Loads multiple Sigmond files (.smp, .bins, .fstream, .hdf5, etc) and combines
them into a single HDF5 output file.

Library functions backing the ``ss combine`` command (see ``sigmondsamplings.cli``).
"""

import logging
from pathlib import Path

from ..sampling import SigmondSampling
from .loader import DEFAULT_GROUP, SigmondLoader
from .writer import SigmondWriter

logger = logging.getLogger(__name__)


def resolve_paths(input_files: list[str], base_path: str | None = None) -> list[str]:
    """
    Resolve input file paths, handling both relative and absolute paths.

    Args:
        input_files: List of input file paths
        base_path: Base path for relative paths (defaults to current directory)

    Returns:
        List of resolved absolute paths
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path).resolve()

    resolved_paths = []
    for file_path in input_files:
        path = Path(file_path)
        if not path.is_absolute():
            path = base_path / path

        resolved_path = path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input file not found: {resolved_path}")

        resolved_paths.append(str(resolved_path))

    return resolved_paths


def load_all_samplings(input_files: list[str], verbose: bool = False) -> dict[str, SigmondSampling]:
    """
    Load all samplings from multiple input files.

    Args:
        input_files: List of resolved input file paths
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary mapping observable keys to SigmondSampling objects
    """
    loader = SigmondLoader()
    all_samplings = {}

    for i, input_file in enumerate(input_files, 1):
        if verbose:
            logger.info(f"Loading file {i}/{len(input_files)}: {Path(input_file).name}")

        try:
            # Load file (loader auto-detects format and path)
            loader.load_file(input_file)

            samplings_collection = loader.observables

            if verbose:
                logger.info(f"  Loaded {len(samplings_collection)} observables")

            # Convert collection to dict with string keys "name index"
            samplings = {
                f"{s.observable_info.name} {s.observable_info.index}": s
                for s in samplings_collection
            }

            # Check for conflicts with existing observables
            conflicts = set(all_samplings.keys()) & set(samplings.keys())
            if conflicts:
                logger.warning(f"Observable conflicts detected in {Path(input_file).name}:")
                for conflict in sorted(conflicts):
                    logger.warning(f"  - {conflict} (overwriting previous)")

            # Add samplings to combined collection
            all_samplings.update(samplings)

        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")
            raise

    return all_samplings


def validate_compatibility(samplings: dict[str, SigmondSampling], verbose: bool = False) -> None:
    """
    Validate that all samplings are compatible for combination.

    Args:
        samplings: Dictionary of samplings to validate
        verbose: Whether to print detailed validation information

    Raises:
        ValueError: If samplings are incompatible
    """
    if not samplings:
        raise ValueError("No samplings to combine")

    # Get reference sampling for compatibility checking
    first_key = next(iter(samplings.keys()))
    reference_sampling = samplings[first_key]
    ref_sampling_info = reference_sampling.sampling_info
    ref_ensemble_info = reference_sampling.observable_info.ensemble_info

    if verbose:
        logger.info(
            f"Reference sampling info: {ref_sampling_info.method}, {ref_sampling_info.num_resamplings} resamplings"
        )
        logger.info(f"Reference ensemble: {ref_ensemble_info.name}")

    incompatible_samplings = []
    incompatible_ensembles = []

    for key, sampling in samplings.items():
        # Check sampling compatibility
        if sampling.sampling_info != ref_sampling_info:
            incompatible_samplings.append(key)

        # Check ensemble compatibility (different ensembles are allowed)
        if sampling.observable_info.ensemble_info != ref_ensemble_info:
            ensemble_name = sampling.observable_info.ensemble_info.name
            if ensemble_name not in incompatible_ensembles:
                incompatible_ensembles.append(ensemble_name)

    if incompatible_samplings:
        logger.error(
            f"Found {len(incompatible_samplings)} samplings with incompatible sampling info:"
        )
        for key in incompatible_samplings[:5]:  # Show first 5
            logger.error(f"  - {key}")
        if len(incompatible_samplings) > 5:
            logger.error(f"  ... and {len(incompatible_samplings) - 5} more")
        raise ValueError("Incompatible sampling information detected")

    if verbose and incompatible_ensembles:
        logger.info("Note: Found multiple ensembles (this is allowed):")
        logger.info(f"  - {ref_ensemble_info.name} (reference)")
        for ensemble in incompatible_ensembles:
            logger.info(f"  - {ensemble}")


def combine_files(
    input_files: list[str],
    output_file: str,
    group: str = DEFAULT_GROUP,
    base_path: str | None = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Combine multiple Sigmond files into a single HDF5 file.

    Args:
        input_files: List of input file paths (relative or absolute)
        output_file: Output HDF5 file path
        group: HDF5 root group to write the combined output under
        base_path: Base path for resolving relative input paths
        verbose: Whether to print detailed progress information
        overwrite: Whether to overwrite existing output file

    Returns:
        Path to the combined output file
    """
    # Resolve input file paths
    if verbose:
        logger.info(f"Resolving {len(input_files)} input file paths...")
    resolved_files = resolve_paths(input_files, base_path)

    if verbose:
        logger.info("Resolved input files:")
        for f in resolved_files:
            logger.info(f"  - {f}")

    # Load all samplings
    logger.info(f"Loading samplings from {len(resolved_files)} files...")
    all_samplings = load_all_samplings(resolved_files, verbose)

    logger.info(f"Loaded {len(all_samplings)} total observables")

    # Validate compatibility
    if verbose:
        logger.info("Validating sampling compatibility...")
    validate_compatibility(all_samplings, verbose)
    logger.info("All samplings are compatible")

    # Check if output file exists
    output_path = SigmondWriter.hdf5_output_path(resolved_files[0], output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --overwrite to replace it."
        )

    # Write combined file using SigmondWriter
    logger.info(f"Writing combined file to {output_path}...")
    writer = SigmondWriter(create_backups=True)

    # Convert dict to list for SigmondWriter
    samplings_list = list(all_samplings.values())

    final_output = writer.write_file(
        filename=str(output_path),
        samplings=samplings_list,
        group=group,
        overwrite=overwrite,
    )

    logger.info(f"Successfully combined {len(input_files)} files into {final_output}")
    logger.info(f"Combined file contains {len(all_samplings)} observables")

    return final_output
