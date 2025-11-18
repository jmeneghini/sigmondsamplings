#!/usr/bin/env python3
"""
Script to combine multiple Sigmond files into a single HDF5 file.

This script loads multiple Sigmond files (.smp, .bins, .fstream, .hdf5, etc) and combines
them into a single HDF5 output file.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from ..writer import SigmondWriter
    from ..loader import SigmondLoader
    from ..sampling import SigmondSampling
except ImportError:
    # Handle direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from writer import SigmondWriter
    from loader import SigmondLoader
    from sampling import SigmondSampling


def resolve_paths(input_files: List[str], base_path: Optional[str] = None) -> List[str]:
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


def load_all_samplings(
    input_files: List[str], verbose: bool = False
) -> Dict[str, SigmondSampling]:
    """
    Load all samplings from multiple input files.

    Args:
        input_files: List of resolved input file paths
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary mapping observable keys to SigmondSampling objects
    """
    loader = SigmondLoader(enable_caching=False)
    all_samplings = {}

    for i, input_file in enumerate(input_files, 1):
        if verbose:
            print(f"Loading file {i}/{len(input_files)}: {Path(input_file).name}")

        try:
            # Handle HDF5 files that might need path specification
            if input_file.lower().endswith(".hdf5") and "[" not in input_file:
                is_valid, file_type, hdf5_paths = loader.check_file_validity(input_file)
                if is_valid and file_type == "hdf5" and hdf5_paths:
                    if len(hdf5_paths) > 1:
                        print(
                            f"Warning: HDF5 file {input_file} has multiple paths. Using: {hdf5_paths[0]}"
                        )
                        if verbose:
                            print(f"Available paths: {', '.join(hdf5_paths)}")
                    input_file_with_path = f"{input_file}[{hdf5_paths[0]}]"
                    loader.load_file(input_file_with_path)
                else:
                    loader.load_file(input_file)
            else:
                loader.load_file(input_file)

            samplings = loader.get_observables()

            if verbose:
                print(f"  Loaded {len(samplings)} observables")

            # Check for conflicts with existing observables
            conflicts = set(all_samplings.keys()) & set(samplings.keys())
            if conflicts:
                print(
                    f"Warning: Observable conflicts detected in {Path(input_file).name}:"
                )
                for conflict in sorted(conflicts):
                    print(f"  - {conflict} (overwriting previous)")

            # Add samplings to combined collection
            all_samplings.update(samplings)

        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            raise

    return all_samplings


def validate_compatibility(
    samplings: Dict[str, SigmondSampling], verbose: bool = False
) -> None:
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
        print(
            f"Reference sampling info: {ref_sampling_info.method}, {ref_sampling_info.num_resamplings} resamplings"
        )
        print(f"Reference ensemble: {ref_ensemble_info.ensemble_name}")

    incompatible_samplings = []
    incompatible_ensembles = []

    for key, sampling in samplings.items():
        # Check sampling compatibility
        if sampling.sampling_info != ref_sampling_info:
            incompatible_samplings.append(key)

        # Check ensemble compatibility (different ensembles are allowed)
        if sampling.observable_info.ensemble_info != ref_ensemble_info:
            ensemble_name = sampling.observable_info.ensemble_info.ensemble_name
            if ensemble_name not in incompatible_ensembles:
                incompatible_ensembles.append(ensemble_name)

    if incompatible_samplings:
        print(
            f"Error: Found {len(incompatible_samplings)} samplings with incompatible sampling info:"
        )
        for key in incompatible_samplings[:5]:  # Show first 5
            print(f"  - {key}")
        if len(incompatible_samplings) > 5:
            print(f"  ... and {len(incompatible_samplings) - 5} more")
        raise ValueError("Incompatible sampling information detected")

    if verbose and incompatible_ensembles:
        print(f"Note: Found multiple ensembles (this is allowed):")
        print(f"  - {ref_ensemble_info.ensemble_name} (reference)")
        for ensemble in incompatible_ensembles:
            print(f"  - {ensemble}")


def combine_files(
    input_files: List[str],
    output_file: str,
    hdf5_root_path: str = "/data/",
    base_path: Optional[str] = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Combine multiple Sigmond files into a single HDF5 file.

    Args:
        input_files: List of input file paths (relative or absolute)
        output_file: Output HDF5 file path
        hdf5_root_path: Root path for HDF5 output
        base_path: Base path for resolving relative input paths
        verbose: Whether to print detailed progress information
        overwrite: Whether to overwrite existing output file

    Returns:
        Path to the combined output file
    """
    # Resolve input file paths
    if verbose:
        print(f"Resolving {len(input_files)} input file paths...")
    resolved_files = resolve_paths(input_files, base_path)

    if verbose:
        print("Resolved input files:")
        for f in resolved_files:
            print(f"  - {f}")

    # Load all samplings
    print(f"Loading samplings from {len(resolved_files)} files...")
    all_samplings = load_all_samplings(resolved_files, verbose)

    print(f"Loaded {len(all_samplings)} total observables")

    # Validate compatibility
    if verbose:
        print("Validating sampling compatibility...")
    validate_compatibility(all_samplings, verbose)
    print("All samplings are compatible")

    # Ensure output is HDF5 format
    if not output_file.lower().endswith(".hdf5"):
        output_file = output_file.rsplit(".", 1)[0] + ".hdf5"
        print(f"Output file adjusted to HDF5 format: {output_file}")

    # Check if output file exists
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_file} already exists. Use --overwrite to replace it."
        )

    # Write combined file using SigmondWriter
    print(f"Writing combined file to {output_file}...")
    writer = SigmondWriter(create_backups=True)

    # Convert dict to list for SigmondWriter
    samplings_list = list(all_samplings.values())

    final_output = writer.write_file(
        filename=output_file,
        samplings=samplings_list,
        root_path=hdf5_root_path,
        overwrite=overwrite,
    )

    print(f"Successfully combined {len(input_files)} files into {final_output}")
    print(f"Combined file contains {len(all_samplings)} observables")

    return final_output


def main():
    """Main entry point for the combine script."""
    parser = argparse.ArgumentParser(
        description="Combine multiple Sigmond files into a single HDF5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine files in current directory
  sigmond-combine file1.smp file2.hdf5 file3.smp -o combined.hdf5

  # Combine with relative paths from a base directory
  sigmond-combine ../data/*.smp results/*.hdf5 -o combined.hdf5 --base-path /path/to/project

  # Combine with custom HDF5 root path and verbose output
  sigmond-combine *.smp -o combined.hdf5 --hdf5-root-path /ensemble_A/ --verbose

  # Overwrite existing output file
  sigmond-combine file1.smp file2.smp -o existing.hdf5 --overwrite

Note: All input files must have compatible sampling information (bootstrap/jackknife
parameters) but can come from different ensembles.
        """,
    )

    parser.add_argument(
        "input_files", nargs="+", help="Input Sigmond files (.smp, .fstream, .hdf5)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--hdf5-root-path",
        default="/data/",
        help="Root path for output HDF5 file (default: /data/)",
    )
    parser.add_argument(
        "--base-path",
        help="Base path for resolving relative input paths (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    args = parser.parse_args()

    try:
        combine_files(
            input_files=args.input_files,
            output_file=args.output,
            hdf5_root_path=args.hdf5_root_path,
            base_path=args.base_path,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
