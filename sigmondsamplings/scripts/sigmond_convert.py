#!/usr/bin/env python3
"""
Script to convert between Sigmond file formats.

This script supports bidirectional conversion between:
- .smp/.fstream files to HDF5 format
- HDF5 files to .smp format

Uses SigmondWriter and SigmondLoader for
format handling and conversion.
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from ..loader import SigmondLoader
    from ..writer import SigmondWriter
except ImportError:
    # Handle direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loader import SigmondLoader
    from writer import SigmondWriter


def detect_output_format(output_file: str) -> str:
    """Detect the desired output format from file extension."""
    output_file_lower = output_file.lower()
    if output_file_lower.endswith(".hdf5"):
        return "hdf5"
    elif output_file_lower.endswith(".smp") or output_file_lower.endswith(".dat"):
        return "smp"
    elif output_file_lower.endswith(".fstream"):
        return "fstream"
    else:
        # Default to HDF5 for unknown extensions
        return "hdf5"


def convert_to_smp(input_file: str, output_file: str, hdf5_path: str | None = None):
    """Convert HDF5 file to .smp format using sigmond_query."""

    # Build input filename with path for sigmond_query
    # Note: sigmond_query still uses the old filename[path] syntax
    input_filename = input_file
    path_to_use = hdf5_path

    if input_file.lower().endswith(".hdf5") and not hdf5_path:
        # Auto-detect path for HDF5 files
        loader = SigmondLoader()
        is_valid, file_type, available_paths = loader.check_file_validity(input_file)
        if is_valid and file_type == "hdf5" and available_paths:
            # Use the first available path
            path_to_use = available_paths[0]
            print(f"Using HDF5 path: {path_to_use}")

    # Construct filename[path] format for sigmond_query command
    if path_to_use:
        input_filename = f"{input_file}[{path_to_use}]"

    print(f"Converting {input_filename} to .smp format...")

    try:
        # Use sigmond_query to convert HDF5 to .smp
        # The -w flag writes to .smp format
        cmd = ["sigmond_query", "-w", output_file, input_filename]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"sigmond_query conversion failed: {error_msg}")

        print(f"Successfully converted to {output_file}")
        return output_file

    except subprocess.TimeoutExpired:
        raise RuntimeError("sigmond_query conversion timed out")
    except FileNotFoundError:
        raise RuntimeError(
            "sigmond_query command not found. Please ensure Sigmond is installed and in PATH."
        )
    except Exception:
        raise


def convert_to_hdf5(
    input_file: str,
    output_file: str,
    hdf5_path: str | None = None,
    hdf5_root_path: str = "/data/",
):
    """Convert a Sigmond file to HDF5 format using SigmondWriter."""

    # Initialize writer
    writer = SigmondWriter()

    # For HDF5 inputs with multiple paths, require explicit path specification
    if input_file.lower().endswith(".hdf5") and not hdf5_path:
        loader = SigmondLoader()
        is_valid, file_type, available_paths = loader.check_file_validity(input_file)
        if is_valid and file_type == "hdf5" and available_paths and len(available_paths) > 1:
            paths_str = "\n".join(available_paths)
            raise ValueError(
                f"HDF5 input file has multiple paths. Please specify one with --hdf5-path. "
                f"Available paths:\n{paths_str}"
            )

    print(f"Converting {input_file} to HDF5 format...")

    try:
        # Use SigmondWriter's convert_format method
        output_path = writer.convert_format(
            input_filename=input_file,
            output_filename=output_file,
            output_format="hdf5",
            hdf5_root_path=hdf5_root_path,
            overwrite=True,
        )
        print(f"Successfully converted to {output_path}")
        return output_path

    except Exception:
        raise


def convert_files(
    input_file: str,
    output_file: str,
    output_format: str | None = None,
    hdf5_path: str | None = None,
    hdf5_root_path: str = "/data/",
):
    """
    Convert between Sigmond file formats.

    Args:
        input_file: Input file path
        output_file: Output file path
        output_format: Explicitly specify output format ('hdf5', 'smp', 'fstream')
        hdf5_path: Path within input HDF5 file (for HDF5 inputs)
        hdf5_root_path: Root path for output HDF5 file
    """

    # Detect output format if not specified
    if output_format is None:
        output_format = detect_output_format(output_file)

    print(f"Target output format: {output_format}")

    if output_format == "hdf5":
        return convert_to_hdf5(input_file, output_file, hdf5_path, hdf5_root_path)
    elif output_format in ["smp", "fstream"]:
        return convert_to_smp(input_file, output_file, hdf5_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert between Sigmond file formats (.smp, .fstream, .hdf5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert .smp to HDF5
  sigmond-convert input.smp output.hdf5

  # Convert HDF5 to .smp
  sigmond-convert input.hdf5 output.smp

  # Convert HDF5 with specific path to .smp
  sigmond-convert input.hdf5 output.smp --hdf5-path /path/in/file

  # Convert .smp to HDF5 with custom root path
  sigmond-convert input.smp output.hdf5 --hdf5-root-path /custom/path/

  # Force specific output format
  sigmond-convert input.smp output.dat --output-format hdf5
        """,
    )

    parser.add_argument("input_file", help="Input Sigmond file (.smp, .fstream, or .hdf5)")
    parser.add_argument("output_file", help="Output file")
    parser.add_argument(
        "--output-format",
        choices=["hdf5", "smp", "fstream"],
        help="Force specific output format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--hdf5-path",
        help="Path within input HDF5 file (required for some HDF5 inputs)",
    )
    parser.add_argument(
        "--hdf5-root-path",
        default="/data/",
        help="Root path for output HDF5 file (default: /data/)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)

    # Check if output file exists (SigmondWriter handles this with backups for HDF5)
    if Path(args.output_file).exists() and not args.force and args.output_format == "smp":
        print(f"Error: Output file {args.output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    try:
        convert_files(
            args.input_file,
            args.output_file,
            args.output_format,
            args.hdf5_path,
            args.hdf5_root_path,
        )
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
