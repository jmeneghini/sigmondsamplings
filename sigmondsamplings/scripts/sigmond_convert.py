#!/usr/bin/env python3
"""
Script to convert Sigmond files to HDF5 format.

Reads any Sigmond file (fstream ``.smp``/``.bins`` or ``.hdf5``) and writes it
out in HDF5 format. HDF5 is the only output format supported by SigmondWriter;
writing fstream is not supported.

Uses SigmondWriter and SigmondLoader for format handling and conversion.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ..loader import DEFAULT_ROOT_PATH, SigmondLoader
    from ..writer import SigmondWriter
except ImportError:
    # Handle direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loader import DEFAULT_ROOT_PATH, SigmondLoader
    from writer import SigmondWriter


def convert_to_hdf5(
    input_file: str,
    output_file: str,
    hdf5_path: str | None = None,
    hdf5_root_path: str = DEFAULT_ROOT_PATH,
) -> Path:
    """Convert a Sigmond file to HDF5 format using SigmondWriter."""

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

    logger.info(f"Converting {input_file} to HDF5 format...")
    output_path = writer.convert_format(
        input_filename=input_file,
        output_filename=output_file,
        output_format="hdf5",
        hdf5_root_path=hdf5_root_path,
        overwrite=True,
    )
    logger.info(f"Successfully converted to {output_path}")
    return output_path


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert a Sigmond file (.smp, .bins, .fstream, .hdf5) to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert .smp to HDF5
  sigmond-convert input.smp output.hdf5

  # Convert an fstream bins file to HDF5
  sigmond-convert input.bins output.hdf5

  # Convert .smp to HDF5 with custom root path
  sigmond-convert input.smp output.hdf5 --hdf5-root-path /custom/path/

  # Re-pack an HDF5 file, selecting a specific input path
  sigmond-convert input.hdf5 output.hdf5 --hdf5-path /path/in/file
        """,
    )

    parser.add_argument("input_file", help="Input Sigmond file (.smp, .bins, .fstream, or .hdf5)")
    parser.add_argument("output_file", help="Output HDF5 file")
    parser.add_argument(
        "--hdf5-path",
        help="Path within input HDF5 file (required for HDF5 inputs with multiple paths)",
    )
    parser.add_argument(
        "--hdf5-root-path",
        default=DEFAULT_ROOT_PATH,
        help=f"Root path for output HDF5 file (default: {DEFAULT_ROOT_PATH})",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not Path(args.input_file).exists():
        logger.error(f"Input file {args.input_file} does not exist")
        sys.exit(1)

    if Path(args.output_file).exists() and not args.force:
        logger.error(f"Output file {args.output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    try:
        convert_to_hdf5(
            args.input_file,
            args.output_file,
            args.hdf5_path,
            args.hdf5_root_path,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
