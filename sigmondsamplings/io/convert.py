"""
Convert Sigmond files to HDF5 format.

Reads any Sigmond file (fstream ``.smp``/``.bins`` or ``.hdf5``) and writes it
out in HDF5 format. HDF5 is the only output format supported by SigmondWriter;
writing fstream is not supported.

Library function backing the ``ss convert`` command (see ``sigmondsamplings.cli``).
"""

import logging
from pathlib import Path

from .loader import DEFAULT_GROUP, SigmondLoader
from .writer import SigmondWriter

logger = logging.getLogger(__name__)


def convert_to_hdf5(
    input_file: str,
    output_file: str,
    in_group: str | None = None,
    out_group: str = DEFAULT_GROUP,
) -> Path:
    """Convert a Sigmond file to HDF5 format using SigmondWriter."""

    writer = SigmondWriter()

    # For HDF5 inputs with multiple root groups, require an explicit input group.
    if input_file.lower().endswith(".hdf5") and not in_group:
        loader = SigmondLoader()
        is_valid, file_type, available_groups = loader.check_file_validity(input_file)
        if is_valid and file_type == "hdf5" and available_groups and len(available_groups) > 1:
            groups_str = "\n".join(available_groups)
            raise ValueError(
                f"HDF5 input file has multiple root groups. Please specify one with --in-group. "
                f"Available groups:\n{groups_str}"
            )

    logger.info(f"Converting {input_file} to HDF5 format...")
    output_path = writer.convert_format(
        input_filename=input_file,
        output_filename=output_file,
        output_format="hdf5",
        group=out_group,
        overwrite=True,
    )
    logger.info(f"Successfully converted to {output_path}")
    return output_path
