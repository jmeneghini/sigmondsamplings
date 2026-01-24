"""
Writer module for Sigmond samplings files.
Automatically converts fstream files to HDF5 format for all operations, providing
full Sigmond format functionality while using HDF5 as the primary working format.
"""

import h5py
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from .sampling import SigmondSampling, ObservableInfo, EnsembleInfo, SamplingInfo
from .loader import SigmondLoader

# Set up logging
logger = logging.getLogger(__name__)


class SigmondWriter:
    """
    Writer for Sigmond samplings files with automatic format conversion.

    This class automatically converts fstream (.smp) files to HDF5 format before
    any operations, ensuring reliable and consistent handling while maintaining
    full Sigmond format compatibility.
    """

    def __init__(self, create_backups: bool = True):
        """
        Initialize the writer.

        Args:
            create_backups: Whether to create numbered backups before modifying existing files
        """
        self.create_backups = create_backups

    def _create_numbered_backup(self, filename: str) -> Optional[Path]:
        """
        Create a numbered backup of the file before modification.

        Args:
            filename: Path to the file to backup

        Returns:
            Path to the backup file, or None if backups are disabled or file doesn't exist
        """
        filepath = Path(filename)

        if not self.create_backups or not filepath.exists():
            return None

        counter = 1
        while (
            backup_path := filepath.with_suffix(
                f"{filepath.suffix}.backup_{counter:03d}"
            )
        ).exists():
            counter += 1

        try:
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup of {filepath}: {e}")
            return None

    def _ensure_hdf5_format(
        self, filename: str, hdf5_root_path: Optional[str] = None
    ) -> Tuple[Path, str]:
        """
        Ensure the input file is in HDF5 format, converting if necessary.

        Args:
            filename: Input file path
            hdf5_root_path: Preferred root path for HDF5 conversion

        Returns:
            Tuple of (hdf5_filename, root_path) ready for operations
        """
        filepath = Path(filename)
        loader = SigmondLoader()

        # Check if already HDF5
        if filepath.suffix.lower() == ".hdf5":
            is_valid, file_type, hdf5_paths = loader.check_file_validity(str(filepath))
            if is_valid and file_type == "hdf5" and hdf5_paths:
                root_path = hdf5_root_path or hdf5_paths[0]
                return filepath, root_path

        # Convert fstream to HDF5
        logger.info(f"Converting {filepath} to HDF5 format for reliable processing...")

        # Create HDF5 filename
        hdf5_filename = filepath.with_stem(f"{filepath.stem}_working").with_suffix(
            ".hdf5"
        )

        # Determine root path
        root_path = hdf5_root_path or "samplings"

        # Load and convert
        loader.load_file(str(filepath), hdf5_path=root_path)
        samplings = list(loader.observables)

        # Write to HDF5 format
        self.write_hdf5(str(hdf5_filename), samplings, root_path, overwrite=True)

        logger.info(f"Conversion complete. Working with: {hdf5_filename}")
        logger.info(f"Original file {filepath} unchanged.")

        return hdf5_filename, root_path

    def _generate_header_xml(
        self, ensemble_info: EnsembleInfo, sampling_info: SamplingInfo
    ) -> str:
        """Generate the XML header for a Sigmond file in the exact format expected."""
        # Use SigmondSamplingsFile as root element to match real format
        root = ET.Element("SigmondSamplingsFile")

        # Add bins info first (matches real format order)
        bins_elem = ET.SubElement(root, "MCBinsInfo")
        ET.SubElement(bins_elem, "MCEnsembleInfo").text = ensemble_info.name
        ET.SubElement(bins_elem, "NumberOfMeasurements").text = str(
            ensemble_info.num_measurements
        )
        ET.SubElement(bins_elem, "NumberOfBins").text = str(ensemble_info.num_bins)

        # Add tweak info if present
        if ensemble_info.tweak_info:
            tweak_elem = ET.SubElement(bins_elem, "TweakEnsemble")
            for key, value in ensemble_info.tweak_info.items():
                ET.SubElement(tweak_elem, key).text = str(value)

        # Add sampling info
        sampling_elem = ET.SubElement(root, "MCSamplingInfo")

        if sampling_info.method == "bootstrap":
            bootstrap_elem = ET.SubElement(sampling_elem, "Bootstrapper")
            ET.SubElement(bootstrap_elem, "NumberResamplings").text = str(
                sampling_info.num_resamplings
            )
            ET.SubElement(bootstrap_elem, "Seed").text = str(sampling_info.seed)
            ET.SubElement(bootstrap_elem, "BootSkip").text = str(
                sampling_info.boot_skip
            )
        elif sampling_info.method == "jackknife":
            if sampling_info.num_resamplings == ensemble_info.num_bins:
                # Simple jackknife
                ET.SubElement(sampling_elem, "Jackknife")
            else:
                # Full jackknife with parameters
                jackknife_elem = ET.SubElement(sampling_elem, "Jackkniffer")
                ET.SubElement(jackknife_elem, "NumberResamplings").text = str(
                    sampling_info.num_resamplings
                )
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_info.method}")

        # Return XML without indentation to match real format (compact)
        return ET.tostring(root, encoding="unicode")

    def _indent_xml(self, elem, level=0):
        """Add proper indentation to XML elements."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _generate_observable_key_xml(self, observable_info: ObservableInfo) -> str:
        """Generate the XML key for an observable."""
        # For HDF5 files, we need to avoid using </> characters that HDF5 interprets as paths
        # For fstream files, we can use standard XML
        # We'll generate both and choose based on context

        info_content = f"{observable_info.name} {observable_info.index} {observable_info.op_type} {observable_info.re_im}"

        # Return standard XML - the write methods will handle format-specific adjustments
        root = ET.Element("MCObservable")
        info_elem = ET.SubElement(root, "Info")
        info_elem.text = info_content

        return ET.tostring(root, encoding="unicode")

    def _make_hdf5_safe_key(self, xml_key: str) -> str:
        """Convert XML key to HDF5-safe format by replacing problematic characters."""
        # Replace closing tags </ with <| to avoid HDF5 path interpretation
        # This converts <Tag>content</Tag> to <Tag>content<|Tag>
        hdf5_key = xml_key.replace("</", "<|")

        # Replace forward slashes with a safe character to avoid HDF5 path interpretation
        # This is critical for observables with names like "PSQ2/G/ecm_0"
        hdf5_key = hdf5_key.replace("/", "|")

        return hdf5_key

    def write_file(
        self,
        filename: str,
        samplings: List[SigmondSampling],
        root_path: str = "/data/",
        overwrite: bool = False,
    ) -> str:
        """
        Write samplings to a Sigmond file.

        All files are written in HDF5 format for reliability and compatibility.

        Args:
            filename: Output file path (will be converted to .hdf5 if needed)
            samplings: List of SigmondSampling objects
            root_path: HDF5 root path for the data
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the written file
        """
        # Ensure output is HDF5 format
        if not filename.lower().endswith(".hdf5"):
            filename = filename.rsplit(".", 1)[0] + ".hdf5"

        self.write_hdf5(filename, samplings, root_path, overwrite)
        return filename

    def write_hdf5(
        self,
        filename: str,
        samplings: List[SigmondSampling],
        root_path: str = "/",
        overwrite: bool = False,
    ) -> None:
        """
        Write samplings to an HDF5 (.hdf5) format file in correct Sigmond format.

        Args:
            filename: Output file path
            samplings: List of SigmondSampling objects
            root_path: HDF5 root path (e.g., "/isosinglet_S0_A1g_1_P0/")
            overwrite: Whether to overwrite existing file
        """
        if Path(filename).exists() and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to overwrite."
            )
        elif Path(filename).exists() and overwrite:
            # Create backup before overwriting
            self._create_numbered_backup(filename)

        if not samplings:
            raise ValueError("No samplings provided")

        # Ensure root_path has proper format and extract group name
        if not root_path.startswith("/"):
            root_path = "/" + root_path
        if not root_path.endswith("/"):
            root_path += "/"

        group_name = root_path.strip("/")

        # Get reference sampling for common info
        first_sampling = samplings[0]
        ensemble_info = first_sampling.observable_info.ensemble_info
        sampling_info = first_sampling.sampling_info

        # Verify all samplings have compatible info
        for i, sampling in enumerate(samplings):
            if sampling.sampling_info != sampling_info:
                raise ValueError(f"Sampling {i} has incompatible sampling info")
            if sampling.observable_info.ensemble_info != ensemble_info:
                raise ValueError(f"Sampling {i} has incompatible ensemble info")

        with h5py.File(filename, "w") as hdf5_file:
            # Create global Info group (required by Sigmond)
            info_group = hdf5_file.create_group("Info")

            # File identifier - match exact format of real files with fixed size
            fid_dtype = h5py.string_dtype(encoding="utf-8", length=23)
            info_group.create_dataset(
                "FIdentifier", data="Sigmond--SamplingsFile", dtype=fid_dtype
            )

            # Endianness - match exact format of real files with fixed size
            end_dtype = h5py.string_dtype(encoding="utf-8", length=2)
            info_group.create_dataset("Endianness", data="L", dtype=end_dtype)

            # Create data group
            data_group = hdf5_file.create_group(group_name)

            # Generate XML header and store only in data group (NOT in Info group!)
            header_xml = self._generate_header_xml(ensemble_info, sampling_info)
            # Use fixed-length UTF-8 encoding to match real files exactly
            header_len = len(header_xml.encode("utf-8")) + 1  # +1 for null termination
            header_dtype = h5py.string_dtype(encoding="utf-8", length=header_len)
            data_group.create_dataset("Header", data=header_xml, dtype=header_dtype)

            # Include checksums flag - match exact format of real files
            cks_dtype = h5py.string_dtype(encoding="utf-8", length=2)
            data_group.create_dataset("IncludeCKS", data="N", dtype=cks_dtype)

            # Create Values group to hold all observables
            values_group = data_group.create_group("Values")

            # Write each sampling as a dataset with XML key as name
            for sampling in samplings:
                if sampling.is_complex:
                    # Write real part
                    re_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        "re",
                        sampling.observable_info.ensemble_info,
                    )
                    re_key_xml = self._generate_observable_key_xml(re_obs_info)
                    re_key_safe = self._make_hdf5_safe_key(re_key_xml)
                    # Ensure data is float64 for compatibility
                    re_data = np.real(sampling.data).astype(np.float64)
                    values_group.create_dataset(re_key_safe, data=re_data)

                    # Write imaginary part
                    im_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        "im",
                        sampling.observable_info.ensemble_info,
                    )
                    im_key_xml = self._generate_observable_key_xml(im_obs_info)
                    im_key_safe = self._make_hdf5_safe_key(im_key_xml)
                    # Ensure data is float64 for compatibility
                    im_data = np.imag(sampling.data).astype(np.float64)
                    values_group.create_dataset(im_key_safe, data=im_data)

                else:
                    # Write real data
                    key_xml = self._generate_observable_key_xml(
                        sampling.observable_info
                    )
                    key_safe = self._make_hdf5_safe_key(key_xml)
                    # Ensure data is float64 for compatibility
                    real_data = sampling.data.astype(np.float64)
                    values_group.create_dataset(key_safe, data=real_data)

    def restore_from_backup(
        self, filename: str, backup_number: Optional[int] = None
    ) -> None:
        """
        Restore a file from its numbered backup.

        Args:
            filename: Path to the original file
            backup_number: Specific backup number to restore (if None, restores latest)
        """
        if backup_number is not None:
            backup_path = f"{filename}.backup_{backup_number:03d}"
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"Backup file {backup_path} does not exist")
        else:
            # Find latest backup
            counter = 1
            latest_backup = None
            while Path(f"{filename}.backup_{counter:03d}").exists():
                latest_backup = f"{filename}.backup_{counter:03d}"
                counter += 1
            if latest_backup is None:
                raise FileNotFoundError(f"No backups found for file {filename}")
            backup_path = latest_backup

        # Restore the backup
        shutil.copy2(backup_path, filename)
        logger.info(f"Restored {filename} from backup {backup_path}")

    def delete_backups(
        self, filename: str, backup_numbers: Optional[List[int] | int] = None
    ) -> None:
        """
        Delete all backups associated with a file.

        Args:
            filename: Path to the original file
            backup_numbers: Specific backup numbers to delete (if None, deletes all)
        """
        if backup_numbers is not None:
            if isinstance(backup_numbers, int):
                backup_numbers = [backup_numbers]
            for number in backup_numbers:
                backup_path = f"{filename}.backup_{number:03d}"
                if Path(backup_path).exists():
                    os.remove(backup_path)
                    logger.info(f"Deleted backup: {backup_path}")
                else:
                    logger.warning(f"Backup file {backup_path} does not exist")
        else:
            # Delete all backups
            counter = 1
            deleted_any = False
            while True:
                backup_path = f"{filename}.backup_{counter:03d}"
                if Path(backup_path).exists():
                    os.remove(backup_path)
                    logger.info(f"Deleted backup: {backup_path}")
                    deleted_any = True
                    counter += 1
                else:
                    break
            if not deleted_any:
                logger.info(f"No backups found for file {filename}")

    def append_to_file(
        self,
        filename: str,
        new_samplings: List[SigmondSampling],
        overwrite: bool = True,
        hdf5_root_path: Optional[str] = None,
    ) -> str:
        """
        Append new samplings to an existing file.

        Automatically converts fstream files to HDF5 format for reliable processing.

        Args:
            filename: Path to existing file
            new_samplings: List of new samplings to add
            overwrite: Whether to overwrite existing observables with same keys
            hdf5_root_path: Root path for HDF5 files (if None, auto-detect)

        Returns:
            Path to the modified file
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        # Ensure we're working with HDF5 format
        hdf5_filename, root_path = self._ensure_hdf5_format(filename, hdf5_root_path)

        # Create backup before modification
        self._create_numbered_backup(hdf5_filename)

        # Perform append on HDF5 file
        self._append_to_hdf5(
            hdf5_filename, new_samplings, overwrite, root_path.strip("/")
        )

        return hdf5_filename

    def _append_to_hdf5(
        self,
        filename: str,
        new_samplings: List[SigmondSampling],
        overwrite: bool,
        root_path: Optional[str] = None,
    ) -> None:
        if not new_samplings:
            raise ValueError("No samplings provided")

        with h5py.File(filename, "r") as f:
            if root_path is None:
                # Find existing data groups (exclude 'Info' group)
                data_groups = [key for key in f.keys() if key != "Info"]
                if not data_groups:
                    raise ValueError("No data groups found in HDF5 file")
                root_path = data_groups[0]

            if root_path not in f:
                raise ValueError(f"Data group {root_path} not found in file")

            data_group = f[root_path]
            if "Values" not in data_group:
                raise ValueError(f"Values group not found in data group {root_path}")
        # Validate compatibility with existing file structure
        # close file before validation to avoid locking issues
        filename_with_path = f"{filename}[{root_path}]"
        self._validate_samplings_compatibility(filename_with_path, new_samplings)

        # open in write mode to append and delete existing datasets if needed
        with h5py.File(filename, "r+") as f:
            # Add new samplings
            data_group = f[root_path]
            values_group = data_group["Values"]
            for sampling in new_samplings:
                if sampling.is_complex:
                    # Write real part
                    re_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        "re",
                        sampling.observable_info.ensemble_info,
                    )
                    re_key_xml = self._generate_observable_key_xml(re_obs_info)
                    re_key_safe = self._make_hdf5_safe_key(re_key_xml)

                    # Handle existing dataset
                    if re_key_safe in values_group:
                        if not overwrite:
                            obs_name = f"{sampling.observable_info.name}[{sampling.observable_info.index}]"
                            raise FileExistsError(
                                f"Observable {obs_name} (real part) already exists. "
                                f"Use overwrite=True to replace it."
                            )
                        del values_group[re_key_safe]

                    # Ensure data is float64 for compatibility
                    re_data = np.real(sampling.data).astype(np.float64)
                    values_group.create_dataset(re_key_safe, data=re_data)

                    # Write imaginary part
                    im_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        "im",
                        sampling.observable_info.ensemble_info,
                    )
                    im_key_xml = self._generate_observable_key_xml(im_obs_info)
                    im_key_safe = self._make_hdf5_safe_key(im_key_xml)

                    # Handle existing dataset
                    if im_key_safe in values_group:
                        if not overwrite:
                            obs_name = f"{sampling.observable_info.name}[{sampling.observable_info.index}]"
                            raise FileExistsError(
                                f"Observable {obs_name} (imaginary part) already exists. "
                                f"Use overwrite=True to replace it."
                            )
                        del values_group[im_key_safe]

                    # Ensure data is float64 for compatibility
                    im_data = np.imag(sampling.data).astype(np.float64)
                    values_group.create_dataset(im_key_safe, data=im_data)

                else:
                    # Write real data
                    key_xml = self._generate_observable_key_xml(
                        sampling.observable_info
                    )
                    key_safe = self._make_hdf5_safe_key(key_xml)

                    # Handle existing dataset
                    if key_safe in values_group:
                        if not overwrite:
                            obs_name = str(sampling.observable_info)
                            raise FileExistsError(
                                f"Observable {obs_name} already exists. "
                                f"Use overwrite=True to replace it."
                            )
                        del values_group[key_safe]

                    # Ensure data is float64 for compatibility
                    real_data = sampling.data.astype(np.float64)
                    values_group.create_dataset(key_safe, data=real_data)

    def _validate_samplings_compatibility(
        self, filename: str, new_samplings: List[SigmondSampling]
    ) -> None:
        """Validate that new samplings are compatible with existing file structure.

        Args:
            filename: Path to existing HDF5 file
            new_samplings: List of new samplings to validate

        Raises:
            ValueError: If samplings are incompatible with existing file
        """
        if not new_samplings:
            return

        # Load existing file - use collection's shared properties
        loader = SigmondLoader(filename=filename)
        existing = loader.observables

        if not existing:
            raise ValueError(f"No existing samplings found in {filename}")

        # Use SingleEnsembleCollection's shared properties
        # These are guaranteed to be consistent across all samplings
        try:
            ref_ensemble = existing.ensemble_info
            ref_sampling_info = existing.sampling_info
        except Exception as e:
            raise ValueError(
                f"Failed to read existing file for compatibility check: {e}"
            )

        # Validate new samplings - check both against file and internal consistency
        first_new = new_samplings[0]
        for i, samp in enumerate(new_samplings):
            if samp.sampling_info != ref_sampling_info:
                raise ValueError(
                    f"New sampling {i} has incompatible sampling info with file"
                )
            if samp.observable_info.ensemble_info != ref_ensemble:
                raise ValueError(
                    f"New sampling {i} has incompatible ensemble info with file"
                )
            # Check internal consistency
            if samp.sampling_info != first_new.sampling_info:
                raise ValueError(f"New sampling {i} has inconsistent sampling info")
            if (
                samp.observable_info.ensemble_info
                != first_new.observable_info.ensemble_info
            ):
                raise ValueError(f"New sampling {i} has inconsistent ensemble info")

    def convert_format(
        self,
        input_filename: str,
        output_filename: str,
        output_format: str = "hdf5",
        hdf5_root_path: str = "/data/",
        overwrite: bool = False,
    ) -> str:
        """
        Convert Sigmond files to HDF5 format (primary supported format).

        Args:
            input_filename: Input file path
            output_filename: Output file path
            output_format: Output format ('hdf5' - only supported format)
            hdf5_root_path: Root path for HDF5 output
            overwrite: Whether to overwrite existing output file

        Returns:
            Path to the converted file
        """
        # Ensure HDF5 output
        outpath = Path(output_filename).with_suffix(".hdf5")

        # Load and convert
        loader = SigmondLoader(filename=input_filename)
        samplings_list = list(loader.observables)

        # Write to HDF5
        self.write_hdf5(
            str(outpath), samplings_list, hdf5_root_path, overwrite=overwrite
        )

        return str(outpath)

    def modify_observable(
        self,
        filename: str,
        observable_name: str,
        observable_index: int,
        new_data: np.ndarray,
        hdf5_root_path: Optional[str] = None,
    ) -> str:
        """
        Modify an existing observable in a file.

        Automatically converts fstream files to HDF5 format for reliable processing.

        Args:
            filename: Path to the file
            observable_name: Name of the observable to modify
            observable_index: Index of the observable
            new_data: New data array
            hdf5_root_path: Root path for HDF5 files (if None, auto-detect)

        Returns:
            Path to the modified file
        """
        from .loader import SigmondLoader

        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        # Ensure we're working with HDF5 format
        hdf5_filename, root_path = self._ensure_hdf5_format(filename, hdf5_root_path)

        # Create backup before modification
        self._create_numbered_backup(str(hdf5_filename))

        # Load samplings from HDF5 file
        loader = SigmondLoader()
        loader.load_file(str(hdf5_filename), hdf5_path=root_path)
        samplings = loader.observables

        # Find the target observable
        original_sampling = samplings.find(name=observable_name, index=observable_index)
        if original_sampling is None:
            # Get first few observables for error message
            available = [f"{s.name} {s.index}" for s in list(samplings)[:5]]
            raise ValueError(
                f"Observable {observable_name} {observable_index} not found in file. "
                f"Available observables: {available}..."
            )

        # Create modified sampling
        modified_sampling = SigmondSampling(
            new_data,
            original_sampling.observable_info,
            original_sampling.sampling_info,
            original_sampling.is_complex,
        )

        # Build new list with modification
        samplings_list = [
            modified_sampling if s is original_sampling else s for s in samplings
        ]

        # Write back to HDF5 file
        self.write_hdf5(str(hdf5_filename), samplings_list, root_path, overwrite=True)

        return str(hdf5_filename)
