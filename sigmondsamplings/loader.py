"""
Loader module for Sigmond samplings files.
"""

import logging
import re
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .ensemble_collection import SingleEnsembleCollection
from .info import EnsembleInfo, KnownEnsembles, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

logger = logging.getLogger(__name__)
SIGMOND_QUERY_CMD = "sigmond_query"


class SigmondLoader:
    """
    Loader for Sigmond samplings files.

    Automatically detects file format (HDF5 or fstream) and uses the appropriate method:
    - HDF5 files: Direct h5py reading (fast)
    - Fstream files: sigmond_query tool (slower but supports legacy format)

    For HDF5 files with a single data path, the path is auto-detected.
    For HDF5 files with multiple paths, you must specify hdf5_path parameter.

    Provides queryable SingleEnsembleCollection of all loaded observables:
        loader.observables - All loaded samplings
        loader.hdf5_path - The HDF5 path used (None for fstream files)

    Use SingleEnsembleCollection filtering methods to query:
        loader.observables.filter(index=0)
        loader.observables.find(lambda obs: 'PSQ' in obs.name)
        loader.observables.find(lambda obs: re.search(r'PSQ.*', obs.name))
    """

    # TODO: some major issues with appending. Creates a 'working' file, still uses sigmond_query, and doesn't replace the original file with the working file.
    def __init__(
        self,
        filename: str = None,
        hdf5_path: str = None,
    ):
        """
        Initialize the loader.

        Args:
            filename: Path to the samplings file to load upon construction (optional)
            hdf5_path: For HDF5 files, the root path to use (default: None = auto-detect)
                      If None and file has a single path, it will be used automatically.
                      If None and file has multiple paths, an error will be raised.
        """
        # Single source of truth for data
        self._filename = None
        self._hdf5_path = hdf5_path
        self._all_samplings = SingleEnsembleCollection([])

        # Load file if provided
        if filename:
            self.load_file(filename, self._hdf5_path)

    @property
    def observables(self) -> SingleEnsembleCollection:
        """Access the loaded samplings collection."""
        return self._all_samplings

    @property
    def hdf5_path(self) -> str | None:
        """Get the HDF5 path used for loading (None for fstream files)."""
        return self._hdf5_path

    def _clean_hdf5_path(self, path: str) -> str:
        """Clean HDF5 path by removing leading/trailing slashes."""
        return path.strip("/")

    def _is_hdf5_file(self, filename: str) -> bool:
        """Check if a file is an HDF5 file."""
        try:
            with h5py.File(filename, "r"):
                return True
        except OSError:
            return False

    def _verify_hdf5_sigmond_file(self, filename: str) -> tuple[bool, list[str] | None]:
        """
        Verify that an HDF5 file is a valid Sigmond samplings file.

        Returns:
            (is_valid, available_paths)
            - is_valid: True if file has correct Sigmond structure
            - available_paths: List of available data paths (e.g., ["samplings"])
        """
        try:
            with h5py.File(filename, "r") as f:
                # Check for Info group with FIdentifier
                if "Info" not in f or "FIdentifier" not in f["Info"]:
                    return False, None

                # Verify it's a Sigmond samplings file
                fid = f["Info"]["FIdentifier"][()].decode("utf-8")
                if fid != "Sigmond--SamplingsFile":
                    return False, None

                # Find available paths (all groups at root except Info)
                available_paths = []
                for key in f.keys():
                    if key != "Info" and isinstance(f[key], h5py.Group):
                        available_paths.append(key)

                return True, available_paths
        except Exception as e:
            logger.debug(f"HDF5 verification failed: {e}")
            return False, None

    def _load_from_hdf5(self, filename: str, path: str = "samplings") -> SingleEnsembleCollection:
        """
        Load samplings directly from HDF5 file using h5py.

        Args:
            filename: Path to the HDF5 file
            path: Root path in HDF5 file (default: "samplings")

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        with h5py.File(filename, "r") as f:
            # Verify the path exists
            if path not in f:
                available_paths = [k for k in f.keys() if k != "Info"]
                raise ValueError(
                    f"Path '{path}' not found in HDF5 file. Available paths: {available_paths}"
                )

            group = f[path]

            # Extract header XML
            if "Header" not in group:
                raise ValueError(f"No Header dataset found in {path}")

            header_xml = group["Header"][()].decode("utf-8")
            ensemble_info, sampling_info = self._parse_header_xml(header_xml)

            # Load all observable data from Values group
            if "Values" not in group:
                raise ValueError(f"No Values group found in {path}")

            values_group = group["Values"]

            # Parse all datasets in Values group
            observable_infos = []
            all_data = []

            for dataset_name in values_group.keys():
                # Dataset name is the XML key, e.g., "<MCObservable><Info>name<|Info><|MCObservable>"
                # Parse it to get ObservableInfo
                try:
                    obs_info = self._parse_observable_key(dataset_name, ensemble_info)
                    observable_infos.append(obs_info)

                    # Load the data
                    data = values_group[dataset_name][:]
                    all_data.append(data)
                except (ValueError, NotImplementedError) as e:
                    logger.debug(f"Skipping dataset {dataset_name}: {e}")
                    continue

            if not observable_infos:
                raise ValueError("No valid observable data found in file")

            # Build samplings collection
            samplings_list = self._build_samplings_list(observable_infos, all_data, sampling_info)

            return SingleEnsembleCollection(samplings_list)

    def _check_sigmond_query(self):
        """Check if sigmond_query is available."""
        try:
            result = subprocess.run(
                [SIGMOND_QUERY_CMD, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sigmond_query command failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"sigmond_query not found or not working: {e}")

    def _run_sigmond_query(self, filename: str, options: str) -> str:
        """Run sigmond_query with given options."""
        cmd = [SIGMOND_QUERY_CMD] + options.split() + [filename]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                # For HDF5 files, useful info might be in stdout even with non-zero exit code
                if "bad root path" in result.stdout:
                    raise RuntimeError(f"sigmond_query failed: {result.stdout}")
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    raise RuntimeError(f"sigmond_query failed: {error_msg}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("sigmond_query timed out")
        except Exception as e:
            raise RuntimeError(f"sigmond_query error with cmd '{' '.join(cmd)}': {e}")

    def check_file_validity(self, filename: str) -> tuple[bool, str | None, list[str] | None]:
        """
        Check if a file is a valid Sigmond samplings file.

        Returns:
            (is_valid, file_type, hdf5_paths)
            - is_valid: True if file is valid
            - file_type: 'fstream', 'hdf5', or None
            - hdf5_paths: List of available paths for HDF5 files, None otherwise
        """
        try:
            output = self._run_sigmond_query(filename, "-i")

            if "This is a Sigmond samplings file in fstreams format" in output:
                return True, "fstream", None
            elif "This is a Sigmond samplings file in HDF5 format" in output:
                return True, "hdf5", None
            elif "This is a Sigmond bins file in fstreams format" in output:
                return True, "fstream", None
            elif "This is a Sigmond bins file in HDF5 format" in output:
                return True, "hdf5", None
            else:
                return False, None, None

        except RuntimeError as e:
            error_msg = str(e)
            if "bad root path" in error_msg:
                # Extract available paths from error message
                paths = []
                lines = error_msg.split("\n")
                for line in lines:
                    if line.strip().startswith("/") and line.strip().endswith("/"):
                        paths.append(line.strip())
                return True, "hdf5", paths
            elif "This file type is not known to Sigmond" in error_msg:
                return False, None, None
            return False, None, None

    def load_file(self, filename: str, hdf5_path: str = None) -> None:
        """
        Load all data from a file.

        Args:
            filename: Path to the samplings file
            hdf5_path: For HDF5 files, the root path to use (default: None = auto-detect)
                      If None and file has a single path, it will be used automatically.
                      If None and file has multiple paths, an error will be raised.
        """
        # Load samplings
        try:
            self._all_samplings = self._load_samplings_impl(filename, hdf5_path)
            logger.info(f"Successfully loaded {len(self._all_samplings)} samplings from {filename}")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

        self._filename = filename

    def _load_samplings_impl(
        self, filename: str, hdf5_path: str = None
    ) -> SingleEnsembleCollection:
        """
        Load all samplings from a file.

        Args:
            filename: Path to the file
            hdf5_path: For HDF5 files, the root path to use (None = auto-detect)

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        # Auto-detect file type
        if self._is_hdf5_file(filename):
            # Verify it's a valid Sigmond HDF5 file
            is_valid, available_paths = self._verify_hdf5_sigmond_file(filename)
            if not is_valid:
                raise ValueError(f"File {filename} is not a valid Sigmond HDF5 file")

            # Handle path selection
            if hdf5_path is None:
                # Auto-detect path
                if len(available_paths) == 1:
                    hdf5_path = available_paths[0]
                    logger.info(f"Auto-detected single path: '{hdf5_path}'")
                elif len(available_paths) > 1:
                    raise ValueError(
                        f"Multiple paths found in HDF5 file. Please specify hdf5_path parameter.\n"
                        f"Available paths: {available_paths}"
                    )
                else:
                    raise ValueError(f"No data paths found in HDF5 file {filename}")
            else:
                hdf5_path = self._clean_hdf5_path(hdf5_path)
                # User specified a path - verify it exists
                if hdf5_path not in available_paths:
                    raise ValueError(
                        f"Path '{hdf5_path}' not found in HDF5 file. "
                        f"Available paths: {available_paths}"
                    )
                logger.info(f"Using specified path: '{hdf5_path}'")

            # Store the path being used
            self._hdf5_path = hdf5_path

            logger.info(f"Loading HDF5 file {filename} using path '{hdf5_path}'")
            return self._load_from_hdf5(filename, hdf5_path)
        else:
            # Use sigmond_query for fstream files
            logger.info(f"Loading fstream file {filename} using sigmond_query")
            return self._load_from_sigmond_query(filename)

    def _load_from_sigmond_query(self, filename: str) -> SingleEnsembleCollection:
        """Load samplings using sigmond_query (for fstream files)."""
        # Get header info
        header_output = self._run_sigmond_query(filename, "-i")
        ensemble_info, sampling_info = self._parse_header_xml(header_output)
        # Get and parse keys
        keys_output = self._run_sigmond_query(filename, "-k")
        observable_infos = self._parse_keys_from_output(keys_output, ensemble_info)
        if not observable_infos:
            raise ValueError("No observable keys found in the file")
        # Get all values
        values_output = self._run_sigmond_query(filename, "-v")
        all_data = self._parse_all_values(values_output)
        # Build samplings collection
        samplings_list = self._build_samplings_list(observable_infos, all_data, sampling_info)
        return SingleEnsembleCollection(samplings_list)

    def _parse_header_xml(self, xml_string: str) -> tuple[EnsembleInfo, SamplingInfo]:
        """Parse the header XML to extract ensemble and sampling info."""
        try:
            root = ET.fromstring(xml_string.strip())
        except ET.ParseError:
            # Try to find the XML part in the string
            xml_start = xml_string.find("<")
            xml_end = xml_string.rfind(">") + 1
            if xml_start >= 0 and xml_end > xml_start:
                xml_part = xml_string[xml_start:xml_end]
                root = ET.fromstring(xml_part)
            else:
                raise ValueError("Could not parse header XML")

        # Extract ensemble info
        bins_info = root.find(".//MCBinsInfo")
        if bins_info is None:
            raise ValueError("MCBinsInfo not found in header")

        ensemble_name = bins_info.find("MCEnsembleInfo").text
        num_measurements = int(bins_info.find("NumberOfMeasurements").text)
        num_bins = int(bins_info.find("NumberOfBins").text)

        # Extract tweak info if present
        tweak_info = {}
        tweak_element = bins_info.find("TweakEnsemble")
        if tweak_element is not None:
            for child in tweak_element:
                tweak_info[child.tag] = child.text

        # Check for ensemble name in config's ensembles XML file
        known_ensembles = KnownEnsembles()
        try:
            ensemble_info = known_ensembles.get(
                name=ensemble_name, num_bins=num_bins, tweak_info=tweak_info
            )
        except (ValueError, KeyError) as e:
            # Fallback to basic ensemble info
            ensemble_info = EnsembleInfo(
                name=ensemble_name,
                num_bins=num_bins,
                num_measurements=num_measurements,
                tweak_info=tweak_info,
            )
            if e is ValueError:
                logger.info(f"KnownEnsembles not configured: {e}. Using basic ensemble info.")

        # Extract sampling info
        sampling_element = root.find(".//MCSamplingInfo")
        if sampling_element is None:
            raise ValueError("MCSamplingInfo not found in header")

        # Check for Bootstrap or Jackknife
        bootstrap = sampling_element.find(".//Bootstrapper")
        jackknife = sampling_element.find(".//Jackkniffer")  # Note: might be misspelled in XML
        jackknife_simple = sampling_element.find(".//Jackknife")  # Simple self-closing tag

        if bootstrap is not None:
            method = "bootstrap"
            num_resamplings = int(bootstrap.find("NumberResamplings").text)
            seed = int(bootstrap.find("Seed").text)
            boot_skip = int(bootstrap.find("BootSkip").text)
            sampling_info = SamplingInfo(method, num_resamplings, seed, boot_skip)
        elif jackknife is not None:
            method = "jackknife"
            num_resamplings = int(jackknife.find("NumberResamplings").text)
            sampling_info = SamplingInfo(method, num_resamplings)
        elif jackknife_simple is not None:
            method = "jackknife"
            # For simple jackknife, num_resamplings equals num_bins
            num_resamplings = num_bins
            sampling_info = SamplingInfo(method, num_resamplings)
        else:
            raise ValueError("No Bootstrap or Jackknife info found")

        return ensemble_info, sampling_info

    def _parse_observable_key(self, key_xml: str, ensemble_info: EnsembleInfo) -> ObservableInfo:
        """Parse an observable key from XML."""
        try:
            # Convert HDF5-safe format back to standard XML for parsing
            standard_xml = key_xml.replace("<|", "</")
            root = ET.fromstring(standard_xml.strip())
            info_element = root.find(".//Info")
            if info_element is not None:
                info_text = info_element.text.strip()
                return ObservableInfo.from_string(info_text, ensemble_info)
            else:
                # Handle more complex XML structure if needed
                raise NotImplementedError("Complex XML key parsing not yet implemented")
        except ET.ParseError as e:
            raise ValueError(f"Could not parse key XML: {e}")

    def _parse_keys_from_output(
        self, keys_output: str, ensemble_info: EnsembleInfo
    ) -> list[ObservableInfo]:
        """Parse observable keys from sigmond_query output."""
        observable_infos = []
        lines = keys_output.split("\n")
        current_key_lines = []
        in_key = False

        for line in lines:
            if line.startswith("Record ") and ":" in line:
                if current_key_lines:
                    # Process previous key
                    key_xml = "\n".join(current_key_lines)
                    try:
                        obs_info = self._parse_observable_key(key_xml, ensemble_info)
                        observable_infos.append(obs_info)
                    except (ValueError, NotImplementedError):
                        pass  # Skip problematic keys
                current_key_lines = []
                in_key = True
            elif in_key and line.strip():
                current_key_lines.append(line)
            elif in_key and not line.strip() and current_key_lines:
                # Process current key when we hit empty line
                key_xml = "\n".join(current_key_lines)
                try:
                    obs_info = self._parse_observable_key(key_xml, ensemble_info)
                    observable_infos.append(obs_info)
                except (ValueError, NotImplementedError):
                    pass  # Skip problematic keys
                current_key_lines = []
                in_key = False

        # Process last key
        if current_key_lines:
            key_xml = "\n".join(current_key_lines)
            try:
                obs_info = self._parse_observable_key(key_xml, ensemble_info)
                observable_infos.append(obs_info)
            except (ValueError, NotImplementedError):
                pass

        return observable_infos

    def _parse_all_values(self, values_output: str) -> list[np.ndarray]:
        """Parse the output of 'sigmond_query -v' into a list of numpy arrays."""
        all_records_values = []
        lines = values_output.split("\n")
        current_values = []

        in_record = False
        for line in lines:
            if line.startswith("Record ") and ":" in line:
                if current_values:
                    all_records_values.append(np.array(current_values))
                current_values = []
                in_record = True
            elif in_record and "[" in line and "]" in line and "=" in line:
                value_str = line.split("=")[1].strip()
                try:
                    value = float(value_str)
                    current_values.append(value)
                except ValueError:
                    try:
                        value = complex(value_str)
                        current_values.append(value)
                    except ValueError:
                        pass

        if current_values:
            all_records_values.append(np.array(current_values))

        return all_records_values

    def _build_samplings_list(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
        sampling_info: SamplingInfo,
    ) -> list[SigmondSampling]:
        """Build the samplings list from parsed data."""
        if len(all_data) != len(observable_infos):
            raise ValueError("Mismatch between number of observables in header and data records.")

        # Group observables by name and index to find complex pairs
        grouped_observables = {}
        for i, obs_info in enumerate(observable_infos):
            key = (obs_info.name, obs_info.index)
            if key not in grouped_observables:
                grouped_observables[key] = {}
            grouped_observables[key][obs_info.re_im] = (obs_info, i)

        result = []
        for key, parts in grouped_observables.items():
            if "re" in parts and "im" in parts:
                re_info, re_idx = parts["re"]
                im_info, im_idx = parts["im"]
                re_data = all_data[re_idx]
                im_data = all_data[im_idx]
                complex_data = re_data + 1j * im_data
                sampling = SigmondSampling(complex_data, re_info, sampling_info, is_complex=True)
                result.append(sampling)
            elif "re" in parts:
                re_info, re_idx = parts["re"]
                re_data = all_data[re_idx]
                sampling = SigmondSampling(
                    re_data, re_info, sampling_info, is_complex=np.iscomplexobj(re_data)
                )
                result.append(sampling)
            elif "im" in parts:
                im_info, im_idx = parts["im"]
                im_data = all_data[im_idx]
                sampling = SigmondSampling(
                    im_data, im_info, sampling_info, is_complex=np.iscomplexobj(im_data)
                )
                result.append(sampling)

        return result

    # Utility methods for backward compatibility
    def get_file_info(
        self, filename: str = None
    ) -> tuple[EnsembleInfo, SamplingInfo, list[ObservableInfo]]:
        """Get header info and list of observable keys from a file."""
        if filename is not None:
            self.load_file(filename)
        elif self._filename is None:
            raise ValueError("No file loaded. Please provide a filename or call load_file() first.")

        return self._ensemble_info, self._sampling_info, self._observable_infos

    @staticmethod
    def get_name_and_index_from_dict_key(key: str) -> tuple[str, int]:
        """Extract the (name, index) pair from keys like 'observable[3]'."""
        m = re.match(r"^(.*)\[(\d+)\]$", key)
        if not m:
            raise ValueError(
                f"Key '{key}' does not end with an integer in brackets (e.g. 'name[0]')."
            )
        name, index_str = m.groups()
        return name, int(index_str)
