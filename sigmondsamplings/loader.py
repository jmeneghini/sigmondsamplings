"""
Loader module for Sigmond samplings files.
"""

import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from .sampling import SigmondSampling, ObservableInfo, EnsembleInfo, SamplingInfo

# Optional cache manager import
try:
    from cache_manager import CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None


class SigmondLoader:
    """Loader for Sigmond samplings files using sigmond_query."""

    def __init__(self, filename: str = None, sigmond_query_cmd: str = "sigmond_query",
                 enable_caching: bool = False, cache_app: str = "sigmond-samplings"):
        """
        Initialize the loader.

        Args:
            filename: Path to the samplings file to load upon construction (optional)
            sigmond_query_cmd: Command to run sigmond_query (default: "sigmond_query")
            enable_caching: Whether to enable disk caching of loaded observables (default: False)
            cache_app: Application name for cache directory (default: "sigmond-samplings")
        """
        self.sigmond_query_cmd = sigmond_query_cmd

        # Initialize cache if requested
        self._cache = None
        if enable_caching:
            if not CACHE_AVAILABLE:
                raise RuntimeError("Caching requested but cache-manager package not available. "
                                 "Install with: pip install cache-manager")
            self._cache = CacheManager(app=cache_app)

        # Single source of truth for data
        self._filename = None
        self._all_samplings = {}

        # Load file if provided
        if filename:
            self.load_file(filename)

    @property
    def _observable_infos(self) -> List[ObservableInfo]:
        """Generate observable infos on-demand from samplings."""
        return [sampling.observable_info for sampling in self._all_samplings.values()]

    @property
    def _all_data(self) -> List[np.ndarray]:
        """Generate data arrays on-demand from samplings."""
        return [sampling.data for sampling in self._all_samplings.values()]

    @property
    def _ensemble_info(self) -> Optional[EnsembleInfo]:
        """Get ensemble info from first sampling."""
        if not self._all_samplings:
            return None
        return next(iter(self._all_samplings.values())).ensemble_info

    @property
    def _sampling_info(self) -> Optional[SamplingInfo]:
        """Get sampling info from first sampling."""
        if not self._all_samplings:
            return None
        return next(iter(self._all_samplings.values())).sampling_info

    @classmethod
    def from_samplings_list(cls, samplings_list: List[SigmondSampling]) -> 'SigmondLoader':
        """
        Create a SigmondLoader instance from a list of SigmondSampling objects.

        Args:
            samplings_list: List of SigmondSampling objects to include in the loader.
        Returns:
            SigmondLoader instance with the provided samplings.
        """
        if not samplings_list:
            raise ValueError("samplings_list cannot be empty")

        loader = cls()
        loader._filename = "<from_samplings_list>"

        # Build samplings dictionary
        for s in samplings_list:
            key = f"{s.observable_info.name} {s.observable_info.index}"
            loader._all_samplings[key] = s

        return loader

    def _check_sigmond_query(self):
        """Check if sigmond_query is available."""
        try:
            result = subprocess.run([self.sigmond_query_cmd, "-h"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"sigmond_query command failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"sigmond_query not found or not working: {e}")

    def _run_sigmond_query(self, filename: str, options: str) -> str:
        """Run sigmond_query with given options."""
        cmd = [self.sigmond_query_cmd] + options.split() + [filename]
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

    def check_file_validity(self, filename: str) -> Tuple[bool, Optional[str], Optional[List[str]]]:
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
                lines = error_msg.split('\n')
                for line in lines:
                    if line.strip().startswith('/') and line.strip().endswith('/'):
                        paths.append(line.strip())
                return True, "hdf5", paths
            elif "This file type is not known to Sigmond" in error_msg:
                return False, None, None
            return False, None, None

    def load_file(self, filename: str) -> None:
        """
        Load all data from a file.

        Args:
            filename: Path to the samplings file (with [path] for HDF5 if needed)
        """

        # Load samplings using cache if available
        if self._cache:
            @self._cache.memory.cache
            def _load_samplings_cached(filename_arg):
                return self._load_samplings_impl(filename_arg)

            self._all_samplings = _load_samplings_cached(filename)
        else:
            try:
                self._all_samplings = self._load_samplings_impl(filename)
            except Exception as e:
                # Diagnose the issue
                is_valid, file_type, hdf5_paths = self.check_file_validity(filename)

                if not is_valid:
                    raise ValueError(f"File {filename} is not a valid Sigmond samplings file")

                if file_type == "hdf5" and '[' not in filename:
                    if hdf5_paths:
                        paths_str = '\n'.join(hdf5_paths)
                        raise ValueError(f"HDF5 file requires path specification. Available paths:\n{paths_str}")
                    else:
                        raise ValueError("HDF5 file requires path specification, but no paths found")
        self._filename = filename
                
                

    def _load_samplings_impl(self, filename: str) -> Dict[str, SigmondSampling]:
        """Load all samplings from a file - the method that gets cached."""
        # Get header info
        header_output = self._run_sigmond_query(filename, "-i")
        ensemble_info, sampling_info = self._parse_header_xml(header_output)

        # Get and parse keys
        keys_output = self._run_sigmond_query(filename, "-k")
        observable_infos = self._parse_keys_from_output(keys_output, ensemble_info)

        # Get all values
        values_output = self._run_sigmond_query(filename, "-v")
        all_data = self._parse_all_values(values_output)

        # Build samplings dictionary
        return self._build_samplings_dict(observable_infos, all_data, ensemble_info, sampling_info)

    def _parse_header_xml(self, xml_string: str) -> Tuple[EnsembleInfo, SamplingInfo]:
        """Parse the header XML to extract ensemble and sampling info."""
        try:
            root = ET.fromstring(xml_string.strip())
        except ET.ParseError:
            # Try to find the XML part in the string
            xml_start = xml_string.find('<')
            xml_end = xml_string.rfind('>') + 1
            if xml_start >= 0 and xml_end > xml_start:
                xml_part = xml_string[xml_start:xml_end]
                root = ET.fromstring(xml_part)
            else:
                raise ValueError("Could not parse header XML")

        # Extract ensemble info
        bins_info = root.find('.//MCBinsInfo')
        if bins_info is None:
            raise ValueError("MCBinsInfo not found in header")

        ensemble_name = bins_info.find('MCEnsembleInfo').text
        num_measurements = int(bins_info.find('NumberOfMeasurements').text)
        num_bins = int(bins_info.find('NumberOfBins').text)

        # Extract tweak info if present
        tweak_info = {}
        tweak_element = bins_info.find('TweakEnsemble')
        if tweak_element is not None:
            for child in tweak_element:
                tweak_info[child.tag] = child.text

        ensemble_info = EnsembleInfo(ensemble_name, num_measurements, num_bins, tweak_info)

        # Extract sampling info
        sampling_element = root.find('.//MCSamplingInfo')
        if sampling_element is None:
            raise ValueError("MCSamplingInfo not found in header")

        # Check for Bootstrap or Jackknife
        bootstrap = sampling_element.find('.//Bootstrapper')
        jackknife = sampling_element.find('.//Jackkniffer')  # Note: might be misspelled in XML
        jackknife_simple = sampling_element.find('.//Jackknife')  # Simple self-closing tag

        if bootstrap is not None:
            method = "bootstrap"
            num_resamplings = int(bootstrap.find('NumberResamplings').text)
            seed = int(bootstrap.find('Seed').text)
            boot_skip = int(bootstrap.find('BootSkip').text)
            sampling_info = SamplingInfo(method, num_resamplings, seed, boot_skip)
        elif jackknife is not None:
            method = "jackknife"
            num_resamplings = int(jackknife.find('NumberResamplings').text)
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
            standard_xml = key_xml.replace('<|', '</')
            root = ET.fromstring(standard_xml.strip())
            info_element = root.find('.//Info')
            if info_element is not None:
                info_text = info_element.text.strip()
                return ObservableInfo.from_string(info_text, ensemble_info)
            else:
                # Handle more complex XML structure if needed
                raise NotImplementedError("Complex XML key parsing not yet implemented")
        except ET.ParseError as e:
            raise ValueError(f"Could not parse key XML: {e}")

    def _parse_keys_from_output(self, keys_output: str, ensemble_info: EnsembleInfo) -> List[ObservableInfo]:
        """Parse observable keys from sigmond_query output."""
        observable_infos = []
        lines = keys_output.split('\n')
        current_key_lines = []
        in_key = False

        for line in lines:
            if line.startswith('Record ') and ':' in line:
                if current_key_lines:
                    # Process previous key
                    key_xml = '\n'.join(current_key_lines)
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
                key_xml = '\n'.join(current_key_lines)
                try:
                    obs_info = self._parse_observable_key(key_xml, ensemble_info)
                    observable_infos.append(obs_info)
                except (ValueError, NotImplementedError):
                    pass  # Skip problematic keys
                current_key_lines = []
                in_key = False

        # Process last key
        if current_key_lines:
            key_xml = '\n'.join(current_key_lines)
            try:
                obs_info = self._parse_observable_key(key_xml, ensemble_info)
                observable_infos.append(obs_info)
            except (ValueError, NotImplementedError):
                pass

        return observable_infos

    def _parse_all_values(self, values_output: str) -> List[np.ndarray]:
        """Parse the output of 'sigmond_query -v' into a list of numpy arrays."""
        all_records_values = []
        lines = values_output.split('\n')
        current_values = []

        in_record = False
        for line in lines:
            if line.startswith('Record ') and ':' in line:
                if current_values:
                    all_records_values.append(np.array(current_values))
                current_values = []
                in_record = True
            elif in_record and '[' in line and ']' in line and '=' in line:
                value_str = line.split('=')[1].strip()
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

    def _build_samplings_dict(self, observable_infos: List[ObservableInfo],
                             all_data: List[np.ndarray], ensemble_info: EnsembleInfo,
                             sampling_info: SamplingInfo) -> Dict[str, SigmondSampling]:
        """Build the samplings dictionary from parsed data."""
        if len(all_data) != len(observable_infos):
            raise ValueError("Mismatch between number of observables in header and data records.")

        # Group observables by name and index to find complex pairs
        grouped_observables = {}
        for i, obs_info in enumerate(observable_infos):
            key = (obs_info.name, obs_info.index)
            if key not in grouped_observables:
                grouped_observables[key] = {}
            grouped_observables[key][obs_info.re_im] = (obs_info, i)

        result = {}
        for key, parts in grouped_observables.items():
            obs_name, obs_index = key
            output_key = f"{obs_name} {obs_index}"

            if 're' in parts and 'im' in parts:
                re_info, re_idx = parts['re']
                im_info, im_idx = parts['im']
                re_data = all_data[re_idx]
                im_data = all_data[im_idx]
                complex_data = re_data + 1j * im_data
                sampling = SigmondSampling(complex_data, re_info, sampling_info, is_complex=True)
                result[output_key] = sampling
            elif 're' in parts:
                re_info, re_idx = parts['re']
                re_data = all_data[re_idx]
                sampling = SigmondSampling(re_data, re_info, sampling_info, is_complex=np.iscomplexobj(re_data))
                result[output_key] = sampling
            elif 'im' in parts:
                im_info, im_idx = parts['im']
                im_data = all_data[im_idx]
                sampling = SigmondSampling(im_data, im_info, sampling_info, is_complex=np.iscomplexobj(im_data))
                result[output_key] = sampling

        return result

    # Simplified public API
    def load_all_observables(self, filename: str = None) -> Dict[str, SigmondSampling]:
        """
        Load all observables from the file.

        Args:
            filename: Path to the samplings file (if None, uses cached data)

        Returns:
            Dictionary mapping observable string to SigmondSampling objects
        """
        if filename is not None:
            self.load_file(filename)
        elif self._filename is None:
            raise ValueError("No file loaded. Please provide a filename or call load_file() first.")

        return self._all_samplings

    def get_observables(self, name_patterns: Union[List[str], str] = None,
                       index: int = None, scalar_type: str = None) -> Dict[str, SigmondSampling]:
        """
        Get observables matching given criteria.

        Args:
            name_patterns: Regex pattern(s) to match observable names
            index: Specific index to match
            scalar_type: Specific scalar type ('re' or 'im')

        Returns:
            Dictionary of matching observables
        """
        if not self._all_samplings:
            raise ValueError("No file loaded. Call load_file() or load_all_observables() first.")

        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        result = {}
        for key, sampling in self._all_samplings.items():
            obs_info = sampling.observable_info
            match = True

            if name_patterns is not None:
                pattern_found = any(re.search(pattern, obs_info.name) for pattern in name_patterns)
                if not pattern_found:
                    match = False

            if index is not None and obs_info.index != index:
                match = False

            if scalar_type is not None and obs_info.re_im != scalar_type:
                match = False

            if match:
                result[key] = sampling

        return result

    # Utility methods for backward compatibility
    def get_file_info(self, filename: str = None) -> Tuple[EnsembleInfo, SamplingInfo, List[ObservableInfo]]:
        """Get header info and list of observable keys from a file."""
        if filename is not None:
            self.load_file(filename)
        elif self._filename is None:
            raise ValueError("No file loaded. Please provide a filename or call load_file() first.")

        return self._ensemble_info, self._sampling_info, self._observable_infos

    @staticmethod
    def get_name_and_index_from_dict_key(key: str) -> Tuple[str, int]:
        """Extract the (name, index) pair from keys like 'observable[3]'."""
        m = re.match(r'^(.*)\[(\d+)\]$', key)
        if not m:
            raise ValueError(f"Key '{key}' does not end with an integer in brackets (e.g. 'name[0]').")
        name, index_str = m.groups()
        return name, int(index_str)