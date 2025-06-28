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


class SigmondLoader:
    """Loader for Sigmond samplings files using sigmond_query."""
    
    def __init__(self, sigmond_query_cmd: str = "sigmond_query"):
        """
        Initialize the loader.
        
        Args:
            sigmond_query_cmd: Command to run sigmond_query (default: "sigmond_query")
        """
        self.sigmond_query_cmd = sigmond_query_cmd
        self._check_sigmond_query()
        self.is_on_mac: bool = self._check_for_mac()
    
    def _check_sigmond_query(self):
        """Check if sigmond_query is available."""
        try:
            result = subprocess.run([self.sigmond_query_cmd, "-h"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"sigmond_query command failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"sigmond_query not found or not working: {e}")
        
    def _check_for_mac(self) -> bool:
        """Check if the user is on macOS."""
        import sys
        return sys.platform == "darwin"
    
    def _run_sigmond_query(self, filename: str, options: str) -> str:
        """Run sigmond_query with given options."""
        # on macOS, we need to surround the filename with quotes for sigmond_query to work correctly
        if self.is_on_mac:
            filename = f'"{filename}"'
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
            root = ET.fromstring(key_xml.strip())
            info_element = root.find('.//Info')
            if info_element is not None:
                info_text = info_element.text.strip()
                return ObservableInfo.from_string(info_text, ensemble_info)
            else:
                # Handle more complex XML structure if needed
                raise NotImplementedError("Complex XML key parsing not yet implemented")
        except ET.ParseError as e:
            raise ValueError(f"Could not parse key XML: {e}")
    
    def get_file_info(self, filename: str) -> Tuple[EnsembleInfo, SamplingInfo, List[ObservableInfo]]:
        """
        Get header info and list of observable keys from a file.
        
        Args:
            filename: Path to the samplings file (with [path] for HDF5 if needed)
            
        Returns:
            Tuple of (ensemble_info, sampling_info, list_of_observable_infos)
        """
        # Check file validity first
        is_valid, file_type, hdf5_paths = self.check_file_validity(filename)
        
        if not is_valid:
            raise ValueError(f"File {filename} is not a valid Sigmond samplings file")
        
        if file_type == "hdf5" and '[' not in filename:
            if hdf5_paths:
                paths_str = '\n'.join(hdf5_paths)
                raise ValueError(f"HDF5 file requires path specification. Available paths:\n{paths_str}")
            else:
                raise ValueError("HDF5 file requires path specification, but no paths found")
        
        # Get header info
        header_output = self._run_sigmond_query(filename, "-i")
        ensemble_info, sampling_info = self._parse_header_xml(header_output)
        
        # Get keys
        keys_output = self._run_sigmond_query(filename, "-k")
        observable_infos = []
        
        # Parse keys from output
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
        
        return ensemble_info, sampling_info, observable_infos
        
        
    
    def load_observable(self, filename: str, observable_info: ObservableInfo) -> SigmondSampling:
        """
        Load a specific observable from the file.
        
        Args:
            filename: Path to the samplings file
            observable_info: Information about the observable to load
            
        Returns:
            SigmondSampling object containing the data
        """
        ensemble_info, sampling_info, observable_infos = self.get_file_info(filename)
        
        try:
            record_index = observable_infos.index(observable_info)
        except ValueError:
            raise ValueError(f"Observable {observable_info} not found in file's list of observables.")

        # Get values for all observables
        values_output = self._run_sigmond_query(filename, "-v")
        
        # Parse the values output to find our specific observable
        lines = values_output.split('\n')
        current_record = -1
        current_values = []
        
        for line in lines:
            if line.startswith('Record ') and ':' in line:
                if current_record == record_index:
                    # We have finished parsing our record, so we can return
                    data = np.array(current_values)
                    is_complex = np.iscomplexobj(data)
                    return SigmondSampling(data, observable_info, 
                                         sampling_info, is_complex)

                current_record += 1
                current_values = []
                
            elif current_record == record_index:
                # We are at the correct record, start parsing values
                if '[' in line and ']' in line and '=' in line:
                    value_str = line.split('=')[1].strip()
                    try:
                        value = float(value_str)
                        current_values.append(value)
                    except ValueError:
                        try:
                            # Try complex number
                            value = complex(value_str)
                            current_values.append(value)
                        except ValueError:
                            pass

        # Handle case where the desired record is the last one in the file
        if current_record == record_index and current_values:
            data = np.array(current_values)
            is_complex = np.iscomplexobj(data)
            return SigmondSampling(data, observable_info, 
                                 sampling_info, is_complex)

        raise ValueError(f"Observable {observable_info} not found in file")
    
    def load_observables(self, filename: str, observable_infos: List[ObservableInfo]) -> Dict[str, SigmondSampling]:
        """
        Load multiple observables from the file, automatically handling complex ones.
        
        Args:
            filename: Path to the samplings file
            observable_infos: List of ObservableInfo objects to load. For complex
                              observables, providing just the 're' or 'im' part
                              is sufficient.
            
        Returns:
            Dictionary mapping observable string to SigmondSampling objects
        """
        all_observables = self.load_all_observables(filename)
        
        result = {}
        requested_keys = set()
        for obs_info in observable_infos:
            # Create key in the same format as load_all_observables uses: "name index"
            key = f"{obs_info.name} {obs_info.index}"
            requested_keys.add(key)
            
        for name, sampling in all_observables.items():
            if name in requested_keys:
                result[name] = sampling
        
        return result
    
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

    def load_all_observables(self, filename: str) -> Dict[str, SigmondSampling]:
        """
        Load all observables from the file, automatically handling complex observables.
        
        Args:
            filename: Path to the samplings file
            
        Returns:
            Dictionary mapping observable string to SigmondSampling objects
        """
        ensemble_info, sampling_info, observable_infos = self.get_file_info(filename)
        
        # Get all values
        values_output = self._run_sigmond_query(filename, "-v")
        all_data = self._parse_all_values(values_output)

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
            
            # Unique key for the output dictionary
            output_key = f"{obs_name} {obs_index}"

            if 're' in parts and 'im' in parts:
                re_info, re_idx = parts['re']
                im_info, im_idx = parts['im']

                re_data = all_data[re_idx]
                im_data = all_data[im_idx]
                complex_data = re_data + 1j * im_data
                
                sampling = SigmondSampling(complex_data, re_info, 
                                         sampling_info, is_complex=True)
                result[output_key] = sampling

            elif 're' in parts:
                re_info, re_idx = parts['re']
                re_data = all_data[re_idx]
                
                sampling = SigmondSampling(re_data, re_info,
                                         sampling_info, is_complex=np.iscomplexobj(re_data))
                result[output_key] = sampling

            elif 'im' in parts:
                im_info, im_idx = parts['im']
                im_data = all_data[im_idx]

                # Store as real array, since it's just one component
                sampling = SigmondSampling(im_data, im_info,
                                         sampling_info, is_complex=np.iscomplexobj(im_data))
                result[output_key] = sampling

        return result
    
    def find_observables(self, filename: str, name_patterns: Union[List[str], str] = None, 
                        index: int = None, scalar_type: str = None) -> List[ObservableInfo]:
        """
        Find observables matching given criteria.
        
        Args:
            filename: Path to the samplings file
            name_pattern: Regex pattern to match observable names
            index: Specific index to match
            scalar_type: Specific scalar type ('re' or 'im')
            
        Returns:
            List of matching ObservableInfo objects
        """
        _, _, observable_infos = self.get_file_info(filename)
        
        results = []
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]
            
        for obs_info in observable_infos:
            match = True
            
            if name_patterns is not None:
                pattern_found = False
                for pattern in name_patterns:
                    if re.search(pattern, obs_info.name):
                        pattern_found = True
                        break
                if not pattern_found:
                    match = False
                    
            
            if index is not None and obs_info.index != index:
                match = False
            
            if scalar_type is not None and obs_info.re_im != scalar_type:
                match = False
            
            if match:
                results.append(obs_info)
        
        return results

    @staticmethod
    def get_name_and_index_from_dict_key(key: str) -> Tuple[str, int]:
        """
        Extract the (name, index) pair from keys like 'observable[3]'.

        The name itself may contain brackets (e.g. 'mass[stat]') so we:
            • Look for a final bracket block that contains only digits.
            • Treat everything before that block as the name.

        Args:
            key: A dictionary key whose last component is an integer in brackets.

        Returns:
            (name, index) where `index` is an int.

        Raises:
            ValueError: if no trailing integer bracket is found.
        """
        m = re.match(r'^(.*)\[(\d+)\]$', key)
        if not m:
            raise ValueError(
                f"Key '{key}' does not end with an integer in brackets (e.g. 'name[0]')."
            )

        name, index_str = m.groups()
        return name, int(index_str)