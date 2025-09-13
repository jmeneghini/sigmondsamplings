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
from typing import List, Optional, Tuple
from pathlib import Path

from .sampling import SigmondSampling, ObservableInfo, EnsembleInfo, SamplingInfo


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
    
    def _create_numbered_backup(self, filename: str) -> Optional[str]:
        """
        Create a numbered backup of the file before modification.
        
        Args:
            filename: Path to the file to backup
            
        Returns:
            Path to the backup file, or None if backups are disabled or file doesn't exist
        """
        if not self.create_backups or not Path(filename).exists():
            return None
        
        counter = 1
        while Path(f"{filename}.backup_{counter:03d}").exists():
            counter += 1
        
        backup_path = f"{filename}.backup_{counter:03d}"
        
        try:
            shutil.copy2(filename, backup_path)
            print(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Warning: Failed to create backup of {filename}: {e}")
            return None
    
    def _ensure_hdf5_format(self, filename: str, hdf5_root_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Ensure the input file is in HDF5 format, converting if necessary.
        
        Args:
            filename: Input file path
            hdf5_root_path: Preferred root path for HDF5 conversion
            
        Returns:
            Tuple of (hdf5_filename, root_path) ready for operations
        """
        from .loader import SigmondLoader
        
        # Check if already HDF5
        if filename.lower().endswith('.hdf5'):
            loader = SigmondLoader()
            is_valid, file_type, hdf5_paths = loader.check_file_validity(filename)
            if is_valid and file_type == 'hdf5' and hdf5_paths:
                root_path = hdf5_root_path or hdf5_paths[0]
                return filename, root_path
        
        # Convert fstream to HDF5
        if filename.lower().endswith('.smp') or not filename.lower().endswith('.hdf5'):
            print(f"Converting {filename} to HDF5 format for reliable processing...")
            
            # Create HDF5 filename
            if filename.lower().endswith('.smp'):
                hdf5_filename = filename.replace('.smp', '_working.hdf5')
            else:
                hdf5_filename = filename + '_working.hdf5'
            
            # Determine root path
            root_path = hdf5_root_path or '/data/'
            
            # Load original data using the loader
            loader = SigmondLoader()
            samplings = loader.load_all_observables(filename)
            
            # Write to HDF5 format
            self.write_hdf5(hdf5_filename, samplings, root_path, overwrite=True)
            
            print(f"Conversion complete. Working with: {hdf5_filename}")
            print(f"Original file {filename} preserved unchanged.")
            
            return hdf5_filename, root_path
        
        raise ValueError(f"Unable to determine format for file: {filename}")
    
    def _generate_header_xml(self, ensemble_info: EnsembleInfo, 
                           sampling_info: SamplingInfo) -> str:
        """Generate the XML header for a Sigmond file in the exact format expected."""
        # Use SigmondSamplingsFile as root element to match real format
        root = ET.Element("SigmondSamplingsFile")
        
        # Add bins info first (matches real format order)
        bins_elem = ET.SubElement(root, "MCBinsInfo")
        ET.SubElement(bins_elem, "MCEnsembleInfo").text = ensemble_info.ensemble_name
        ET.SubElement(bins_elem, "NumberOfMeasurements").text = str(ensemble_info.num_measurements)
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
            ET.SubElement(bootstrap_elem, "NumberResamplings").text = str(sampling_info.num_resamplings)
            ET.SubElement(bootstrap_elem, "Seed").text = str(sampling_info.seed)
            ET.SubElement(bootstrap_elem, "BootSkip").text = str(sampling_info.boot_skip)
        elif sampling_info.method == "jackknife":
            if sampling_info.num_resamplings == ensemble_info.num_bins:
                # Simple jackknife
                ET.SubElement(sampling_elem, "Jackknife")
            else:
                # Full jackknife with parameters
                jackknife_elem = ET.SubElement(sampling_elem, "Jackkniffer")
                ET.SubElement(jackknife_elem, "NumberResamplings").text = str(sampling_info.num_resamplings)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_info.method}")
        
        # Return XML without indentation to match real format (compact)
        return ET.tostring(root, encoding='unicode')
    
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
        
        return ET.tostring(root, encoding='unicode')
    
    def _make_hdf5_safe_key(self, xml_key: str) -> str:
        """Convert XML key to HDF5-safe format by replacing problematic characters."""
        # Replace closing tags </ with <| to avoid HDF5 path interpretation
        # This converts <Tag>content</Tag> to <Tag>content<|Tag>
        hdf5_key = xml_key.replace('</', '<|')
        
        # Replace forward slashes with a safe character to avoid HDF5 path interpretation
        # This is critical for observables with names like "PSQ2/G/ecm_0"
        hdf5_key = hdf5_key.replace('/', '|')
        
        return hdf5_key
    
    def write_file(self, filename: str, samplings: List[SigmondSampling], 
                   root_path: str = "/data/", overwrite: bool = False) -> str:
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
        if not filename.lower().endswith('.hdf5'):
            filename = filename.rsplit('.', 1)[0] + '.hdf5'
        
        self.write_hdf5(filename, samplings, root_path, overwrite)
        return filename
    
    def write_hdf5(self, filename: str, samplings: List[SigmondSampling],
                   root_path: str = "/", overwrite: bool = False) -> None:
        """
        Write samplings to an HDF5 (.hdf5) format file in correct Sigmond format.
        
        Args:
            filename: Output file path
            samplings: List of SigmondSampling objects
            root_path: HDF5 root path (e.g., "/isosinglet_S0_A1g_1_P0/")
            overwrite: Whether to overwrite existing file
        """
        if Path(filename).exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to overwrite.")
        elif Path(filename).exists() and overwrite:
            # Create backup before overwriting
            self._create_numbered_backup(filename)
        
        if not samplings:
            raise ValueError("No samplings provided")
        
        # Ensure root_path has proper format and extract group name
        if not root_path.startswith('/'):
            root_path = '/' + root_path
        if not root_path.endswith('/'):
            root_path += '/'
        
        group_name = root_path.strip('/')
        
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
        
        with h5py.File(filename, 'w') as hdf5_file:
            # Create global Info group (required by Sigmond)
            info_group = hdf5_file.create_group('Info')
            
            # File identifier - match exact format of real files with fixed size
            fid_dtype = h5py.string_dtype(encoding='utf-8', length=23)
            info_group.create_dataset('FIdentifier', data='Sigmond--SamplingsFile', dtype=fid_dtype)
            
            # Endianness - match exact format of real files with fixed size  
            end_dtype = h5py.string_dtype(encoding='utf-8', length=2)
            info_group.create_dataset('Endianness', data='L', dtype=end_dtype)
            
            # Create data group
            data_group = hdf5_file.create_group(group_name)
            
            # Generate XML header and store only in data group (NOT in Info group!)
            header_xml = self._generate_header_xml(ensemble_info, sampling_info)
            # Use fixed-length UTF-8 encoding to match real files exactly
            header_len = len(header_xml.encode('utf-8')) + 1  # +1 for null termination
            header_dtype = h5py.string_dtype(encoding='utf-8', length=header_len)
            data_group.create_dataset('Header', data=header_xml, dtype=header_dtype)
            
            # Include checksums flag - match exact format of real files
            cks_dtype = h5py.string_dtype(encoding='utf-8', length=2)
            data_group.create_dataset('IncludeCKS', data='N', dtype=cks_dtype)
            
            # Create Values group to hold all observables
            values_group = data_group.create_group('Values')
            
            # Write each sampling as a dataset with XML key as name
            for sampling in samplings:
                if sampling.is_complex:
                    # Write real part
                    re_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        're',
                        sampling.observable_info.ensemble_info
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
                        'im',
                        sampling.observable_info.ensemble_info
                    )
                    im_key_xml = self._generate_observable_key_xml(im_obs_info)
                    im_key_safe = self._make_hdf5_safe_key(im_key_xml)
                    # Ensure data is float64 for compatibility
                    im_data = np.imag(sampling.data).astype(np.float64)
                    values_group.create_dataset(im_key_safe, data=im_data)
                    
                else:
                    # Write real data
                    key_xml = self._generate_observable_key_xml(sampling.observable_info)
                    key_safe = self._make_hdf5_safe_key(key_xml)
                    # Ensure data is float64 for compatibility
                    real_data = sampling.data.astype(np.float64)
                    values_group.create_dataset(key_safe, data=real_data)
    
    def append_to_file(self, filename: str, new_samplings: List[SigmondSampling],
                      hdf5_root_path: Optional[str] = None) -> str:
        """
        Append new samplings to an existing file.
        
        Automatically converts fstream files to HDF5 format for reliable processing.
        
        Args:
            filename: Path to existing file
            new_samplings: List of new samplings to add
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
        self._append_to_hdf5(hdf5_filename, new_samplings, root_path.strip('/'))
        
        return hdf5_filename
    
    def _append_to_hdf5(self, filename: str, new_samplings: List[SigmondSampling],
                       root_path: Optional[str] = None) -> None:
        """Append samplings to existing HDF5 file."""
        with h5py.File(filename, 'a') as f:
            if root_path is None:
                # Find existing data groups (exclude 'Info' group)
                data_groups = [key for key in f.keys() if key != 'Info']
                if not data_groups:
                    raise ValueError("No data groups found in HDF5 file")
                root_path = data_groups[0]
            
            if root_path not in f:
                raise ValueError(f"Data group {root_path} not found in file")
            
            data_group = f[root_path]
            if 'Values' not in data_group:
                raise ValueError(f"Values group not found in data group {root_path}")
            
            values_group = data_group['Values']
            
            # Add new samplings
            for sampling in new_samplings:
                if sampling.is_complex:
                    # Write real part
                    re_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        're',
                        sampling.observable_info.ensemble_info
                    )
                    re_key_xml = self._generate_observable_key_xml(re_obs_info)
                    re_key_safe = self._make_hdf5_safe_key(re_key_xml)
                    
                    if re_key_safe in values_group:
                        del values_group[re_key_safe]
                    values_group.create_dataset(re_key_safe, data=np.real(sampling.data))
                    
                    # Write imaginary part
                    im_obs_info = ObservableInfo(
                        sampling.observable_info.name,
                        sampling.observable_info.index,
                        sampling.observable_info.op_type,
                        'im',
                        sampling.observable_info.ensemble_info
                    )
                    im_key_xml = self._generate_observable_key_xml(im_obs_info)
                    im_key_safe = self._make_hdf5_safe_key(im_key_xml)
                    
                    if im_key_safe in values_group:
                        del values_group[im_key_safe]
                    values_group.create_dataset(im_key_safe, data=np.imag(sampling.data))
                    
                else:
                    # Write real data
                    key_xml = self._generate_observable_key_xml(sampling.observable_info)
                    key_safe = self._make_hdf5_safe_key(key_xml)
                    
                    if key_safe in values_group:
                        del values_group[key_safe]
                    values_group.create_dataset(key_safe, data=sampling.data)
    
    
    def convert_format(self, input_filename: str, output_filename: str,
                      output_format: str = 'hdf5', hdf5_root_path: str = "/data/",
                      overwrite: bool = False) -> str:
        """
        Convert Sigmond files to HDF5 format (primary supported format).
        
        Args:
            input_filename: Input file path
            output_filename: Output file path
            output_format: Output format ('hdf5' - other formats deprecated)
            hdf5_root_path: Root path for HDF5 output
            overwrite: Whether to overwrite existing output file
            
        Returns:
            Path to the converted file
        """
        from .loader import SigmondLoader
        
        # Only support HDF5 output format
        if output_format.lower() != 'hdf5':
            print(f"Warning: {output_format} format deprecated. Converting to HDF5 instead.")
            output_format = 'hdf5'
        
        # Ensure output filename has correct extension
        if not output_filename.lower().endswith('.hdf5'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.hdf5'
        
        # Load all samplings from input file
        loader = SigmondLoader()
        
        # Handle HDF5 input with path specification
        if input_filename.lower().endswith('.hdf5') and '[' not in input_filename:
            # Check available paths
            is_valid, file_type, hdf5_paths = loader.check_file_validity(input_filename)
            if is_valid and file_type == 'hdf5' and hdf5_paths:
                # Use first available path
                input_filename_with_path = f"{input_filename}[{hdf5_paths[0]}]"
                samplings = loader.load_all_observables(input_filename_with_path)
            else:
                samplings = loader.load_all_observables(input_filename)
        else:
            samplings = loader.load_all_observables(input_filename)
        
        # Write in HDF5 format (convert dict to list)
        samplings_list = list(samplings.values())
        self.write_hdf5(output_filename, samplings_list, hdf5_root_path, overwrite=overwrite)
        
        return output_filename
    
    def modify_observable(self, filename: str, observable_name: str, observable_index: int,
                         new_data: np.ndarray, hdf5_root_path: Optional[str] = None) -> str:
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
        self._create_numbered_backup(hdf5_filename)
        
        # Load samplings from HDF5 file
        loader = SigmondLoader()
        full_hdf5_path = f"{hdf5_filename}[{root_path}]"
        samplings = loader.load_all_observables(full_hdf5_path)
        
        # Find and modify the target observable
        target_key = f"{observable_name} {observable_index}"
        if target_key not in samplings:
            raise ValueError(f"Observable {target_key} not found in file. Available observables: {list(samplings.keys())[:5]}...")
        
        # Update the data
        original_sampling = samplings[target_key]
        modified_sampling = SigmondSampling(
            new_data,
            original_sampling.observable_info,
            original_sampling.sampling_info,
            original_sampling.is_complex
        )
        samplings[target_key] = modified_sampling
        
        # Write back to HDF5 file (convert dict to list)
        samplings_list = list(samplings.values())
        self.write_hdf5(hdf5_filename, samplings_list, root_path, overwrite=True)
        
        return hdf5_filename
    
    def get_info(self) -> str:
        """
        Get information about the SigmondWriter class capabilities.
        
        Returns:
            Information string about the writer
        """
        info = """
SigmondWriter - Automatic Format Conversion & HDF5-First Approach

Key Features:
- Automatic fstream (.smp) to HDF5 conversion for all operations
- Preserves original files unchanged during modifications
- HDF5-first approach for reliability and compatibility
- Full Sigmond format support with proper XML structure
- Complex number handling (separate real/imaginary parts)

Available Methods:
- write_file(): Write list of samplings to HDF5 format
- append_to_file(): Add new observables as list (auto-converts fstream)
- modify_observable(): Modify existing data (auto-converts fstream)  
- convert_format(): Convert any format to HDF5
- write_hdf5(): Direct HDF5 writing with full format compliance
        """
        return info.strip()