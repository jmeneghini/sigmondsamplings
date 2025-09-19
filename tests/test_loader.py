"""
Unit tests for the loader module using actual Sigmond files.
"""

import unittest
import os
from unittest.mock import patch, MagicMock
from sigmondsamplings.loader import SigmondLoader
from sigmondsamplings.sampling import ObservableInfo, EnsembleInfo, SamplingInfo


class TestSigmondLoader(unittest.TestCase):
    """Test SigmondLoader class."""
    
    def setUp(self):
        """Set up test with actual files."""
        self.test_files = {
            'fstream': 'example_samplings/k-pi_scatteringDrew.smp',
            'hdf5_without_path': 'example_samplings/fit_spectrum_fitparams-5-27-25-interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5',
            'hdf5_with_path': 'example_samplings/fit_spectrum_fitparams-5-27-25-interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5[/isosinglet S=0 A1g_1 PSQ=0/]',
            'valid_hdf5_without_path': 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5',
            'valid_hdf5_with_path': 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5[/samplings/]',
            'jackknife_hdf5_without_path': 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_J-samplings.hdf5',
            'jackknife_hdf5_with_path': 'example_samplings/fit_spectrum_levels_sigmond-5-27-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_J-samplings.hdf5[/samplings/]'
        }
        
        # Check if files exist
        for key, path in self.test_files.items():
            if not key.endswith('_with_path'):
                base_path = path.split('[')[0] if '[' in path else path
                if not os.path.exists(base_path):
                    self.skipTest(f"Test file {base_path} not found")
    
    def test_loader_creation(self):
        """Test loader creation."""
        loader = SigmondLoader()
        self.assertIsInstance(loader, SigmondLoader)
        self.assertEqual(loader.sigmond_query_cmd, "sigmond_query")
    
    def test_custom_command(self):
        """Test loader with custom command."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="help text")
            loader = SigmondLoader("custom_sigmond_query")
            self.assertEqual(loader.sigmond_query_cmd, "custom_sigmond_query")
    
    def test_check_sigmond_query_not_found(self):
        """Test error when sigmond_query not found."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with self.assertRaises(RuntimeError):
                SigmondLoader("nonexistent_command")
    
    def test_check_file_validity_fstream(self):
        """Test file validity check for fstream file."""
        loader = SigmondLoader()
        is_valid, file_type, hdf5_paths = loader.check_file_validity(self.test_files['fstream'])
        
        self.assertTrue(is_valid)
        self.assertEqual(file_type, "fstream")
        self.assertIsNone(hdf5_paths)
    
    def test_check_file_validity_hdf5_no_path(self):
        """Test file validity check for HDF5 file without path."""
        loader = SigmondLoader()
        is_valid, file_type, hdf5_paths = loader.check_file_validity(self.test_files['hdf5_without_path'])
        
        self.assertTrue(is_valid)
        self.assertEqual(file_type, "hdf5")
        self.assertIsInstance(hdf5_paths, list)
        self.assertGreater(len(hdf5_paths), 0)
        # Check that paths start and end with '/'
        for path in hdf5_paths:
            self.assertTrue(path.startswith('/'))
            self.assertTrue(path.endswith('/'))
    
    def test_check_file_validity_hdf5_with_path(self):
        """Test file validity check for HDF5 file with path."""
        loader = SigmondLoader()
        is_valid, file_type, hdf5_paths = loader.check_file_validity(self.test_files['hdf5_with_path'])
        
        self.assertTrue(is_valid)
        self.assertEqual(file_type, "hdf5")
        self.assertIsNone(hdf5_paths)
    
    def test_get_file_info_fstream(self):
        """Test getting file info from fstream file."""
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info(self.test_files['fstream'])
        
        # Check ensemble info
        self.assertIsInstance(ensemble_info, EnsembleInfo)
        self.assertEqual(ensemble_info.ensemble_name, "clover_s32_t256_ud860_s743")
        self.assertEqual(ensemble_info.num_measurements, 412)
        self.assertEqual(ensemble_info.num_bins, 412)
        
        # Check sampling info
        self.assertIsInstance(sampling_info, SamplingInfo)
        self.assertEqual(sampling_info.method, "bootstrap")
        self.assertEqual(sampling_info.num_resamplings, 1000)
        self.assertEqual(sampling_info.seed, 6754)
        
        # Check observables
        self.assertIsInstance(observable_infos, list)
        self.assertGreater(len(observable_infos), 0)
        
        # Check first few observables match expected format
        if len(observable_infos) >= 3:
            self.assertEqual(observable_infos[0].name, "kaon")
            self.assertEqual(observable_infos[1].name, "pion")
            self.assertEqual(observable_infos[2].name, "kaon_mpi")
    
    def test_get_file_info_hdf5(self):
        """Test getting file info from HDF5 file."""
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info(self.test_files['hdf5_with_path'])
        
        # Check ensemble info
        self.assertIsInstance(ensemble_info, EnsembleInfo)
        self.assertIn("phirho", ensemble_info.ensemble_name)
        
        # Check sampling info
        self.assertIsInstance(sampling_info, SamplingInfo)
        self.assertEqual(sampling_info.method, "bootstrap")
        
        # Check observables
        self.assertIsInstance(observable_infos, list)
        self.assertGreater(len(observable_infos), 0)
    
    def test_get_file_info_valid_hdf5(self):
        """Test getting file info from valid Sigmond HDF5 file."""
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info(self.test_files['valid_hdf5_with_path'])
        
        # Check ensemble info
        self.assertIsInstance(ensemble_info, EnsembleInfo)
        self.assertIn("phirho", ensemble_info.ensemble_name)
        self.assertEqual(ensemble_info.num_measurements, 700000)
        self.assertEqual(ensemble_info.num_bins, 28000)
        
        # Check sampling info
        self.assertIsInstance(sampling_info, SamplingInfo)
        self.assertEqual(sampling_info.method, "bootstrap")
        self.assertEqual(sampling_info.num_resamplings, 2000)
        self.assertEqual(sampling_info.seed, 0)
        
        # Check observables
        self.assertIsInstance(observable_infos, list)
        self.assertGreater(len(observable_infos), 0)
    
    def test_check_file_validity_valid_hdf5_no_path(self):
        """Test file validity check for valid Sigmond HDF5 file without path."""
        loader = SigmondLoader()
        is_valid, file_type, hdf5_paths = loader.check_file_validity(self.test_files['valid_hdf5_without_path'])
        
        self.assertTrue(is_valid)
        self.assertEqual(file_type, "hdf5")
        self.assertIsInstance(hdf5_paths, list)
        self.assertEqual(len(hdf5_paths), 1)
        self.assertEqual(hdf5_paths[0], '/samplings/')
    
    def test_get_file_info_jackknife_hdf5(self):
        """Test getting file info from jackknife Sigmond HDF5 file."""
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info(self.test_files['jackknife_hdf5_with_path'])
        
        # Check ensemble info
        self.assertIsInstance(ensemble_info, EnsembleInfo)
        self.assertIn("phirho", ensemble_info.ensemble_name)
        self.assertEqual(ensemble_info.num_measurements, 700000)
        self.assertEqual(ensemble_info.num_bins, 28000)
        
        # Check sampling info - should be jackknife
        self.assertIsInstance(sampling_info, SamplingInfo)
        self.assertEqual(sampling_info.method, "jackknife")
        self.assertEqual(sampling_info.num_resamplings, 28000)  # For jackknife, num_resamplings = num_bins
        
        # Check observables
        self.assertIsInstance(observable_infos, list)
        self.assertGreater(len(observable_infos), 0)
    
    def test_check_file_validity_jackknife_hdf5_no_path(self):
        """Test file validity check for jackknife Sigmond HDF5 file without path."""
        loader = SigmondLoader()
        is_valid, file_type, hdf5_paths = loader.check_file_validity(self.test_files['jackknife_hdf5_without_path'])
        
        self.assertTrue(is_valid)
        self.assertEqual(file_type, "hdf5")
        self.assertIsInstance(hdf5_paths, list)
        self.assertEqual(len(hdf5_paths), 1)
        self.assertEqual(hdf5_paths[0], '/samplings/')
    
    def test_hdf5_requires_path_error(self):
        """Test error when HDF5 file used without path."""
        loader = SigmondLoader()
        
        with self.assertRaises(ValueError) as cm:
            loader.get_file_info(self.test_files['hdf5_without_path'])
        
        error_msg = str(cm.exception)
        self.assertIn("HDF5 file requires path specification", error_msg)
        self.assertIn("Available paths", error_msg)
    
    def test_find_observables(self):
        """Test finding observables by pattern."""
        loader = SigmondLoader()
        
        # Test with fstream file
        matches = loader.find_observables(self.test_files['fstream'], name_pattern="kaon")
        self.assertGreater(len(matches), 0)
        
        # All matches should contain "kaon" in name
        for match in matches:
            self.assertIn("kaon", match.name.lower())
    
    def test_find_observables_by_scalar_type(self):
        """Test finding observables by scalar type."""
        loader = SigmondLoader()
        
        matches = loader.find_observables(self.test_files['fstream'], scalar_type="re")
        self.assertGreater(len(matches), 0)
        
        # All matches should be real
        for match in matches:
            self.assertEqual(match.re_im, "re")
    
    @unittest.skip("Skip loading full data for faster tests - enable for full integration testing")
    def test_load_observable(self):
        """Test loading a specific observable."""
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info(self.test_files['fstream'])
        
        if observable_infos:
            # Load first observable
            sampling = loader.load_observable(self.test_files['fstream'], observable_infos[0])
            
            self.assertEqual(sampling.observable_info, observable_infos[0])
            self.assertEqual(sampling.ensemble_info, ensemble_info)
            self.assertEqual(sampling.sampling_info, sampling_info)
            self.assertGreater(len(sampling.data), 1)
            self.assertEqual(len(sampling.resampled_values), sampling_info.num_resamplings)
    
    @unittest.skip("Skip loading all data for faster tests - enable for full integration testing")
    def test_load_all_observables(self):
        """Test loading all observables."""
        loader = SigmondLoader()
        
        # Use a smaller test case
        result = loader.load_all_observables(self.test_files['fstream'])
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        
        # Check that all values are SigmondSampling objects
        for name, sampling in result.items():
            self.assertIsInstance(sampling, type(list(result.values())[0]))


class TestLoaderMocked(unittest.TestCase):
    """Test loader with mocked sigmond_query responses."""
    
    def setUp(self):
        """Set up mocked responses."""
        self.mock_header_response = '''
This is a Sigmond samplings file in fstreams format

<SigmondSamplingsFile>
   <MCBinsInfo>
      <MCEnsembleInfo>test_ensemble</MCEnsembleInfo>
      <NumberOfMeasurements>100</NumberOfMeasurements>
      <NumberOfBins>50</NumberOfBins>
   </MCBinsInfo>
   <MCSamplingInfo>
      <Bootstrapper>
         <NumberResamplings>1000</NumberResamplings>
         <Seed>1234</Seed>
         <BootSkip>0</BootSkip>
      </Bootstrapper>
   </MCSamplingInfo>
</SigmondSamplingsFile>
'''
        
        self.mock_keys_response = '''
This is a Sigmond samplings file in fstreams format
Record 0:

<MCObservable>
   <Info>test_obs 0 n re</Info>
</MCObservable>

Record 1:

<MCObservable>
   <Info>test_obs2 0 n re</Info>
</MCObservable>
'''
        
        self.mock_values_response = '''
This is a Sigmond samplings file in fstreams format
Record 0:
Full Sampling Mean Value = 1.0

[0] = 1.0
[1] = 1.1
[2] = 0.9

Record 1:
Full Sampling Mean Value = 2.0

[0] = 2.0
[1] = 2.1
[2] = 1.9
'''
    
    @patch('subprocess.run')
    def test_mocked_file_info_parsing(self, mock_run):
        """Test parsing of file info with mocked responses."""
        def side_effect(cmd, **kwargs):
            if '-h' in cmd:
                return MagicMock(returncode=0, stdout="help")
            elif '-i' in cmd:
                return MagicMock(returncode=0, stdout=self.mock_header_response)
            elif '-k' in cmd:
                return MagicMock(returncode=0, stdout=self.mock_keys_response)
            elif '-v' in cmd:
                return MagicMock(returncode=0, stdout=self.mock_values_response)
            else:
                return MagicMock(returncode=0, stdout="")
        
        mock_run.side_effect = side_effect
        
        loader = SigmondLoader()
        ensemble_info, sampling_info, observable_infos = loader.get_file_info("test_file.smp")
        
        # Check parsed info
        self.assertEqual(ensemble_info.ensemble_name, "test_ensemble")
        self.assertEqual(ensemble_info.num_measurements, 100)
        self.assertEqual(ensemble_info.num_bins, 50)
        
        self.assertEqual(sampling_info.method, "bootstrap")
        self.assertEqual(sampling_info.num_resamplings, 1000)
        self.assertEqual(sampling_info.seed, 1234)
        
        self.assertEqual(len(observable_infos), 2)
        self.assertEqual(observable_infos[0].name, "test_obs")
        self.assertEqual(observable_infos[1].name, "test_obs2")


if __name__ == '__main__':
    unittest.main() 