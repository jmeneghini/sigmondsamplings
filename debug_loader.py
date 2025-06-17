#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from SigmondSamplings.loader import SigmondLoader
import subprocess

def main():
    loader = SigmondLoader()
    
    # Test fstream file
    print("Testing fstream file...")
    try:
        is_valid, file_type, paths = loader.check_file_validity('example_samplings/k-pi_scatteringDrew.smp')
        print(f"Valid: {is_valid}, Type: {file_type}, Paths: {paths}")
        
        if is_valid:
            ensemble_info, sampling_info, observable_infos = loader.get_file_info('example_samplings/k-pi_scatteringDrew.smp')
            print(f"Ensemble: {ensemble_info}")
            print(f"Sampling: {sampling_info}")
            print(f"Number of observables: {len(observable_infos)}")
            if observable_infos:
                print(f"First observable: {observable_infos[0]}")
                print(f"Second observable: {observable_infos[1]}")
                print(f"Third observable: {observable_infos[2]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test HDF5 file without path - debug the actual subprocess call
    print("\nTesting HDF5 file without path...")
    hdf5_file = 'example_samplings/fit_spectrum_fitparams-5-27-25-interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5'
    print(f"Testing file: {hdf5_file}")
    
    # Direct subprocess call
    try:
        result = subprocess.run(['sigmond_query', '-i', hdf5_file], 
                              capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {repr(result.stdout)}")
        print(f"Stderr: {repr(result.stderr)}")
    except Exception as e:
        print(f"Subprocess error: {e}")
    
    # Now test with loader
    try:
        is_valid, file_type, paths = loader.check_file_validity(hdf5_file)
        print(f"Valid: {is_valid}, Type: {file_type}, Paths: {paths}")
    except Exception as e:
        print(f"Loader error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 