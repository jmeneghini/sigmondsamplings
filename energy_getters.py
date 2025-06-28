# energy getters

import re
from typing import Dict, List, Union, Tuple
import SigmondSamplings as ss

loader = ss.SigmondLoader()

PSQ_map = {0: "PSQ=0", 1: "P=(0,0,1)", 2: "P=(0,1,1)", 3: "P=(1,1,1)"}

def get_energies_for_ensemble(file_path: str, level_indices_for_each_psq: Dict[int, Union[int, List[int]]],
                              energy_types: Union[str, List[str]], ref: bool) -> Tuple[ss.EnsembleInfo, ss.SamplingInfo,
                                                                            Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]]:
    """
    Get the energies for a given ensemble from the specified file.
    
    Parameters
    ----------
    file_path : str
        Full path to the SigmondLoader file containing the energies.
    level_indices_for_each_psq : Dict[int, Union[int, List[int]]]
        A dictionary mapping each momentum squared (psq) to the level indices
        for which energies should be retrieved. The level indices can be a single
        integer or a list of integers.
    energy_types : Union[str, List[str]]
        The types of energies to retrieve. This can be a single energy type or a list of energy types.
    ref : bool
        Whether to include "_ref" suffix in the pattern matching.
        
    Returns
    -------
    Tuple[ss.EnsembleInfo, ss.SamplingInfo, Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]]
        A tuple containing:
        - EnsembleInfo: Information about the ensemble
        - SamplingInfo: Information about the sampling method
        - Dictionary where:
            - Key: momentum squared (psq)
            - Value: Dictionary where:
                - Key: energy type (e.g., "elab", "dElab", "ecm")
                - Value: Dictionary where:
                    - Key: level index
                    - Value: SigmondSampling object
    """
    
    # Load all observables once
    all_observables = loader.load_all_observables(file_path)
    
    # Get ensemble and sampling info
    ensemble_info, sampling_info, _ = loader.get_file_info(file_path)
    
    # Ensure energy_types is a list
    if isinstance(energy_types, str):
        energy_types = [energy_types]
    
    # Initialize output structure
    out_results: Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]] = {}
    
    # Parse all observables and categorize them
    for obs_name, sampling in all_observables.items():
        # Try to extract information from the observable name
        parsed_info = _parse_observable_name(obs_name, ref)
        if parsed_info is None:
            continue
            
        psq, energy_type, level_index = parsed_info
        
        # Check if this psq is requested
        if psq not in level_indices_for_each_psq:
            continue
            
        # Check if this energy type is requested
        if energy_type not in energy_types:
            continue
            
        # Check if this level index is requested
        requested_levels = level_indices_for_each_psq[psq]
        if isinstance(requested_levels, int):
            requested_levels = [requested_levels]
        if level_index not in requested_levels:
            continue
            
        # Add to results
        if psq not in out_results:
            out_results[psq] = {}
        if energy_type not in out_results[psq]:
            out_results[psq][energy_type] = {}
        out_results[psq][energy_type][level_index] = sampling
    
    # Validate that we found all requested observables
    _validate_results(out_results, level_indices_for_each_psq, energy_types)
    
    return ensemble_info, sampling_info, out_results


def _parse_observable_name(obs_name: str, ref: bool) -> Union[Tuple[int, str, int], None]:
    """
    Parse an observable name to extract PSQ, energy type, and level index.
    
    Parameters
    ----------
    obs_name : str
        Observable name like "isosinglet_S=0_A1g_1_PSQ=0_elab_1_ref 0"
    ref : bool
        Whether to expect "_ref" suffix
        
    Returns
    -------
    Union[Tuple[int, str, int], None]
        Tuple of (psq, energy_type, level_index) or None if parsing fails
    """
    # Extract the actual observable name (before the space and index)
    obs_parts = obs_name.split()
    if len(obs_parts) != 2:
        return None
    actual_name = obs_parts[0]
    
    # Build the regex pattern based on whether ref is expected
    ref_suffix = "_ref" if ref else ""
    
    # Pattern to match PSQ format: PSQ=N
    psq_pattern = r"PSQ=(\d+)"
    psq_match = re.search(psq_pattern, actual_name)
    
    if psq_match:
        psq = int(psq_match.group(1))
        # Pattern to match energy type and level: _{energy_type}_{level}{_ref}
        energy_pattern = rf"_([a-zA-Z]+)_(\d+){re.escape(ref_suffix)}$"
        energy_match = re.search(energy_pattern, actual_name)
        if energy_match:
            energy_type = energy_match.group(1)
            level_index = int(energy_match.group(2))
            return psq, energy_type, level_index
    else:
        # Try P=(x,y,z) format
        p_pattern = r"P=\(([^)]+)\)"
        p_match = re.search(p_pattern, actual_name)
        if p_match:
            p_coords = p_match.group(1).split(',')
            psq = sum(int(x)**2 for x in p_coords)
            # Pattern to match energy type and level
            energy_pattern = rf"_([a-zA-Z]+)_(\d+){re.escape(ref_suffix)}$"
            energy_match = re.search(energy_pattern, actual_name)
            if energy_match:
                energy_type = energy_match.group(1)
                level_index = int(energy_match.group(2))
                return psq, energy_type, level_index
    
    return None


def _validate_results(results: Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]], 
                     requested_psqs: Dict[int, Union[int, List[int]]], 
                     requested_energy_types: List[str]) -> None:
    """
    Validate that all requested observables were found.
    
    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]
        The parsed results
    requested_psqs : Dict[int, Union[int, List[int]]]
        The requested PSQ values and levels
    requested_energy_types : List[str]
        The requested energy types
        
    Raises
    ------
    ValueError
        If any requested observable was not found
    """
    missing_observables = []
    
    for psq, requested_levels in requested_psqs.items():
        if isinstance(requested_levels, int):
            requested_levels = [requested_levels]
            
        for energy_type in requested_energy_types:
            for level_index in requested_levels:
                if (psq not in results or 
                    energy_type not in results[psq] or 
                    level_index not in results[psq][energy_type]):
                    missing_observables.append(f"PSQ={psq}, {energy_type}, level={level_index}")
    
    if missing_observables:
        raise ValueError(f"Missing observables: {', '.join(missing_observables)}")


def get_energy_levels_summary(results: Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]) -> Dict[int, Dict[str, List[int]]]:
    """
    Get a summary of available energy levels for each PSQ and energy type.
    
    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]
        Results from get_energies_for_ensemble
        
    Returns
    -------
    Dict[int, Dict[str, List[int]]]
        Dictionary mapping PSQ -> energy_type -> list of available levels
    """
    summary = {}
    for psq, energy_data in results.items():
        summary[psq] = {}
        for energy_type, level_data in energy_data.items():
            summary[psq][energy_type] = sorted(level_data.keys())
    return summary


def print_energy_summary(results: Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]) -> None:
    """
    Print a nice summary of the loaded energy data.
    
    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[int, ss.SigmondSampling]]]
        Results from get_energies_for_ensemble
    """
    summary = get_energy_levels_summary(results)
    
    print("Energy Data Summary:")
    print("=" * 50)
    
    for psq in sorted(summary.keys()):
        print(f"PSQ = {psq}:")
        for energy_type in sorted(summary[psq].keys()):
            levels = summary[psq][energy_type]
            print(f"  {energy_type}: levels {levels}")
        print()


# Example usage:
if __name__ == "__main__":
    # Example of how to use the function
    file_path = "path/to/your/sampling/file.hdf5"
    
    # Request specific levels for each PSQ
    level_indices = {
        0: [1, 2, 3],  # PSQ=0, levels 1, 2, 3
        1: [1, 2],     # PSQ=1, levels 1, 2
        2: 1           # PSQ=2, level 1 only
    }
    
    # Request specific energy types
    energy_types = ["elab", "dElab", "ecm"]
    
    try:
        ensemble_info, sampling_info, energy_data = get_energies_for_ensemble(
            file_path, level_indices, energy_types, ref=True
        )
        
        print(f"Ensemble: {ensemble_info.ensemble_name}")
        print(f"Sampling: {sampling_info.method} with {sampling_info.num_resamplings} resamplings")
        print()
        
        print_energy_summary(energy_data)
        
        # Access specific energy data
        if 0 in energy_data and "elab" in energy_data[0] and 1 in energy_data[0]["elab"]:
            elab_level1_psq0 = energy_data[0]["elab"][1]
            print(f"Example: PSQ=0, elab, level 1 = {elab_level1_psq0.mean:.6f} ± {elab_level1_psq0.error:.6f}")
            
    except ValueError as e:
        print(f"Error: {e}") 