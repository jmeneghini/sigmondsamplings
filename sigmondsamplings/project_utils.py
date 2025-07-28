"""
Utility functions for lattice QCD project analysis.

This module contains helper functions that are useful across different
lattice QCD analysis projects.
"""

import os
import platform
import glob
import re
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .sampling import SigmondSampling


@dataclass(frozen=True, slots=True)
class OSInfo:
    """Basic operating-system information."""
    name: str                 # 'Windows', 'Darwin', 'Linux', …
    version: str              # user-friendly version string
    distro: Optional[str] = None   # only for Linux, e.g. 'Ubuntu 24.04'
    

@dataclass(frozen=True, slots=True)
class LinuxDistro:
    """Linux distribution information."""
    name: str                 # e.g. 'Ubuntu'
    version: str              # e.g. '24.04 LTS'


def _linux_distribution() -> LinuxDistro:
    """Return LinuxDistro without requiring lsb_release."""
    try:
        import distro
        return LinuxDistro(
            name=distro.name(pretty=True),  # e.g. 'Ubuntu'
            version=distro.version(pretty=True)  # e.g. '24.04 LTS'
        )
    except ImportError:
        # Fallback if distro package is not available
        return LinuxDistro(name="Unknown", version="Unknown")
    

def get_os_info() -> OSInfo:
    """Return OSInfo with name, version, and (on Linux) distribution."""
    system = platform.system()          # 'Windows' | 'Darwin' | 'Linux' | …
    if system == "Linux":
        distro_info = _linux_distribution()
        # kernel version (e.g. '6.8.9-arch1-1')
        kernel = platform.release()
        return OSInfo(name=system, version=kernel, distro=distro_info.name if distro_info else None)

    if system == "Darwin":              # macOS
        return OSInfo(
            name="macOS",
            version=platform.mac_ver()[0] or platform.release()
        )

    if system in {"Windows", "CYGWIN_NT"}:
        raise NotImplementedError("No Windows support")

    # catch-all for the rest
    return OSInfo(name=system, version=platform.release())


def string_of_list_to_list(string: str) -> List[str]:
    """
    Convert a string representation of a list to an actual list.

    Parameters
    ----------
    string : str
        The string representation of a list, e.g. "['Phi', 'Rho', 'Pi']".

    Returns
    -------
    List[str]
        The converted list.
    """
    # Remove the outer brackets and split by ','
    return [s.strip().replace("'", "").lower() for s in string.strip('[]').split(",")]


def get_gamma_from_elab_and_ecm(elab: Union[float, SigmondSampling], 
                               ecm: Union[float, SigmondSampling]) -> Union[float, SigmondSampling]:
    """
    Calculate the gamma factor from elab and ecm energies.

    Parameters
    ----------
    elab : Union[float, SigmondSampling]
        The laboratory energy.
    ecm : Union[float, SigmondSampling]
        The center-of-mass energy.

    Returns
    -------
    Union[float, SigmondSampling]
        The gamma factor calculated from elab and ecm.
    """
    # Gamma is defined as the ratio of the center-of-mass energy to the laboratory energy
    if isinstance(elab, (int, float)) and elab == 0:
        raise ValueError("elab cannot be zero for gamma calculation.")
    
    gamma = elab / ecm
    return gamma


def get_g_ref_from_Gamma_ref(Gamma_ref: Union[float, SigmondSampling], 
                             rho_mass_ref: Union[float, SigmondSampling]) -> Union[float, SigmondSampling]:
    """
    Calculate the g_ref from Gamma_ref and rho_mass_ref.

    Parameters
    ----------
    Gamma_ref : Union[float, SigmondSampling]
        The reference decay width.
    rho_mass_ref : Union[float, SigmondSampling]
        The reference mass of the rho meson.

    Returns
    -------
    Union[float, SigmondSampling]
        The calculated g_ref value.
    """
    mass_rho_ref_sqr = rho_mass_ref**2
    return np.sqrt(32 * np.pi * mass_rho_ref_sqr * Gamma_ref/np.sqrt(mass_rho_ref_sqr - 4.0))


def get_Gamma_ref_from_g_ref(g_ref: Union[float, SigmondSampling], 
                             rho_mass_ref: Union[float, SigmondSampling]) -> Union[float, SigmondSampling]:
    """
    Calculate the Gamma_ref from g_ref and rho_mass_ref.

    Parameters
    ----------
    g_ref : Union[float, SigmondSampling]
        The reference coupling constant.
    rho_mass_ref : Union[float, SigmondSampling]
        The reference mass of the rho meson.

    Returns
    -------
    Union[float, SigmondSampling]
        The calculated Gamma_ref value.
    """
    mass_rho_ref_sqr = rho_mass_ref**2
    return g_ref**2 / (32 * np.pi * mass_rho_ref_sqr) * np.sqrt(mass_rho_ref_sqr - 4.0)


def find_files_with_pattern(base_dir: Union[str, Path], 
                           pattern: str,
                           recursive: bool = True) -> List[str]:
    """
    Find files matching a pattern in a directory.
    
    Parameters
    ----------
    base_dir : Union[str, Path]
        Base directory to search in.
    pattern : str
        Glob pattern to match files.
    recursive : bool, optional
        Whether to search recursively in subdirectories.
        
    Returns
    -------
    List[str]
        List of matching file paths.
    """
    base_path = Path(base_dir)
    if recursive:
        return [str(p) for p in base_path.rglob(pattern)]
    else:
        return [str(p) for p in base_path.glob(pattern)]


def extract_numeric_values_from_filename(filename: str, 
                                        patterns: Union[str, List[str]]) -> Dict[str, Union[int, float]]:
    """
    Extract numeric values from filename using regex patterns.
    
    Parameters
    ----------
    filename : str
        The filename to parse.
    patterns : Union[str, List[str]]
        Regex patterns with named groups to extract values.
        
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary mapping group names to extracted numeric values.
        
    Examples
    --------
    >>> extract_numeric_values_from_filename(
    ...     "data_L32_beta6.0_m0.01.dat",
    ...     [r"L(?P<L>\d+)", r"beta(?P<beta>\d+\.\d+)", r"m(?P<mass>\d+\.\d+)"]
    ... )
    {'L': 32, 'beta': 6.0, 'mass': 0.01}
    """
    if isinstance(patterns, str):
        patterns = [patterns]
        
    results = {}
    filename_base = Path(filename).name
    
    for pattern in patterns:
        match = re.search(pattern, filename_base)
        if match:
            for name, value in match.groupdict().items():
                try:
                    # Try to convert to int first, then float
                    if '.' in value:
                        results[name] = float(value)
                    else:
                        results[name] = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    results[name] = value
                    
    return results


def get_momentum_squared_from_momentum(momentum: Union[Tuple[int, int, int], List[int]]) -> int:
    """
    Calculate momentum squared from momentum vector.
    
    Parameters
    ----------
    momentum : Union[Tuple[int, int, int], List[int]]
        Momentum vector (px, py, pz).
        
    Returns
    -------
    int
        Momentum squared (px^2 + py^2 + pz^2).
    """
    return sum(p**2 for p in momentum)


def get_momentum_from_momentum_squared(psq: int) -> List[Tuple[int, int, int]]:
    """
    Get all possible momentum vectors for a given momentum squared.
    
    Parameters
    ----------
    psq : int
        Momentum squared value.
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of all momentum vectors (px, py, pz) that give the specified psq.
    """
    momenta = []
    max_p = int(np.sqrt(psq)) + 1
    
    for px in range(-max_p, max_p + 1):
        for py in range(-max_p, max_p + 1):
            for pz in range(-max_p, max_p + 1):
                if px*px + py*py + pz*pz == psq:
                    momenta.append((px, py, pz))
                    
    return momenta


def group_observables_by_momentum(observables: Dict[str, SigmondSampling],
                                 observable_parser: callable) -> Dict[int, Dict[str, SigmondSampling]]:
    """
    Group observables by momentum squared using a parser function.
    
    Parameters
    ----------
    observables : Dict[str, SigmondSampling]
        Dictionary of observables to group.
    observable_parser : callable
        Function that takes an observable name and returns (psq, energy_type, level_index) or None.
        
    Returns
    -------
    Dict[int, Dict[str, SigmondSampling]]
        Dictionary grouped by momentum squared.
    """
    grouped = {}
    
    for obs_name, sampling in observables.items():
        parsed = observable_parser(obs_name)
        if parsed is not None:
            psq, energy_type, level_index = parsed
            
            if psq not in grouped:
                grouped[psq] = {}
            grouped[psq][obs_name] = sampling
            
    return grouped