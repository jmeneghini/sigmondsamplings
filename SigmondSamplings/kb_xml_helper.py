"""
KBfitXMLHelper module for generating XML configurations for KB fits.

This module provides functionality to generate XML configurations for:
- Determinant residual fits (detres)
- Single channel fits
- Print tasks

It takes observables, ensemble information, and other parameters to create
the appropriate XML structure for KB fitting tasks.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import re

from .sampling import EnsembleInfo, SamplingInfo, ObservableInfo


class BoxQuantizationInfo:
    """Information about box quantization for KBBlocks."""
    
    def __init__(self, momentum_ray: str, momentum_int_squared: int, 
                 lg_irrep: str, lmax_values: Union[int, str] = 0):
        self.momentum_ray = momentum_ray
        self.momentum_int_squared = momentum_int_squared
        self.lg_irrep = lg_irrep
        self.lmax_values = str(lmax_values)


class ParticleInfo:
    """Information about particles for decay channels."""
    
    def __init__(self, name: str, spin_times_two: int, identical: bool = False):
        self.name = name
        self.spin_times_two = spin_times_two
        self.identical = identical


class KElementInfo:
    """Information for K-matrix elements."""
    
    def __init__(self, j_times_two: int, k_index1: str, k_index2: str):
        self.j_times_two = j_times_two
        self.k_index1 = k_index1
        self.k_index2 = k_index2


class FitParameterInfo:
    """Information for fit parameters."""
    
    def __init__(self, parameter_name: str, starting_value: float, 
                 k_element_info: Optional[KElementInfo] = None,
                 polynomial_power: Optional[int] = None):
        self.parameter_name = parameter_name
        self.starting_value = starting_value
        self.k_element_info = k_element_info
        self.polynomial_power = polynomial_power


class KBfitXMLHelper:
    """Helper class for generating KBfit XML configurations."""
    
    def __init__(self):
        self.namespace = None
        
    def _create_element(self, tag: str, text: Optional[str] = None, 
                       parent: Optional[ET.Element] = None) -> ET.Element:
        """Create an XML element with optional text content."""
        if parent is not None:
            elem = ET.SubElement(parent, tag)
        else:
            elem = ET.Element(tag)
        
        if text is not None:
            elem.text = str(text)
        
        return elem
    
    def _create_mcsamplinginfo_element(self, sampling_info: SamplingInfo) -> ET.Element:
        """Create MCSamplingInfo element from SamplingInfo object."""
        sampling_elem = self._create_element("MCSamplingInfo")
        
        if sampling_info.method.lower() == "bootstrap":
            bootstrap_elem = self._create_element("Bootstrapper", parent=sampling_elem)
            self._create_element("NumberResamplings", str(sampling_info.num_resamplings), bootstrap_elem)
            self._create_element("Seed", str(sampling_info.seed), bootstrap_elem)
            self._create_element("BootSkip", str(sampling_info.boot_skip), bootstrap_elem)
        elif sampling_info.method.lower() == "jackknife":
            if sampling_info.num_resamplings:
                jackknife_elem = self._create_element("Jackkniffer", parent=sampling_elem)
                self._create_element("NumberResamplings", str(sampling_info.num_resamplings), jackknife_elem)
            else:
                self._create_element("Jackknife", parent=sampling_elem)
        
        return sampling_elem
    
    def _create_mcbinsinfo_element(self, ensemble_info: EnsembleInfo) -> ET.Element:
        """Create MCBinsInfo element from EnsembleInfo object."""
        bins_elem = self._create_element("MCBinsInfo")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, bins_elem)
        self._create_element("NumberOfMeasurements", str(ensemble_info.num_measurements), bins_elem)
        self._create_element("NumberOfBins", str(ensemble_info.num_bins), bins_elem)
        
        if ensemble_info.tweak_info:
            tweak_elem = self._create_element("TweakEnsemble", parent=bins_elem)
            for key, value in ensemble_info.tweak_info.items():
                self._create_element(key, str(value), tweak_elem)
        
        return bins_elem
    
    def _create_mcensemble_parameters(self, ensemble_info: EnsembleInfo, 
                                    reference_particle: str,
                                    particle_masses: Dict[str, float]) -> ET.Element:
        """Create MCEnsembleParameters element."""
        params_elem = self._create_element("MCEnsembleParameters")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, params_elem)
        
        # Reference mass
        ref_mass_elem = self._create_element("ReferenceMassTimeSpacingProduct", parent=params_elem)
        mcobs_elem = self._create_element("MCObs", f"{reference_particle}(0)_elab 0", ref_mass_elem)
        
        # Particle masses
        for particle_name, mass_value in particle_masses.items():
            mass_elem = self._create_element("ParticleMass", parent=params_elem)
            self._create_element("n", particle_name, mass_elem)
            self._create_element("FixedValue", str(mass_value), mass_elem)
        
        return params_elem
    
    def _create_kbblock(self, ensemble_info: EnsembleInfo, 
                       box_quantization: BoxQuantizationInfo,
                       lab_frame_energies: List[str]) -> ET.Element:
        """Create KBBlock element."""
        kb_block = self._create_element("KBBlock")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, kb_block)
        
        # Box quantization
        box_quant_elem = self._create_element("BoxQuantization", parent=kb_block)
        self._create_element("TotalMomentumRay", box_quantization.momentum_ray, box_quant_elem)
        self._create_element("TotalMomentumIntSquared", str(box_quantization.momentum_int_squared), box_quant_elem)
        self._create_element("LGIrrep", box_quantization.lg_irrep, box_quant_elem)
        self._create_element("LmaxValues", box_quantization.lmax_values, box_quant_elem)
        
        # Lab frame energies
        for energy_obs in lab_frame_energies:
            energy_elem = self._create_element("LabFrameEnergy", parent=kb_block)
            self._create_element("MCObs", energy_obs, energy_elem)
        
        return kb_block
    
    def _create_print_kbblock(self, ensemble_info: EnsembleInfo,
                             box_quantization: BoxQuantizationInfo,
                             energy_range: Tuple[float, float, float]) -> ET.Element:
        """Create KBBlock element for print tasks."""
        kb_block = self._create_element("KBBlock")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, kb_block)
        
        # Box quantization
        box_quant_elem = self._create_element("BoxQuantization", parent=kb_block)
        self._create_element("TotalMomentumRay", box_quantization.momentum_ray, box_quant_elem)
        self._create_element("TotalMomentumIntSquared", str(box_quantization.momentum_int_squared), box_quant_elem)
        self._create_element("LGIrrep", box_quantization.lg_irrep, box_quant_elem)
        self._create_element("LmaxValues", box_quantization.lmax_values, box_quant_elem)
        
        # Energy range for print
        self._create_element("LabFrameEnergyMin", str(energy_range[0]), kb_block)
        self._create_element("LabFrameEnergyMax", str(energy_range[1]), kb_block)
        self._create_element("LabFrameEnergyInc", str(energy_range[2]), kb_block)
        
        return kb_block
    
    def _create_decay_channels(self, particles: List[ParticleInfo]) -> ET.Element:
        """Create DecayChannels element."""
        channels_elem = self._create_element("DecayChannels")
        
        for particle in particles:
            channel_elem = self._create_element("DecayChannelInfo", parent=channels_elem)
            self._create_element("Particle1Name", particle.name, channel_elem)
            self._create_element("Spin1TimesTwo", str(particle.spin_times_two), channel_elem)
            if particle.identical:
                self._create_element("Identical", parent=channel_elem)
        
        return channels_elem
    
    def _create_kbobservables(self, sampling_info: SamplingInfo,
                             ensemble_info: EnsembleInfo,
                             sampling_files: List[str]) -> ET.Element:
        """Create KBObservables element."""
        kb_obs = self._create_element("KBObservables")
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        kb_obs.append(sampling_elem)
        
        # Bins info (optional)
        bins_elem = self._create_mcbinsinfo_element(ensemble_info)
        kb_obs.append(bins_elem)
        
        # Verbose flag
        self._create_element("Verbose", parent=kb_obs)
        
        # Sampling data files
        sampling_data_elem = self._create_element("SamplingData", parent=kb_obs)
        for file_path in sampling_files:
            self._create_element("FileName", file_path, sampling_data_elem)
        
        return kb_obs
    
    def _create_ktilde_matrix_inverse(self, k_elements: List[KElementInfo],
                                    fit_expressions: Dict[str, str],
                                    decay_channels: List[ParticleInfo],
                                    starting_values: List[FitParameterInfo]) -> ET.Element:
        """Create KtildeMatrixInverse element for detres fits."""
        ktilde_elem = self._create_element("KtildeMatrixInverse")
        
        # K-matrix elements
        for k_element in k_elements:
            element_elem = self._create_element("Element", parent=ktilde_elem)
            k_elem_info = self._create_element("KElementInfo", parent=element_elem)
            self._create_element("JTimesTwo", str(k_element.j_times_two), k_elem_info)
            self._create_element("KIndex", k_element.k_index1, k_elem_info)
            self._create_element("KIndex", k_element.k_index2, k_elem_info)
            
            # Fit form
            if k_element.k_index1 in fit_expressions:
                fit_form_elem = self._create_element("FitForm", parent=element_elem)
                expr_elem = self._create_element("Expression", parent=fit_form_elem)
                self._create_element("String", fit_expressions[k_element.k_index1], expr_elem)
        
        # Decay channels
        channels_elem = self._create_decay_channels(decay_channels)
        ktilde_elem.append(channels_elem)
        
        # Starting values
        start_vals_elem = self._create_element("StartingValues", parent=ktilde_elem)
        for param in starting_values:
            param_elem = self._create_element("KFitParamInfo", parent=start_vals_elem)
            
            # String expression parameter
            str_expr_elem = self._create_element("StringExpressionParameter", parent=param_elem)
            self._create_element("ParameterName", param.parameter_name, str_expr_elem)
            
            if param.k_element_info:
                k_elem_info = self._create_element("KElementInfo", parent=str_expr_elem)
                self._create_element("JTimesTwo", str(param.k_element_info.j_times_two), k_elem_info)
                self._create_element("KIndex", param.k_element_info.k_index1, k_elem_info)
                self._create_element("KIndex", param.k_element_info.k_index2, k_elem_info)
            
            self._create_element("StartingValue", str(param.starting_value), param_elem)
        
        return ktilde_elem
    
    def _create_ktilde_matrix(self, k_elements: List[KElementInfo],
                            polynomial_powers: Dict[str, List[int]],
                            decay_channels: List[ParticleInfo],
                            starting_values: List[FitParameterInfo]) -> ET.Element:
        """Create KtildeMatrix element for print tasks."""
        ktilde_elem = self._create_element("KtildeMatrix")
        
        # K-matrix elements with polynomial fit forms
        for k_element in k_elements:
            element_elem = self._create_element("Element", parent=ktilde_elem)
            k_elem_info = self._create_element("KElementInfo", parent=element_elem)
            self._create_element("JTimesTwo", str(k_element.j_times_two), k_elem_info)
            self._create_element("KIndex", k_element.k_index1, k_elem_info)
            self._create_element("KIndex", k_element.k_index2, k_elem_info)
            
            # Polynomial fit form
            if k_element.k_index1 in polynomial_powers:
                fit_form_elem = self._create_element("FitForm", parent=element_elem)
                poly_elem = self._create_element("Polynomial", parent=fit_form_elem)
                powers_str = " ".join(map(str, polynomial_powers[k_element.k_index1]))
                self._create_element("Powers", powers_str, poly_elem)
        
        # Decay channels
        channels_elem = self._create_decay_channels(decay_channels)
        ktilde_elem.append(channels_elem)
        
        # Starting values
        start_vals_elem = self._create_element("StartingValues", parent=ktilde_elem)
        for param in starting_values:
            param_elem = self._create_element("KFitParamInfo", parent=start_vals_elem)
            
            # Polynomial term
            poly_term_elem = self._create_element("PolynomialTerm", parent=param_elem)
            self._create_element("Power", str(param.polynomial_power), poly_term_elem)
            
            if param.k_element_info:
                k_elem_info = self._create_element("KElementInfo", parent=poly_term_elem)
                self._create_element("JTimesTwo", str(param.k_element_info.j_times_two), k_elem_info)
                self._create_element("KIndex", param.k_element_info.k_index1, k_elem_info)
                self._create_element("KIndex", param.k_element_info.k_index2, k_elem_info)
            
            self._create_element("StartingValue", str(param.starting_value), param_elem)
        
        return ktilde_elem
    
    def create_detres_xml(self,
                         project_name: str,
                         observables: List[ObservableInfo],
                         sampling_info: SamplingInfo,
                         reference_particle: str,
                         particle_masses: Dict[str, float],
                         box_quantizations: List[BoxQuantizationInfo],
                         sampling_files: List[str],
                         omega_mu: float = 8.0,
                         quantization_condition: str = "KtildeinvB",
                         default_energy_format: str = "reference_ratio",
                         fit_expressions: Optional[Dict[str, str]] = None,
                         k_elements: Optional[List[KElementInfo]] = None,
                         decay_channels: Optional[List[ParticleInfo]] = None,
                         starting_values: Optional[List[FitParameterInfo]] = None,
                         use_ktilde_inverse: bool = True,
                         output_file: Optional[str] = None) -> str:
        """
        Create XML for determinant residual fits.
        
        Args:
            project_name: Name of the project
            observables: List of observable infos (ensemble info extracted from these)
            sampling_info: Sampling information
            reference_particle: Name of reference particle
            particle_masses: Dictionary mapping particle names to masses
            box_quantizations: List of BoxQuantizationInfo objects (contains lmax values)
            sampling_files: List of paths to sampling files
            omega_mu: Omega mu parameter
            quantization_condition: Quantization condition type
            default_energy_format: Default energy format ("reference_ratio" or "time_spacing_product")
            fit_expressions: Dictionary mapping K-indices to fit expressions
            k_elements: List of K-matrix element info
            decay_channels: List of decay channel particles
            starting_values: List of fit parameter starting values
            use_ktilde_inverse: Whether to use KtildeInverse (True) or Ktilde (False)
            output_file: Optional output file path
        
        Returns:
            XML string
        """
        # Extract unique ensemble infos from observables
        ensemble_infos = []
        for obs in observables:
            if obs.ensemble_info not in ensemble_infos:
                ensemble_infos.append(obs.ensemble_info)
        
        # Group observables by momentum to create KB blocks
        momentum_groups = self.group_observables_by_momentum(observables)
        
        # Create KB blocks from momentum groups and box quantizations
        kb_blocks = []
        for i, (momentum_key, obs_list) in enumerate(momentum_groups.items()):
            if i < len(box_quantizations):
                box_quant = box_quantizations[i]
                # Extract lab frame energies from observable names
                lab_energies = [obs.name for obs in obs_list]
                # Get ensemble from first observable in group
                ensemble_info = obs_list[0].ensemble_info
                kb_blocks.append((ensemble_info, box_quant, lab_energies))
        
        # Create root element
        root = self._create_element("KBFit")
        
        # Initialize section
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        self._create_element("EchoXML", parent=init_elem)
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        init_elem.append(sampling_elem)
        
        # Task sequence
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoFit", task)
        self._create_element("Type", "DeterminantResidualFit", task)
        
        # Minimizer info
        minimizer = self._create_element("MinimizerInfo", parent=task)
        self._create_element("Method", "NL2Sno", minimizer)
        self._create_element("ParameterRelTol", "1e-6", minimizer)
        self._create_element("ChiSquareRelTol", "1e-4", minimizer)
        self._create_element("MaximumIterations", "1024", minimizer)
        self._create_element("Verbosity", "Low", minimizer)
        
        # Output files
        self._create_element("OutSamplingsFile", "samplings.hdf5[/samplings]", task)
        self._create_element("EcmQcmBoxSamplingsStub", "ecm_qcm_box_samplings", task)
        
        # Determinant residual fit section
        detres_elem = self._create_element("DeterminantResidualFit", parent=task)
        self._create_element("OmegaMu", str(omega_mu), detres_elem)
        self._create_element("QuantizationCondition", quantization_condition, detres_elem)
        self._create_element("Verbose", parent=detres_elem)
        
        # K-tilde matrix inverse or K-tilde matrix (if provided)
        if k_elements and decay_channels and starting_values:
            if fit_expressions is None:
                fit_expressions = {}
            
            if use_ktilde_inverse:
                ktilde_elem = self._create_ktilde_matrix_inverse(
                    k_elements, fit_expressions, decay_channels, starting_values
                )
            else:
                # Convert lmax values from box quantizations to polynomial powers format
                polynomial_powers = {}
                for i, box_quant in enumerate(box_quantizations):
                    lmax_str = str(box_quant.lmax_values)
                    if ' ' in lmax_str:
                        # Multiple l values (space separated)
                        lmax_vals = [int(x) for x in lmax_str.split()]
                        for j, max_l in enumerate(lmax_vals):
                            key = f"block_{i}_elem_{j}"
                            polynomial_powers[key] = list(range(max_l + 1))
                    else:
                        # Single l value
                        max_l = int(lmax_str)
                        key = f"block_{i}"
                        polynomial_powers[key] = list(range(max_l + 1))
                
                ktilde_elem = self._create_ktilde_matrix(
                    k_elements, polynomial_powers, decay_channels, starting_values
                )
            detres_elem.append(ktilde_elem)
        
        # Default energy format
        self._create_element("DefaultEnergyFormat", default_energy_format, detres_elem)
        
        # MC ensemble parameters for each ensemble
        for ensemble_info in ensemble_infos:
            params_elem = self._create_mcensemble_parameters(
                ensemble_info, reference_particle, particle_masses
            )
            detres_elem.append(params_elem)
        
        # KB blocks
        for ensemble_info, box_quant, lab_energies in kb_blocks:
            kb_block = self._create_kbblock(ensemble_info, box_quant, lab_energies)
            detres_elem.append(kb_block)
        
        # KB observables
        kb_obs = self._create_kbobservables(sampling_info, ensemble_infos[0], sampling_files)
        detres_elem.append(kb_obs)
        
        # Convert to string
        xml_str = self._prettify_xml(root)
        
        if output_file:
            Path(output_file).write_text(xml_str)
        
        return xml_str
    
    def create_single_channel_xml(self,
                                 project_name: str,
                                 observables: List[ObservableInfo],
                                 sampling_info: SamplingInfo,
                                 reference_particle: str,
                                 particle_masses: Dict[str, float],
                                 decay_channels: List[ParticleInfo],
                                 box_quantizations: List[BoxQuantizationInfo],
                                 sampling_files: List[str],
                                 output_stub: str,
                                 default_energy_format: str = "reference_ratio",
                                 output_mode: str = "resampled",
                                 output_file: Optional[str] = None) -> str:
        """
        Create XML for single channel fits.
        
        Args:
            project_name: Name of the project
            observables: List of observable infos (ensemble info extracted from these)
            sampling_info: Sampling information
            reference_particle: Name of reference particle
            particle_masses: Dictionary mapping particle names to masses
            decay_channels: List of decay channel particles
            box_quantizations: List of BoxQuantizationInfo objects (contains lmax values)
            sampling_files: List of paths to sampling files
            output_stub: Output file stub
            default_energy_format: Default energy format ("reference_ratio" or "time_spacing_product")
            output_mode: Output mode ("resampled" or "full")
            output_file: Optional output file path
        
        Returns:
            XML string
        """
        # Extract ensemble info from first observable (assuming single ensemble for single channel)
        ensemble_info = observables[0].ensemble_info
        
        # Group observables by momentum to create KB blocks
        momentum_groups = self.group_observables_by_momentum(observables)
        
        # Create KB blocks from momentum groups and box quantizations
        kb_blocks = []
        for i, (momentum_key, obs_list) in enumerate(momentum_groups.items()):
            if i < len(box_quantizations):
                box_quant = box_quantizations[i]
                # Extract lab frame energies from observable names
                lab_energies = [obs.name for obs in obs_list]
                kb_blocks.append((ensemble_info, box_quant, lab_energies))
        
        # Create root element
        root = self._create_element("KBFit")
        
        # Initialize section
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        self._create_element("EchoXML", parent=init_elem)
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        init_elem.append(sampling_elem)
        
        # Bins info
        bins_elem = self._create_mcbinsinfo_element(ensemble_info)
        init_elem.append(bins_elem)
        
        # Task sequence
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoSingleChannel", task)
        
        # Output configuration
        self._create_element("OutputStub", output_stub, task)
        self._create_element("OutputMode", output_mode, task)
        self._create_element("SamplingsOutputStub", output_stub, task)
        
        # Decay channels
        channels_elem = self._create_decay_channels(decay_channels)
        task.append(channels_elem)
        
        # Default energy format
        self._create_element("DefaultEnergyFormat", default_energy_format, task)
        
        # MC ensemble parameters
        params_elem = self._create_mcensemble_parameters(
            ensemble_info, reference_particle, particle_masses
        )
        task.append(params_elem)
        
        # KB blocks
        for ensemble_info, box_quant, lab_energies in kb_blocks:
            kb_block = self._create_kbblock(ensemble_info, box_quant, lab_energies)
            task.append(kb_block)
        
        # KB observables
        kb_obs = self._create_kbobservables(sampling_info, ensemble_info, sampling_files)
        task.append(kb_obs)
        
        # Convert to string
        xml_str = self._prettify_xml(root)
        
        if output_file:
            Path(output_file).write_text(xml_str)
        
        return xml_str
    
    def create_print_xml(self,
                        project_name: str,
                        observables: List[ObservableInfo],
                        sampling_info: SamplingInfo,
                        reference_particle: str,
                        particle_masses: Dict[str, float],
                        energy_range: Tuple[float, float, float],
                        decay_channels: List[ParticleInfo],
                        k_elements: List[KElementInfo],
                        polynomial_powers: Dict[str, List[int]],
                        starting_values: List[FitParameterInfo],
                        sampling_files: List[str],
                        output_stub: str,
                        omega_mu: float = 0.5,
                        quantization_condition: str = "StildeCB",
                        default_energy_format: str = "reference_ratio",
                        output_mode: str = "full",
                        output_file: Optional[str] = None) -> str:
        """
        Create XML for print tasks.
        
        Args:
            project_name: Name of the project
            observables: List of observable infos (ensemble info extracted from these)
            sampling_info: Sampling information
            reference_particle: Name of reference particle
            particle_masses: Dictionary mapping particle names to masses
            energy_range: Tuple of (min_energy, max_energy, step_size) for print range
            decay_channels: List of decay channel particles
            k_elements: List of K-matrix element info
            polynomial_powers: Dictionary mapping K-indices to polynomial powers
            starting_values: List of fit parameter starting values
            sampling_files: List of paths to sampling files
            output_stub: Output file stub
            omega_mu: Omega mu parameter
            quantization_condition: Quantization condition type
            default_energy_format: Default energy format ("reference_ratio" or "time_spacing_product")
            output_mode: Output mode ("full" or "resampled")
            output_file: Optional output file path
        
        Returns:
            XML string
        """
        # Extract ensemble info from first observable
        ensemble_info = observables[0].ensemble_info
        
        # Create box quantization from first observable's momentum
        momentum_groups = self.group_observables_by_momentum(observables)
        first_momentum_key = list(momentum_groups.keys())[0]
        psq, irrep = first_momentum_key
        box_quantization = self.create_box_quantization_from_momentum(psq, irrep)
        
        # Create root element
        root = self._create_element("KBFit")
        
        # Initialize section
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        self._create_element("EchoXML", parent=init_elem)
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        init_elem.append(sampling_elem)
        
        # Bins info
        bins_elem = self._create_mcbinsinfo_element(ensemble_info)
        init_elem.append(bins_elem)
        
        # Task sequence
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoPrint", task)
        
        # Output configuration
        self._create_element("OutputStub", output_stub, task)
        self._create_element("OutputMode", output_mode, task)
        
        # Print section
        print_elem = self._create_element("Print", parent=task)
        self._create_element("OmegaMu", str(omega_mu), print_elem)
        self._create_element("QuantizationCondition", quantization_condition, print_elem)
        
        # K-tilde matrix
        ktilde_elem = self._create_ktilde_matrix(
            k_elements, polynomial_powers, decay_channels, starting_values
        )
        print_elem.append(ktilde_elem)
        
        # Default energy format
        self._create_element("DefaultEnergyFormat", default_energy_format, print_elem)
        
        # MC ensemble parameters
        params_elem = self._create_mcensemble_parameters(
            ensemble_info, reference_particle, particle_masses
        )
        print_elem.append(params_elem)
        
        # KB block for print with energy range
        kb_block = self._create_print_kbblock(ensemble_info, box_quantization, energy_range)
        print_elem.append(kb_block)
        
        # KB observables
        kb_obs = self._create_kbobservables(sampling_info, ensemble_info, sampling_files)
        print_elem.append(kb_obs)
        
        # Convert to string
        xml_str = self._prettify_xml(root)
        
        if output_file:
            Path(output_file).write_text(xml_str)
        
        return xml_str
    
    def _prettify_xml(self, element: ET.Element) -> str:
        """Convert ET.Element to formatted XML string."""
        rough_string = ET.tostring(element, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="    ")
        
        # Remove empty lines and fix formatting
        lines = [line for line in pretty.split('\n') if line.strip()]
        if lines[0].startswith('<?xml'):
            lines = lines[1:]  # Remove XML declaration
        
        return '\n'.join(lines)
    
    @staticmethod
    def extract_momentum_info_from_observable(observable_name: str) -> Optional[Tuple[int, str]]:
        """
        Extract momentum squared and irrep from observable name.
        
        Args:
            observable_name: Observable name like "isosinglet_S=0_A1g_1_PSQ=0_elab_2_ref"
        
        Returns:
            Tuple of (psq, irrep) or None if not found
        """
        psq_match = re.search(r'PSQ=(\d+)', observable_name)
        irrep_match = re.search(r'_(A\d+g?)_', observable_name)
        
        if psq_match and irrep_match:
            psq = int(psq_match.group(1))
            irrep = irrep_match.group(1)
            return psq, irrep
        
        # Try alternative format
        p_match = re.search(r'P=\(([^)]+)\)', observable_name)
        if p_match:
            p_coords = p_match.group(1).split(',')
            psq = sum(int(x)**2 for x in p_coords)
            irrep_match = re.search(r'_(A\d+)_', observable_name)
            if irrep_match:
                irrep = irrep_match.group(1)
                return psq, irrep
        
        return None
    
    @staticmethod
    def group_observables_by_momentum(observables: List[ObservableInfo]) -> Dict[Tuple[int, str], List[ObservableInfo]]:
        """
        Group observables by momentum squared and irrep.
        
        Args:
            observables: List of observable infos
        
        Returns:
            Dictionary mapping (psq, irrep) to list of observables
        """
        grouped = {}
        
        for obs in observables:
            momentum_info = KBfitXMLHelper.extract_momentum_info_from_observable(obs.name)
            if momentum_info:
                psq, irrep = momentum_info
                key = (psq, irrep)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(obs)
        
        return grouped
    
    @staticmethod
    def create_box_quantization_from_momentum(psq: int, irrep: str) -> BoxQuantizationInfo:
        """
        Create BoxQuantizationInfo from momentum squared and irrep.
        
        Args:
            psq: Momentum squared
            irrep: Irreducible representation
        
        Returns:
            BoxQuantizationInfo object
        """
        # Mapping of momentum squared to momentum ray (simplified)
        momentum_ray_map = {
            0: "ar",
            1: "oa", 
            2: "pd",
            3: "cd"
        }
        
        momentum_ray = momentum_ray_map.get(psq, "ar")
        
        return BoxQuantizationInfo(
            momentum_ray=momentum_ray,
            momentum_int_squared=psq,
            lg_irrep=irrep,
            lmax_values=0
        ) 