"""
KBfitXMLHelper module for generating XML configurations for KB fits.

This module provides functionality to generate XML configurations for:
- Determinant residual fits (detres)
- Spectrum fits 
- Print tasks

It takes observables, ensemble information, and other parameters to create
the appropriate XML structure for KB fitting tasks.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .sampling import EnsembleInfo, SamplingInfo, ObservableInfo, SigmondSampling


class TaskType(Enum):
    """Enumeration of KB fit task types."""
    FIT = "DoFit"
    PRINT = "DoPrint"
    SINGLE_CHANNEL = "DoSingleChannel"
    
class FitType(Enum):
    """Enumeration of fit types."""
    DETERMINANT_RESIDUAL = "DeterminantResidualFit"
    SPECTRUM = "SpectrumFit"


class MinimizerMethod(Enum):
    """Enumeration of minimizer methods."""
    NL2SNO = "NL2Sno"
    MINUIT2_MIGRAD = "Minuit2Migrad"
    MINUIT2_SIMPLEX = "Minuit2Simplex"


class QuantizationCondition(Enum):
    """Enumeration of quantization conditions."""
    KTILDE_INV_B = "KtildeinvB"
    KTILDE_B = "KtildeB"
    STILDE_CB = "StildeCB"
    STILDE_INV_CB = "StildeinvCB"


class EnergyFormat(Enum):
    """Enumeration of energy formats."""
    REFERENCE_RATIO = "reference_ratio"
    TIME_SPACING_PRODUCT = "time_spacing_product"


class OutputMode(Enum):
    """Enumeration of output modes."""
    FULL = "full"
    RESAMPLED = "resampled"


class Verbosity(Enum):
    """Enumeration of minimizer verbosity levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class BoxQuantizationInfo:
    """Information about box quantization for KBBlocks."""
    momentum_ray: str
    momentum_int_squared: int
    lg_irrep: str
    lmax_values: List[int] | int
    
    def __post_init__(self):
        self.lmax_values = str(self.lmax_values)
        
    def create_xml_content(self, box_quant_elem: ET.Element) -> None:
        """Create XML content for box quantization."""
        box_quant_elem = ET.SubElement(box_quant_elem, "BoxQuantization")
        ET.SubElement(box_quant_elem, "TotalMomentumRay").text = self.momentum_ray
        ET.SubElement(box_quant_elem, "TotalMomentumIntSquared").text = str(self.momentum_int_squared)
        ET.SubElement(box_quant_elem, "LGIrrep").text = self.lg_irrep
        ET.SubElement(box_quant_elem, "LmaxValues").text = self.lmax_values if isinstance(self.lmax_values, int) else " ".join(str(l) for l in self.lmax_values)

@dataclass
class DecayChannelInfo:
    """Information about particles for decay channels."""
    particle_1: str
    spin_1_times_two: int
    particle_2: str
    spin_2_times_two: int
    identical: bool = False
    
    def create_xml_content(self, channels_elem: ET.Element) -> None:
        """Create XML content for decay channel."""
        channel_elem = ET.SubElement(channels_elem, "DecayChannelInfo")
        ET.SubElement(channel_elem, "Particle1Name").text = self.particle_1
        ET.SubElement(channel_elem, "Spin1TimesTwo").text = str(self.spin_1_times_two)
        if self.identical:
            ET.SubElement(channel_elem, "Identical")
        else:
            ET.SubElement(channel_elem, "Particle2Name").text = self.particle_2
            ET.SubElement(channel_elem, "Spin2TimesTwo").text = str(self.spin_2_times_two)

@dataclass
class KElementInfo:
    """Information for K-matrix elements."""
    j_times_two: int
    k_index1: str
    k_index2: str
    
    def create_xml_content(self, elem: ET.Element) -> None:
        """Create XML content for K-matrix element."""
        k_info_elem = ET.SubElement(elem, "KElementInfo")
        ET.SubElement(k_info_elem, "JTimesTwo").text = str(self.j_times_two)
        ET.SubElement(k_info_elem, "KIndex").text = self.k_index1
        ET.SubElement(k_info_elem, "KIndex").text = self.k_index2

class FitForm(ABC):
    """Abstract base class for all fit form types."""
    
    @abstractmethod
    def create_element_info_xml_content(self, kmatrix_elem: ET.Element) -> None:
        """Create XML content for the <Element> element."""
        pass
    
    @abstractmethod
    def create_starting_values_xml_content(self, starting_values_elem: ET.Element) -> None:
        """Create XML content for the <StartingValues> element."""
        pass


@dataclass
class ExpressionFitForm(FitForm):
    """Expression fit form configuration."""
    expression: str
    params_with_starting_values: Dict[str, float] # parameter name -> starting value
    k_index: KElementInfo
    
    def create_element_info_xml_content(self, kmatrix_elem: ET.Element) -> None:
        """Create XML content for expression fit form."""
        # create the <Element>
        elem_elem = ET.SubElement(kmatrix_elem, "Element")
        # create the <KElementInfo>
        self.k_index.create_xml_content(elem_elem)
        # create the <FitForm>
        fit_form_elem = ET.SubElement(elem_elem, "FitForm")
        # create the <Expression>
        expr_elem = ET.SubElement(fit_form_elem, "Expression")
        ET.SubElement(expr_elem, "String").text = self.expression
        
    def create_starting_values_xml_content(self, starting_values_elem: ET.Element) -> None:
        """Create XML content for starting values."""
        for param_name, starting_value in self.params_with_starting_values.items():
            # create and populate the <KFitParamInfo>
            kfit_param_info_elem = ET.SubElement(starting_values_elem, "KFitParamInfo")
            string_expr_param_elem = ET.SubElement(kfit_param_info_elem, "StringExpressionParameter")
            ET.SubElement(string_expr_param_elem, "ParameterName").text = param_name
            self.k_index.create_xml_content(string_expr_param_elem)
            # create and populate the <StartingValue>
            ET.SubElement(kfit_param_info_elem, "StartingValue").text = str(starting_value)
            
@dataclass
class ParticleInfo:
    """Information for particles."""
    name: str
    mass: ObservableInfo | float = 1.0 # if float, then the mass is fixed
    psq: int = 0 # used for non-interacting pairs
    
    def create_particle_mass_xml_content(self, mc_ensemble_parameters_elem: ET.Element) -> None:
        """Create XML content for particle mass."""
        particle_mass_elem = ET.SubElement(mc_ensemble_parameters_elem, "ParticleMass")
        ET.SubElement(particle_mass_elem, "Name").text = self.name
        if isinstance(self.mass, float):
            ET.SubElement(particle_mass_elem, "FixedValue").text = str(self.mass)
        else:
            ET.SubElement(particle_mass_elem, "MCObs").text = str(self.mass)

@dataclass
class LabFrameEnergyShiftInfo:
    """Information for lab frame energy shifts (spectrum fits)."""
    mcobs: ObservableInfo
    non_interacting_pair: List[ParticleInfo]
    
    def create_xml_content(self, shift_elem: ET.Element) -> None:
        """Create XML content for lab frame energy shift."""
        shift_elem_new = ET.SubElement(shift_elem, "LabFrameEnergyShift")
        ET.SubElement(shift_elem_new, "MCObs").text = str(self.mcobs)
        ni_str = "".join(f"{p.name}({p.psq})" for p in self.non_interacting_pair)
        ET.SubElement(shift_elem_new, "NonInteractingPair").text = ni_str
        
@dataclass
class LabFrameEnergyInfo:
    """Information for lab frame energies (print tasks)."""
    mcobs: ObservableInfo
    
    def create_xml_content(self, energy_elem: ET.Element) -> None:
        """Create XML content for lab frame energy."""
        energy_elem_new = ET.SubElement(energy_elem, "LabFrameEnergy")
        ET.SubElement(energy_elem_new, "MCObs").text = str(self.mcobs)

@dataclass
class LabFrameEnergyRangeInfo:
    """Information for lab frame energy range (print tasks)."""
    min: float
    max: float
    inc: float
    
    def create_xml_content(self, energy_range_elem: ET.Element) -> None:
        """Create XML content for lab frame energy range."""
        ET.SubElement(energy_range_elem, "LabFrameEnergyMin").text = str(self.min)
        ET.SubElement(energy_range_elem, "LabFrameEnergyMax").text = str(self.max)
        ET.SubElement(energy_range_elem, "LabFrameEnergyInc").text = str(self.inc)

@dataclass
class MinimizerInfo:
    """Configuration for minimizer."""
    method: MinimizerMethod = MinimizerMethod.MINUIT2_MIGRAD
    parameter_rel_tol: float = 1e-6
    chisquare_rel_tol: float = 1e-4
    maximum_iterations: int = 1024
    verbosity: Verbosity = Verbosity.LOW
    
    def create_xml_content(self, minimizer_elem: ET.Element) -> None:
        """Create XML content for minimizer."""
        minimizer_elem = ET.SubElement(minimizer_elem, "MinimizerInfo")
        ET.SubElement(minimizer_elem, "Method").text = self.method.value
        ET.SubElement(minimizer_elem, "ParameterRelTol").text = str(self.parameter_rel_tol)
        ET.SubElement(minimizer_elem, "ChiSquareRelTol").text = str(self.chisquare_rel_tol)
        ET.SubElement(minimizer_elem, "MaximumIterations").text = str(self.maximum_iterations)


@dataclass
class RootFinderConfig:
    """Configuration for root finder (required for spectrum fits and optional for print)."""
    initial_step_percent: float = 0.01
    abs_x_tolerance: float = 1e-6
    abs_residual_tolerance: float = 1e-12
    min_step_percent: float = 1e-5
    max_step_percent: float = 5e-3
    step_scale_limit: float = 10.0
    plateau_mod2_threshold: float = 1e-8
    plateau_count_before_jump: int = 5
    
    def create_xml_content(self, spectrum_elem: ET.Element) -> None:
        """Create XML content for root finder."""
        root_finder_elem = ET.SubElement(spectrum_elem, "RootFinder")
        ET.SubElement(root_finder_elem, "AdaptiveBracket")
        ET.SubElement(root_finder_elem, "InitialStepPercent").text = str(self.initial_step_percent)
        ET.SubElement(root_finder_elem, "AbsXTolerance").text = str(self.abs_x_tolerance)
        ET.SubElement(root_finder_elem, "AbsResidualTolerance").text = str(self.abs_residual_tolerance)
        ET.SubElement(root_finder_elem, "MinStepPercent").text = str(self.min_step_percent)
        ET.SubElement(root_finder_elem, "MaxStepPercent").text = str(self.max_step_percent)


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
            if sampling_info.num_resamplings and sampling_info.num_resamplings > 0:
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
                                    particle_masses: List[ParticleInfo]) -> ET.Element:
        """Create MCEnsembleParameters element."""
        params_elem = self._create_element("MCEnsembleParameters")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, params_elem)
        
        # Reference mass
        ref_mass_elem = self._create_element("ReferenceMassTimeSpacingProduct", parent=params_elem)
        self._create_element("MCObs", f"{reference_particle}(0)_elab 0", ref_mass_elem)
        
        # Particle masses
        for particle in particle_masses:
            particle.create_particle_mass_xml_content(params_elem)
        
        return params_elem
    
    def _create_kbblock_xml(self, ensemble_info: EnsembleInfo, box_quantization: BoxQuantizationInfo) -> ET.Element:
        """Create common KBBlock elements for all tasks."""
        kb_block = self._create_element("KBBlock")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, kb_block)
        box_quantization.create_xml_content(kb_block)
        return kb_block
    
    def _create_kbblock_detres(self, ensemble_info: EnsembleInfo, 
                              box_quantization: BoxQuantizationInfo,
                              lab_frame_energies: List[LabFrameEnergyInfo]) -> ET.Element:
        """Create KBBlock element for determinant residual fits."""
        kb_block = self._create_kbblock_xml(ensemble_info, box_quantization)
        
        # Lab frame energies
        for lab_frame_energy in lab_frame_energies:
            lab_frame_energy.create_xml_content(kb_block)
        
        return kb_block
    
    def _create_kbblock_spectrum(self, ensemble_info: EnsembleInfo,
                                box_quantization: BoxQuantizationInfo,
                                energy_shifts: List[LabFrameEnergyShiftInfo],
                                cm_energy_range: Tuple[float, float]) -> ET.Element:
        """Create KBBlock element for spectrum fits."""
        kb_block = self._create_kbblock_xml(ensemble_info, box_quantization)
        
        # Lab frame energy shifts
        for shift in energy_shifts:
            shift.create_xml_content(kb_block)
        
        # CM frame energy range for the root finder
        self._create_element("CMFrameEnergyMin", str(cm_energy_range[0]), kb_block)
        self._create_element("CMFrameEnergyMax", str(cm_energy_range[1]), kb_block)
        
        return kb_block
    
    def _create_kbblock_print(self, ensemble_info: EnsembleInfo,
                             box_quantization: BoxQuantizationInfo,
                             energy_range: LabFrameEnergyRangeInfo) -> ET.Element:
        """Create KBBlock element for print tasks."""
        kb_block = self._create_kbblock_xml(ensemble_info, box_quantization)
        
        # Lab frame energy range
        energy_range.create_xml_content(kb_block)
        
        return kb_block
    
    def _create_decay_channels(self, channels: List[DecayChannelInfo]) -> ET.Element:
        """Create DecayChannels element."""
        channels_elem = self._create_element("DecayChannels")

        for channel in channels:
            channel.create_xml_content(channels_elem)

        return channels_elem
    
    def _create_kbobservables(self, sampling_info: SamplingInfo,
                             sampling_files: List[str],
                             verbose: bool = True) -> ET.Element:
        """Create KBObservables element."""
        kb_obs = self._create_element("KBObservables")
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        kb_obs.append(sampling_elem)
        
        # Verbose flag
        if verbose:
            self._create_element("Verbose", parent=kb_obs)
        
        # Sampling data files
        sampling_data_elem = self._create_element("SamplingData", parent=kb_obs)
        for file_path in sampling_files:
            self._create_element("FileName", file_path, sampling_data_elem)
        
        return kb_obs
    
    def _create_ktilde_matrix_or_inverse(self, fit_forms: List[FitForm],
                                         decay_channels: List[DecayChannelInfo],
                                         make_inverse: bool) -> ET.Element:
        """Create KtildeMatrix/KtildeMatrixInverse element"""
        ktilde_elem = self._create_element("KtildeMatrixInverse" if make_inverse else "KtildeMatrix")
        
        # K-matrix elements
        for fit_form in fit_forms:
            fit_form.create_element_info_xml_content(ktilde_elem)
        
        # Decay channels
        channels_elem = self._create_decay_channels(decay_channels)
        ktilde_elem.append(channels_elem)
        
        # Starting values
        start_vals_elem = self._create_element("StartingValues", parent=ktilde_elem)
        for fit_form in fit_forms:
            fit_form.create_starting_values_xml_content(start_vals_elem)
        
        return ktilde_elem
    
    def _create_ensemble_particle_masses(self, particle_masses: List[ParticleInfo] | List[Tuple[EnsembleInfo, List[ParticleInfo]]],
                                         unique_ensemble_infos: List[EnsembleInfo]) -> List[Tuple[EnsembleInfo, List[ParticleInfo]]]:
        """Create ensemble particle masses from a list of particle masses."""
         # setup particle masses to be of the form [(EnsembleInfo, List[ParticleInfo])]
        if isinstance(particle_masses, ParticleInfo):
            # Single particle for all ensembles
            ensemble_particle_infos = [(ens_info, [particle_masses]) for ens_info in unique_ensemble_infos]
        elif isinstance(particle_masses, list):
            if len(particle_masses) == 0:
                raise ValueError("`particle_masses` cannot be empty.")
            
            # Check what type of list we have
            first_item = particle_masses[0]
            if isinstance(first_item, ParticleInfo):
                # List of ParticleInfo - same for all ensembles
                ensemble_particle_infos = [(ens_info, particle_masses) for ens_info in unique_ensemble_infos]
            elif isinstance(first_item, tuple) and len(first_item) == 2:
                # List of tuples - check if (EnsembleInfo, ParticleInfo) or (EnsembleInfo, List[ParticleInfo])
                ens_info, particle_info = first_item
                if isinstance(particle_info, ParticleInfo):
                    # [(EnsembleInfo, ParticleInfo)]
                    ensemble_particle_infos = [(ens_info, [particle_info]) for ens_info, particle_info in particle_masses]
                elif isinstance(particle_info, list):
                    # [(EnsembleInfo, List[ParticleInfo])]
                    ensemble_particle_infos = particle_masses
                else:
                    raise ValueError("`particle_masses` tuple format not recognized.")
            else:
                raise ValueError("`particle_masses` has been provided in an unexpected format.")
        else:
            raise ValueError("`particle_masses` must be ParticleInfo, List[ParticleInfo], or List[Tuple[EnsembleInfo, ParticleInfo]].")
        
        return ensemble_particle_infos
    
    def create_detres_xml(
        self,
        xml_output_file: Optional[str],
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, List[LabFrameEnergyInfo]]],
        reference_particle: str,
        particle_masses: ParticleInfo | List[ParticleInfo] | List[Tuple[EnsembleInfo, ParticleInfo]] | List[Tuple[EnsembleInfo, List[ParticleInfo]]],
        sampling_files: List[str],
        fit_forms: List[FitForm],
        decay_channels: List[DecayChannelInfo],
        omega_mu: float = 30.0,
        quantization_condition: QuantizationCondition = QuantizationCondition.KTILDE_INV_B,
        use_inverse_k_matrix: bool = True,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        minimizer_info: MinimizerInfo = MinimizerInfo(),
        output_directory: str = ".",
        echo_xml: bool = True,
        verbose: bool = True,
        output_samplings_file: str = "fit_param_samplings.hdf5[/samplings]"
    ) -> str:
        """Create determinant residual fit XML directly from structured ensemble data.

        Parameters
        ----------
        xml_output_file
            Optional path to write the generated XML file.
        project_name
            Human-readable project label.
        ensemble_data
            A list where each element is a tuple ``(ensemble_info, sampling_info, lab_energies)``.
            ``lab_energies`` is a list of ``LabFrameEnergyInfo`` objects.
        reference_particle
            The name of the reference particle.
        particle_masses
            A list of ``ParticleInfo`` objects or a list of tuples, where each tuple contains
            an ``EnsembleInfo`` and a list of ``ParticleInfo`` objects. If a single list is provided,
            then all ensembles will use the same particle masses.
        sampling_files
            A list of sampling data files.
        fit_forms
            A list of ``FitForm`` objects.
        decay_channels
            A list of ``DecayChannelInfo`` objects.
        omega_mu
            The value of omega_mu for the fit.
        quantization_condition
            The quantization condition for the fit.
        use_inverse_k_matrix
            Whether to use the inverse K matrix for the fit.
        default_energy_format
            The default energy format for the fit.
        minimizer_info
            The minimizer info for the fit.
        output_directory
            The directory to write the output file to.
        echo_xml
            Whether to echo the XML to the console.
        verbose
            Whether to print verbose output.
        output_samplings_file
            The name of the output samplings file.

        Returns
        -------
        The KBfit XML for the determinant residual fit.
        """
        
        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")

        ref_sampling_info = ensemble_data[0][1]
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}
        lab_energies_dict: Dict[Tuple[str, int, str], List[LabFrameEnergyInfo]] = {}

        for ens_info, samp_info, lab_energies in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info

            for lab_energy_info in lab_energies:
                obs_info = lab_energy_info.mcobs
                
                momentum_info = self.extract_momentum_info_from_observable(obs_info.name)
                if not momentum_info:
                    continue
                
                psq_extracted, irrep = momentum_info
                
                key = (ens_info.ensemble_name, psq_extracted, irrep)
                if key not in lab_energies_dict:
                    lab_energies_dict[key] = []
                lab_energies_dict[key].append(lab_energy_info)

        unique_ensemble_infos = list(all_ensemble_infos.values())
        
        # setup particle masses to be of the form [(EnsembleInfo, List[ParticleInfo])]
        ensemble_particle_infos = self._create_ensemble_particle_masses(particle_masses, unique_ensemble_infos)

        # 2. Build the XML structure
        task_type = TaskType.FIT
        fit_type = FitType.DETERMINANT_RESIDUAL
        
        root = self._create_element("KBFit")
        
        self.create_initialize_element(root, project_name,
                                        output_directory, f"{project_name}.log",
                                        echo_xml, ref_sampling_info)
        
        task_elem = self.create_task_and_header_elements(root, task_type, fit_type)
        
        self.create_fit_task_elements(task_elem, minimizer_info, output_samplings_file, add_ecm_qcm_stub=True)
        
        detres_elem = self._create_element(fit_type.value, parent=task_elem)
        
        self.create_common_task_elements(detres_elem, fit_forms, decay_channels, omega_mu,
                                         quantization_condition, verbose, use_inverse_k_matrix,  # make_inverse=True for detres
                                         default_energy_format, reference_particle, ensemble_particle_infos)
        
        # Create KBBlocks data for the helper method
        block_data = []
        for (ensemble_name, psq, irrep), lab_energies in lab_energies_dict.items():
            ensemble_info = all_ensemble_infos[ensemble_name]
            box_quant = self.create_box_quantization_from_momentum(psq, irrep)
            block_data.append((ensemble_info, box_quant, lab_energies))
        
        self.create_kbblocks_and_observables(detres_elem, ref_sampling_info, sampling_files, verbose, block_data)
        
        xml_str = self._prettify_xml(root)
        
        if xml_output_file:
            Path(xml_output_file).write_text(xml_str)
        
        return xml_str
    
    def create_print_xml(
        self,
        xml_output_file: Optional[str],
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, List[Tuple[BoxQuantizationInfo, LabFrameEnergyRangeInfo]]]],
        reference_particle: str,
        particle_masses: ParticleInfo | List[ParticleInfo] | List[Tuple[EnsembleInfo, ParticleInfo]] | List[Tuple[EnsembleInfo, List[ParticleInfo]]],
        sampling_files: List[str],
        fit_forms: List[FitForm],
        decay_channels: List[DecayChannelInfo],
        output_stub: str,
        omega_mu: float = 0.5,
        quantization_condition: QuantizationCondition = QuantizationCondition.STILDE_CB,
        use_inverse_k_matrix: bool = True,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        output_mode: OutputMode = OutputMode.FULL,
        root_finder_config: Optional[RootFinderConfig] = None,
        output_directory: str = ".",
        echo_xml: bool = True,
        verbose: bool = True,
    ) -> str:
        """Create XML for print tasks from structured ensemble data.

        This method uses the simplified data-driven API to configure a print
        job. It discovers the ensembles and momentum shells from your data,
        and you only need to provide the energy ranges for the print grid.

        Parameters
        ----------
        xml_output_file
            Optional path to write the generated XML file.
        project_name
            Human-readable project label.
        ensemble_data
            A list where each element is a tuple ``(ensemble_info, sampling_info, block_configs)``.
            ``block_configs`` is a list of tuples, where each tuple contains the
            ``BoxQuantizationInfo`` and ``LabFrameEnergyRangeInfo`` for a KBBlock.
        reference_particle
            The name of the reference particle.
        particle_masses
            A list of ``ParticleInfo`` objects or a list of tuples, where each tuple contains
            an ``EnsembleInfo`` and a list of ``ParticleInfo`` objects. If a single list is provided,
            then all ensembles will use the same particle masses.
        sampling_files
            A list of sampling data files.
        fit_forms
            A list of ``FitForm`` objects.
        decay_channels
            A list of ``DecayChannelInfo`` objects.
        output_stub
            The base name for the output data files.
        omega_mu, quantization_condition, default_energy_format, output_mode,
        root_finder_config
            Optional settings for the print task.
        output_directory, echo_xml, verbose
            Optional settings for output and logging.

        Returns
        -------
        The KBfit XML document as a string.
        """
        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")

        ref_sampling_info = ensemble_data[0][1]
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}
        # block_configs_dict: key=ensemble_name, val=List[(BoxQuantizationInfo, LabFrameEnergyRangeInfo)]
        block_configs_dict: Dict[str, List[Tuple[BoxQuantizationInfo, LabFrameEnergyRangeInfo]]] = {}

        for ens_info, samp_info, block_configs in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info
                block_configs_dict[ens_info.ensemble_name] = []
            
            block_configs_dict[ens_info.ensemble_name].extend(block_configs)
        
        unique_ensemble_infos = list(all_ensemble_infos.values())

        # setup particle masses to be of the form [(EnsembleInfo, List[ParticleInfo])]
        ensemble_particle_infos = self._create_ensemble_particle_masses(particle_masses, unique_ensemble_infos)
        
        # 2. Build the XML structure
        root = self._create_element("KBFit")
        
        self.create_initialize_element(root, project_name,
                                        output_directory, f"{project_name}.log",
                                        echo_xml, ref_sampling_info)
        
        task_elem = self.create_task_and_header_elements(root, TaskType.PRINT)
        
        self.create_print_task_elements(task_elem, output_stub, output_mode, root_finder_config)
        
        self.create_common_task_elements(task_elem, fit_forms, decay_channels, omega_mu,
                                         quantization_condition, verbose, use_inverse_k_matrix,  # make_inverse=False for print
                                         default_energy_format, reference_particle, ensemble_particle_infos)
        
        # Create KBBlocks data for the helper method
        block_data = []
        for ens_name, block_configs in block_configs_dict.items():
            ensemble_info = all_ensemble_infos[ens_name]
            for box_quant, energy_range in block_configs:
                block_data.append((ensemble_info, box_quant, energy_range))
        
        self.create_kbblocks_and_observables(task_elem, ref_sampling_info, sampling_files, verbose, block_data)
        
        xml_str = self._prettify_xml(root)
        
        if xml_output_file:
            Path(xml_output_file).write_text(xml_str)
        
        return xml_str
    
    def create_spectrum_xml(
        self,
        xml_output_file: str,
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, List[LabFrameEnergyShiftInfo]]],
        reference_particle: str,
        ensemble_particle_infos: ParticleInfo | List[ParticleInfo] | List[Tuple[EnsembleInfo, ParticleInfo]] | List[Tuple[EnsembleInfo, List[ParticleInfo]]],
        sampling_files: List[str],
        fit_forms: List[FitForm],   
        decay_channels: List[DecayChannelInfo],
        omega_mu: float = 0.8,
        use_inverse_k_matrix: bool = True,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        minimizer_info: MinimizerInfo = MinimizerInfo(),
        root_finder_config: RootFinderConfig = RootFinderConfig(),
        cm_energy_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        output_directory: str = ".",
        echo_xml: bool = True,
        verbose: bool = True,
        output_samplings_file: str = "fit_param_samplings.hdf5[/samplings]"
    ) -> str:
        """Create spectrum XML directly from structured ensemble data.

        This high-level convenience wrapper dramatically reduces the amount of
        user boiler-plate required by consuming a rich data structure that is
        typically produced by a data loading helper function.

        It automatically discovers ensembles, momenta, and energy shift levels
        from the data, and constructs all necessary ``KBBlock`` elements.

        Parameters
        ----------
        xml_output_file
            The name of the output XML file.
        project_name
            Human-readable project label.
        ensemble_data
            A list where each element is a tuple ``(ensemble_info, sampling_info, energy_shifts)``.
            ``energy_shifts`` is a list of ``LabFrameEnergyShiftInfo`` objects.
        reference_particle
            The name of the reference particle.
        ensemble_particle_infos
            Either a single ``ParticleInfo`` object,  where all ensembles use the same single particle mass,
            or a list of ``ParticleInfo`` objects, where each ensemble uses the set of particle masses provided.
            If a list of tuples of the form ``(EnsembleInfo, ParticleInfo)`` is provided, then each ensemble uses the
            associated particle mass. Finally, if a list of tuples of the form ``(EnsembleInfo, List[ParticleInfo])`` is provided,
            then each ensemble uses the set of particle masses provided.
        sampling_files
            A list of sampling data files.
        fit_forms
            A list of ``FitForm`` objects.
        decay_channels
            A list of ``DecayChannelInfo`` objects.
        omega_mu
            The value of omega_mu for the fit.
        use_inverse_k_matrix
            Whether to use the inverse K-matrix for the fit.
        default_energy_format
            The default energy format for the fit.
        minimizer_info
            The minimizer info for the fit.
        root_finder_config
            The root finder config for the fit.
        cm_energy_ranges
            The energy ranges for the root finder.
        output_directory
            The directory to write the output file to.
        echo_xml
            Whether to echo the XML to the console.
        verbose
            Whether to print verbose output.
        output_samplings_file
            The name of the output samplings file.

        Returns
        -------
        The KBfit XML document as a string.
        """
        # qc must be set to STILDE_CB for spectrum fits
        quantization_condition: QuantizationCondition = QuantizationCondition.STILDE_CB
        
        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")
        
        ref_sampling_info = ensemble_data[0][1]
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}
        # key is (ensemble_name, psq, irrep)
        energy_shifts_dict: Dict[Tuple[str, int, str], List[LabFrameEnergyShiftInfo]] = {}

        for ens_info, samp_info, energy_shifts in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info

            for shift_info in energy_shifts:
                obs_info = shift_info.mcobs
                obs_name = str(obs_info)
                
                momentum_info = self.extract_momentum_info_from_observable(obs_name)
                if not momentum_info:
                    continue 
                
                psq, irrep = momentum_info
                
                key = (ens_info.ensemble_name, psq, irrep)
                if key not in energy_shifts_dict:
                    energy_shifts_dict[key] = []
                energy_shifts_dict[key].append(shift_info)

        unique_ensemble_infos = list(all_ensemble_infos.values())
        
        # setup particle masses to be of the form [(EnsembleInfo, List[ParticleInfo])]
        ensemble_particle_infos = self._create_ensemble_particle_masses(ensemble_particle_infos, unique_ensemble_infos)
        
        # 2. Build the XML structure
        task_type = TaskType.FIT
        fit_type = FitType.SPECTRUM
        
        root = self._create_element("KBFit")
        
        self.create_initialize_element(root, project_name,
                                        output_directory, f"{project_name}.log",
                                        echo_xml, ref_sampling_info)
        
        task_elem = self.create_task_and_header_elements(root, task_type, fit_type)
        
        self.create_fit_task_elements(task_elem, minimizer_info, output_samplings_file, add_ecm_qcm_stub=False)

        # enter into spectrum fit task
        spectrum_elem = self._create_element(fit_type.value, parent=task_elem)

        root_finder_config.create_xml_content(spectrum_elem)
  
        self.create_common_task_elements(spectrum_elem, fit_forms, decay_channels, omega_mu,
                                         quantization_condition, verbose, use_inverse_k_matrix,
                                         default_energy_format, reference_particle, ensemble_particle_infos)
        
        # Create KBBlocks data for the helper method
        block_data = []
        for (ensemble_name, psq, irrep), shifts in energy_shifts_dict.items():
            ensemble_info = all_ensemble_infos[ensemble_name]
            box_quant = self.create_box_quantization_from_momentum(psq, irrep)
            
            try:
                cm_range = cm_energy_ranges.get(psq, (2.50, 2.90))
            except:
                cm_range = (2.50, 2.90)

            block_data.append((ensemble_info, box_quant, (shifts, cm_range)))
        
        self.create_kbblocks_and_observables(spectrum_elem, ref_sampling_info, sampling_files, verbose, block_data)
        
        xml_str = self._prettify_xml(root)
        
        if xml_output_file:
            Path(xml_output_file).write_text(xml_str)
        
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
    

    def create_initialize_element(self, root: ET.Element, project_name: str, output_directory: str, log_file: str,
                                  echo_xml: bool, common_sampling_info: SamplingInfo) -> None:
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        
        if output_directory:
            self._create_element("OutputDirectory", output_directory, init_elem)

        self._create_element("LogFile", log_file, init_elem)
        
        if echo_xml:
            self._create_element("EchoXML", parent=init_elem)
        
        sampling_elem = self._create_mcsamplinginfo_element(common_sampling_info)
        init_elem.append(sampling_elem)
        
    
    def create_task_and_header_elements(self, root: ET.Element, task_type: TaskType,
                                        fit_type: Optional[FitType] = None) -> ET.Element:
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", task_type.value, task)
        
        if task_type == TaskType.FIT:
            self._create_element("Type", fit_type.value, task)
        
        return task
    
    def create_common_task_elements(self, task_elem: ET.Element,
                                   fit_forms: List[FitForm],
                                   decay_channels: List[DecayChannelInfo],
                                   omega_mu: float,
                                   quantization_condition: QuantizationCondition,
                                   verbose: bool, make_inverse: bool,
                                   default_energy_format: EnergyFormat,
                                   reference_particle_str: str,
                                   ensemble_particle_infos: List[Tuple[EnsembleInfo, List[ParticleInfo]]]):
        self._create_element("OmegaMu", str(omega_mu), task_elem)
        self._create_element("QuantizationCondition", quantization_condition.value, task_elem)
        
        if verbose:
            self._create_element("Verbose", parent=task_elem)
        
        ktilde_elem = self._create_ktilde_matrix_or_inverse(fit_forms, decay_channels, make_inverse)
        task_elem.append(ktilde_elem)
        
        self._create_element("DefaultEnergyFormat", default_energy_format.value, task_elem)
        
        for ensemble_info, particle_infos in ensemble_particle_infos:
            ensemble_params_elem = self._create_mcensemble_parameters(
                ensemble_info, reference_particle_str, particle_infos
            )
            task_elem.append(ensemble_params_elem)

    def create_fit_task_elements(self, task_elem: ET.Element,
                                minimizer_info: MinimizerInfo,
                                output_samplings_file: str,
                                add_ecm_qcm_stub: bool = False) -> None:
        """Create fit-specific task elements (minimizer info, output files)."""
        minimizer_info.create_xml_content(task_elem)
        self._create_element("OutSamplingsFile", output_samplings_file, task_elem)
        
        if add_ecm_qcm_stub:
            self._create_element("EcmQcmBoxSamplingsStub", "ecm_qcm_box_samplings", task_elem)
    
    def create_print_task_elements(self, task_elem: ET.Element,
                                  output_stub: str,
                                  output_mode: OutputMode,
                                  root_finder_config: Optional[RootFinderConfig] = None) -> None:
        """Create print-specific task elements (output configuration)."""
        self._create_element("OutputStub", output_stub, task_elem)
        self._create_element("OutputMode", output_mode.value, task_elem)
        
        if root_finder_config:
            root_finder_config.create_xml_content(task_elem)
    
    def create_kbblocks_and_observables(self, parent_elem: ET.Element,
                                       ref_sampling_info: SamplingInfo,
                                       sampling_files: List[str],
                                       verbose: bool,
                                       block_data: List[Tuple[EnsembleInfo, BoxQuantizationInfo, Any]]) -> None:
        """Create KBBlocks and KBObservables elements.
        
        Parameters
        ----------
        parent_elem
            The parent element to append blocks to
        ref_sampling_info
            Sampling info for observables
        sampling_files
            List of sampling data files
        verbose
            Whether to include verbose flag
        block_data
            List of tuples containing (ensemble_info, box_quantization, block_specific_data)
            where block_specific_data can be:
            - (energy_shifts, cm_range) tuple for spectrum
            - List[LabFrameEnergyInfo] for detres
            - LabFrameEnergyRangeInfo for print
        """
        # Create all KBBlocks
        for ensemble_info, box_quant, block_specific_data in block_data:
            if isinstance(block_specific_data, tuple) and len(block_specific_data) == 2:
                # Spectrum: (energy_shifts, cm_range)
                energy_shifts, cm_range = block_specific_data
                kb_block = self._create_kbblock_spectrum(ensemble_info, box_quant, energy_shifts, cm_range)
            elif isinstance(block_specific_data, list):
                # Detres: list of LabFrameEnergyInfo (may be empty)
                kb_block = self._create_kbblock_detres(ensemble_info, box_quant, block_specific_data)
            elif isinstance(block_specific_data, LabFrameEnergyRangeInfo):
                # Print: LabFrameEnergyRangeInfo
                kb_block = self._create_kbblock_print(ensemble_info, box_quant, block_specific_data)
            else:
                raise ValueError(f"Unsupported block_specific_data type: {type(block_specific_data)}. "
                               f"Expected tuple (for spectrum), list (for detres), or LabFrameEnergyRangeInfo (for print).")
            
            parent_elem.append(kb_block)
        
        # Create KBObservables
        kb_obs = self._create_kbobservables(ref_sampling_info, sampling_files, verbose)
        parent_elem.append(kb_obs)

    @staticmethod
    def extract_momentum_info_from_observable(observable_name: str) -> Optional[Tuple[int, str]]:
        """
        Extract momentum squared and irrep from observable name.
        
        Args:
            observable_name: Observable name like "isosinglet_S=0_A1g_1_PSQ=0_elab_2_ref 0"
        
        Returns:
            Tuple of (psq, irrep) or None if not found
        """
        psq_match_type_1 = re.search(r'PSQ=(\d+)', observable_name)
        psq_match_type_2 = re.search(r'P=\(([^)]+)\)', observable_name)
        irrep_match = re.search(r'S=\d+_(A\d+g?)', observable_name)
        
        if irrep_match:
            irrep = irrep_match.group(1)
        else:
            warnings.warn(f"Could not extract irrep from observable name: {observable_name}")
            return None
        
        if psq_match_type_1:
            psq = int(psq_match_type_1.group(1))
            return psq, irrep
        elif psq_match_type_2:
            p_coords = psq_match_type_2.group(1).split(',')
            psq = sum(int(x)**2 for x in p_coords)
            irrep_match = re.search(r'S=\d+_(A\d+)', observable_name)
            if irrep_match:
                irrep = irrep_match.group(1)
                return psq, irrep
        else:
            warnings.warn(f"Could not extract momentum squared from observable name: {observable_name}")
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
    def create_box_quantization_from_momentum(psq: int, irrep: str, lmax: int = 0) -> BoxQuantizationInfo:
        """
        Create BoxQuantizationInfo from momentum squared and irrep.
        
        Args:
            psq: Momentum squared
            irrep: Irreducible representation
            lmax: Maximum L value
        
        Returns:
            BoxQuantizationInfo object
        """
        # Mapping of momentum squared to momentum ray
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
            lmax_values=lmax
        )
    
    @staticmethod
    def get_default_cm_energy_ranges() -> Dict[int, Tuple[float, float]]:
        """
        Get default CM energy ranges for different momentum squared values.
        
        Returns:
            Dictionary mapping psq to (min_energy, max_energy)
        """
        return {
            0: (2.05, 3.0),
            1: (2.05, 3.0),
            2: (2.05, 3.0),
            3: (2.05, 3.0)
        }
