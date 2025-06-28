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
from dataclasses import dataclass
from enum import Enum

from .sampling import EnsembleInfo, SamplingInfo, ObservableInfo, SigmondSampling


class TaskType(Enum):
    """Enumeration of KB fit task types."""
    DETERMINANT_RESIDUAL = "DeterminantResidualFit"
    SPECTRUM = "SpectrumFit"
    PRINT = "DoPrint"


class MinimizerMethod(Enum):
    """Enumeration of minimizer methods."""
    NL2SNO = "NL2Sno"
    MINUIT2_MIGRAD = "Minuit2Migrad"


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
    lmax_values: Union[int, str] = 0
    
    def __post_init__(self):
        self.lmax_values = str(self.lmax_values)


@dataclass
class DecayChannelInfo:
    """Information about particles for decay channels."""
    particle_1: str
    spin_1_times_two: int
    particle_2: str
    spin_2_times_two: int
    identical: bool = False
    


@dataclass
class KElementInfo:
    """Information for K-matrix elements."""
    j_times_two: int
    k_index1: str
    k_index2: str


class FitFormType(Enum):
    """Enumeration of fit form types."""
    EXPRESSION = "Expression"
    POLYNOMIAL = "Polynomial"
    SUM_OF_POLES = "SumOfPoles"
    SUM_OF_POLES_PLUS_POLYNOMIAL = "SumOfPolesPlusPolynomial"


@dataclass
class ExpressionFitForm:
    """Expression fit form configuration."""
    expression: str


@dataclass
class PolynomialFitForm:
    """Polynomial fit form configuration."""
    degree: Optional[int] = None
    powers: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.degree is None and self.powers is None:
            raise ValueError("Either degree or powers must be specified for polynomial fit form")
        if self.degree is not None and self.powers is not None:
            raise ValueError("Cannot specify both degree and powers for polynomial fit form")


@dataclass
class SumOfPolesFitForm:
    """Sum of poles fit form configuration."""
    number_of_poles: Optional[int] = None
    pole_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.number_of_poles is None and self.pole_indices is None:
            raise ValueError("Either number_of_poles or pole_indices must be specified")
        if self.number_of_poles is not None and self.pole_indices is not None:
            raise ValueError("Cannot specify both number_of_poles and pole_indices")


@dataclass
class SumOfPolesPlusPolynomialFitForm:
    """Sum of poles plus polynomial fit form configuration."""
    sum_of_poles: SumOfPolesFitForm
    polynomial: PolynomialFitForm


@dataclass
class FitParameterInfo:
    """Information for fit parameters."""
    parameter_name: str
    starting_value: float
    k_element_info: Optional[KElementInfo] = None
    fit_form_type: FitFormType = FitFormType.EXPRESSION


@dataclass
class LabFrameEnergyShiftInfo:
    """Information for lab frame energy shifts (spectrum fits)."""
    mcobs: str
    non_interacting_pair: str


@dataclass
class MinimizerInfo:
    """Configuration for minimizer."""
    method: MinimizerMethod = MinimizerMethod.MINUIT2_MIGRAD
    parameter_rel_tol: float = 1e-6
    chisquare_rel_tol: float = 1e-4
    maximum_iterations: int = 1024
    verbosity: Verbosity = Verbosity.LOW


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
                                    particle_masses: Dict[str, float]) -> ET.Element:
        """Create MCEnsembleParameters element."""
        params_elem = self._create_element("MCEnsembleParameters")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, params_elem)
        
        # Reference mass
        ref_mass_elem = self._create_element("ReferenceMassTimeSpacingProduct", parent=params_elem)
        self._create_element("MCObs", f"{reference_particle}(0)_elab 0", ref_mass_elem)
        
        # Particle masses
        for particle_name, mass_value in particle_masses.items():
            mass_elem = self._create_element("ParticleMass", parent=params_elem)
            self._create_element("Name", particle_name, mass_elem)
            self._create_element("FixedValue", str(mass_value), mass_elem)
        
        return params_elem
    
    def _create_minimizer_info(self, minimizer_info: MinimizerInfo) -> ET.Element:
        """Create MinimizerInfo element."""
        minimizer_elem = self._create_element("MinimizerInfo")
        self._create_element("Method", minimizer_info.method.value, minimizer_elem)
        self._create_element("ParameterRelTol", str(minimizer_info.parameter_rel_tol), minimizer_elem)
        self._create_element("ChiSquareRelTol", str(minimizer_info.chisquare_rel_tol), minimizer_elem)
        self._create_element("MaximumIterations", str(minimizer_info.maximum_iterations), minimizer_elem)
        self._create_element("Verbosity", minimizer_info.verbosity.value, minimizer_elem)
        return minimizer_elem
    
    def _create_root_finder_config(self, config: RootFinderConfig) -> ET.Element:
        """Create RootFinder element."""
        root_elem = self._create_element("RootFinder")
        self._create_element("AdaptiveBracket", parent=root_elem)
        self._create_element("InitialStepPercent", str(config.initial_step_percent), root_elem)
        self._create_element("AbsXTolerance", str(config.abs_x_tolerance), root_elem)
        self._create_element("AbsResidualTolerance", str(config.abs_residual_tolerance), root_elem)
        self._create_element("MinStepPercent", str(config.min_step_percent), root_elem)
        self._create_element("MaxStepPercent", str(config.max_step_percent), root_elem)
        self._create_element("StepScaleLimit", str(config.step_scale_limit), root_elem)
        self._create_element("PlateauMod2Threshold", str(config.plateau_mod2_threshold), root_elem)
        self._create_element("PlateauCountBeforeJump", str(config.plateau_count_before_jump), root_elem)
        return root_elem
    
    def _create_fit_form_content(self, fit_form: Union[ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm], 
                                parent: ET.Element) -> None:
        """Create fit form content based on the fit form type."""
        if isinstance(fit_form, ExpressionFitForm):
            expr_elem = self._create_element("Expression", parent=parent)
            self._create_element("String", fit_form.expression, expr_elem)
            
        elif isinstance(fit_form, PolynomialFitForm):
            poly_elem = self._create_element("Polynomial", parent=parent)
            if fit_form.degree is not None:
                self._create_element("Degree", str(fit_form.degree), poly_elem)
            elif fit_form.powers is not None:
                powers_str = " ".join(str(p) for p in fit_form.powers)
                self._create_element("Powers", powers_str, poly_elem)
                
        elif isinstance(fit_form, SumOfPolesFitForm):
            poles_elem = self._create_element("SumOfPoles", parent=parent)
            if fit_form.number_of_poles is not None:
                self._create_element("NumberOfPoles", str(fit_form.number_of_poles), poles_elem)
            elif fit_form.pole_indices is not None:
                indices_str = " ".join(str(i) for i in fit_form.pole_indices)
                self._create_element("PoleIndices", indices_str, poles_elem)
                
        elif isinstance(fit_form, SumOfPolesPlusPolynomialFitForm):
            combo_elem = self._create_element("SumOfPolesPlusPolynomial", parent=parent)
            
            # Add SumOfPoles part
            poles_elem = self._create_element("SumOfPoles", parent=combo_elem)
            if fit_form.sum_of_poles.number_of_poles is not None:
                self._create_element("NumberOfPoles", str(fit_form.sum_of_poles.number_of_poles), poles_elem)
            elif fit_form.sum_of_poles.pole_indices is not None:
                indices_str = " ".join(str(i) for i in fit_form.sum_of_poles.pole_indices)
                self._create_element("PoleIndices", indices_str, poles_elem)
            
            # Add Polynomial part
            poly_elem = self._create_element("Polynomial", parent=combo_elem)
            if fit_form.polynomial.degree is not None:
                self._create_element("Degree", str(fit_form.polynomial.degree), poly_elem)
            elif fit_form.polynomial.powers is not None:
                powers_str = " ".join(str(p) for p in fit_form.polynomial.powers)
                self._create_element("Powers", powers_str, poly_elem)
    
    def _create_kbblock_detres(self, ensemble_info: EnsembleInfo, 
                              box_quantization: BoxQuantizationInfo,
                              lab_frame_energies: List[str]) -> ET.Element:
        """Create KBBlock element for determinant residual fits."""
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
    
    def _create_kbblock_spectrum(self, ensemble_info: EnsembleInfo,
                                box_quantization: BoxQuantizationInfo,
                                energy_shifts: List[LabFrameEnergyShiftInfo],
                                cm_energy_range: Tuple[float, float]) -> ET.Element:
        """Create KBBlock element for spectrum fits."""
        kb_block = self._create_element("KBBlock")
        self._create_element("MCEnsembleInfo", ensemble_info.ensemble_name, kb_block)
        
        # Box quantization
        box_quant_elem = self._create_element("BoxQuantization", parent=kb_block)
        self._create_element("TotalMomentumRay", box_quantization.momentum_ray, box_quant_elem)
        self._create_element("TotalMomentumIntSquared", str(box_quantization.momentum_int_squared), box_quant_elem)
        self._create_element("LGIrrep", box_quantization.lg_irrep, box_quant_elem)
        self._create_element("LmaxValues", box_quantization.lmax_values, box_quant_elem)
        
        # Lab frame energy shifts
        for shift in energy_shifts:
            shift_elem = self._create_element("LabFrameEnergyShift", parent=kb_block)
            self._create_element("MCObs", shift.mcobs, shift_elem)
            self._create_element("NonInteractingPair", shift.non_interacting_pair, shift_elem)
        
        # CM frame energy range
        self._create_element("CMFrameEnergyMin", str(cm_energy_range[0]), kb_block)
        self._create_element("CMFrameEnergyMax", str(cm_energy_range[1]), kb_block)
        
        return kb_block
    
    def _create_kbblock_print(self, ensemble_info: EnsembleInfo,
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
    
    def _create_decay_channels(self, channels: List[DecayChannelInfo]) -> ET.Element:
        """Create DecayChannels element."""
        channels_elem = self._create_element("DecayChannels")

        for channel in channels:
            channel_elem = self._create_element("DecayChannelInfo", parent=channels_elem)
            self._create_element("Particle1Name", channel.particle_1, channel_elem)
            self._create_element("Spin1TimesTwo", str(channel.spin_1_times_two), channel_elem)
            if channel.identical:
                self._create_element("Identical", parent=channel_elem)
            else:
                self._create_element("Particle2Name", channel.particle_2, channel_elem)
                self._create_element("Spin2TimesTwo", str(channel.spin_2_times_two), channel_elem)

        return channels_elem
    
    def _create_kbobservables(self, sampling_info: SamplingInfo,
                             ensemble_info: Optional[EnsembleInfo],
                             sampling_files: List[str],
                             verbose: bool = True) -> ET.Element:
        """Create KBObservables element."""
        kb_obs = self._create_element("KBObservables")
        
        # Sampling info
        sampling_elem = self._create_mcsamplinginfo_element(sampling_info)
        kb_obs.append(sampling_elem)
        
        # Bins info (optional)
        if ensemble_info:
            bins_elem = self._create_mcbinsinfo_element(ensemble_info)
            kb_obs.append(bins_elem)
        
        # Verbose flag
        if verbose:
            self._create_element("Verbose", parent=kb_obs)
        
        # Sampling data files
        sampling_data_elem = self._create_element("SamplingData", parent=kb_obs)
        for file_path in sampling_files:
            self._create_element("FileName", file_path, sampling_data_elem)
        
        return kb_obs
    
    def _create_ktilde_matrix_inverse(self, k_elements: List[KElementInfo],
                                    fit_forms: List[Union[ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm]],
                                    decay_channels: List[DecayChannelInfo],
                                    starting_values: List[FitParameterInfo]) -> ET.Element:
        """Create KtildeMatrixInverse element for detres and spectrum fits."""
        ktilde_elem = self._create_element("KtildeMatrixInverse")
        
        # K-matrix elements
        for k_element, fit_form in zip(k_elements, fit_forms):
            element_elem = self._create_element("Element", parent=ktilde_elem)
            k_elem_info = self._create_element("KElementInfo", parent=element_elem)
            self._create_element("JTimesTwo", str(k_element.j_times_two), k_elem_info)
            self._create_element("KIndex", k_element.k_index1, k_elem_info)
            self._create_element("KIndex", k_element.k_index2, k_elem_info)
            
            if fit_form:
                fit_form_elem = self._create_element("FitForm", parent=element_elem)
                self._create_fit_form_content(fit_form, fit_form_elem)
        
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
    
    def create_detres_xml(
        self,
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, Dict[int, Dict[str, Dict[int, Tuple[List[str], 'SigmondSampling']]]]]],
        reference_particle: str,
        particle_masses: Dict[str, float],
        sampling_files: List[str],
        k_elements: List[KElementInfo],
        fit_forms: List[Union[ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm]],
        decay_channels: List[DecayChannelInfo],
        starting_values: List[FitParameterInfo],
        omega_mu: float = 30.0,
        quantization_condition: QuantizationCondition = QuantizationCondition.KTILDE_INV_B,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        minimizer_info: Optional[MinimizerInfo] = None,
        output_directory: Optional[str] = None,
        echo_xml: bool = True,
        verbose: bool = True,
        output_file: Optional[str] = None
    ) -> str:
        """Create determinant residual fit XML directly from structured ensemble data.

        This method follows the simplified API of ``create_spectrum_xml``,
        consuming a rich data structure to automatically generate all
        necessary ``KBBlock`` elements for a determinant residual fit.

        Parameters
        ----------
        project_name
            Human-readable project label.
        ensemble_data
            A list of tuples, where each tuple contains the ``EnsembleInfo``,
            ``SamplingInfo``, and the nested ``psq_dict`` for an ensemble.
        reference_particle, particle_masses, sampling_files, k_elements, fit_forms,
        decay_channels, starting_values
            Core physics and fit model parameters.
        omega_mu, quantization_condition, default_energy_format, minimizer_info
            Optional settings for the fit and numerical procedures.
        output_directory, echo_xml, verbose, output_file
            Optional settings for output and logging.

        Returns
        -------
        The pretty-printed KBfit XML document as a string.
        """
        if minimizer_info is None:
            minimizer_info = MinimizerInfo()

        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")

        ref_sampling_info = ensemble_data[0][1]
        all_observables: List[ObservableInfo] = []
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}
        lab_energies_dict: Dict[Tuple[str, int, str], List[str]] = {}

        for ens_info, samp_info, psq_dict in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info

            for psq, energy_type_map in psq_dict.items():
                # We are only interested in 'elab' for determinant residual fits
                if 'elab' not in energy_type_map:
                    continue
                
                level_map = energy_type_map['elab']
                for level, (_, sigmond_sampling) in level_map.items():
                    obs_info = sigmond_sampling.observable_info
                    all_observables.append(obs_info)
                    
                    momentum_info = self.extract_momentum_info_from_observable(obs_info.name)
                    if not momentum_info:
                        continue
                    
                    _, irrep = momentum_info
                    
                    mcobs_name = f"{obs_info.name} 0"
                    key = (ens_info.ensemble_name, psq, irrep)
                    if key not in lab_energies_dict:
                        lab_energies_dict[key] = []
                    lab_energies_dict[key].append(mcobs_name)

        unique_ensemble_infos = list(all_ensemble_infos.values())
        ensemble_momentum_groups = self.group_observables_by_ensemble_and_momentum(all_observables)

        # 2. Build the XML structure
        root = self._create_element("KBFit")
        
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        
        if output_directory:
            self._create_element("OutputDirectory", output_directory, init_elem)
        
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        
        if echo_xml:
            self._create_element("EchoXML", parent=init_elem)
        
        sampling_elem = self._create_mcsamplinginfo_element(ref_sampling_info)
        init_elem.append(sampling_elem)
        
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoFit", task)
        self._create_element("Type", TaskType.DETERMINANT_RESIDUAL.value, task)
        
        minimizer_elem = self._create_minimizer_info(minimizer_info)
        task.append(minimizer_elem)
        
        self._create_element("OutSamplingsFile", "fit_param_samplings.hdf5[/samplings]", task)
        self._create_element("EcmQcmBoxSamplingsStub", "ecm_qcm_box_samplings", task)
        
        detres_elem = self._create_element("DeterminantResidualFit", parent=task)
        self._create_element("OmegaMu", str(omega_mu), detres_elem)
        self._create_element("QuantizationCondition", quantization_condition.value, detres_elem)
        
        if verbose:
            self._create_element("Verbose", parent=detres_elem)
        
        ktilde_elem = self._create_ktilde_matrix_inverse(
            k_elements, fit_forms, decay_channels, starting_values
        )
        detres_elem.append(ktilde_elem)
        
        self._create_element("DefaultEnergyFormat", default_energy_format.value, detres_elem)
        
        for ensemble_info in unique_ensemble_infos:
            params_elem = self._create_mcensemble_parameters(
                ensemble_info, reference_particle, particle_masses
            )
            detres_elem.append(params_elem)
        
        for (ensemble_name, psq, irrep), obs_list in ensemble_momentum_groups.items():
            ensemble_info = all_ensemble_infos[ensemble_name]
            box_quant = self.create_box_quantization_from_momentum(psq, irrep)
            
            key = (ensemble_name, psq, irrep)
            lab_energies = lab_energies_dict.get(key, [])
            
            kb_block = self._create_kbblock_detres(ensemble_info, box_quant, lab_energies)
            detres_elem.append(kb_block)
        
        kb_obs = self._create_kbobservables(ref_sampling_info, unique_ensemble_infos[0] if unique_ensemble_infos else None, sampling_files, verbose)
        detres_elem.append(kb_obs)
        
        xml_str = self._prettify_xml(root)
        
        if output_file:
            Path(output_file).write_text(xml_str)
        
        return xml_str
    
    def create_print_xml(
        self,
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, Dict[int, Dict[str, Dict[int, Tuple[List[str], 'SigmondSampling']]]]]],
        reference_particle: str,
        particle_masses: Dict[str, float],
        sampling_files: List[str],
        k_elements: List[KElementInfo],
        fit_forms: List[Union[ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm]],
        decay_channels: List[DecayChannelInfo],
        starting_values: List[FitParameterInfo],
        output_stub: str,
        energy_ranges: Optional[Dict[int, Tuple[float, float, float]]] = None,
        omega_mu: float = 0.5,
        quantization_condition: QuantizationCondition = QuantizationCondition.STILDE_CB,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        output_mode: OutputMode = OutputMode.FULL,
        root_finder_config: Optional[RootFinderConfig] = None,
        output_directory: Optional[str] = None,
        echo_xml: bool = True,
        verbose: bool = True,
        output_file: Optional[str] = None
    ) -> str:
        """Create XML for print tasks from structured ensemble data.

        This method uses the simplified data-driven API to configure a print
        job. It discovers the ensembles and momentum shells from your data,
        and you only need to provide the energy ranges for the print grid.

        Parameters
        ----------
        project_name
            Human-readable project label.
        ensemble_data
            A list of tuples, where each tuple contains the ``EnsembleInfo``,
            ``SamplingInfo``, and the nested ``psq_dict`` for an ensemble.
        reference_particle, particle_masses, sampling_files, k_elements, fit_forms,
        decay_channels, starting_values
            Core physics and fit model parameters.
        output_stub
            The base name for the output data files.
        energy_ranges
            Optional dictionary mapping momentum-squared (``psq``) to a tuple
            ``(min_energy, max_energy, step_size)`` for the print grid.
            Sensible defaults are used if not provided.
        omega_mu, quantization_condition, default_energy_format, output_mode,
        root_finder_config
            Optional settings for the print task.
        output_directory, echo_xml, verbose, output_file
            Optional settings for output and logging.

        Returns
        -------
        The pretty-printed KBfit XML document as a string.
        """
        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")

        ref_sampling_info = ensemble_data[0][1]
        all_observables: List[ObservableInfo] = []
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}

        for ens_info, samp_info, psq_dict in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info
            
            for psq, energy_type_map in psq_dict.items():
                for energy_type, level_map in energy_type_map.items():
                    for level, (_, sigmond_sampling) in level_map.items():
                        all_observables.append(sigmond_sampling.observable_info)

        unique_ensemble_infos = list(all_ensemble_infos.values())
        # We group by momentum to find all unique (psq, irrep) shells
        momentum_groups = self.group_observables_by_momentum(all_observables)
        
        if energy_ranges is None:
            energy_ranges = {
                psq: (2.0, 4.0, 0.01) for psq, _ in momentum_groups.keys()
            }

        # 2. Build the XML structure
        root = self._create_element("KBFit")
        
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        
        if output_directory:
            self._create_element("OutputDirectory", output_directory, init_elem)
        
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        
        if echo_xml:
            self._create_element("EchoXML", parent=init_elem)
        
        sampling_elem = self._create_mcsamplinginfo_element(ref_sampling_info)
        init_elem.append(sampling_elem)
        
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoPrint", task)
        
        self._create_element("OutputStub", output_stub, task)
        self._create_element("OutputMode", output_mode.value, task)
        
        self._create_element("OmegaMu", str(omega_mu), task)
        self._create_element("QuantizationCondition", quantization_condition.value, task)
        
        if root_finder_config:
            root_finder_elem = self._create_root_finder_config(root_finder_config)
            task.append(root_finder_elem)
        
        if verbose:
            self._create_element("Verbose", parent=task)
        
        ktilde_elem = self._create_ktilde_matrix_inverse(
            k_elements, fit_forms, decay_channels, starting_values
        )
        task.append(ktilde_elem)
        
        self._create_element("DefaultEnergyFormat", default_energy_format.value, task)
        
        for ensemble_info in unique_ensemble_infos:
            params_elem = self._create_mcensemble_parameters(
                ensemble_info, reference_particle, particle_masses
            )
            task.append(params_elem)
        
        # Use the first ensemble for all KBBlocks, as the print job is often
        # considered universal for the given K-matrix.
        first_ensemble = unique_ensemble_infos[0] if unique_ensemble_infos else None

        if first_ensemble:
            for (psq, irrep), obs_list in momentum_groups.items():
                box_quant = self.create_box_quantization_from_momentum(psq, irrep)
                energy_range = energy_ranges.get(psq, (2.0, 4.0, 0.01))
                kb_block = self._create_kbblock_print(first_ensemble, box_quant, energy_range)
                task.append(kb_block)
        
        kb_obs = self._create_kbobservables(ref_sampling_info, first_ensemble, sampling_files, verbose)
        task.append(kb_obs)
        
        xml_str = self._prettify_xml(root)
        
        if output_file:
            Path(output_file).write_text(xml_str)
        
        return xml_str
    
    def create_spectrum_xml(
        self,
        project_name: str,
        ensemble_data: List[Tuple[EnsembleInfo, SamplingInfo, Dict[int, Dict[str, Dict[int, Tuple[List[str], SigmondSampling]]]]]],
        reference_particle: str,
        particle_masses: Dict[str, float],
        sampling_files: List[str],
        k_elements: List[KElementInfo],
        fit_forms: List[Union[ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm]],
        decay_channels: List[DecayChannelInfo],
        starting_values: List[FitParameterInfo],
        omega_mu: float = 0.8,
        quantization_condition: QuantizationCondition = QuantizationCondition.STILDE_CB,
        default_energy_format: EnergyFormat = EnergyFormat.REFERENCE_RATIO,
        minimizer_info: Optional[MinimizerInfo] = None,
        root_finder_config: Optional[RootFinderConfig] = None,
        cm_energy_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        output_directory: Optional[str] = None,
        echo_xml: bool = True,
        verbose: bool = True,
        output_file: Optional[str] = None
    ) -> str:
        """Create spectrum XML directly from structured ensemble data.

        This high-level convenience wrapper dramatically reduces the amount of
        user boiler-plate required by consuming a rich data structure that is
        typically produced by a data loading helper function.

        It automatically discovers ensembles, momenta, and energy shift levels
        from the data, and constructs all necessary ``KBBlock`` elements.

        Parameters
        ----------
        project_name
            Human-readable project label.
        ensemble_data
            A list where each element is a tuple ``(ensemble_info, sampling_info, psq_dict)``.
            ``psq_dict`` is a nested dictionary mapping from momentum-squared (int)
            to energy type (str) to level (int) to a final tuple containing
            a list of non-interacting pair strings and the ``SigmondSampling`` object.
        reference_particle
            Name of the particle to use for the reference mass scale.
        particle_masses
            Dictionary mapping particle names to their masses in reference units.
        sampling_files
            List of paths to the HDF5 sampling files.
        k_elements
            List of ``KElementInfo`` objects defining the K-matrix elements.
        fit_forms
            List of fit form objects, one for each corresponding k-element.
        decay_channels
            List of ``DecayChannelInfo`` objects.
        starting_values
            List of ``FitParameterInfo`` objects for the K-matrix fit.
        omega_mu, quantization_condition, default_energy_format
            Optional physics settings for the fit.
        minimizer_info, root_finder_config
            Optional configurations for the numerical procedures.
        cm_energy_ranges
            Optional dictionary mapping momentum-squared to a ``(min, max)``
            energy range tuple for the root search. Defaults are provided if ``None``.
        output_directory, echo_xml, verbose, output_file
            Optional settings for output and logging.

        Returns
        -------
        The pretty-printed KBfit XML document as a string.
        """
        if minimizer_info is None:
            minimizer_info = MinimizerInfo()

        if root_finder_config is None:
            root_finder_config = RootFinderConfig()

        # 1. Extract and collate information from ensemble_data
        if not ensemble_data:
            raise ValueError("`ensemble_data` must contain at least one entry.")

        ref_sampling_info = ensemble_data[0][1]
        all_observables: List[ObservableInfo] = []
        all_ensemble_infos: Dict[str, EnsembleInfo] = {}
        energy_shifts_dict: Dict[Tuple[str, int, str], List[LabFrameEnergyShiftInfo]] = {}

        for ens_info, samp_info, psq_dict in ensemble_data:
            if samp_info != ref_sampling_info:
                raise ValueError("All entries in `ensemble_data` must share the same SamplingInfo.")
            if ens_info.ensemble_name not in all_ensemble_infos:
                all_ensemble_infos[ens_info.ensemble_name] = ens_info

            for psq, energy_type_map in psq_dict.items():
                for energy_type, level_map in energy_type_map.items():
                    # We are only interested in energy shifts for spectrum fits
                    if "delab" not in energy_type.lower():
                        continue
                    for level, (ni_list, sigmond_sampling) in level_map.items():
                        obs_info = sigmond_sampling.observable_info
                        all_observables.append(obs_info)
                        
                        momentum_info = self.extract_momentum_info_from_observable(obs_info.name)
                        if not momentum_info:
                            continue 
                        
                        _, irrep = momentum_info
                        
                        if not ni_list:
                            raise ValueError(f"Non-interacting pair list is empty for observable: {obs_info.name}")
                        
                        mcobs_name = f"{obs_info.name} 0"
                        merged_ni_list = "".join(ni_list)
                        energy_shift = LabFrameEnergyShiftInfo(mcobs_name, merged_ni_list)
                        
                        key = (ens_info.ensemble_name, psq, irrep)
                        if key not in energy_shifts_dict:
                            energy_shifts_dict[key] = []
                        energy_shifts_dict[key].append(energy_shift)

        unique_ensemble_infos = list(all_ensemble_infos.values())
        ensemble_momentum_groups = self.group_observables_by_ensemble_and_momentum(all_observables)

        # 2. Build the XML structure
        root = self._create_element("KBFit")
        
        init_elem = self._create_element("Initialize", parent=root)
        self._create_element("ProjectName", project_name, init_elem)
        
        if output_directory:
            self._create_element("OutputDirectory", output_directory, init_elem)
        
        self._create_element("LogFile", f"{project_name}.log", init_elem)
        
        if echo_xml:
            self._create_element("EchoXML", parent=init_elem)
        
        sampling_elem = self._create_mcsamplinginfo_element(ref_sampling_info)
        init_elem.append(sampling_elem)
        
        task_seq = self._create_element("TaskSequence", parent=root)
        task = self._create_element("Task", parent=task_seq)
        self._create_element("Action", "DoFit", task)
        self._create_element("Type", TaskType.SPECTRUM.value, task)
        
        minimizer_elem = self._create_minimizer_info(minimizer_info)
        task.append(minimizer_elem)
        
        self._create_element("OutSamplingsFile", "fit_param_samplings.hdf5[/samplings]", task)
        
        spectrum_elem = self._create_element("SpectrumFit", parent=task)
        self._create_element("OmegaMu", str(omega_mu), spectrum_elem)
        self._create_element("QuantizationCondition", quantization_condition.value, spectrum_elem)
        
        root_finder_elem = self._create_root_finder_config(root_finder_config)
        spectrum_elem.append(root_finder_elem)
        
        if verbose:
            self._create_element("Verbose", parent=spectrum_elem)
        
        ktilde_elem = self._create_ktilde_matrix_inverse(
            k_elements, fit_forms, decay_channels, starting_values
        )
        spectrum_elem.append(ktilde_elem)
        
        self._create_element("DefaultEnergyFormat", default_energy_format.value, spectrum_elem)
        
        for ensemble_info in unique_ensemble_infos:
            params_elem = self._create_mcensemble_parameters(
                ensemble_info, reference_particle, particle_masses
            )
            spectrum_elem.append(params_elem)
        
        for (ensemble_name, psq, irrep), obs_list in ensemble_momentum_groups.items():
            ensemble_info = all_ensemble_infos[ensemble_name]
            box_quant = self.create_box_quantization_from_momentum(psq, irrep)
            
            key = (ensemble_name, psq, irrep)
            shifts = energy_shifts_dict.get(key, [])
            
            try:
                cm_range = cm_energy_ranges.get(psq, (2.55, 2.85))
            except:
                cm_range = (2.55, 2.85)

            kb_block = self._create_kbblock_spectrum(ensemble_info, box_quant, shifts, cm_range)
            spectrum_elem.append(kb_block)
        
        kb_obs = self._create_kbobservables(ref_sampling_info, unique_ensemble_infos[0] if unique_ensemble_infos else None, sampling_files, verbose)
        spectrum_elem.append(kb_obs)
        
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
    
    @staticmethod
    def group_observables_by_ensemble_and_momentum(observables: List[ObservableInfo]) -> Dict[Tuple[str, int, str], List[ObservableInfo]]:
        """
        Group observables by ensemble, momentum squared, and irrep.
        
        Args:
            observables: List of observable infos
        
        Returns:
            Dictionary mapping (ensemble_name, psq, irrep) to list of observables
        """
        grouped = {}
        
        for obs in observables:
            momentum_info = KBfitXMLHelper.extract_momentum_info_from_observable(obs.name)
            if momentum_info:
                psq, irrep = momentum_info
                ensemble_name = obs.ensemble_info.ensemble_name
                key = (ensemble_name, psq, irrep)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(obs)
        
        return grouped 