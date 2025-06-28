"""
Tests for KBfitXMLHelper module.
"""

import pytest
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

from SigmondSamplings.kb_xml_helper import (
    KBfitXMLHelper, BoxQuantizationInfo, ParticleInfo, KElementInfo, 
    FitParameterInfo, LabFrameEnergyShiftInfo, MinimizerInfo, RootFinderConfig,
    TaskType, MinimizerMethod, QuantizationCondition, EnergyFormat, OutputMode, Verbosity,
    FitFormType, ExpressionFitForm, PolynomialFitForm, SumOfPolesFitForm, SumOfPolesPlusPolynomialFitForm
)
from SigmondSamplings.sampling import EnsembleInfo, SamplingInfo, ObservableInfo


@pytest.fixture
def sample_ensemble_info():
    """Create sample ensemble info for testing."""
    return EnsembleInfo("test_ensemble", 1000, 25)


@pytest.fixture
def sample_sampling_info():
    """Create sample sampling info for testing."""
    return SamplingInfo("bootstrap", 2000, seed=12345, boot_skip=0)


@pytest.fixture
def sample_observables(sample_ensemble_info):
    """Create sample observables for testing."""
    return [
        ObservableInfo("isosinglet_S=0_A1g_1_PSQ=0_elab_2_ref", 0, "mcobs", "re", sample_ensemble_info),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,0,1)_elab_2_ref", 0, "mcobs", "re", sample_ensemble_info),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_elab_3_ref", 0, "mcobs", "re", sample_ensemble_info),
    ]


@pytest.fixture
def sample_box_quantizations():
    """Create sample box quantizations for testing."""
    return [
        BoxQuantizationInfo("ar", 0, "A1g", 0),
        BoxQuantizationInfo("oa", 1, "A1", 0),
        BoxQuantizationInfo("pd", 2, "A1", 0),
    ]


@pytest.fixture
def sample_k_elements():
    """Create sample K-matrix elements for testing."""
    return [
        KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)")
    ]


@pytest.fixture
def sample_decay_channels():
    """Create sample decay channels for testing."""
    return [
        ParticleInfo("phi", 0, True)
    ]


@pytest.fixture
def sample_fit_parameters(sample_k_elements):
    """Create sample fit parameters for testing."""
    return [
        FitParameterInfo("rho_mass_ref", 2.70, sample_k_elements[0]),
        FitParameterInfo("Gamma_ref", 0.001, sample_k_elements[0])
    ]


@pytest.fixture
def sample_fit_expressions():
    """Create sample fit expressions for testing."""
    return {
        "L(0) 2S(0) chan(0)": "0.5*sqrt(x^2 - 4.0)*(rho_mass_ref^2 - x^2)/(rho_mass_ref * Gamma_ref)"
    }

@pytest.fixture
def sample_fit_forms():
    """Create sample fit forms for testing."""
    return {
        "L(0) 2S(0) chan(0)": ExpressionFitForm("0.5*sqrt(x^2 - 4.0)*(rho_mass_ref^2 - x^2)/(rho_mass_ref * Gamma_ref)")
    }


@pytest.fixture
def kb_helper():
    """Create KBfitXMLHelper instance."""
    return KBfitXMLHelper()


class TestKBfitXMLHelper:
    """Test class for KBfitXMLHelper."""

    def test_create_element(self, kb_helper):
        """Test basic element creation."""
        elem = kb_helper._create_element("TestTag", "test_value")
        assert elem.tag == "TestTag"
        assert elem.text == "test_value"

    def test_create_mcsamplinginfo_bootstrap(self, kb_helper, sample_sampling_info):
        """Test MCSamplingInfo creation for bootstrap."""
        elem = kb_helper._create_mcsamplinginfo_element(sample_sampling_info)
        assert elem.tag == "MCSamplingInfo"
        
        bootstrap = elem.find("Bootstrapper")
        assert bootstrap is not None
        assert bootstrap.find("NumberResamplings").text == "2000"
        assert bootstrap.find("Seed").text == "12345"
        assert bootstrap.find("BootSkip").text == "0"

    def test_create_mcsamplinginfo_jackknife(self, kb_helper):
        """Test MCSamplingInfo creation for jackknife."""
        jackknife_info = SamplingInfo("jackknife", 25)
        elem = kb_helper._create_mcsamplinginfo_element(jackknife_info)
        
        assert elem.tag == "MCSamplingInfo"
        jackknife = elem.find("Jackkniffer")
        assert jackknife is not None
        assert jackknife.find("NumberResamplings").text == "25"

    def test_create_mcsamplinginfo_simple_jackknife(self, kb_helper):
        """Test MCSamplingInfo creation for simple jackknife."""
        jackknife_info = SamplingInfo("jackknife", 0)  # Simple jackknife
        elem = kb_helper._create_mcsamplinginfo_element(jackknife_info)
        
        assert elem.tag == "MCSamplingInfo"
        jackknife = elem.find("Jackknife")
        assert jackknife is not None

    def test_create_mcbinsinfo(self, kb_helper, sample_ensemble_info):
        """Test MCBinsInfo creation."""
        elem = kb_helper._create_mcbinsinfo_element(sample_ensemble_info)
        
        assert elem.tag == "MCBinsInfo"
        assert elem.find("MCEnsembleInfo").text == "test_ensemble"
        assert elem.find("NumberOfMeasurements").text == "1000"
        assert elem.find("NumberOfBins").text == "25"

    def test_create_mcensemble_parameters(self, kb_helper, sample_ensemble_info):
        """Test MCEnsembleParameters creation."""
        particle_masses = {"phi": 1.0, "pi": 0.139}
        elem = kb_helper._create_mcensemble_parameters(
            sample_ensemble_info, "Phi", particle_masses
        )
        
        assert elem.tag == "MCEnsembleParameters"
        assert elem.find("MCEnsembleInfo").text == "test_ensemble"
        
        ref_mass = elem.find("ReferenceMassTimeSpacingProduct")
        assert ref_mass.find("MCObs").text == "Phi(0)_elab 0"
        
        masses = elem.findall("ParticleMass")
        assert len(masses) == 2

    def test_create_minimizer_info(self, kb_helper):
        """Test MinimizerInfo creation."""
        minimizer_info = MinimizerInfo(
            method=MinimizerMethod.NL2SNO,
            parameter_rel_tol=1e-5,
            verbosity=Verbosity.HIGH
        )
        elem = kb_helper._create_minimizer_info(minimizer_info)
        
        assert elem.tag == "MinimizerInfo"
        assert elem.find("Method").text == "NL2Sno"
        assert elem.find("ParameterRelTol").text == "1e-05"
        assert elem.find("Verbosity").text == "High"

    def test_create_root_finder_config(self, kb_helper):
        """Test RootFinder configuration creation."""
        config = RootFinderConfig(
            abs_x_tolerance=1e-7,
            min_step_percent=1e-6
        )
        elem = kb_helper._create_root_finder_config(config)
        
        assert elem.tag == "RootFinder"
        assert elem.find("AdaptiveBracket") is not None
        assert elem.find("AbsXTolerance").text == "1e-07"
        assert elem.find("MinStepPercent").text == "1e-06"

    def test_create_kbblock_detres(self, kb_helper, sample_ensemble_info, sample_box_quantizations):
        """Test KBBlock creation for detres."""
        lab_energies = ["test_energy_1", "test_energy_2"]
        elem = kb_helper._create_kbblock_detres(
            sample_ensemble_info, sample_box_quantizations[0], lab_energies
        )
        
        assert elem.tag == "KBBlock"
        assert elem.find("MCEnsembleInfo").text == "test_ensemble"
        
        box_quant = elem.find("BoxQuantization")
        assert box_quant.find("TotalMomentumRay").text == "ar"
        assert box_quant.find("TotalMomentumIntSquared").text == "0"
        assert box_quant.find("LGIrrep").text == "A1g"
        
        energies = elem.findall("LabFrameEnergy")
        assert len(energies) == 2

    def test_create_kbblock_spectrum(self, kb_helper, sample_ensemble_info, sample_box_quantizations):
        """Test KBBlock creation for spectrum."""
        energy_shifts = [
            LabFrameEnergyShiftInfo("test_shift_1", "phi(1)phi(1)"),
            LabFrameEnergyShiftInfo("test_shift_2", "phi(2)phi(1)")
        ]
        cm_range = (2.0, 3.0)
        
        elem = kb_helper._create_kbblock_spectrum(
            sample_ensemble_info, sample_box_quantizations[0], energy_shifts, cm_range
        )
        
        assert elem.tag == "KBBlock"
        shifts = elem.findall("LabFrameEnergyShift")
        assert len(shifts) == 2
        assert shifts[0].find("NonInteractingPair").text == "phi(1)phi(1)"
        
        assert elem.find("CMFrameEnergyMin").text == "2.0"
        assert elem.find("CMFrameEnergyMax").text == "3.0"

    def test_create_kbblock_print(self, kb_helper, sample_ensemble_info, sample_box_quantizations):
        """Test KBBlock creation for print."""
        energy_range = (2.0, 4.0, 0.01)
        elem = kb_helper._create_kbblock_print(
            sample_ensemble_info, sample_box_quantizations[0], energy_range
        )
        
        assert elem.tag == "KBBlock"
        assert elem.find("LabFrameEnergyMin").text == "2.0"
        assert elem.find("LabFrameEnergyMax").text == "4.0"
        assert elem.find("LabFrameEnergyInc").text == "0.01"

    def test_create_decay_channels(self, kb_helper, sample_decay_channels):
        """Test DecayChannels creation."""
        elem = kb_helper._create_decay_channels(sample_decay_channels)
        
        assert elem.tag == "DecayChannels"
        channels = elem.findall("DecayChannelInfo")
        assert len(channels) == 1
        assert channels[0].find("Particle1Name").text == "phi"
        assert channels[0].find("Spin1TimesTwo").text == "0"
        assert channels[0].find("Identical") is not None

    def test_create_kbobservables(self, kb_helper, sample_sampling_info, sample_ensemble_info):
        """Test KBObservables creation."""
        sampling_files = ["file1.hdf5", "file2.hdf5"]
        elem = kb_helper._create_kbobservables(
            sample_sampling_info, sample_ensemble_info, sampling_files
        )
        
        assert elem.tag == "KBObservables"
        assert elem.find("MCSamplingInfo") is not None
        assert elem.find("MCBinsInfo") is not None
        assert elem.find("Verbose") is not None
        
        files = elem.find("SamplingData").findall("FileName")
        assert len(files) == 2

    def test_create_ktilde_matrix_inverse(self, kb_helper, sample_k_elements, 
                                        sample_fit_forms, sample_decay_channels,
                                        sample_fit_parameters):
        """Test KtildeMatrixInverse creation."""
        elem = kb_helper._create_ktilde_matrix_inverse(
            sample_k_elements, sample_fit_forms, 
            sample_decay_channels, sample_fit_parameters
        )
        
        assert elem.tag == "KtildeMatrixInverse"
        
        elements = elem.findall("Element")
        assert len(elements) == 1
        
        k_elem_info = elements[0].find("KElementInfo")
        assert k_elem_info.find("JTimesTwo").text == "0"
        
        fit_form = elements[0].find("FitForm")
        assert fit_form is not None
        
        assert elem.find("DecayChannels") is not None
        
        start_vals = elem.find("StartingValues")
        params = start_vals.findall("KFitParamInfo")
        assert len(params) == 2

    def test_create_detres_xml(self, kb_helper, sample_observables, sample_sampling_info,
                              sample_box_quantizations, sample_k_elements, 
                              sample_fit_forms, sample_decay_channels,
                              sample_fit_parameters):
        """Test complete detres XML creation."""
        particle_masses = {"phi": 1.0}
        sampling_files = ["test.hdf5"]
        
        xml_str = kb_helper.create_detres_xml(
            project_name="TestDetRes",
            observables=sample_observables,
            sampling_info=sample_sampling_info,
            reference_particle="Phi",
            particle_masses=particle_masses,
            box_quantizations=sample_box_quantizations,
            sampling_files=sampling_files,
            fit_forms=sample_fit_forms,
            k_elements=sample_k_elements,
            decay_channels=sample_decay_channels,
            starting_values=sample_fit_parameters,
            omega_mu=30.0,
            quantization_condition=QuantizationCondition.KTILDE_INV_B
        )
        
        # Parse XML to verify structure
        root = ET.fromstring(xml_str)
        assert root.tag == "KBFit"
        
        # Check Initialize section
        init = root.find("Initialize")
        assert init.find("ProjectName").text == "TestDetRes"
        assert init.find("LogFile").text == "TestDetRes.log"
        assert init.find("EchoXML") is not None
        
        # Check TaskSequence
        task_seq = root.find("TaskSequence")
        task = task_seq.find("Task")
        assert task.find("Action").text == "DoFit"
        assert task.find("Type").text == "DeterminantResidualFit"
        
        # Check MinimizerInfo
        minimizer = task.find("MinimizerInfo")
        assert minimizer.find("Method").text == "Minuit2Migrad"  # default
        
        # Check DeterminantResidualFit section
        detres = task.find("DeterminantResidualFit")
        assert detres.find("OmegaMu").text == "30.0"
        assert detres.find("QuantizationCondition").text == "KtildeinvB"
        
        # Check KtildeMatrixInverse
        ktilde = detres.find("KtildeMatrixInverse")
        assert ktilde is not None
        
        # Check MCEnsembleParameters
        ensemble_params = detres.findall("MCEnsembleParameters")
        assert len(ensemble_params) >= 1
        
        # Check KBBlocks
        kb_blocks = detres.findall("KBBlock")
        assert len(kb_blocks) >= 1
        
        # Check KBObservables
        kb_obs = detres.find("KBObservables")
        assert kb_obs is not None

    def test_create_spectrum_xml(self, kb_helper, sample_observables, sample_sampling_info,
                                sample_box_quantizations, sample_k_elements,
                                sample_fit_forms, sample_decay_channels,
                                sample_fit_parameters):
        """Test complete spectrum XML creation."""
        particle_masses = {"phi": 1.0}
        sampling_files = ["test.hdf5"]
        
        energy_shifts = {
            0: [LabFrameEnergyShiftInfo("shift_1", "phi(1)phi(1)")],
            1: [LabFrameEnergyShiftInfo("shift_2", "phi(2)phi(1)")]
        }
        cm_energy_ranges = {
            0: (2.0, 3.0),
            1: (2.1, 3.1)
        }
        
        xml_str = kb_helper.create_spectrum_xml(
            project_name="TestSpectrum",
            observables=sample_observables,
            sampling_info=sample_sampling_info,
            reference_particle="Phi",
            particle_masses=particle_masses,
            box_quantizations=sample_box_quantizations[:2],  # Use first 2
            energy_shifts=energy_shifts,
            cm_energy_ranges=cm_energy_ranges,
            sampling_files=sampling_files,
            fit_forms=sample_fit_forms,
            k_elements=sample_k_elements,
            decay_channels=sample_decay_channels,
            starting_values=sample_fit_parameters,
            omega_mu=0.8
        )
        
        # Parse XML to verify structure
        root = ET.fromstring(xml_str)
        assert root.tag == "KBFit"
        
        # Check TaskSequence
        task_seq = root.find("TaskSequence")
        task = task_seq.find("Task")
        assert task.find("Action").text == "DoFit"
        assert task.find("Type").text == "SpectrumFit"
        
        # Check SpectrumFit section
        spectrum = task.find("SpectrumFit")
        assert spectrum.find("OmegaMu").text == "0.8"
        assert spectrum.find("QuantizationCondition").text == "StildeCB"  # default
        
        # Check RootFinder
        root_finder = spectrum.find("RootFinder")
        assert root_finder is not None
        assert root_finder.find("AdaptiveBracket") is not None
        
        # Check KBBlocks have energy shifts
        kb_blocks = spectrum.findall("KBBlock")
        assert len(kb_blocks) >= 1
        
        # Check first block has energy shifts
        first_block = kb_blocks[0]
        shifts = first_block.findall("LabFrameEnergyShift")
        assert len(shifts) >= 1
        assert first_block.find("CMFrameEnergyMin") is not None
        assert first_block.find("CMFrameEnergyMax") is not None

    def test_create_print_xml(self, kb_helper, sample_sampling_info, sample_box_quantizations,
                             sample_k_elements, sample_fit_forms,
                             sample_decay_channels, sample_fit_parameters):
        """Test complete print XML creation."""
        particle_masses = {"phi": 1.0}
        sampling_files = ["test.hdf5"]
        ensemble_infos = [EnsembleInfo("test_ensemble", 1000, 25)]
        
        energy_ranges = {
            0: (2.0, 4.0, 0.01),
            1: (2.5, 4.5, 0.01)
        }
        
        xml_str = kb_helper.create_print_xml(
            project_name="TestPrint",
            sampling_info=sample_sampling_info,
            reference_particle="Phi",
            particle_masses=particle_masses,
            box_quantizations=sample_box_quantizations[:2],
            energy_ranges=energy_ranges,
            sampling_files=sampling_files,
            fit_forms=sample_fit_forms,
            k_elements=sample_k_elements,
            decay_channels=sample_decay_channels,
            starting_values=sample_fit_parameters,
            ensemble_infos=ensemble_infos,
            output_stub="test_output",
            omega_mu=0.5
        )
        
        # Parse XML to verify structure
        root = ET.fromstring(xml_str)
        assert root.tag == "KBFit"
        
        # Check TaskSequence
        task_seq = root.find("TaskSequence")
        task = task_seq.find("Task")
        assert task.find("Action").text == "DoPrint"
        
        # Check output configuration
        assert task.find("OutputStub").text == "test_output"
        assert task.find("OutputMode").text == "full"  # default
        
        # Check print configuration
        assert task.find("OmegaMu").text == "0.5"
        assert task.find("QuantizationCondition").text == "StildeCB"  # default
        
        # Check KBBlocks have energy ranges
        kb_blocks = task.findall("KBBlock")
        assert len(kb_blocks) >= 1
        
        # Check first block has energy range
        first_block = kb_blocks[0]
        assert first_block.find("LabFrameEnergyMin") is not None
        assert first_block.find("LabFrameEnergyMax") is not None
        assert first_block.find("LabFrameEnergyInc") is not None

    def test_extract_momentum_info_from_observable(self):
        """Test momentum extraction from observable names."""
        # Test PSQ format
        obs_name = "isosinglet_S=0_A1g_1_PSQ=0_elab_2_ref"
        psq, irrep = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
        assert psq == 0
        assert irrep == "A1g"
        
        # Test P= format
        obs_name = "isosinglet_S=0_A1_1_P=(0,0,1)_elab_2_ref"
        psq, irrep = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
        assert psq == 1
        assert irrep == "A1"
        
        # Test invalid format
        result = KBfitXMLHelper.extract_momentum_info_from_observable("invalid_name")
        assert result is None

    def test_group_observables_by_momentum(self, sample_observables):
        """Test grouping observables by momentum."""
        grouped = KBfitXMLHelper.group_observables_by_momentum(sample_observables)
        
        # Should have groups for PSQ=0, PSQ=1, PSQ=2
        assert (0, "A1g") in grouped
        assert (1, "A1") in grouped
        assert (2, "A1") in grouped

    def test_create_box_quantization_from_momentum(self):
        """Test box quantization creation from momentum."""
        box_quant = KBfitXMLHelper.create_box_quantization_from_momentum(1, "A1", 2)
        
        assert box_quant.momentum_ray == "oa"
        assert box_quant.momentum_int_squared == 1
        assert box_quant.lg_irrep == "A1"
        assert box_quant.lmax_values == "2"

    def test_prettify_xml(self, kb_helper):
        """Test XML prettification."""
        elem = kb_helper._create_element("Root")
        child = kb_helper._create_element("Child", "value", elem)
        
        xml_str = kb_helper._prettify_xml(elem)
        
        # Should be properly formatted
        lines = xml_str.split('\n')
        assert lines[0] == "<Root>"
        assert "    <Child>value</Child>" in lines
        assert lines[-1] == "</Root>"

    def test_xml_file_output(self, kb_helper, sample_observables, sample_sampling_info,
                            sample_box_quantizations, sample_k_elements,
                            sample_fit_forms, sample_decay_channels,
                            sample_fit_parameters):
        """Test XML output to file."""
        particle_masses = {"phi": 1.0}
        sampling_files = ["test.hdf5"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            output_file = f.name
        
        try:
            xml_str = kb_helper.create_detres_xml(
                project_name="TestDetRes",
                observables=sample_observables,
                sampling_info=sample_sampling_info,
                reference_particle="Phi",
                particle_masses=particle_masses,
                box_quantizations=sample_box_quantizations,
                sampling_files=sampling_files,
                fit_forms=sample_fit_forms,
                k_elements=sample_k_elements,
                decay_channels=sample_decay_channels,
                starting_values=sample_fit_parameters,
                output_file=output_file
            )
            
            # Check file was created and has correct content
            file_content = Path(output_file).read_text()
            assert file_content == xml_str
            
            # Verify XML is parseable
            root = ET.fromstring(file_content)
            assert root.tag == "KBFit"
            
        finally:
            Path(output_file).unlink()

    def test_enum_values(self):
        """Test that enums have correct values."""
        assert TaskType.DETERMINANT_RESIDUAL.value == "DeterminantResidualFit"
        assert TaskType.SPECTRUM.value == "SpectrumFit"
        assert TaskType.PRINT.value == "DoPrint"
        
        assert MinimizerMethod.NL2SNO.value == "NL2Sno"
        assert MinimizerMethod.MINUIT2_MIGRAD.value == "Minuit2Migrad"
        
        assert QuantizationCondition.KTILDE_INV_B.value == "KtildeinvB"
        assert QuantizationCondition.STILDE_CB.value == "StildeCB"
        
        assert EnergyFormat.REFERENCE_RATIO.value == "reference_ratio"
        assert OutputMode.FULL.value == "full"

    def test_dataclass_defaults(self):
        """Test that dataclasses have sensible defaults."""
        box_quant = BoxQuantizationInfo("ar", 0, "A1g")
        assert box_quant.lmax_values == "0"
        
        particle = ParticleInfo("phi", 0)
        assert particle.identical is False
        
        minimizer = MinimizerInfo()
        assert minimizer.method == MinimizerMethod.MINUIT2_MIGRAD
        assert minimizer.verbosity == Verbosity.LOW
        
        root_finder = RootFinderConfig()
        assert root_finder.abs_x_tolerance == 1e-6
        assert root_finder.min_step_percent == 1e-5
        
        # FitParameterInfo
        param = FitParameterInfo("test_param", 1.0)
        assert param.fit_form_type == FitFormType.EXPRESSION

    def test_fit_form_classes(self):
        """Test fit form dataclasses."""
        # ExpressionFitForm
        expr_form = ExpressionFitForm("a*x^2 + b")
        assert expr_form.expression == "a*x^2 + b"
        
        # PolynomialFitForm with degree
        poly_form1 = PolynomialFitForm(degree=3)
        assert poly_form1.degree == 3
        assert poly_form1.powers is None
        
        # PolynomialFitForm with powers
        poly_form2 = PolynomialFitForm(powers=[0, 2, 4])
        assert poly_form2.powers == [0, 2, 4]
        assert poly_form2.degree is None
        
        # SumOfPolesFitForm with number of poles
        poles_form1 = SumOfPolesFitForm(number_of_poles=2)
        assert poles_form1.number_of_poles == 2
        assert poles_form1.pole_indices is None
        
        # SumOfPolesFitForm with pole indices
        poles_form2 = SumOfPolesFitForm(pole_indices=[0, 2])
        assert poles_form2.pole_indices == [0, 2]
        assert poles_form2.number_of_poles is None
        
        # SumOfPolesPlusPolynomialFitForm
        combo_form = SumOfPolesPlusPolynomialFitForm(poles_form1, poly_form1)
        assert combo_form.sum_of_poles.number_of_poles == 2
        assert combo_form.polynomial.degree == 3

    def test_fit_form_validation(self):
        """Test fit form validation."""
        # PolynomialFitForm should require either degree or powers
        with pytest.raises(ValueError, match="Either degree or powers must be specified"):
            PolynomialFitForm()
        
        with pytest.raises(ValueError, match="Cannot specify both degree and powers"):
            PolynomialFitForm(degree=3, powers=[0, 1, 2])
        
        # SumOfPolesFitForm should require either number_of_poles or pole_indices  
        with pytest.raises(ValueError, match="Either number_of_poles or pole_indices must be specified"):
            SumOfPolesFitForm()
        
        with pytest.raises(ValueError, match="Cannot specify both number_of_poles and pole_indices"):
            SumOfPolesFitForm(number_of_poles=2, pole_indices=[0, 1])

    def test_create_fit_form_content_expression(self, kb_helper):
        """Test creation of expression fit form content."""
        parent = kb_helper._create_element("Parent")
        
        expr_form = ExpressionFitForm("a*x^2 + b*x + c")
        kb_helper._create_fit_form_content(expr_form, parent)
        
        # Check XML structure
        expr_elem = parent.find("Expression")
        assert expr_elem is not None
        string_elem = expr_elem.find("String")
        assert string_elem is not None
        assert string_elem.text == "a*x^2 + b*x + c"

    def test_create_fit_form_content_polynomial(self, kb_helper):
        """Test creation of polynomial fit form content."""
        # Test with degree
        parent1 = kb_helper._create_element("Parent")
        poly_form1 = PolynomialFitForm(degree=3)
        kb_helper._create_fit_form_content(poly_form1, parent1)
        
        poly_elem1 = parent1.find("Polynomial")
        assert poly_elem1 is not None
        degree_elem = poly_elem1.find("Degree")
        assert degree_elem is not None
        assert degree_elem.text == "3"
        
        # Test with powers
        parent2 = kb_helper._create_element("Parent")
        poly_form2 = PolynomialFitForm(powers=[0, 2, 4])
        kb_helper._create_fit_form_content(poly_form2, parent2)
        
        poly_elem2 = parent2.find("Polynomial")
        assert poly_elem2 is not None
        powers_elem = poly_elem2.find("Powers")
        assert powers_elem is not None
        assert powers_elem.text == "0 2 4"

    def test_create_fit_form_content_sum_of_poles(self, kb_helper):
        """Test creation of sum of poles fit form content."""
        # Test with number of poles
        parent1 = kb_helper._create_element("Parent")
        poles_form1 = SumOfPolesFitForm(number_of_poles=2)
        kb_helper._create_fit_form_content(poles_form1, parent1)
        
        poles_elem1 = parent1.find("SumOfPoles")
        assert poles_elem1 is not None
        num_poles_elem = poles_elem1.find("NumberOfPoles")
        assert num_poles_elem is not None
        assert num_poles_elem.text == "2"
        
        # Test with pole indices
        parent2 = kb_helper._create_element("Parent")
        poles_form2 = SumOfPolesFitForm(pole_indices=[0, 2, 5])
        kb_helper._create_fit_form_content(poles_form2, parent2)
        
        poles_elem2 = parent2.find("SumOfPoles")
        assert poles_elem2 is not None
        indices_elem = poles_elem2.find("PoleIndices")
        assert indices_elem is not None
        assert indices_elem.text == "0 2 5"

    def test_create_fit_form_content_combined(self, kb_helper):
        """Test creation of combined sum of poles plus polynomial fit form content."""
        parent = kb_helper._create_element("Parent")
        
        poles_form = SumOfPolesFitForm(number_of_poles=1)
        poly_form = PolynomialFitForm(degree=2)
        combo_form = SumOfPolesPlusPolynomialFitForm(poles_form, poly_form)
        
        kb_helper._create_fit_form_content(combo_form, parent)
        
        # Check XML structure
        combo_elem = parent.find("SumOfPolesPlusPolynomial")
        assert combo_elem is not None
        
        poles_elem = combo_elem.find("SumOfPoles")
        assert poles_elem is not None
        num_poles_elem = poles_elem.find("NumberOfPoles")
        assert num_poles_elem is not None
        assert num_poles_elem.text == "1"
        
        poly_elem = combo_elem.find("Polynomial")
        assert poly_elem is not None
        degree_elem = poly_elem.find("Degree")
        assert degree_elem is not None
        assert degree_elem.text == "2"


if __name__ == "__main__":
    pytest.main([__file__]) 