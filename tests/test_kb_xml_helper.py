"""
Unit tests for the KBfitXMLHelper module.
"""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import os

from SigmondSamplings import (
    KBfitXMLHelper, BoxQuantizationInfo, ParticleInfo, 
    KElementInfo, FitParameterInfo, SamplingInfo, EnsembleInfo, ObservableInfo
)


class TestBoxQuantizationInfo(unittest.TestCase):
    """Test BoxQuantizationInfo class."""
    
    def test_creation(self):
        """Test basic creation."""
        box_quant = BoxQuantizationInfo("ar", 0, "A1g", 0)
        self.assertEqual(box_quant.momentum_ray, "ar")
        self.assertEqual(box_quant.momentum_int_squared, 0)
        self.assertEqual(box_quant.lg_irrep, "A1g")
        self.assertEqual(box_quant.lmax_values, "0")
    
    def test_lmax_values_conversion(self):
        """Test that lmax_values is converted to string."""
        box_quant = BoxQuantizationInfo("oa", 1, "A1", 2)
        self.assertEqual(box_quant.lmax_values, "2")


class TestParticleInfo(unittest.TestCase):
    """Test ParticleInfo class."""
    
    def test_creation(self):
        """Test basic creation."""
        particle = ParticleInfo("lambda", 1, True)
        self.assertEqual(particle.name, "lambda")
        self.assertEqual(particle.spin_times_two, 1)
        self.assertTrue(particle.identical)
    
    def test_default_identical(self):
        """Test default identical value."""
        particle = ParticleInfo("phi", 0)
        self.assertFalse(particle.identical)


class TestKElementInfo(unittest.TestCase):
    """Test KElementInfo class."""
    
    def test_creation(self):
        """Test basic creation."""
        k_elem = KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)")
        self.assertEqual(k_elem.j_times_two, 0)
        self.assertEqual(k_elem.k_index1, "L(0) 2S(0) chan(0)")
        self.assertEqual(k_elem.k_index2, "L(0) 2S(0) chan(0)")


class TestFitParameterInfo(unittest.TestCase):
    """Test FitParameterInfo class."""
    
    def test_creation_with_k_element(self):
        """Test creation with K element info."""
        k_elem = KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)")
        param = FitParameterInfo("mass", 2.71, k_elem)
        self.assertEqual(param.parameter_name, "mass")
        self.assertEqual(param.starting_value, 2.71)
        self.assertEqual(param.k_element_info, k_elem)
        self.assertIsNone(param.polynomial_power)
    
    def test_creation_with_polynomial_power(self):
        """Test creation with polynomial power."""
        param = FitParameterInfo("coeff", 3.9, polynomial_power=0)
        self.assertEqual(param.polynomial_power, 0)


class TestKBfitXMLHelper(unittest.TestCase):
    """Test KBfitXMLHelper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.helper = KBfitXMLHelper()
        
        # Create test sampling info
        self.sampling_info = SamplingInfo("bootstrap", 2000, 1234, 0)
        
        # Create test ensemble info
        self.ensemble_info = EnsembleInfo(
            "test_ensemble|1000|1|24|24|24|48", 
            1000, 100, 
            {"Rebin": "10"}
        )
        
        # Create test observable info
        self.observable_info = ObservableInfo(
            "isosinglet_S=0_A1g_PSQ=0_elab_1_ref", 0, "n", "re", self.ensemble_info
        )
        
        # Create test box quantization
        self.box_quant = BoxQuantizationInfo("ar", 0, "A1g", 0)
        
        # Create test particle info
        self.particle = ParticleInfo("lambda", 1, True)
        
        # Create test K element info
        self.k_element = KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)")
        
        # Create test fit parameter
        self.fit_param = FitParameterInfo("mass_ref", 2.71, self.k_element)
    
    def test_create_element(self):
        """Test _create_element method."""
        elem = self.helper._create_element("TestTag", "test_text")
        self.assertEqual(elem.tag, "TestTag")
        self.assertEqual(elem.text, "test_text")
    
    def test_create_element_with_parent(self):
        """Test _create_element with parent."""
        parent = ET.Element("Parent")
        child = self.helper._create_element("Child", "child_text", parent)
        self.assertEqual(len(parent), 1)
        self.assertEqual(parent[0].tag, "Child")
        self.assertEqual(parent[0].text, "child_text")
    
    def test_create_mcsamplinginfo_bootstrap(self):
        """Test _create_mcsamplinginfo_element for bootstrap."""
        elem = self.helper._create_mcsamplinginfo_element(self.sampling_info)
        self.assertEqual(elem.tag, "MCSamplingInfo")
        
        bootstrap = elem.find("Bootstrapper")
        self.assertIsNotNone(bootstrap)
        self.assertEqual(bootstrap.find("NumberResamplings").text, "2000")
        self.assertEqual(bootstrap.find("Seed").text, "1234")
        self.assertEqual(bootstrap.find("BootSkip").text, "0")
    
    def test_create_mcsamplinginfo_jackknife(self):
        """Test _create_mcsamplinginfo_element for jackknife."""
        jk_sampling = SamplingInfo("jackknife", 100)
        elem = self.helper._create_mcsamplinginfo_element(jk_sampling)
        self.assertEqual(elem.tag, "MCSamplingInfo")
        
        jackknife = elem.find("Jackkniffer")
        self.assertIsNotNone(jackknife)
        self.assertEqual(jackknife.find("NumberResamplings").text, "100")
    
    def test_create_mcsamplinginfo_jackknife_simple(self):
        """Test _create_mcsamplinginfo_element for simple jackknife."""
        jk_sampling = SamplingInfo("jackknife", 0)
        elem = self.helper._create_mcsamplinginfo_element(jk_sampling)
        self.assertEqual(elem.tag, "MCSamplingInfo")
        
        jackknife = elem.find("Jackknife")
        self.assertIsNotNone(jackknife)
    
    def test_create_mcbinsinfo_element(self):
        """Test _create_mcbinsinfo_element method."""
        elem = self.helper._create_mcbinsinfo_element(self.ensemble_info)
        self.assertEqual(elem.tag, "MCBinsInfo")
        self.assertEqual(elem.find("MCEnsembleInfo").text, self.ensemble_info.ensemble_name)
        self.assertEqual(elem.find("NumberOfMeasurements").text, str(self.ensemble_info.num_measurements))
        self.assertEqual(elem.find("NumberOfBins").text, str(self.ensemble_info.num_bins))
        
        tweak = elem.find("TweakEnsemble")
        self.assertIsNotNone(tweak)
        self.assertEqual(tweak.find("Rebin").text, "10")
    
    def test_create_mcensemble_parameters(self):
        """Test _create_mcensemble_parameters method."""
        particle_masses = {"lambda": 1.0, "phi": 1.0}
        elem = self.helper._create_mcensemble_parameters(
            self.ensemble_info, "Lambda", particle_masses
        )
        
        self.assertEqual(elem.tag, "MCEnsembleParameters")
        self.assertEqual(elem.find("MCEnsembleInfo").text, self.ensemble_info.ensemble_name)
        
        ref_mass = elem.find("ReferenceMassTimeSpacingProduct")
        self.assertIsNotNone(ref_mass)
        self.assertEqual(ref_mass.find("MCObs").text, "Lambda(0)_elab 0")
        
        # Check particle masses
        masses = elem.findall("ParticleMass")
        self.assertEqual(len(masses), 2)
    
    def test_create_kbblock(self):
        """Test _create_kbblock method."""
        lab_energies = ["obs_1", "obs_2"]
        elem = self.helper._create_kbblock(self.ensemble_info, self.box_quant, lab_energies)
        
        self.assertEqual(elem.tag, "KBBlock")
        self.assertEqual(elem.find("MCEnsembleInfo").text, self.ensemble_info.ensemble_name)
        
        box_quant = elem.find("BoxQuantization")
        self.assertIsNotNone(box_quant)
        self.assertEqual(box_quant.find("TotalMomentumRay").text, "ar")
        self.assertEqual(box_quant.find("TotalMomentumIntSquared").text, "0")
        self.assertEqual(box_quant.find("LGIrrep").text, "A1g")
        self.assertEqual(box_quant.find("LmaxValues").text, "0")
        
        energies = elem.findall("LabFrameEnergy")
        self.assertEqual(len(energies), 2)
    
    def test_create_print_kbblock(self):
        """Test _create_print_kbblock method."""
        energy_range = (1.9, 2.25, 0.001)
        elem = self.helper._create_print_kbblock(self.ensemble_info, self.box_quant, energy_range)
        
        self.assertEqual(elem.tag, "KBBlock")
        self.assertEqual(elem.find("LabFrameEnergyMin").text, "1.9")
        self.assertEqual(elem.find("LabFrameEnergyMax").text, "2.25")
        self.assertEqual(elem.find("LabFrameEnergyInc").text, "0.001")
    
    def test_create_decay_channels(self):
        """Test _create_decay_channels method."""
        particles = [self.particle, ParticleInfo("phi", 0, False)]
        elem = self.helper._create_decay_channels(particles)
        
        self.assertEqual(elem.tag, "DecayChannels")
        channels = elem.findall("DecayChannelInfo")
        self.assertEqual(len(channels), 2)
        
        # Check first particle (identical)
        first_channel = channels[0]
        self.assertEqual(first_channel.find("Particle1Name").text, "lambda")
        self.assertEqual(first_channel.find("Spin1TimesTwo").text, "1")
        self.assertIsNotNone(first_channel.find("Identical"))
        
        # Check second particle (not identical)
        second_channel = channels[1]
        self.assertEqual(second_channel.find("Particle1Name").text, "phi")
        self.assertEqual(second_channel.find("Spin1TimesTwo").text, "0")
        self.assertIsNone(second_channel.find("Identical"))
    
    def test_create_kbobservables(self):
        """Test _create_kbobservables method."""
        sampling_files = ["/path/to/file1.hdf5[/samplings]", "/path/to/file2.hdf5[/samplings]"]
        elem = self.helper._create_kbobservables(self.sampling_info, self.ensemble_info, sampling_files)
        
        self.assertEqual(elem.tag, "KBObservables")
        
        # Check sampling info
        sampling_elem = elem.find("MCSamplingInfo")
        self.assertIsNotNone(sampling_elem)
        
        # Check bins info
        bins_elem = elem.find("MCBinsInfo")
        self.assertIsNotNone(bins_elem)
        
        # Check verbose flag
        verbose = elem.find("Verbose")
        self.assertIsNotNone(verbose)
        
        # Check sampling data files
        sampling_data = elem.find("SamplingData")
        self.assertIsNotNone(sampling_data)
        files = sampling_data.findall("FileName")
        self.assertEqual(len(files), 2)
    
    def test_create_single_channel_xml(self):
        """Test create_single_channel_xml method."""
        decay_channels = [self.particle]
        box_quantizations = [self.box_quant]
        sampling_files = ["/path/to/file.hdf5[/samplings]"]
        particle_masses = {"lambda": 1.0}
        observables = [self.observable_info]
        
        xml_str = self.helper.create_single_channel_xml(
            project_name="TestSingleChannel",
            observables=observables,
            sampling_info=self.sampling_info,
            reference_particle="Lambda",
            particle_masses=particle_masses,
            decay_channels=decay_channels,
            box_quantizations=box_quantizations,
            sampling_files=sampling_files,
            output_stub="test_output"
        )
        
        # Parse the XML to check structure
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "KBFit")
        
        # Check Initialize section
        init = root.find("Initialize")
        self.assertIsNotNone(init)
        self.assertEqual(init.find("ProjectName").text, "TestSingleChannel")
        
        # Check Task section
        task = root.find("TaskSequence/Task")
        self.assertIsNotNone(task)
        self.assertEqual(task.find("Action").text, "DoSingleChannel")
        self.assertEqual(task.find("OutputStub").text, "test_output")
    
    def test_create_detres_xml(self):
        """Test create_detres_xml method."""
        observables = [self.observable_info]
        particle_masses = {"phi": 1.0}
        box_quantizations = [self.box_quant]
        sampling_files = ["/path/to/file.hdf5[/samplings]"]
        
        xml_str = self.helper.create_detres_xml(
            project_name="TestDetRes",
            observables=observables,
            sampling_info=self.sampling_info,
            reference_particle="Phi",
            particle_masses=particle_masses,
            box_quantizations=box_quantizations,
            sampling_files=sampling_files
        )
        
        # Parse the XML to check structure
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "KBFit")
        
        # Check Initialize section
        init = root.find("Initialize")
        self.assertIsNotNone(init)
        self.assertEqual(init.find("ProjectName").text, "TestDetRes")
        
        # Check Task section
        task = root.find("TaskSequence/Task")
        self.assertIsNotNone(task)
        self.assertEqual(task.find("Action").text, "DoFit")
        self.assertEqual(task.find("Type").text, "DeterminantResidualFit")
        
        # Check DeterminantResidualFit section
        detres = task.find("DeterminantResidualFit")
        self.assertIsNotNone(detres)
        self.assertEqual(detres.find("OmegaMu").text, "8.0")
        self.assertEqual(detres.find("QuantizationCondition").text, "KtildeinvB")
    
    def test_create_print_xml(self):
        """Test create_print_xml method."""
        decay_channels = [self.particle]
        k_elements = [self.k_element]
        polynomial_powers = {"L(0) 2S(0) chan(0)": [0, 2]}
        starting_values = [FitParameterInfo("coeff_0", 3.9, self.k_element, 0)]
        energy_range = (1.9, 2.25, 0.001)
        sampling_files = ["/path/to/file.hdf5[/samplings]"]
        particle_masses = {"lambda": 1.0}
        observables = [self.observable_info]
        
        xml_str = self.helper.create_print_xml(
            project_name="TestPrint",
            observables=observables,
            sampling_info=self.sampling_info,
            reference_particle="Lambda",
            particle_masses=particle_masses,
            energy_range=energy_range,
            decay_channels=decay_channels,
            k_elements=k_elements,
            polynomial_powers=polynomial_powers,
            starting_values=starting_values,
            sampling_files=sampling_files,
            output_stub="test_print"
        )
        
        # Parse the XML to check structure
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "KBFit")
        
        # Check Task section
        task = root.find("TaskSequence/Task")
        self.assertIsNotNone(task)
        self.assertEqual(task.find("Action").text, "DoPrint")
        
        # Check Print section
        print_elem = task.find("Print")
        self.assertIsNotNone(print_elem)
        self.assertEqual(print_elem.find("OmegaMu").text, "0.5")
        self.assertEqual(print_elem.find("QuantizationCondition").text, "StildeCB")
        
        # Check KtildeMatrix
        ktilde = print_elem.find("KtildeMatrix")
        self.assertIsNotNone(ktilde)
    
    def test_prettify_xml(self):
        """Test _prettify_xml method."""
        elem = ET.Element("Root")
        child = ET.SubElement(elem, "Child")
        child.text = "text"
        
        pretty_xml = self.helper._prettify_xml(elem)
        self.assertIn("<Root>", pretty_xml)
        self.assertIn("    <Child>text</Child>", pretty_xml)
        self.assertIn("</Root>", pretty_xml)
    
    def test_extract_momentum_info_from_observable(self):
        """Test extract_momentum_info_from_observable static method."""
        # Test PSQ format
        obs_name = "isosinglet_S=0_A1g_1_PSQ=2_elab_1_ref"
        result = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
        self.assertEqual(result, (2, "A1g"))
        
        # Test P= format
        obs_name = "isosinglet_S=0_A1_1_P=(0,1,1)_elab_1_ref"
        result = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
        self.assertEqual(result, (2, "A1"))
        
        # Test no match
        obs_name = "invalid_observable_name"
        result = KBfitXMLHelper.extract_momentum_info_from_observable(obs_name)
        self.assertIsNone(result)
    
    def test_group_observables_by_momentum(self):
        """Test group_observables_by_momentum static method."""
        obs1 = ObservableInfo("isosinglet_S=0_A1g_PSQ=0_elab_1_ref", 0, "n", "re", self.ensemble_info)
        obs2 = ObservableInfo("isosinglet_S=0_A1g_PSQ=0_elab_2_ref", 0, "n", "re", self.ensemble_info)
        obs3 = ObservableInfo("isosinglet_S=0_A1_P=(0,0,1)_elab_1_ref", 0, "n", "re", self.ensemble_info)
        
        grouped = KBfitXMLHelper.group_observables_by_momentum([obs1, obs2, obs3])
        
        self.assertIn((0, "A1g"), grouped)
        self.assertIn((1, "A1"), grouped)
        self.assertEqual(len(grouped[(0, "A1g")]), 2)
        self.assertEqual(len(grouped[(1, "A1")]), 1)
    
    def test_create_box_quantization_from_momentum(self):
        """Test create_box_quantization_from_momentum static method."""
        box_quant = KBfitXMLHelper.create_box_quantization_from_momentum(0, "A1g")
        self.assertEqual(box_quant.momentum_ray, "ar")
        self.assertEqual(box_quant.momentum_int_squared, 0)
        self.assertEqual(box_quant.lg_irrep, "A1g")
        
        box_quant = KBfitXMLHelper.create_box_quantization_from_momentum(2, "A1")
        self.assertEqual(box_quant.momentum_ray, "pd")
        self.assertEqual(box_quant.momentum_int_squared, 2)
        self.assertEqual(box_quant.lg_irrep, "A1")
    
    def test_xml_file_output(self):
        """Test XML output to file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_output.xml"
            
            observables = [self.observable_info]
            particle_masses = {"phi": 1.0}
            box_quantizations = [self.box_quant]
            sampling_files = ["/path/to/file.hdf5[/samplings]"]
            
            xml_str = self.helper.create_detres_xml(
                project_name="TestFileOutput",
                observables=observables,
                sampling_info=self.sampling_info,
                reference_particle="Phi",
                particle_masses=particle_masses,
                box_quantizations=box_quantizations,
                sampling_files=sampling_files,
                output_file=str(output_file)
            )
            
            # Check file was created
            self.assertTrue(output_file.exists())
            
            # Check file content matches returned string
            file_content = output_file.read_text()
            self.assertEqual(file_content, xml_str)


if __name__ == '__main__':
    unittest.main() 