# SigmondSamplings: New Modular KB XML Helper API

## 🎉 Overview

The KB XML helper has been completely refactored with a new modular design that provides:

- **✅ Clean, organized imports** - No more messy import statements
- **✅ Self-contained data structures** - Everything needed is in one place
- **✅ Modular helper methods** - Reusable components for all XML types
- **✅ Dual data structures** - Keep your convenient dicts AND get XML-ready objects
- **✅ Consistent API patterns** - All three XML types follow the same structure

## 📦 Improved Import Structure

### Before (messy):
```python
from SigmondSamplings import KBfitXMLHelper, MinimizerInfo, MinimizerMethod, RootFinderConfig, DecayChannelInfo, KElementInfo, ExpressionFitForm, FitParameterInfo
```

### After (clean & organized):
```python
from SigmondSamplings import (
    # Core data classes
    KBfitXMLHelper, EnsembleInfo, SamplingInfo, ObservableInfo,
    # XML generation classes  
    ExpressionFitForm, KElementInfo, DecayChannelInfo, ParticleInfo,
    LabFrameEnergyShiftInfo, LabFrameEnergyInfo, LabFrameEnergyRangeInfo,
    # Configuration classes
    MinimizerInfo, MinimizerMethod, RootFinderConfig, 
    QuantizationCondition, EnergyFormat, OutputMode, Verbosity
)
```

## 🔧 Key Improvements

### 1. Self-Contained ExpressionFitForm
**Before:** Required separate `FitParameterInfo` objects
```python
fit_form = ExpressionFitForm(expression="...")
param1 = FitParameterInfo(name="param1", value=1.0, k_element=k_info)
param2 = FitParameterInfo(name="param2", value=2.0, k_element=k_info)
```

**After:** Everything in one place
```python
fit_forms = [ExpressionFitForm(
    expression="0.5*sqrt(x^2 - 4.0)*(rho_mass_ref^2 - x^2)/(rho_mass_ref * Gamma_ref)",
    params_with_starting_values={
        "rho_mass_ref": 2.705,
        "Gamma_ref": 0.0015
    },
    k_index=k_element_info
)]
```

### 2. Modular Helper Methods
The new API provides reusable helper methods:

- `create_fit_task_elements()` - For fit-specific configuration
- `create_print_task_elements()` - For print-specific configuration  
- `create_common_task_elements()` - For shared configuration
- `create_kbblocks_and_observables()` - For unified block/observable creation

### 3. Dual Data Structures
Keep your convenient analysis structures AND get XML-ready objects:

```python
# Your convenient dict structure (unchanged)
energy_dict = {
    psq: {
        energy_type: {
            level: (NI_list, SigmondSampling)
        }
    }
}

# PLUS new XML-ready structures
energy_shift_infos = [LabFrameEnergyShiftInfo(...), ...]
spectrum_ensemble_data = [(ensemble_info, sampling_info, energy_shifts), ...]
```

## 📋 Updated Usage Pattern

### Data Loading (maintains backward compatibility)
```python
def get_relevant_ensemble_data(L, resamp_method, energy_types, ref):
    # ... your existing logic ...
    
    # NEW: Also create XML-ready structures during processing
    energy_shift_infos = []
    if energy_type == "dElab":
        ni_particles = [ParticleInfo(name=p.split('(')[0], psq=extract_psq(p)) 
                       for p in NI]
        energy_shift_info = LabFrameEnergyShiftInfo(
            mcobs=sampling.observable_info,
            non_interacting_pair=ni_particles
        )
        energy_shift_infos.append(energy_shift_info)
    
    return ensemble_info, sampling_info, energy_dict, energy_shift_infos
```

### XML Generation (much cleaner)
```python
# Configure components
minimizer_config = MinimizerInfo(method=MinimizerMethod.MINUIT2_MIGRAD, ...)
root_finder_config = RootFinderConfig(initial_step_percent=1e-2, ...)
decay_channels = [DecayChannelInfo(particle_1="phi", ...)]
fit_forms = [ExpressionFitForm(...)]
particle_masses = [ParticleInfo(name="phi", mass=1.0)]

# Generate XML with new clean API
xml_helper = KBfitXMLHelper()
xml_content = xml_helper.create_spectrum_xml(
    xml_output_file="output.xml",
    project_name="MyProject",
    ensemble_data=spectrum_ensemble_data,  # New XML-ready structure
    reference_particle="Phi",
    ensemble_particle_infos=particle_masses,
    sampling_files=sampling_files,
    fit_forms=fit_forms,
    decay_channels=decay_channels,
    # ... other parameters
)
```

## 🔍 Data Access Examples

The convenient dictionary structure is preserved for easy analysis:

```python
# Access energy levels for analysis
L = 40
if L in relevant_data:
    ensemble_info, sampling_info, energy_dict, energy_shifts = relevant_data[L]
    
    # Get specific energy values
    for psq in energy_dict:
        if 'dElab' in energy_dict[psq]:
            for level in energy_dict[psq]['dElab']:
                ni_list, sampling = energy_dict[psq]['dElab'][level]
                print(f"PSQ={psq}, level={level}: dElab = {sampling.mean:.6f} ± {sampling.error:.6f}")
                print(f"Non-interacting: {ni_list}")

# XML-ready objects are also available
print(f"Generated {len(energy_shifts)} energy shift objects for XML")
```

## 🎯 All Three XML Types Now Consistent

### Spectrum Fit
```python
xml_content = xml_helper.create_spectrum_xml(
    xml_output_file="spectrum.xml",
    ensemble_data=spectrum_ensemble_data,
    # ... parameters
)
```

### Determinant Residual Fit  
```python
xml_content = xml_helper.create_detres_xml(
    project_name="MyDetRes",
    ensemble_data=detres_ensemble_data,
    # ... parameters  
)
```

### Print Task
```python
xml_content = xml_helper.create_print_xml(
    project_name="MyPrint", 
    ensemble_data=print_ensemble_data,
    output_stub="output_stub",
    # ... parameters
)
```

## ✨ Benefits

1. **Cleaner Code**: Organized imports and self-contained structures
2. **Better Maintainability**: Modular design with reusable components
3. **Backward Compatibility**: Your existing data access patterns still work
4. **Forward Compatibility**: New XML-ready structures for generation
5. **Consistency**: All three XML types follow the same pattern
6. **Flexibility**: Easy to extend with new fit forms or task types

## 🚀 Migration Guide

1. **Update imports** - Use the new organized import structure
2. **Update fit forms** - Use the new self-contained `ExpressionFitForm`
3. **Update data loading** - Modify to return both dict and XML structures
4. **Update XML generation** - Use the new clean API calls
5. **Test thoroughly** - Verify both analysis and XML generation work

The new API maintains all the functionality you need while providing a much cleaner, more maintainable structure! 