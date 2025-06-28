#!/usr/bin/env python3
"""
Example demonstrating the improved KBfitXMLHelper API with automatic KBBlock generation.

This example shows how the new API reduces user input while automatically creating
KBBlocks for each ensemble and momentum combination.
"""

from SigmondSamplings import (
    EnsembleInfo, ObservableInfo, SamplingInfo,
    KBfitXMLHelper, BoxQuantizationInfo, DecayChannelInfo, 
    KElementInfo, FitParameterInfo, ExpressionFitForm,
    QuantizationCondition, EnergyFormat
)

def main():
    # Create ensemble infos
    ensemble_s46 = EnsembleInfo(
        "phirho_s46_t48_mp0100_mr0310|750000|1|46|46|46|48",
        750000, 25
    )
    
    ensemble_s47 = EnsembleInfo(
        "phirho_s47_t48_mp0100_mr0310|700000|1|47|47|47|48", 
        700000, 25
    )
    
    # Create observable infos (these would typically come from loading sampling files)
    observables = [
        # s46 energy shifts - PSQ=0
        ObservableInfo("isosinglet_S=0_A1g_1_PSQ=0_dElab_2_ref", 0, "correlator", "re", ensemble_s46),
        ObservableInfo("isosinglet_S=0_A1g_1_PSQ=0_dElab_3_ref", 0, "correlator", "re", ensemble_s46),
        
        # s46 energy shifts - PSQ=1
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,0,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s46),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,0,1)_dElab_3_ref", 0, "correlator", "re", ensemble_s46),
        
        # s46 energy shifts - PSQ=2
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s46),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_2_ref", 0, "correlator", "re", ensemble_s46),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_4_ref", 0, "correlator", "re", ensemble_s46),
        
        # s46 energy shifts - PSQ=3
        ObservableInfo("isosinglet_S=0_A1_1_P=(1,1,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s46),
        ObservableInfo("isosinglet_S=0_A1_1_P=(1,1,1)_dElab_2_ref", 0, "correlator", "re", ensemble_s46),
        
        # s47 energy shifts - PSQ=0
        ObservableInfo("isosinglet_S=0_A1g_1_PSQ=0_dElab_2_ref", 0, "correlator", "re", ensemble_s47),
        ObservableInfo("isosinglet_S=0_A1g_1_PSQ=0_dElab_3_ref", 0, "correlator", "re", ensemble_s47),
        
        # s47 energy shifts - PSQ=1
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,0,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s47),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,0,1)_dElab_3_ref", 0, "correlator", "re", ensemble_s47),
        
        # s47 energy shifts - PSQ=2
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s47),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_2_ref", 0, "correlator", "re", ensemble_s47),
        ObservableInfo("isosinglet_S=0_A1_1_P=(0,1,1)_dElab_4_ref", 0, "correlator", "re", ensemble_s47),
        
        # s47 energy shifts - PSQ=3
        ObservableInfo("isosinglet_S=0_A1_1_P=(1,1,1)_dElab_1_ref", 0, "correlator", "re", ensemble_s47),
        ObservableInfo("isosinglet_S=0_A1_1_P=(1,1,1)_dElab_2_ref", 0, "correlator", "re", ensemble_s47),
    ]
    
    # Define non-interacting pairs for each energy shift observable
    # The user only needs to specify this physics information
    non_interacting_pairs = {}
    
    # Map observables to their non-interacting pairs
    for obs in observables:
        if "dElab_1_ref" in obs.name:
            if "PSQ=0" in obs.name:
                non_interacting_pairs[obs] = "phi(0)phi(0)"
            elif "P=(0,0,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(1)phi(0)"
            elif "P=(0,1,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(0)phi(2)"
            elif "P=(1,1,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(0)phi(3)"
                
        elif "dElab_2_ref" in obs.name:
            if "PSQ=0" in obs.name:
                non_interacting_pairs[obs] = "phi(1)phi(1)"
            elif "P=(0,1,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(1)phi(1)"
            elif "P=(1,1,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(2)phi(1)"
                
        elif "dElab_3_ref" in obs.name:
            if "PSQ=0" in obs.name:
                non_interacting_pairs[obs] = "phi(1)phi(1)"
            elif "P=(0,0,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(2)phi(1)"
                
        elif "dElab_4_ref" in obs.name:
            if "P=(0,1,1)" in obs.name:
                non_interacting_pairs[obs] = "phi(1)phi(3)"
    
    # Sampling configuration
    sampling_info = SamplingInfo("bootstrap", 2000, seed=0, boot_skip=0)
    
    # Physics configuration
    reference_particle = "Phi"
    particle_masses = {"phi": 1.0}
    
    # Sampling files
    sampling_files = [
        "/pi-mnt/latticeQCD/spectrum_analysis/channels/phirho/levels/john_results/s46/3fit_spectrum/data/samples/fit_spectrum_levels_sigmond-6-14-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5[/samplings]",
        "/pi-mnt/latticeQCD/spectrum_analysis/channels/phirho/levels/john_results/s47/3fit_spectrum/data/samples/fit_spectrum_levels_sigmond-6-14-25-no_ratio_interacting_vev_sub-Nbin25-SP-3tN-3t0-6tD_B-samplings.hdf5[/samplings]"
    ]
    
    # K-matrix configuration (same as before)
    k_elements = [
        KElementInfo(j_times_two=0, k_index1="L(0) 2S(0) chan(0)", k_index2="L(0) 2S(0) chan(0)")
    ]
    
    fit_forms = [
        ExpressionFitForm("0.5*sqrt(x^2 - 4.0)*(rho_mass_ref^2 - x^2)/(rho_mass_ref * Gamma_ref)")
    ]
    
    decay_channels = [
        DecayChannelInfo("phi", 0, "", 0, identical=True)
    ]
    
    starting_values = [
        FitParameterInfo("rho_mass_ref", 2.70425, 
                        KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)")),
        FitParameterInfo("Gamma_ref", 0.00151675,
                        KElementInfo(0, "L(0) 2S(0) chan(0)", "L(0) 2S(0) chan(0)"))
    ]
    
    # Create XML helper
    helper = KBfitXMLHelper()
    
    # Generate XML with automatic KBBlock creation!
    # The helper will:
    # 1. Extract unique ensembles from observables
    # 2. Group observables by (ensemble, momentum, irrep)
    # 3. Create BoxQuantization info from momentum patterns
    # 4. Auto-extract energy shifts using provided non-interacting pairs
    # 5. Generate KBBlocks for each combination
    xml_content = helper.create_spectrum_xml(
        project_name="PhiRhoSpectrum_s46_s47_auto",
        observables=observables,
        sampling_info=sampling_info,
        reference_particle=reference_particle,
        particle_masses=particle_masses,
        sampling_files=sampling_files,
        fit_forms=fit_forms,
        k_elements=k_elements,
        decay_channels=decay_channels,
        starting_values=starting_values,
        non_interacting_pairs=non_interacting_pairs,  # Only physics input needed!
        # Everything else uses sensible defaults:
        # - CM energy ranges are auto-set based on momentum
        # - Box quantizations are auto-created from observable names
        # - KBBlocks are auto-generated for each (ensemble, momentum) pair
        output_directory="../results/PhiRhoSpectrum_s46_s47_auto",
        output_file="input_spectrum_s46_47_auto.xml"
    )
    
    print("Generated spectrum XML with automatic KBBlock creation!")
    print(f"Output written to: input_spectrum_s46_47_auto.xml")
    print(f"Number of observables processed: {len(observables)}")
    
    # Show what was automatically detected
    ensemble_momentum_groups = helper.group_observables_by_ensemble_and_momentum(observables)
    print(f"\nAutomatically detected {len(ensemble_momentum_groups)} (ensemble, momentum) combinations:")
    for (ensemble_name, psq, irrep), obs_list in ensemble_momentum_groups.items():
        print(f"  - {ensemble_name}: PSQ={psq}, irrep={irrep} ({len(obs_list)} observables)")
    
    print("\nCompare this to the manual approach which would require:")
    print("  - Manually creating BoxQuantizationInfo for each momentum")
    print("  - Manually grouping observables by ensemble and momentum") 
    print("  - Manually creating LabFrameEnergyShiftInfo objects")
    print("  - Manually specifying CM energy ranges")
    print("  - Much more boilerplate code!")


if __name__ == "__main__":
    main() 