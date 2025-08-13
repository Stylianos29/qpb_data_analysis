#!/usr/bin/env python3
"""
Script to analyze HDF5 correlator data and create plots grouped by KL_diagonal_order.

This script uses the HDF5Analyzer to:
1. Load and inspect the HDF5 file structure
2. Group data by KL_diagonal_order parameter
3. Extract PCAC mass correlator data with automatic gvar merging
4. Create plots with different colors/shapes for each group
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from library.data.hdf5_analyzer import HDF5Analyzer

def analyze_and_plot_correlators(hdf5_file_path):
    """
    Main function to analyze HDF5 correlator data and create plots.
    
    Args:
        hdf5_file_path: Path to the HDF5 file
    """
    # Initialize the analyzer
    print("Loading HDF5 file...")
    analyzer = HDF5Analyzer(hdf5_file_path)
    
    # Print basic information about the file
    print(f"\n{analyzer}")
    print("\n" + "="*60)
    
    # Generate and display uniqueness report
    print("PARAMETER AND DATASET ANALYSIS:")
    print("="*60)
    print(analyzer.generate_uniqueness_report(max_width=80, separate_by_type=True))
    
    # Check if KL_diagonal_order exists and is multivalued
    if 'KL_diagonal_order' not in analyzer.list_of_multivalued_tunable_parameter_names:
        print("\nWarning: 'KL_diagonal_order' not found in multivalued parameters.")
        print("Available multivalued parameters:")
        for param in analyzer.list_of_multivalued_tunable_parameter_names:
            print(f"  - {param}")
        
        # Let user choose an alternative or continue anyway
        return analyzer
    
    # Get unique values of KL_diagonal_order
    kl_values = analyzer.unique_values('KL_diagonal_order', print_output=True)
    print(f"\nFound {len(kl_values)} unique KL_diagonal_order values: {kl_values}")
    
    # Group data by all multivalued parameters EXCEPT KL_diagonal_order
    print("\nGrouping data by parameters (excluding KL_diagonal_order)...")
    other_multivalued_params = [p for p in analyzer.reduced_multivalued_tunable_parameter_names_list 
                               if p != 'KL_diagonal_order']
    
    grouped_data = analyzer.group_by_multivalued_tunable_parameters(
        filter_out_parameters_list=['KL_diagonal_order', 'Number_of_gauge_configurations'],  # Exclude KL_diagonal_order from grouping
        verbose=True
    )
    
    print(f"Found {len(grouped_data)} unique parameter combinations (excluding KL_diagonal_order)")
    
    # Reorganize data: for each unique combination of other parameters,
    # collect all KL_diagonal_order values
    plot_groups = {}  # Structure: {other_params_key: {kl_value: [correlators]}}
    plot_group_info = {}  # Store parameter info for each plot group
    
    for group_key, group_paths in grouped_data.items():
        # Get parameter information for this group
        multivalued_params = [p for p in analyzer.reduced_multivalued_tunable_parameter_names_list 
                             if p not in ['KL_diagonal_order', 'Number_of_gauge_configurations']]
        
        # Create parameter info dictionary (excluding KL_diagonal_order)
        group_params = {}
        if group_key and len(group_key) == len(multivalued_params):
            for i, param_name in enumerate(multivalued_params):
                group_params[param_name] = group_key[i]
        
        print(f"\nProcessing parameter combination: {group_params} ({len(group_paths)} configurations)")
        
        # For this parameter combination, collect data for each KL_diagonal_order value
        kl_data = {}  # {kl_value: [correlators]}
        
        for group_path in group_paths:
            try:
                # Get all parameters for this specific group to find KL_diagonal_order
                full_params = analyzer.parameters_for_group(group_path)
                kl_value = full_params.get('KL_diagonal_order')
                
                if kl_value is not None:
                    # Get the gvar-merged PCAC mass correlator data
                    pcac_data = analyzer.dataset_values(
                        'Jackknife_average_of_pion_effective_mass_correlator',
                        return_gvar=True,
                        group_path=group_path
                    )
                    
                    if kl_value not in kl_data:
                        kl_data[kl_value] = []
                    kl_data[kl_value].append(pcac_data)
                
            except ValueError as e:
                print(f"  Warning: Could not extract data from {group_path}: {e}")
                continue
        
        if kl_data:
            plot_groups[group_key] = kl_data
            plot_group_info[group_key] = group_params
            
            # Print summary for this group
            for kl_val, correlators in kl_data.items():
                print(f"  n={kl_val}: {len(correlators)} configurations")
    
    # Create plots
    if plot_groups:
        create_correlator_plots(plot_groups, plot_group_info)
    else:
        print("No correlator data found to plot!")
    
    return analyzer

def create_correlator_plots(plot_groups, plot_group_info):
    """
    Create plots of PCAC mass correlators with all KL_diagonal_order values in each plot.
    
    Args:
        plot_groups: Dictionary with parameter combinations as keys
                    and {kl_value: [correlators]} as values
        plot_group_info: Dictionary with parameter info for each plot group
    """
    from pathlib import Path
    
    # Create output directory
    output_dir = Path("./test_HDF5Plotter_output")
    output_dir.mkdir(exist_ok=True)
    
    # Color and marker cycles for KL_diagonal_order values (n=1 to n=7)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))  # 7 colors for n=1 to n=7
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h']
    
    print(f"\nCreating plots for {len(plot_groups)} parameter combination groups...")
    
    plots_created = 0
    
    for group_key, kl_data in plot_groups.items():
        # Set up single plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        
        # Get parameter info for this plot (excluding KL_diagonal_order)
        group_info = plot_group_info[group_key]
        
        print(f"  Plot {plots_created + 1}: {group_info}")
        print(f"    KL_diagonal_order values in this plot: {sorted(kl_data.keys())}")
        
        # Plot each KL_diagonal_order value with consistent colors/markers
        for kl_value, correlators in kl_data.items():
            # Use consistent color/marker based on KL_diagonal_order
            color_idx = int(kl_value) - 1  # KL values are 1-7, array indices 0-6
            color = colors[color_idx]
            marker = markers[color_idx % len(markers)]
            
            print(f"      n={kl_value}: {len(correlators)} configurations")
            
            # Plot individual correlators (semi-transparent)
            for j, correlator in enumerate(correlators):
                if hasattr(correlator, '__len__') and len(correlator) > 0:
                    # Check if this is a gvar array
                    try:
                        import gvar
                        if isinstance(correlator[0], gvar.GVar):
                            # Extract mean and error for plotting
                            mean_vals = gvar.mean(correlator)
                            error_vals = gvar.sdev(correlator)
                            x_vals = np.arange(len(mean_vals))
                            
                            # Plot with error bars (individual configs - lighter)
                            ax.errorbar(x_vals, mean_vals, yerr=error_vals, 
                                      color=color, marker=marker,
                                        # alpha=0.4, 
                                      markersize=6,
                                      capsize=5,
                                      linestyle='None',
                                        # markerfacecolor="none",
                                    #   linewidth=1,
                                      label=f'{kl_value}')
                                        # if j == 0 else "")
                        else:
                            # Regular array
                            x_vals = np.arange(len(correlator))
                            # ax.plot(x_vals, correlator, color=color, marker=marker, 
                            #       alpha=0.4, markersize=6, linewidth=1,
                            #       label=f'n={kl_value}' if j == 0 else "")
                            
                    except ImportError:
                        # gvar not available, treat as regular array
                        x_vals = np.arange(len(correlator))
                        # ax.plot(x_vals, correlator, color=color, marker=marker, 
                        #       alpha=0.4, markersize=6, linewidth=1,
                        #       label=f'n={kl_value}' if j == 0 else "")
            
            # Compute and plot average (if multiple correlators)
            if len(correlators) > 1:
                try:
                    import gvar
                    # Stack correlators and compute average
                    stacked = np.array(correlators)
                    if len(stacked.shape) == 2:  # 2D array of correlators
                        avg_correlator = np.mean(stacked, axis=0)
                        x_vals = np.arange(len(avg_correlator))
                        
                        # Check if gvar
                        if hasattr(avg_correlator, '__len__') and len(avg_correlator) > 0:
                            if isinstance(avg_correlator[0], gvar.GVar):
                                mean_vals = gvar.mean(avg_correlator)
                                error_vals = gvar.sdev(avg_correlator)
                                
                                ax.errorbar(x_vals, mean_vals, yerr=error_vals, 
                                          color=color, marker=marker, alpha=1.0, 
                                          markersize=7, linestyle='None',
                                          label=f'n={kl_value} (avg)')
                            else:
                                ax.plot(x_vals, avg_correlator, color=color, marker=marker, 
                                      alpha=1.0, markersize=7, linestyle='None',
                                      label=f'n={kl_value} (avg)')
                except:
                    pass  # Skip averaging if it fails
        
        # Create title with other multivalued parameters (excluding KL_diagonal_order)
        title_parts = []
        MSCG_epsilon = 0
        Bare_mass = ''
        Kernel_operator_type = ''
        for param, value in group_info.items():
            if param in ['Bare_mass', 'Kernel_operator_type', 'MSCG_epsilon']:
                if param == 'Bare_mass':
                    Bare_mass = value
                elif param == 'Kernel_operator_type':
                    Kernel_operator_type = value
                elif param == 'MSCG_epsilon':
                    MSCG_epsilon = value
                    
                    # param='a$\\mathbf{m}$'
        #             continue
        #         title_parts.append(f"{param}={value}")
        title = f"{Kernel_operator_type} Kernel, a$\\mathbf{{m}}$ = {Bare_mass}"
        # title = "PCAC Mass Correlators by n (KL_diagonal_order)"
        # title = ""
        # if title_parts:
        #     title += f"\n{', '.join(title_parts)}"
        
        # Customize plot
        ax.set_xlabel('$t/a$', fontsize=15)
        ax.set_ylabel('a$m_{\mathrm{eff.}}(t)$', fontsize=15)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3.0)
        
        # Sort legend by n values
        handles, labels = ax.get_legend_handles_labels()
        # Sort by the n value in the label
        sorted_items = sorted(zip(handles, labels), key=lambda x: int(x[1].split()[0]))
        sorted_handles, sorted_labels = zip(*sorted_items) if sorted_items else ([], [])
        
        legend = ax.legend(sorted_handles, sorted_labels, fontsize=13, loc='upper center', ncols=3)
        legend.set_title('n=', prop={"size": 14})
        
        plt.tight_layout()
        
        # Save plot
        filename = f"Pion_effective_mass_by_KL_diagonal_order_{Kernel_operator_type}_{MSCG_epsilon}_{Bare_mass}.png" #_{plots_created + 1:03d}
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filepath}")
        
        plt.close()  # Close figure to save memory
        plots_created += 1
    
    print(f"\nTotal plots created: {plots_created}")
    print(f"Output directory: {output_dir.absolute()}")

def inspect_file_structure(hdf5_file_path):
    """
    Helper function to inspect the HDF5 file structure in detail.
    
    Args:
        hdf5_file_path: Path to the HDF5 file
    """
    analyzer = HDF5Analyzer(hdf5_file_path)
    
    print("DETAILED FILE INSPECTION:")
    print("="*60)
    print(f"File: {hdf5_file_path}")
    print(f"Total groups: {len(analyzer._all_deepest_groups)}")
    print()
    
    print("TUNABLE PARAMETERS:")
    print("-" * 30)
    print("Single-valued:")
    for param in analyzer.list_of_single_valued_tunable_parameter_names:
        value = analyzer.unique_value_columns_dictionary[param]
        print(f"  {param}: {value}")
    
    print("\nMulti-valued:")
    for param in analyzer.list_of_multivalued_tunable_parameter_names:
        count = analyzer.multivalued_columns_count_dictionary[param]
        values = analyzer.unique_values(param)
        print(f"  {param}: {count} values -> {values}")
    
    print("\nOUTPUT QUANTITIES (DATASETS):")
    print("-" * 30)
    for dataset in analyzer.list_of_output_quantity_names_from_dataframe[:10]:  # Show first 10
        if dataset in analyzer.list_of_single_valued_output_quantity_names:
            print(f"  {dataset}: single-valued")
        else:
            count = analyzer.multivalued_columns_count_dictionary.get(dataset, 'unknown')
            print(f"  {dataset}: multi-valued ({count})")
    
    if len(analyzer.list_of_output_quantity_names_from_dataframe) > 10:
        remaining = len(analyzer.list_of_output_quantity_names_from_dataframe) - 10
        print(f"  ... and {remaining} more datasets")
    
    # Check for gvar pairs
    print(f"\nGVAR DATASET PAIRS:")
    print("-" * 30)
    if hasattr(analyzer, '_gvar_dataset_pairs'):
        for base_name, (mean_name, error_name) in analyzer._gvar_dataset_pairs.items():
            print(f"  {base_name}: {mean_name} + {error_name}")
    else:
        print("  No gvar pairs detected")
    
    return analyzer

# Example usage
if __name__ == "__main__":
    # Update this path to your actual HDF5 file
    file_path = "../../../data_files/processed/invert/KL_several_config_varying_n/correlators_jackknife_analysis.h5"
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with the correct path to your HDF5 file.")
    else:
        # First, inspect the file structure
        print("STEP 1: Inspecting file structure...")
        inspect_analyzer = inspect_file_structure(file_path)
        print("\n" + "="*80 + "\n")
        
        # Then run the full analysis and plotting
        print("STEP 2: Running full analysis and plotting...")
        main_analyzer = analyze_and_plot_correlators(file_path)
        
        # Clean up
        inspect_analyzer.close()
        main_analyzer.close()