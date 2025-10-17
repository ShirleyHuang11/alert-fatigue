#!/usr/bin/env python3
"""
Batch simulation runner for Alert Fatigue Bellman Equation

This script runs simulations for all parameter sets in parameter_set.csv
and saves all results in combined CSV files in the simulation_results/ folder.
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_batch_simulations(csv_file="parameter_set.csv"):
    """Run simulations for all parameter sets in the CSV file."""
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: Parameter CSV file '{csv_file}' not found.")
        return
    
    # Read the CSV to get number of rows
    try:
        df = pd.read_csv(csv_file)
        num_rows = len(df)
        print(f"Found {num_rows} parameter sets in {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create results directory
    os.makedirs("simulation_results", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STARTING BATCH SIMULATION")
    print("=" * 60)
    
    successful_runs = 0
    failed_runs = []
    all_results = []
    
    # Run simulation for each parameter set
    for i in tqdm(range(num_rows), desc="Running simulations"):
        print(f"\n--- Running simulation {i+1}/{num_rows} ---")
        
        try:
            # Run the simulation with --no-plots to avoid individual CSV files
            cmd = [
                sys.executable, "bellman_simulation.py",
                "--parameter-csv", csv_file,
                "--run-index", str(i),
                "--no-plots"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Simulation {i} completed successfully")
                successful_runs += 1
                
                # Load the results from the individual files (temporarily created)
                results_file = f"simulation_results/row_{i}_comprehensive_results.csv"
                
                if os.path.exists(results_file):
                    result_df = pd.read_csv(results_file)
                    result_df['row_index'] = i  # Add row index
                    all_results.append(result_df)
                
            else:
                print(f"✗ Simulation {i} failed")
                print(f"Error: {result.stderr}")
                failed_runs.append(i)
                
        except Exception as e:
            print(f"✗ Simulation {i} failed with exception: {e}")
            failed_runs.append(i)
    
    # Generate plots for successful runs
    if successful_runs > 0:
        print("\n" + "=" * 40)
        print("GENERATING PLOTS")
        print("=" * 40)
        
        for i in range(num_rows):
            if i not in failed_runs:
                print(f"Generating plot for row {i}...")
                cmd = [
                    sys.executable, "bellman_simulation.py",
                    "--parameter-csv", csv_file,
                    "--run-index", str(i)
                ]
                # Run without --no-plots to generate the plot
                subprocess.run(cmd, capture_output=True)
    
    # Combine all results into single CSV files
    print("\n" + "=" * 40)
    print("CREATING COMBINED RESULTS")
    print("=" * 40)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_file = "simulation_results/comprehensive_results.csv"
        combined_results.to_csv(combined_file, index=False)
        print(f"✓ Combined comprehensive results saved: {combined_file}")
        print(f"  - {len(combined_results)} parameter sets")
    
    # Clean up individual result files
    print("\nCleaning up individual CSV files...")
    for i in range(num_rows):
        files_to_remove = [
            f"simulation_results/row_{i}_comprehensive_results.csv"
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
    
    print("Individual CSV files removed.")
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Successful runs: {successful_runs}/{num_rows}")
    
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
    else:
        print("All simulations completed successfully!")
    
    # List final output files
    print("\nFinal output files:")
    print("  - simulation_results/comprehensive_results.csv: All parameter sets and results")
    print("  - simulation_results/row_X_results.png: Individual plots")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "parameter_set.csv"
    run_batch_simulations(csv_file)
