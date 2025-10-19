#!/usr/bin/env python3
"""
Comprehensive comparison script that:
1. Runs numerical simulation for each parameter set
2. Computes all 4 closed-form solutions (A, B, C, D)
3. Selects the best closed-form based on:
   - Non-decreasing constraint
   - Closest V(1) to numerical simulation
4. Creates comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Tuple, List
from tqdm import tqdm

# Import all solvers
from bellman_simulation import BellmanSimulator
from closed_form_A11 import ClosedFormA11
from closed_form_B1_1 import ClosedFormB11
from closed_form_C_11 import ClosedFormC11
from closed_form_D_1_1 import ClosedFormD11

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')


class ComprehensiveComparison:
    """Compare numerical simulation with all closed-form solutions."""
    
    def __init__(self, params_row: pd.Series, output_dir: str = "comparison_results"):
        """Initialize with a parameter row from parameter_set.csv."""
        self.params = params_row
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract parameters
        self.beta = params_row['beta']
        self.alpha = params_row['alpha']
        self.gamma = params_row['gamma']
        
        self.tau_values = {
            '11': params_row['tau11'],
            '10': params_row['tau10'],
            '01': params_row['tau01'],
            '00': params_row['tau00']
        }
        
        self.p_values = {
            '11': params_row['p11'],
            '10': params_row['p10'],
            '01': params_row['p01'],
            '00': params_row['p00']
        }
        
        self.row_index = int(params_row.get('row_index', 0))
        
    def run_numerical_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run numerical simulation and return trust grid and V values."""
        print(f"  Running numerical simulation...")
        
        sim = BellmanSimulator(
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
            tau_values=self.tau_values,
            p_values=self.p_values,
            trust_range=(0, 1),
            grid_size=500
        )
        
        sim.solve_value_iteration(verbose=False)
        
        return sim.trust_grid, sim.V
    
    def compute_closed_form_A(self, trust_grid: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Compute closed-form A and return V values, is_non_decreasing, V(1)."""
        try:
            solver = ClosedFormA11(beta=self.beta, tau_values=self.tau_values, p_values=self.p_values)
            V_values = np.full_like(trust_grid, solver.V_constant)
            
            # Check non-decreasing (constant is always non-decreasing)
            is_non_decreasing = True
            V_at_1 = solver.V_constant
            
            return V_values, is_non_decreasing, V_at_1
        except Exception as e:
            print(f"    Error in Branch A: {e}")
            return np.full_like(trust_grid, np.nan), False, np.nan
    
    def compute_closed_form_B(self, trust_grid: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Compute closed-form B and return V values, is_non_decreasing, V(1)."""
        try:
            solver = ClosedFormB11(
                beta=self.beta,
                alpha=self.alpha,
                gamma=self.gamma,
                tau_values=self.tau_values,
                p_values=self.p_values
            )
            
            # Get V values over trust grid
            V_values = np.array([solver.get_value_function(t) for t in trust_grid])
            
            # Check non-decreasing
            is_non_decreasing = np.all(np.diff(V_values) >= -1e-6)
            V_at_1 = solver.get_value_function(1.0)
            
            return V_values, is_non_decreasing, V_at_1
        except Exception as e:
            print(f"    Error in Branch B: {e}")
            return np.full_like(trust_grid, np.nan), False, np.nan
    
    def compute_closed_form_C(self, trust_grid: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Compute closed-form C and return V values, is_non_decreasing, V(1)."""
        try:
            solver = ClosedFormC11(
                beta=self.beta,
                alpha=self.alpha,
                gamma=self.gamma,
                tau_values=self.tau_values,
                p_values=self.p_values
            )
            
            # Get V values over trust grid
            V_values = np.array([solver.get_value_function(t) for t in trust_grid])
            
            # Check non-decreasing
            is_non_decreasing = np.all(np.diff(V_values) >= -1e-6)
            V_at_1 = solver.get_value_function(1.0)
            
            return V_values, is_non_decreasing, V_at_1
        except Exception as e:
            print(f"    Error in Branch C: {e}")
            return np.full_like(trust_grid, np.nan), False, np.nan
    
    def compute_closed_form_D(self, trust_grid: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Compute closed-form D and return V values, is_non_decreasing, V(1)."""
        try:
            solver = ClosedFormD11(
                beta=self.beta,
                alpha=self.alpha,
                gamma=self.gamma,
                tau_values=self.tau_values,
                p_values=self.p_values
            )
            
            # Get V values over trust grid
            V_values = solver.get_value_function(trust_grid)
            
            # Check non-decreasing
            is_non_decreasing = np.all(np.diff(V_values) >= -1e-6)
            V_at_1 = solver.get_value_function(np.array([1.0]))[0]
            
            return V_values, is_non_decreasing, V_at_1
        except Exception as e:
            print(f"    Error in Branch D: {e}")
            return np.full_like(trust_grid, np.nan), False, np.nan
    
    def select_best_branch(self, numerical_V_at_1: float, 
                          branches: Dict[str, Tuple[np.ndarray, bool, float]]) -> str:
        """
        Select the best branch based on:
        1. Must be non-decreasing
        2. Closest V(1) to numerical simulation
        """
        candidates = []
        
        for branch_name, (V_values, is_non_decreasing, V_at_1) in branches.items():
            if is_non_decreasing and not np.isnan(V_at_1):
                error = abs(V_at_1 - numerical_V_at_1)
                candidates.append((branch_name, error, V_at_1))
        
        if not candidates:
            print("    WARNING: No valid non-decreasing branches found!")
            return None
        
        # Sort by error and pick the best
        candidates.sort(key=lambda x: x[1])
        best_branch, best_error, best_V_at_1 = candidates[0]
        
        print(f"    Numerical V(1) = {numerical_V_at_1:.6f}")
        for branch_name, error, V_at_1 in candidates:
            marker = "✓ SELECTED" if branch_name == best_branch else ""
            print(f"    Branch {branch_name}: V(1) = {V_at_1:.6f}, error = {error:.6f} {marker}")
        
        return best_branch
    
    def create_comparison_plot(self, trust_grid: np.ndarray, numerical_V: np.ndarray,
                               branches: Dict[str, Tuple[np.ndarray, bool, float]],
                               best_branch: str):
        """Create comprehensive comparison plot."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot numerical simulation
        ax.plot(trust_grid, numerical_V, 'k-', linewidth=3, label='Numerical Simulation', zorder=10)
        
        # Plot all four closed-form solutions
        colors = {'A': 'blue', 'B': 'orange', 'C': 'green', 'D': 'red'}
        linestyles = {'A': '--', 'B': '--', 'C': '--', 'D': '--'}
        alphas = {'A': 0.6, 'B': 0.6, 'C': 0.6, 'D': 0.6}
        
        for branch_name, (V_values, is_non_decreasing, V_at_1) in branches.items():
            if not np.all(np.isnan(V_values)):
                label = f"Branch {branch_name}"
                if not is_non_decreasing:
                    label += " (decreasing ✗)"
                
                ax.plot(trust_grid, V_values, 
                       color=colors[branch_name],
                       linestyle=linestyles[branch_name],
                       linewidth=2,
                       alpha=alphas[branch_name],
                       label=label,
                       zorder=5)
        
        # Highlight the best branch
        if best_branch and best_branch in branches:
            V_best, _, _ = branches[best_branch]
            ax.plot(trust_grid, V_best,
                   color=colors[best_branch],
                   linestyle='-',
                   linewidth=3,
                   alpha=0.9,
                   label=f'Branch {best_branch} (SELECTED ✓)',
                   zorder=8)
        
        ax.set_xlabel('Trust Level (t)', fontsize=12)
        ax.set_ylabel('Value Function V(t)', fontsize=12)
        ax.set_title(f'Comparison: Numerical vs All Closed-Forms (Row {self.row_index})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add parameter info
        param_text = (f'β={self.beta}, α={self.alpha}, γ={self.gamma}\n'
                     f'p¹¹={self.p_values["11"]}, p¹⁰={self.p_values["10"]}, '
                     f'p⁰¹={self.p_values["01"]}, p⁰⁰={self.p_values["00"]}')
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'comparison_row_{self.row_index}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def get_simulation_dataframe(self) -> pd.DataFrame:
        """Get full simulation results as DataFrame."""
        sim = BellmanSimulator(
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
            tau_values=self.tau_values,
            p_values=self.p_values,
            trust_range=(0, 1),
            grid_size=500
        )
        
        sim.solve_value_iteration(verbose=False)
        
        # Get the comprehensive results dataframe
        sim_df = sim.create_results_dataframe(plot_path=None)
        
        return sim_df
    
    def get_closed_form_dataframe(self, branch: str) -> pd.DataFrame:
        """Get full closed-form results as DataFrame for the selected branch."""
        if branch == 'A':
            solver = ClosedFormA11(beta=self.beta, tau_values=self.tau_values, p_values=self.p_values)
            return solver.create_results_dataframe(plot_path=None)
        elif branch == 'B':
            solver = ClosedFormB11(
                beta=self.beta, alpha=self.alpha, gamma=self.gamma,
                tau_values=self.tau_values, p_values=self.p_values
            )
            return solver.create_results_dataframe(plot_path=None)
        elif branch == 'C':
            solver = ClosedFormC11(
                beta=self.beta, alpha=self.alpha, gamma=self.gamma,
                tau_values=self.tau_values, p_values=self.p_values
            )
            return solver.create_results_dataframe(plot_path=None)
        elif branch == 'D':
            solver = ClosedFormD11(
                beta=self.beta, alpha=self.alpha, gamma=self.gamma,
                tau_values=self.tau_values, p_values=self.p_values
            )
            return solver.create_results_dataframe(plot_path=None)
        else:
            return pd.DataFrame()
    
    def run_comparison(self) -> Dict:
        """Run complete comparison and return results."""
        print(f"\n{'='*60}")
        print(f"Processing Row {self.row_index}")
        print(f"{'='*60}")
        
        # 1. Run numerical simulation
        trust_grid, numerical_V = self.run_numerical_simulation()
        numerical_V_at_1_idx = np.argmin(np.abs(trust_grid - 1.0))
        numerical_V_at_1 = numerical_V[numerical_V_at_1_idx]
        
        # 2. Compute all closed-form solutions
        print(f"  Computing closed-form solutions...")
        branches = {}
        
        print(f"    Branch A...")
        branches['A'] = self.compute_closed_form_A(trust_grid)
        
        print(f"    Branch B...")
        branches['B'] = self.compute_closed_form_B(trust_grid)
        
        print(f"    Branch C...")
        branches['C'] = self.compute_closed_form_C(trust_grid)
        
        print(f"    Branch D...")
        branches['D'] = self.compute_closed_form_D(trust_grid)
        
        # 3. Select best branch
        print(f"  Selecting best branch...")
        best_branch = self.select_best_branch(numerical_V_at_1, branches)
        
        # 4. Create comparison plot
        print(f"  Creating comparison plot...")
        plot_path = self.create_comparison_plot(trust_grid, numerical_V, branches, best_branch)
        print(f"  ✓ Plot saved: {plot_path}")
        
        # 5. Get full dataframes
        print(f"  Collecting comprehensive data...")
        sim_df = self.get_simulation_dataframe()
        
        # 6. Compile results - start with basic info
        results = {
            'row_index': self.row_index,
            'best_branch': best_branch if best_branch else "None",
            'comparison_plot_path': plot_path
        }
        
        # Add branch-specific summary info
        for branch_name, (V_values, is_non_decreasing, V_at_1) in branches.items():
            results[f'{branch_name}_V_at_1'] = V_at_1
            results[f'{branch_name}_non_decreasing'] = is_non_decreasing
            results[f'{branch_name}_error'] = abs(V_at_1 - numerical_V_at_1) if not np.isnan(V_at_1) else np.nan
        
        # Add simulation results (prefix with 'sim_' to avoid conflicts)
        for col in sim_df.columns:
            if col not in ['plot_path']:  # Skip plot_path from simulation
                results[f'sim_{col}'] = sim_df[col].iloc[0]
        
        # Add closed-form results for the selected branch (prefix with 'cf_')
        if best_branch:
            cf_df = self.get_closed_form_dataframe(best_branch)
            if not cf_df.empty:
                for col in cf_df.columns:
                    # Skip duplicate parameters that are already in sim_
                    if col not in ['beta', 'alpha', 'gamma', 'tau_11', 'tau_10', 'tau_01', 'tau_00',
                                   'tau11', 'tau10', 'tau01', 'tau00',
                                   'p_11', 'p_10', 'p_01', 'p_00', 'p11', 'p10', 'p01', 'p00',
                                   'plot_path']:
                        results[f'cf_{col}'] = cf_df[col].iloc[0]
        
        return results


def main():
    """Main function to process all parameter sets."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run comprehensive comparison of numerical simulation vs closed-form solutions'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='parameter_set.csv',
        help='Input CSV file with parameter sets (default: parameter_set.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_results',
        help='Output directory for results (default: comparison_results)'
    )
    
    args = parser.parse_args()
    
    # Read parameter sets
    param_file = args.input
    if not os.path.exists(param_file):
        print(f"Error: {param_file} not found!")
        return
    
    params_df = pd.read_csv(param_file)
    
    # Add row_index if not present
    if 'row_index' not in params_df.columns:
        params_df['row_index'] = range(len(params_df))
    
    print(f"Found {len(params_df)} parameter sets in {param_file}")
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each parameter set
    all_results = []
    
    for idx, row in tqdm(params_df.iterrows(), total=len(params_df), desc="Processing parameter sets"):
        try:
            comparison = ComprehensiveComparison(row, output_dir)
            results = comparison.run_comparison()
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERROR processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Processed {len(all_results)}/{len(params_df)} parameter sets")
    print(f"✓ Results saved: {results_path}")
    print(f"✓ Plots saved in: {output_dir}/")
    
    # Print summary
    print(f"\nBranch Selection Summary:")
    if len(all_results) > 0:
        branch_counts = results_df['best_branch'].value_counts()
        for branch, count in branch_counts.items():
            print(f"  Branch {branch}: {count} times")


if __name__ == '__main__':
    main()

