"""
Closed-Form Solution for Alert Fatigue Bellman Equation
Winning Branch A: (+1, +1) - Both conflict states choose Forward option

This implements the analytical solution when:
- 10-branch (σ₁₀ = +1): Choose Forward → τ¹⁰ + βV(t+α)
- 01-branch (σ₀₁ = +1): Choose Forward → βV(t+α)

The closed-form solution is:
V(t) = (p¹¹τ¹¹ + p¹⁰τ¹⁰) / (1-β)

This solution is independent of trust level t.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

class ClosedFormA11:
    """
    Closed-form solution for the (+1, +1) winning branch.
    """
    
    def __init__(self, 
                 beta: float = 0.95,
                 tau_values: Dict[str, float] = None,
                 p_values: Dict[str, float] = None):
        """
        Initialize the closed-form solver.
        
        Args:
            beta: Discount factor (0 < beta < 1)
            tau_values: Treatment effects for each type
            p_values: Probabilities for each type
        """
        self.beta = beta
        
        # Default treatment effects
        self.tau_values = tau_values or {
            '11': 1.0,   # Both human and AI choose 1 (positive effect)
            '10': 0.5,   # Human chooses 1, AI chooses 0
            '01': -0.3,  # Human chooses 0, AI chooses 1
            '00': -0.8   # Both choose 0 (negative effect)
        }
        
        # Default probabilities (must sum to 1)
        self.p_values = p_values or {
            '11': 0.3,
            '10': 0.2,
            '01': 0.25,
            '00': 0.25
        }
        
        # Validate probabilities sum to 1
        if abs(sum(self.p_values.values()) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1")
        
        # Validate beta
        if not 0 < beta < 1:
            raise ValueError("Beta must be in (0, 1)")
        
        # Calculate the closed-form solution
        self.C0 = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
        self.V_constant = self.C0 / (1 - self.beta)
        
        print(f"Closed-form solution parameters:")
        print(f"  C₀ = p¹¹τ¹¹ + p¹⁰τ¹⁰ = {self.p_values['11']}×{self.tau_values['11']} + {self.p_values['10']}×{self.tau_values['10']} = {self.C0:.4f}")
        print(f"  V(t) = C₀/(1-β) = {self.C0:.4f}/(1-{self.beta}) = {self.V_constant:.4f}")
    
    def get_value_function(self, t: float) -> float:
        """
        Get the value function at trust level t.
        
        For the (+1, +1) branch, V(t) is constant for all t.
        
        Args:
            t: Trust level (not used in this branch)
            
        Returns:
            Constant value function V(t) = C₀/(1-β)
        """
        return self.V_constant
    
    def get_policy(self, t: float) -> str:
        """
        Get the optimal policy at trust level t.
        
        For the (+1, +1) branch, both conflict states choose Forward.
        
        Args:
            t: Trust level
            
        Returns:
            Policy description
        """
        return "Forward for both 10 and 01 states"
    
    def calculate_branch_parameters(self) -> Dict:
        """
        Calculate the branch-specific parameters according to the table.
        
        Returns:
            Dictionary with branch parameters
        """
        return {
            'sigma_10': +1,  # 10-branch chooses Forward
            'sigma_01': +1,  # 01-branch chooses Forward
            'kappa': self.p_values['11'] + self.p_values['00'] + self.p_values['10'] + self.p_values['01'],
            'tau_10_sigma': self.tau_values['10'],  # τ¹⁰
            'tau_01_sigma': 0.0,  # For Forward branch in 01 state
            'branch_name': '(+1, +1)',
            'description': 'Both conflict states choose Forward'
        }
    
    def plot_solution(self, trust_range: Tuple[float, float] = (0.0, 1.0), 
                     save_path: str = None) -> str:
        """
        Plot the closed-form solution.
        
        Args:
            trust_range: Range of trust values to plot
            save_path: Path to save the plot
        """
        t_min, t_max = trust_range
        trust_values = np.linspace(t_min, t_max, 100)
        
        # Calculate value function (constant for this branch)
        value_function = np.full_like(trust_values, self.V_constant)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Value function
        ax1.plot(trust_values, value_function, 'b-', linewidth=3, 
                label=f'V(t) = {self.V_constant:.4f}')
        ax1.set_xlabel('Trust Level (t)')
        ax1.set_ylabel('Value Function V(t)')
        ax1.set_title(f'Closed-Form Solution: {self.calculate_branch_parameters()["branch_name"]} Branch')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(t_min, t_max)
        
        # Add formula annotation
        formula_text = f'V(t) = $\\frac{{p^{{11}}\\tau^{{11}} + p^{{10}}\\tau^{{10}}}}{{1-\\beta}} = \\frac{{{self.C0:.4f}}}{{{1-self.beta}}} = {self.V_constant:.4f}$'
        ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        # Plot 2: Policy (constant for this branch)
        policy_colors = ['green'] * len(trust_values)
        ax2.scatter(trust_values, np.ones_like(trust_values), c=policy_colors, 
                   s=20, alpha=0.7, label='Forward (both 10 and 01)')
        ax2.set_xlabel('Trust Level (t)')
        ax2.set_ylabel('Policy')
        ax2.set_title('Optimal Policy: Always Forward for Conflict States')
        ax2.set_yticks([1])
        ax2.set_yticklabels(['Forward'])
        ax2.set_xlim(t_min, t_max)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add parameter information
        param_text = f'Parameters:\nβ = {self.beta}\nτ¹¹ = {self.tau_values["11"]}\nτ¹⁰ = {self.tau_values["10"]}\np¹¹ = {self.p_values["11"]}\np¹⁰ = {self.p_values["10"]}'
        ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.close()
        return save_path if save_path else None
    
    def compare_with_numerical(self, numerical_file: str = None) -> pd.DataFrame:
        """
        Compare closed-form solution with numerical results.
        
        Args:
            numerical_file: Path to numerical results CSV file
            
        Returns:
            DataFrame with comparison results
        """
        if not numerical_file or not os.path.exists(numerical_file):
            print("No numerical file provided or file doesn't exist")
            return pd.DataFrame()
        
        # Load numerical results
        try:
            numerical_df = pd.read_csv(numerical_file)
            print(f"Loaded numerical results from {numerical_file}")
            print(f"Found {len(numerical_df)} parameter sets")
        except Exception as e:
            print(f"Error loading numerical file: {e}")
            return pd.DataFrame()
        
        # Compare with each parameter set
        comparisons = []
        for idx, row in numerical_df.iterrows():
            row_index = row.get('row_index', idx)
            
            # Extract parameters from numerical results
            beta_num = row['beta']
            tau11_num = row['tau11']
            tau10_num = row['tau10']
            p11_num = row['p11']
            p10_num = row['p10']
            
            # Calculate closed-form solution for these parameters
            C0_num = p11_num * tau11_num + p10_num * tau10_num
            V_closed_form = C0_num / (1 - beta_num)
            
            # Extract numerical value (assuming it's constant)
            if 'value_function' in row:
                # Parse the array string
                import ast
                value_array = ast.literal_eval(row['value_function'])
                V_numerical = np.mean(value_array)  # Take mean since it should be constant
            
            comparison = {
                'row_index': row_index,
                'beta': beta_num,
                'tau11': tau11_num,
                'tau10': tau10_num,
                'p11': p11_num,
                'p10': p10_num,
                'C0': C0_num,
                'V_closed_form': V_closed_form,
                'V_numerical': V_numerical if 'V_numerical' in locals() else None,
                'error': abs(V_closed_form - V_numerical) if 'V_numerical' in locals() else None,
                'relative_error': abs(V_closed_form - V_numerical) / V_numerical if 'V_numerical' in locals() else None
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def create_results_dataframe(self, plot_path: str = None) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all parameters and results.
        
        Args:
            plot_path: Path to the generated plot file
            
        Returns:
            DataFrame with all parameters, results, and metadata
        """
        branch_params = self.calculate_branch_parameters()
        
        # Create the mathematical formula string
        formula = f"V(t) = (p¹¹τ¹¹ + p¹⁰τ¹⁰) / (1-β) = ({self.C0:.6f}) / ({1-self.beta}) = {self.V_constant:.6f}"
        
        results_data = {
            # All parameters
            'beta': self.beta,
            'tau11': self.tau_values['11'],
            'tau10': self.tau_values['10'],
            'tau01': self.tau_values['01'],
            'tau00': self.tau_values['00'],
            'p11': self.p_values['11'],
            'p10': self.p_values['10'],
            'p01': self.p_values['01'],
            'p00': self.p_values['00'],
            
            # Branch parameters
            'sigma_10': branch_params['sigma_10'],
            'sigma_01': branch_params['sigma_01'],
            'kappa': branch_params['kappa'],
            'tau_10_sigma': branch_params['tau_10_sigma'],
            'tau_01_sigma': branch_params['tau_01_sigma'],
            'branch_name': branch_params['branch_name'],
            'description': branch_params['description'],
            
            # Closed-form solution
            'C0': self.C0,
            'V_constant': self.V_constant,
            'formula': formula,
            
            # File paths
            'plot_path': plot_path if plot_path else None,
            
            # Metadata
            'solution_type': 'Closed-Form',
            'branch_type': 'A11 (+1, +1)',
            'is_constant': True,
            'independent_of_t': True
        }
        
        return pd.DataFrame([results_data])

def main():
    """Main function to run the closed-form analysis."""
    parser = argparse.ArgumentParser(description="Closed-form solution for (+1, +1) branch")
    parser.add_argument('--beta', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--tau11', type=float, default=1.0, help='Treatment effect for type (1,1)')
    parser.add_argument('--tau10', type=float, default=0.5, help='Treatment effect for type (1,0)')
    parser.add_argument('--tau01', type=float, default=-0.3, help='Treatment effect for type (0,1)')
    parser.add_argument('--tau00', type=float, default=-0.8, help='Treatment effect for type (0,0)')
    parser.add_argument('--p11', type=float, default=0.3, help='Probability of type (1,1)')
    parser.add_argument('--p10', type=float, default=0.2, help='Probability of type (1,0)')
    parser.add_argument('--p01', type=float, default=0.25, help='Probability of type (0,1)')
    parser.add_argument('--p00', type=float, default=0.25, help='Probability of type (0,0)')
    parser.add_argument('--output-prefix', type=str, default='closed_form_A11', help='Output file prefix')
    parser.add_argument('--compare-numerical', type=str, help='Path to numerical results CSV for comparison')
    
    args = parser.parse_args()
    
    # Create closed_form_results directory
    os.makedirs("closed_form_results", exist_ok=True)
    
    print("=" * 60)
    print("CLOSED-FORM SOLUTION: (+1, +1) WINNING BRANCH")
    print("=" * 60)
    
    # Validate probabilities
    prob_sum = args.p11 + args.p10 + args.p01 + args.p00
    if abs(prob_sum - 1.0) > 1e-6:
        print(f"Warning: Probabilities sum to {prob_sum:.6f}, not 1.0")
    
    # Initialize solver
    tau_values = {'11': args.tau11, '10': args.tau10, '01': args.tau01, '00': args.tau00}
    p_values = {'11': args.p11, '10': args.p10, '01': args.p01, '00': args.p00}
    
    solver = ClosedFormA11(beta=args.beta, tau_values=tau_values, p_values=p_values)
    
    # Calculate branch parameters
    branch_params = solver.calculate_branch_parameters()
    print(f"\nBranch Parameters:")
    for key, value in branch_params.items():
        print(f"  {key}: {value}")
    
    # Plot the solution
    plot_path = f"closed_form_results/{args.output_prefix}_solution.png"
    plot_file = solver.plot_solution(save_path=plot_path)
    
    # Create comprehensive results DataFrame
    print("\n" + "=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    
    results_df = solver.create_results_dataframe(plot_file)
    
    # Save comprehensive results
    results_file = f"closed_form_results/{args.output_prefix}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"✓ Comprehensive results saved: {results_file}")
    
    # Compare with numerical results if provided
    if args.compare_numerical:
        print(f"\nComparing with numerical results from {args.compare_numerical}...")
        comparison_df = solver.compare_with_numerical(args.compare_numerical)
        
        if not comparison_df.empty:
            comparison_file = f"closed_form_results/{args.output_prefix}_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"✓ Comparison results saved: {comparison_file}")
            
            if 'error' in comparison_df.columns:
                print(f"\nComparison Summary:")
                print(f"  Mean absolute error: {comparison_df['error'].mean():.6f}")
                print(f"  Mean relative error: {comparison_df['relative_error'].mean():.4%}")
                print(f"  Max absolute error: {comparison_df['error'].max():.6f}")
    
    print(f"\n" + "=" * 60)
    print("CLOSED-FORM SOLUTION COMPLETE")
    print("=" * 60)
    print(f"V(t) = {solver.V_constant:.6f} for all trust levels t")
    print(f"\nFiles created in closed_form_results/ folder:")
    print(f"  - {os.path.basename(plot_path)}: Solution visualization")
    print(f"  - {os.path.basename(results_file)}: Comprehensive results with formula")
    if args.compare_numerical:
        print(f"  - {os.path.basename(comparison_file)}: Comparison with numerical results")

if __name__ == "__main__":
    main()
