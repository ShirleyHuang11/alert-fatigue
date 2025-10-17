"""
Numerical Simulation for Alert Fatigue Bellman Equation

This script implements the numerical solution for the Bellman equation:
V(t) = p^{11}(τ^{11} + βV(t+α)) + p^{00}(βV(t+α))
     + p^{10} * max{Forward, Backward, Current}
     + p^{01} * max{Forward, Backward, Current}

Where the options are:
- Forward: τ^{10/01} + βV(t+α) or βV(t+α)
- Backward: (1-t̃)τ^{10/01} + βV(t-α) or t̃τ^{01} + βV(t-α)  
- Current: γτ^{10/01} + βV(t) or (1-γ)τ^{01} + βV(t)

Usage:
    python bellman_simulation.py --beta 0.95 --alpha 0.1 --gamma 0.7
    python bellman_simulation.py --config config.json
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import pandas as pd
import argparse
import json
import sys
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class BellmanSimulator:
    """
    Numerical simulator for the alert fatigue Bellman equation.
    """
    
    def __init__(self, 
                 beta: float = 0.95,
                 alpha: float = 0.1,
                 gamma: float = 0.7,
                 tau_values: Dict[str, float] = None,
                 p_values: Dict[str, float] = None,
                 trust_range: Tuple[float, float] = (0, 1.0),
                 grid_size: int = 500):
        """
        Initialize the Bellman simulator.
        
        Args:
            beta: Discount factor (0 < beta < 1)
            alpha: Trust update magnitude (0 < alpha << 1)
            gamma: Human decision probability when no recommendation
            tau_values: Treatment effects for each type
            p_values: Probabilities for each type
            trust_range: Range of trust values to consider
            grid_size: Number of grid points for discretization
        """
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        
        # Default treatment effects (assuming τ^{11} > 0, τ^{00} < 0)
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
        
        # Create trust grid
        self.trust_min, self.trust_max = trust_range
        self.grid_size = grid_size
        self.trust_grid = np.linspace(self.trust_min, self.trust_max, grid_size)
        self.dt = (self.trust_max - self.trust_min) / (grid_size - 1)
        
        # Initialize value function
        self.V = np.zeros(grid_size)
        self.policy_10 = np.zeros(grid_size, dtype=int)  # Policy for 10-branch: 0=Backward, 1=Current, 2=Forward
        self.policy_01 = np.zeros(grid_size, dtype=int)  # Policy for 01-branch: 0=Backward, 1=Current, 2=Forward
        
        # Convergence parameters
        self.tolerance = 1e-6
        self.max_iterations = 10000
        
    def get_trust_clamped(self, t: float) -> float:
        """Get clamped trust value: min(max(t, 0), 1)"""
        return min(max(t, 0), 1)
    
    def interpolate_value(self, t: float) -> float:
        """Interpolate value function at trust level t"""
        if t <= self.trust_min:
            return self.V[0]
        elif t >= self.trust_max:
            return self.V[-1]
        else:
            idx = (t - self.trust_min) / self.dt
            idx_low = int(np.floor(idx))
            idx_high = min(idx_low + 1, self.grid_size - 1)
            weight = idx - idx_low
            return (1 - weight) * self.V[idx_low] + weight * self.V[idx_high]
    
    def compute_bellman_operator(self, t_idx: int) -> Tuple[float, int, int]:
        """
        Compute the Bellman operator for trust level at index t_idx.
        
        Returns:
            (value, policy_10, policy_01): The optimal value and corresponding policies for both branches
        """
        t = self.trust_grid[t_idx]
        t_tilde = self.get_trust_clamped(t)
        
        # Get interpolated values
        V_t_plus = self.interpolate_value(t + self.alpha)
        V_t_minus = self.interpolate_value(t - self.alpha)
        V_t = self.V[t_idx]
        
        # Compute the deterministic parts
        deterministic_part = (
            self.p_values['11'] * (self.tau_values['11'] + self.beta * V_t_plus) +
            self.p_values['00'] * (self.beta * V_t_plus)
        )
        
        # Compute options for p^{10} case
        p10_forward = self.tau_values['10'] + self.beta * V_t_plus
        p10_backward = (1 - t_tilde) * self.tau_values['10'] + self.beta * V_t_minus
        p10_current = self.gamma * self.tau_values['10'] + self.beta * V_t
        
        p10_options = [p10_backward, p10_current, p10_forward]  # Reorder: 0=B, 1=C, 2=F
        p10_best_idx = np.argmax(p10_options)
        p10_best_value = p10_options[p10_best_idx]
        
        # Compute options for p^{01} case
        p01_forward = self.beta * V_t_plus
        p01_backward = t_tilde * self.tau_values['01'] + self.beta * V_t_minus
        p01_current = (1 - self.gamma) * self.tau_values['01'] + self.beta * V_t
        
        p01_options = [p01_backward, p01_current, p01_forward]  # Reorder: 0=B, 1=C, 2=F
        p01_best_idx = np.argmax(p01_options)
        p01_best_value = p01_options[p01_best_idx]
        
        # Total value
        total_value = deterministic_part + self.p_values['10'] * p10_best_value + self.p_values['01'] * p01_best_value
        
        return total_value, p10_best_idx, p01_best_idx
    
    def solve_value_iteration(self, verbose: bool = True) -> Dict:
        """
        Solve the Bellman equation using value iteration.
        
        Returns:
            Dictionary with convergence information
        """
        if verbose:
            print("Starting value iteration...")
        
        for iteration in tqdm(range(self.max_iterations), disable=not verbose):
            V_old = self.V.copy()
            
            # Update value function
            for t_idx in range(self.grid_size):
                self.V[t_idx], self.policy_10[t_idx], self.policy_01[t_idx] = self.compute_bellman_operator(t_idx)
            
            # Check convergence
            max_change = np.max(np.abs(self.V - V_old))
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Max change = {max_change:.2e}")
            
            if max_change < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return {
            'iterations': iteration + 1,
            'max_change': max_change,
            'converged': max_change < self.tolerance
        }
    
    def analyze_policy(self) -> pd.DataFrame:
        """Analyze the optimal policy across trust levels."""
        policy_names = ['B', 'C', 'F']  # Backward, Current, Forward
        
        data = []
        for i, t in enumerate(self.trust_grid):
            data.append({
                'trust': t,
                'value': self.V[i],
                'policy_10_branch': policy_names[self.policy_10[i]],
                'policy_01_branch': policy_names[self.policy_01[i]],
                'policy_10_code': self.policy_10[i],
                'policy_01_code': self.policy_01[i]
            })
        
        return pd.DataFrame(data)
    
    def plot_results(self, save_path: str = None):
        """Plot the value function and optimal policy (matching closed-form format)."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Find indices for t=0 and t=1
        idx_0 = np.argmin(np.abs(self.trust_grid - 0.0))
        idx_1 = np.argmin(np.abs(self.trust_grid - 1.0))
        
        # Get boundary values and policies
        V_0 = self.V[idx_0]
        V_1 = self.V[idx_1]
        policy_10_at_0 = self.policy_10[idx_0]
        policy_01_at_0 = self.policy_01[idx_0]
        policy_10_at_1 = self.policy_10[idx_1]
        policy_01_at_1 = self.policy_01[idx_1]
        
        # Create adjusted value function and policies with proper boundary conditions
        V_adjusted = self.V.copy()
        policy_10_adjusted = self.policy_10.copy()
        policy_01_adjusted = self.policy_01.copy()
        
        # For t < 0: V(t) = V(0), policy(t) = policy(0)
        mask_below_0 = self.trust_grid < 0
        V_adjusted[mask_below_0] = V_0
        policy_10_adjusted[mask_below_0] = policy_10_at_0
        policy_01_adjusted[mask_below_0] = policy_01_at_0
        
        # For t > 1: V(t) = V(1), policy(t) = policy(1)
        mask_above_1 = self.trust_grid > 1
        V_adjusted[mask_above_1] = V_1
        policy_10_adjusted[mask_above_1] = policy_10_at_1
        policy_01_adjusted[mask_above_1] = policy_01_at_1
        
        # Plot 1: Value function
        ax1.plot(self.trust_grid, V_adjusted, 'b-', linewidth=3, label='Numerical V(t)')
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='t=0 boundary')
        ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='t=1 boundary')
        ax1.set_xlabel('Trust Level (t)', fontsize=12)
        ax1.set_ylabel('Value Function V(t)', fontsize=12)
        ax1.set_title('Numerical Solution: Value Function', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Policy Regions (matching combined branch policies below)
        # Show both 10-branch and 01-branch regions
        color_map = {0: 'lightcoral', 1: 'lightyellow', 2: 'lightgreen'}
        
        # Plot 10-branch regions in upper half
        for i in range(len(self.trust_grid) - 1):
            t = self.trust_grid[i]
            t_next = self.trust_grid[i + 1]
            policy = policy_10_adjusted[i]
            ax2.fill_between([t, t_next], [0.05, 0.05], [0.5, 0.5], 
                            color=color_map.get(policy, 'lightgray'), alpha=0.5)
        
        # Plot 01-branch regions in lower half
        for i in range(len(self.trust_grid) - 1):
            t = self.trust_grid[i]
            t_next = self.trust_grid[i + 1]
            policy = policy_01_adjusted[i]
            ax2.fill_between([t, t_next], [-0.5, -0.5], [-0.05, -0.05], 
                            color=color_map.get(policy, 'lightgray'), alpha=0.5)
        
        ax2.set_xlim(self.trust_grid[0], self.trust_grid[-1])
        ax2.set_ylim(-0.55, 0.55)
        ax2.set_xlabel('Trust Level (t)', fontsize=12)
        ax2.set_title('Policy Regions: 10-branch (top) and 01-branch (bottom)', fontsize=12, fontweight='bold')
        ax2.set_yticks([0.275, -0.275])
        ax2.set_yticklabels(['10-branch', '01-branch'])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
        ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
        
        # Add custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.5, label='Backward (B)'),
            Patch(facecolor='lightyellow', alpha=0.5, label='Current (C)'),
            Patch(facecolor='lightgreen', alpha=0.5, label='Forward (F)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Plot 3: Combined Branch Policies (10-branch and 01-branch)
        ax3.plot(self.trust_grid, policy_10_adjusted, 'b-', linewidth=2, label='10-branch policy', alpha=0.7)
        ax3.plot(self.trust_grid, policy_01_adjusted, 'r--', linewidth=2, label='01-branch policy', alpha=0.7)
        
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Disagree (Backward)', 'Silence (Current)', 'Agree (Forward)'])
        ax3.set_xlabel('Trust Level (t)', fontsize=12)
        ax3.set_ylabel('Optimal Policy', fontsize=12)
        ax3.set_title('Combined Branch Policies', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
        ax3.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close the figure to free memory
    
    def create_results_dataframe(self, plot_path: str = None) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all simulation results.
        
        Args:
            plot_path: Path to the generated plot file
            
        Returns:
            DataFrame with all parameters and results
        """
        policy_names = ['B', 'C', 'F']  # Backward, Current, Forward
        policy_10_array = [policy_names[p] for p in self.policy_10]
        policy_01_array = [policy_names[p] for p in self.policy_01]
        
        results_data = {
            # All parameters
            'beta': self.beta,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau11': self.tau_values['11'],
            'tau10': self.tau_values['10'],
            'tau01': self.tau_values['01'],
            'tau00': self.tau_values['00'],
            'p11': self.p_values['11'],
            'p10': self.p_values['10'],
            'p01': self.p_values['01'],
            'p00': self.p_values['00'],
            'trust_min': self.trust_min,
            'trust_max': self.trust_max,
            'grid_size': self.grid_size,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            
            # Results as arrays
            'trust_grid': [self.trust_grid.tolist()],
            'value_function': [self.V.tolist()],
            'policy_10_branch': [policy_10_array],
            'policy_01_branch': [policy_01_array],
            
            # Plot path
            'plot_path': plot_path if plot_path else None
        }
        
        return pd.DataFrame([results_data])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Numerical simulation for Alert Fatigue Bellman Equation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument('--beta', type=float, default=0.95,
                       help='Discount factor (0 < beta < 1)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Trust update magnitude (0 < alpha << 1)')
    parser.add_argument('--gamma', type=float, default=0.7,
                       help='Human decision probability when no recommendation')
    
    # Treatment effects
    parser.add_argument('--tau11', type=float, default=1.0,
                       help='Treatment effect for type (1,1)')
    parser.add_argument('--tau10', type=float, default=0.5,
                       help='Treatment effect for type (1,0)')
    parser.add_argument('--tau01', type=float, default=-0.3,
                       help='Treatment effect for type (0,1)')
    parser.add_argument('--tau00', type=float, default=-0.8,
                       help='Treatment effect for type (0,0)')
    
    # Type probabilities
    parser.add_argument('--p11', type=float, default=0.3,
                       help='Probability of type (1,1)')
    parser.add_argument('--p10', type=float, default=0.2,
                       help='Probability of type (1,0)')
    parser.add_argument('--p01', type=float, default=0.25,
                       help='Probability of type (0,1)')
    parser.add_argument('--p00', type=float, default=0.25,
                       help='Probability of type (0,0)')
    
    # Grid parameters
    parser.add_argument('--trust-min', type=float, default=0.0,
                       help='Minimum trust level')
    parser.add_argument('--trust-max', type=float, default=1.0,
                       help='Maximum trust level')
    parser.add_argument('--grid-size', type=int, default=500,
                       help='Number of grid points for discretization')
    
    # Convergence parameters
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance for value iteration')
    parser.add_argument('--max-iterations', type=int, default=10000,
                       help='Maximum number of iterations')
    
    # Output options
    parser.add_argument('--output-prefix', type=str, default='bellman',
                       help='Prefix for output files')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Configuration file options
    parser.add_argument('--config', type=str,
                       help='JSON configuration file path')
    parser.add_argument('--parameter-csv', type=str,
                       help='CSV file with parameter sets to run')
    parser.add_argument('--run-index', type=int, default=0,
                       help='Index of parameter set to run from CSV (0-based)')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def load_parameters_from_csv(csv_path: str, run_index: int = 0) -> Dict:
    """Load parameters from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if run_index >= len(df):
            print(f"Error: Run index {run_index} is out of range. CSV has {len(df)} rows.")
            sys.exit(1)
        
        params = df.iloc[run_index].to_dict()
        return params
    except FileNotFoundError:
        print(f"Error: Parameter CSV file '{csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not load parameters from CSV: {e}")
        sys.exit(1)

def validate_parameters(args) -> None:
    """Validate parameter values."""
    errors = []
    
    # Validate probabilities sum to 1
    prob_sum = args.p11 + args.p10 + args.p01 + args.p00
    if abs(prob_sum - 1.0) > 1e-6:
        errors.append(f"Probabilities must sum to 1, got {prob_sum:.6f}")
    
    # Validate parameter ranges
    if not 0 < args.beta < 1:
        errors.append(f"Beta must be in (0,1), got {args.beta}")
    
    if args.alpha <= 0:
        errors.append(f"Alpha must be positive, got {args.alpha}")
    
    if not 0 <= args.gamma <= 1:
        errors.append(f"Gamma must be in [0,1], got {args.gamma}")
    
    if args.grid_size <= 0:
        errors.append(f"Grid size must be positive, got {args.grid_size}")
    
    if args.tolerance <= 0:
        errors.append(f"Tolerance must be positive, got {args.tolerance}")
    
    if args.max_iterations <= 0:
        errors.append(f"Max iterations must be positive, got {args.max_iterations}")
    
    if errors:
        print("Parameter validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

def run_simulation_with_parameters(args):
    """Run simulation with given parameters."""
    
    print("=" * 60)
    print("ALERT FATIGUE BELLMAN EQUATION SIMULATION")
    print("=" * 60)
    
    # Validate parameters
    validate_parameters(args)
    
    # Prepare parameter dictionaries
    tau_values = {
        '11': args.tau11,
        '10': args.tau10,
        '01': args.tau01,
        '00': args.tau00
    }
    
    p_values = {
        '11': args.p11,
        '10': args.p10,
        '01': args.p01,
        '00': args.p00
    }
    
    # Initialize simulator
    simulator = BellmanSimulator(
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        tau_values=tau_values,
        p_values=p_values,
        trust_range=(args.trust_min, args.trust_max),
        grid_size=args.grid_size
    )
    
    # Update convergence parameters
    simulator.tolerance = args.tolerance
    simulator.max_iterations = args.max_iterations
    
    print("\nParameters:")
    print(f"  β (discount factor): {simulator.beta}")
    print(f"  α (trust update): {simulator.alpha}")
    print(f"  γ (human decision prob): {simulator.gamma}")
    print(f"  Treatment effects: {simulator.tau_values}")
    print(f"  Type probabilities: {simulator.p_values}")
    print(f"  Trust range: [{simulator.trust_min}, {simulator.trust_max}]")
    print(f"  Grid size: {simulator.grid_size}")
    print(f"  Tolerance: {simulator.tolerance}")
    print(f"  Max iterations: {simulator.max_iterations}")
    
    # Solve the Bellman equation
    print("\n" + "=" * 40)
    print("SOLVING BELLMAN EQUATION")
    print("=" * 40)
    
    convergence_info = simulator.solve_value_iteration(verbose=args.verbose)
    
    print(f"\nConvergence Results:")
    print(f"  Iterations: {convergence_info['iterations']}")
    print(f"  Max change: {convergence_info['max_change']:.2e}")
    print(f"  Converged: {convergence_info['converged']}")
    
    if not convergence_info['converged']:
        print("WARNING: Simulation did not converge!")
    
    # Analyze results
    print("\n" + "=" * 40)
    print("ANALYZING RESULTS")
    print("=" * 40)
    
    policy_df = simulator.analyze_policy()
    
    print(f"\nPolicy Distribution (10-branch):")
    policy_10_counts = policy_df['policy_10_branch'].value_counts()
    for policy, count in policy_10_counts.items():
        percentage = count / len(policy_df) * 100
        policy_full = {'B': 'Backward', 'C': 'Current', 'F': 'Forward'}[policy]
        print(f"  {policy_full}: {count} points ({percentage:.1f}%)")
    
    print(f"\nPolicy Distribution (01-branch):")
    policy_01_counts = policy_df['policy_01_branch'].value_counts()
    for policy, count in policy_01_counts.items():
        percentage = count / len(policy_df) * 100
        policy_full = {'B': 'Backward', 'C': 'Current', 'F': 'Forward'}[policy]
        print(f"  {policy_full}: {count} points ({percentage:.1f}%)")
    
    # Find trust thresholds where policy changes (10-branch)
    policy_10_changes = []
    for i in range(1, len(policy_df)):
        if policy_df.iloc[i]['policy_10_code'] != policy_df.iloc[i-1]['policy_10_code']:
            trust_val = policy_df.iloc[i]['trust']
            old_policy = policy_df.iloc[i-1]['policy_10_branch']
            new_policy = policy_df.iloc[i]['policy_10_branch']
            policy_10_changes.append((trust_val, old_policy, new_policy))
    
    # Find trust thresholds where policy changes (01-branch)
    policy_01_changes = []
    for i in range(1, len(policy_df)):
        if policy_df.iloc[i]['policy_01_code'] != policy_df.iloc[i-1]['policy_01_code']:
            trust_val = policy_df.iloc[i]['trust']
            old_policy = policy_df.iloc[i-1]['policy_01_branch']
            new_policy = policy_df.iloc[i]['policy_01_branch']
            policy_01_changes.append((trust_val, old_policy, new_policy))
    
    print(f"\nPolicy Change Points (10-branch):")
    for trust, old_pol, new_pol in policy_10_changes:
        policy_map = {'B': 'Backward', 'C': 'Current', 'F': 'Forward'}
        print(f"  t = {trust:.3f}: {policy_map[old_pol]} → {policy_map[new_pol]}")
    
    print(f"\nPolicy Change Points (01-branch):")
    for trust, old_pol, new_pol in policy_01_changes:
        policy_map = {'B': 'Backward', 'C': 'Current', 'F': 'Forward'}
        print(f"  t = {trust:.3f}: {policy_map[old_pol]} → {policy_map[new_pol]}")
    
    # Value function statistics
    print(f"\nValue Function Statistics:")
    print(f"  Min value: {np.min(simulator.V):.4f}")
    print(f"  Max value: {np.max(simulator.V):.4f}")
    print(f"  Mean value: {np.mean(simulator.V):.4f}")
    print(f"  Std value: {np.std(simulator.V):.4f}")
    
    # Create simulation_results directory
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate plots if requested
    plot_file = None
    if not args.no_plots:
        print("\n" + "=" * 40)
        print("GENERATING PLOTS")
        print("=" * 40)
        
        plot_file = os.path.join(results_dir, f"{args.output_prefix}_results.png")
        simulator.plot_results(plot_file)
        print(f"  Plot saved: {plot_file}")
    
    # Create comprehensive results DataFrame
    print("\n" + "=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    
    results_df = simulator.create_results_dataframe(plot_file)
    
    # Save comprehensive results
    results_file = os.path.join(results_dir, f"{args.output_prefix}_comprehensive_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"  Comprehensive results saved: {results_file}")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("Files created in simulation_results/ folder:")
    if plot_file:
        print(f"  - {os.path.basename(plot_file)}: Visualization plots")
    print(f"  - {os.path.basename(results_file)}: Comprehensive results with all parameters and arrays")
    
    return simulator, results_df

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load parameters from CSV if provided
    if args.parameter_csv:
        params = load_parameters_from_csv(args.parameter_csv, args.run_index)
        print(f"Loading parameters from CSV row {args.run_index}")
        print(f"Parameters: {params}")
        
        # Update args with CSV parameter values
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        # Override output_prefix with row number if using CSV
        args.output_prefix = f"row_{args.run_index}"
        print(f"Using output prefix: {args.output_prefix}")
    
    # Load configuration file if provided
    elif args.config:
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Run the simulation
    simulator, results = run_simulation_with_parameters(args)
