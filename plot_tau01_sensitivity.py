#!/usr/bin/env python3
"""
Sensitivity Analysis: How optimal policies vary with tau01 across trust levels.

This script generates two sets of heatmaps:
1. Based on numerical simulation (value iteration)
2. Based on closed-form solutions (when available)

Each set shows how the optimal policy changes as a function of trust level (x-axis) 
and tau01 value (y-axis), with all other parameters fixed.

Usage:
    python plot_tau01_sensitivity.py --beta 0.9 --alpha 0.02 --gamma 0.9 \
        --tau11 1.0 --tau10 1.0 --tau00 -1.0 \
        --p11 0.2 --p10 0.4 --p01 0.2 --p00 0.2 \
        --tau01-min -1.0 --tau01-max 2.0 --tau01-steps 20 \
        --output-prefix sensitivity_results/tau01
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from datetime import datetime
from bellman_simulation import BellmanSimulator
from run_simulation_w_cf import ComprehensiveComparison
import pandas as pd

class Logger:
    """Simple logger that writes to both console and file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = open(log_file, 'w')
    
    def log(self, message):
        """Write message to both console and log file."""
        print(message)
        self.file_handle.write(message + '\n')
        self.file_handle.flush()
    
    def close(self):
        """Close the log file."""
        self.file_handle.close()

def create_sensitivity_plots(
    beta, alpha, gamma,
    tau11, tau10, tau00,
    p11, p10, p01, p00,
    tau01_min, tau01_max, tau01_steps,
    trust_grid_size=100,
    output_prefix='tau01_sensitivity',
    logger=None
):
    """
    Create sensitivity analysis plots for tau01 using both simulation and closed-form.
    
    Parameters:
    -----------
    All Bellman parameters except tau01 which varies
    tau01_min, tau01_max, tau01_steps: Range and resolution for tau01
    trust_grid_size: Number of trust levels to evaluate
    output_prefix: Prefix for output files (will create _simulation.png and _closedform.png)
    logger: Logger instance for output (if None, prints to console only)
    """
    
    # Create tau01 values to test
    tau01_values = np.linspace(tau01_min, tau01_max, tau01_steps)
    
    # Create trust grid (only [0, 1] range)
    trust_grid = np.linspace(0, 1, trust_grid_size)
    
    # Initialize arrays to store policies
    # Simulation policies
    sim_policy_10_grid = np.zeros((tau01_steps, trust_grid_size))
    sim_policy_01_grid = np.zeros((tau01_steps, trust_grid_size))
    
    # Closed-form policies
    cf_policy_10_grid = np.zeros((tau01_steps, trust_grid_size))
    cf_policy_01_grid = np.zeros((tau01_steps, trust_grid_size))
    
    # Track which branch was selected for each tau01
    selected_branches = []
    
    log = logger.log if logger else print
    
    log(f"Running sensitivity analysis...")
    log(f"  tau01 range: [{tau01_min:.2f}, {tau01_max:.2f}] with {tau01_steps} steps")
    log(f"  Trust levels: {trust_grid_size} points in [0, 1]")
    log(f"  Fixed parameters: β={beta}, α={alpha}, γ={gamma}")
    log(f"  τ={{11:{tau11}, 10:{tau10}, 00:{tau00}}}")
    log(f"  p={{11:{p11}, 10:{p10}, 01:{p01}, 00:{p00}}}")
    log("")
    
    # Run simulation and closed-form for each tau01 value
    for i, tau01 in enumerate(tau01_values):
        log(f"  Processing tau01 = {tau01:.3f} ({i+1}/{tau01_steps})...")
        
        tau_values = {
            '11': tau11,
            '10': tau10,
            '01': tau01,
            '00': tau00
        }
        
        p_values = {
            '11': p11,
            '10': p10,
            '01': p01,
            '00': p00
        }
        
        # 1. Run numerical simulation
        sim = BellmanSimulator(
            beta=beta,
            alpha=alpha,
            gamma=gamma,
            tau_values=tau_values,
            p_values=p_values,
            trust_range=(0, 1),
            grid_size=trust_grid_size
        )
        sim.solve_value_iteration(verbose=False)
        sim_policy_10_grid[i, :] = sim.policy_10
        sim_policy_01_grid[i, :] = sim.policy_01
        
        # 2. Run closed-form comparison to get best branch
        params_row = pd.Series({
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'tau11': tau11,
            'tau10': tau10,
            'tau01': tau01,
            'tau00': tau00,
            'p11': p11,
            'p10': p10,
            'p01': p01,
            'p00': p00,
            'row_index': i
        })
        
        comp = ComprehensiveComparison(params_row, 'temp_sensitivity')
        
        # Get closed-form policies from the selected branch
        trust_grid_comp, numerical_V_comp = comp.run_numerical_simulation()
        branches = {}
        branches['A'] = comp.compute_closed_form_A(trust_grid_comp)
        branches['B'] = comp.compute_closed_form_B(trust_grid_comp)
        branches['C'] = comp.compute_closed_form_C(trust_grid_comp)
        branches['D'] = comp.compute_closed_form_D(trust_grid_comp)
        
        # Get numerical V(1) for selection
        numerical_V_at_1_idx = np.argmin(np.abs(trust_grid_comp - 1.0))
        numerical_V_at_1 = numerical_V_comp[numerical_V_at_1_idx]
        best_branch = comp.select_best_branch(numerical_V_at_1, branches)
        selected_branches.append(best_branch)
        
        # Get policies from the best closed-form branch
        if best_branch:
            # Get the V_values from the branches dictionary
            V_values, is_non_decreasing, V_at_1 = branches[best_branch]
            
            # Compute policies from the closed-form V_values
            # We need to evaluate which action is optimal at each trust level
            # Create a temporary simulator to compute policies from V_values
            temp_sim = BellmanSimulator(
                beta=beta,
                alpha=alpha,
                gamma=gamma,
                tau_values=tau_values,
                p_values=p_values,
                trust_range=(0, 1),
                grid_size=trust_grid_size
            )
            
            # Set the value function to the closed-form solution
            temp_sim.V = V_values
            temp_sim.trust_grid = trust_grid_comp
            
            # Compute policies from this V function
            cf_policy_10 = np.zeros(trust_grid_size, dtype=int)
            cf_policy_01 = np.zeros(trust_grid_size, dtype=int)
            
            for j, t in enumerate(trust_grid):
                # Find closest point in trust_grid_comp
                idx = np.argmin(np.abs(trust_grid_comp - t))
                t_comp = trust_grid_comp[idx]
                
                # Compute Q-values for each action
                # 10-branch (human=1, AI=0)
                t_forward = min(max(t_comp + alpha, 0), 1)
                t_backward = min(max(t_comp - alpha, 0), 1)
                t_current = t_comp
                
                V_forward = np.interp(t_forward, trust_grid_comp, V_values)
                V_backward = np.interp(t_backward, trust_grid_comp, V_values)
                V_current = np.interp(t_current, trust_grid_comp, V_values)
                
                Q_forward_10 = tau_values['10'] + beta * V_forward
                Q_backward_10 = (1 - min(max(t_comp, 0), 1)) * tau_values['10'] + beta * V_backward
                Q_current_10 = gamma * tau_values['10'] + beta * V_current
                
                cf_policy_10[j] = np.argmax([Q_backward_10, Q_current_10, Q_forward_10])
                
                # 01-branch (human=0, AI=1)
                Q_forward_01 = beta * V_forward
                Q_backward_01 = min(max(t_comp, 0), 1) * tau_values['01'] + beta * V_backward
                Q_current_01 = (1 - gamma) * tau_values['01'] + beta * V_current
                
                cf_policy_01[j] = np.argmax([Q_backward_01, Q_current_01, Q_forward_01])
            
            cf_policy_10_grid[i, :] = cf_policy_10
            cf_policy_01_grid[i, :] = cf_policy_01
        else:
            # If no branch selected, use simulation policies
            cf_policy_10_grid[i, :] = sim_policy_10_grid[i, :]
            cf_policy_01_grid[i, :] = sim_policy_01_grid[i, :]
    
    log("")
    log(f"  ✓ Completed all simulations and closed-form computations!")
    log(f"  Branch selection summary: {dict(pd.Series(selected_branches).value_counts())}")
    
    # Create two separate plots
    create_plot(sim_policy_10_grid, sim_policy_01_grid, tau01_values, trust_grid,
                tau01_min, tau01_max, beta, alpha, gamma, tau11, tau10, tau00,
                p11, p10, p01, p00, f"{output_prefix}_simulation.png",
                "Numerical Simulation", None, log)
    
    create_plot(cf_policy_10_grid, cf_policy_01_grid, tau01_values, trust_grid,
                tau01_min, tau01_max, beta, alpha, gamma, tau11, tau10, tau00,
                p11, p10, p01, p00, f"{output_prefix}_closedform.png",
                "Closed-Form Solution", selected_branches, log)
    
    return sim_policy_10_grid, sim_policy_01_grid, cf_policy_10_grid, cf_policy_01_grid


def create_plot(policy_10_grid, policy_01_grid, tau01_values, trust_grid,
                tau01_min, tau01_max, beta, alpha, gamma, tau11, tau10, tau00,
                p11, p10, p01, p00, output_path, title_prefix, selected_branches=None, log_func=print):
    """Create a single sensitivity plot."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color map: 0=Disagree/Backward (red), 1=Silent/Current (yellow), 2=Agree/Forward (green)
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot 1: 10-branch policy
    im1 = ax1.imshow(
        policy_10_grid,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=[trust_grid[0], trust_grid[-1], tau01_min, tau01_max]
    )
    ax1.set_xlabel('Trust Level (t)', fontsize=12)
    ax1.set_ylabel('τ⁰¹ Value', fontsize=12)
    title1 = f'10-Branch Policy ({title_prefix}): How Optimal Policy Changes with τ⁰¹ and Trust Level'
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for 10-branch
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.set_label('Policy', fontsize=11)
    cbar1.ax.set_yticklabels(['Disagree (B)', 'Silent (C)', 'Agree (F)'])
    
    # Plot 2: 01-branch policy
    im2 = ax2.imshow(
        policy_01_grid,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=[trust_grid[0], trust_grid[-1], tau01_min, tau01_max]
    )
    ax2.set_xlabel('Trust Level (t)', fontsize=12)
    ax2.set_ylabel('τ⁰¹ Value', fontsize=12)
    title2 = f'01-Branch Policy ({title_prefix}): How Optimal Policy Changes with τ⁰¹ and Trust Level'
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for 01-branch
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar2.set_label('Policy', fontsize=11)
    cbar2.ax.set_yticklabels(['Disagree (B)', 'Silent (C)', 'Agree (F)'])
    
    # Add parameter info
    param_text = (
        f'Fixed Parameters: β={beta}, α={alpha}, γ={gamma}, '
        f'τ¹¹={tau11}, τ¹⁰={tau10}, τ⁰⁰={tau00}, '
        f'p¹¹={p11}, p¹⁰={p10}, p⁰¹={p01}, p⁰⁰={p00}'
    )
    
    # Add branch selection info for closed-form plot
    if selected_branches:
        branch_counts = pd.Series(selected_branches).value_counts()
        branch_text = ', '.join([f'{b}: {c}' for b, c in branch_counts.items()])
        param_text += f'\nSelected Branches: {branch_text}'
    
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_func(f"  ✓ Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis: How optimal policies vary with tau01'
    )
    
    # Fixed parameters
    parser.add_argument('--beta', type=float, required=True,
                       help='Discount factor (0 < β < 1)')
    parser.add_argument('--alpha', type=float, required=True,
                       help='Trust update magnitude')
    parser.add_argument('--gamma', type=float, required=True,
                       help='Human decision probability')
    
    # Fixed tau values (except tau01)
    parser.add_argument('--tau11', type=float, required=True,
                       help='Treatment effect for type (1,1)')
    parser.add_argument('--tau10', type=float, required=True,
                       help='Treatment effect for type (1,0)')
    parser.add_argument('--tau00', type=float, required=True,
                       help='Treatment effect for type (0,0)')
    
    # Probabilities
    parser.add_argument('--p11', type=float, required=True,
                       help='Probability of type (1,1)')
    parser.add_argument('--p10', type=float, required=True,
                       help='Probability of type (1,0)')
    parser.add_argument('--p01', type=float, required=True,
                       help='Probability of type (0,1)')
    parser.add_argument('--p00', type=float, required=True,
                       help='Probability of type (0,0)')
    
    # tau01 range
    parser.add_argument('--tau01-min', type=float, default=-1.0,
                       help='Minimum tau01 value (default: -1.0)')
    parser.add_argument('--tau01-max', type=float, default=2.0,
                       help='Maximum tau01 value (default: 2.0)')
    parser.add_argument('--tau01-steps', type=int, default=20,
                       help='Number of tau01 values to test (default: 20)')
    
    # Grid parameters
    parser.add_argument('--trust-grid-size', type=int, default=100,
                       help='Number of trust levels to evaluate (default: 100)')
    
    # Output
    parser.add_argument('--output-prefix', type=str, default='sensitivity_results/tau01',
                       help='Output file prefix (will create _simulation.png and _closedform.png)')
    
    args = parser.parse_args()
    
    # Validate probabilities sum to 1
    prob_sum = args.p11 + args.p10 + args.p01 + args.p00
    if not np.isclose(prob_sum, 1.0):
        print(f"Error: Probabilities must sum to 1.0 (got {prob_sum})")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{args.output_prefix}_log_{timestamp}.txt"
    logger = Logger(log_file)
    
    logger.log(f"Sensitivity Analysis Log")
    logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_file}")
    logger.log("=" * 70)
    logger.log("")
    
    try:
        # Create the plots
        create_sensitivity_plots(
            beta=args.beta,
            alpha=args.alpha,
            gamma=args.gamma,
            tau11=args.tau11,
            tau10=args.tau10,
            tau00=args.tau00,
            p11=args.p11,
            p10=args.p10,
            p01=args.p01,
            p00=args.p00,
            tau01_min=args.tau01_min,
            tau01_max=args.tau01_max,
            tau01_steps=args.tau01_steps,
            trust_grid_size=args.trust_grid_size,
            output_prefix=args.output_prefix,
            logger=logger
        )
        
        logger.log("")
        logger.log("=" * 70)
        logger.log(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"All results saved with prefix: {args.output_prefix}")
        
    finally:
        logger.close()


if __name__ == '__main__':
    main()

