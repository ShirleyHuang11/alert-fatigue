#!/usr/bin/env python3
"""
2D Sensitivity Analysis: Proportion of Agreement across τ¹⁰ and τ⁰¹.

This script generates heatmaps showing the proportion of "Agree (F)" policy
across the trust range [0, 1] as a function of both τ¹⁰ and τ⁰¹.

Creates separate plots for:
1. 10-branch agreement proportion
2. 01-branch agreement proportion

Usage:
    python plot_tau10_tau01_agreement.py --beta 0.9 --alpha 0.02 --gamma 0.9 \
        --tau11 1.0 --tau00 -1.0 \
        --p11 0.2 --p10 0.4 --p01 0.2 --p00 0.2 \
        --tau10-min -1.0 --tau10-max 1.0 --tau10-steps 20 \
        --tau01-min -1.0 --tau01-max 1.0 --tau01-steps 20 \
        --trust-grid-size 100 \
        --output-prefix sensitivity_results/tau10_tau01_agreement
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from datetime import datetime
from bellman_simulation import BellmanSimulator
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

def create_agreement_heatmap(
    beta, alpha, gamma,
    tau11, tau00,
    p11, p10, p01, p00,
    tau10_min, tau10_max, tau10_steps,
    tau01_min, tau01_max, tau01_steps,
    trust_grid_size=100,
    output_prefix='tau10_tau01_agreement',
    logger=None
):
    """
    Create 2D heatmap showing proportion of agreement across τ¹⁰ and τ⁰¹.
    
    Parameters:
    -----------
    All Bellman parameters
    tau10_min, tau10_max, tau10_steps: Range and resolution for tau10
    tau01_min, tau01_max, tau01_steps: Range and resolution for tau01
    trust_grid_size: Number of trust levels to evaluate
    output_prefix: Prefix for output files
    logger: Logger instance for output
    """
    
    # Create parameter grids
    tau10_values = np.linspace(tau10_min, tau10_max, tau10_steps)
    tau01_values = np.linspace(tau01_min, tau01_max, tau01_steps)
    
    # Create trust grid
    trust_grid = np.linspace(0, 1, trust_grid_size)
    
    # Initialize arrays to store agreement proportions
    agreement_10_grid = np.zeros((tau01_steps, tau10_steps))  # Note: rows=tau01, cols=tau10
    agreement_01_grid = np.zeros((tau01_steps, tau10_steps))
    
    log = logger.log if logger else print
    
    log(f"Running 2D sensitivity analysis...")
    log(f"  tau10 range: [{tau10_min:.2f}, {tau10_max:.2f}] with {tau10_steps} steps")
    log(f"  tau01 range: [{tau01_min:.2f}, {tau01_max:.2f}] with {tau01_steps} steps")
    log(f"  Trust levels: {trust_grid_size} points in [0, 1]")
    log(f"  Fixed parameters: β={beta}, α={alpha}, γ={gamma}")
    log(f"  τ={{11:{tau11}, 00:{tau00}}}")
    log(f"  p={{11:{p11}, 10:{p10}, 01:{p01}, 00:{p00}}}")
    log(f"  Total simulations: {tau10_steps * tau01_steps}")
    log("")
    
    # Run simulation for each (tau10, tau01) combination
    total_sims = tau10_steps * tau01_steps
    sim_count = 0
    
    for i, tau01 in enumerate(tau01_values):
        for j, tau10 in enumerate(tau10_values):
            sim_count += 1
            log(f"  Processing ({sim_count}/{total_sims}): τ¹⁰={tau10:.3f}, τ⁰¹={tau01:.3f}...")
            
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
            
            # Run numerical simulation
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
            
            # Calculate proportion of "Agree (F)" policy (policy value = 2)
            # Policy: 0=Disagree(B), 1=Silent(C), 2=Agree(F)
            agreement_10_grid[i, j] = np.mean(sim.policy_10 == 2)
            agreement_01_grid[i, j] = np.mean(sim.policy_01 == 2)
    
    log("")
    log(f"  ✓ Completed all {total_sims} simulations!")
    log(f"  10-branch agreement range: [{agreement_10_grid.min():.3f}, {agreement_10_grid.max():.3f}]")
    log(f"  01-branch agreement range: [{agreement_01_grid.min():.3f}, {agreement_01_grid.max():.3f}]")
    
    # Create plots
    create_heatmap_plot(
        agreement_10_grid, agreement_01_grid,
        tau10_values, tau01_values,
        tau10_min, tau10_max, tau01_min, tau01_max,
        beta, alpha, gamma, tau11, tau00,
        p11, p10, p01, p00,
        output_prefix, log
    )
    
    return agreement_10_grid, agreement_01_grid


def create_heatmap_plot(
    agreement_10_grid, agreement_01_grid,
    tau10_values, tau01_values,
    tau10_min, tau10_max, tau01_min, tau01_max,
    beta, alpha, gamma, tau11, tau00,
    p11, p10, p01, p00,
    output_prefix, log_func=print
):
    """Create 2D heatmap plots."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Use a colormap that goes from red (0% agreement) to green (100% agreement)
    cmap = plt.cm.RdYlGn
    
    # Plot 1: 10-branch agreement proportion
    im1 = ax1.imshow(
        agreement_10_grid,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin='lower',
        extent=[tau10_min, tau10_max, tau01_min, tau01_max]
    )
    ax1.set_xlabel('τ¹⁰ Value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('τ⁰¹ Value', fontsize=14, fontweight='bold')
    ax1.set_title('10-Branch: Proportion of Agree (F) Policy Across Trust Range', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add colorbar for 10-branch
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Proportion of Agree (F)', fontsize=12)
    
    # Add contour lines
    contour1 = ax1.contour(
        tau10_values, tau01_values, agreement_10_grid,
        levels=[0.25, 0.5, 0.75],
        colors='black',
        linewidths=1,
        alpha=0.4
    )
    ax1.clabel(contour1, inline=True, fontsize=8)
    
    # Plot 2: 01-branch agreement proportion
    im2 = ax2.imshow(
        agreement_01_grid,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin='lower',
        extent=[tau10_min, tau10_max, tau01_min, tau01_max]
    )
    ax2.set_xlabel('τ¹⁰ Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('τ⁰¹ Value', fontsize=14, fontweight='bold')
    ax2.set_title('01-Branch: Proportion of Agree (F) Policy Across Trust Range', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add colorbar for 01-branch
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Proportion of Agree (F)', fontsize=12)
    
    # Add contour lines
    contour2 = ax2.contour(
        tau10_values, tau01_values, agreement_01_grid,
        levels=[0.25, 0.5, 0.75],
        colors='black',
        linewidths=1,
        alpha=0.4
    )
    ax2.clabel(contour2, inline=True, fontsize=8)
    
    # Add parameter info
    param_text = (
        f'Fixed Parameters: β={beta}, α={alpha}, γ={gamma}, '
        f'τ¹¹={tau11}, τ⁰⁰={tau00} | '
        f'p¹¹={p11}, p¹⁰={p10}, p⁰¹={p01}, p⁰⁰={p00}\n'
        f'Agreement proportion calculated across trust range [0, 1]'
    )
    
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    output_path = f"{output_prefix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_func(f"  ✓ Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='2D Sensitivity: Agreement proportion across tau10 and tau01'
    )
    
    # Fixed parameters
    parser.add_argument('--beta', type=float, required=True,
                       help='Discount factor (0 < β < 1)')
    parser.add_argument('--alpha', type=float, required=True,
                       help='Trust update magnitude')
    parser.add_argument('--gamma', type=float, required=True,
                       help='Human decision probability')
    
    # Fixed tau values
    parser.add_argument('--tau11', type=float, required=True,
                       help='Treatment effect for type (1,1)')
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
    
    # tau10 range
    parser.add_argument('--tau10-min', type=float, default=-1.0,
                       help='Minimum tau10 value (default: -1.0)')
    parser.add_argument('--tau10-max', type=float, default=1.0,
                       help='Maximum tau10 value (default: 1.0)')
    parser.add_argument('--tau10-steps', type=int, default=20,
                       help='Number of tau10 values to test (default: 20)')
    
    # tau01 range
    parser.add_argument('--tau01-min', type=float, default=-1.0,
                       help='Minimum tau01 value (default: -1.0)')
    parser.add_argument('--tau01-max', type=float, default=1.0,
                       help='Maximum tau01 value (default: 1.0)')
    parser.add_argument('--tau01-steps', type=int, default=20,
                       help='Number of tau01 values to test (default: 20)')
    
    # Grid parameters
    parser.add_argument('--trust-grid-size', type=int, default=100,
                       help='Number of trust levels to evaluate (default: 100)')
    
    # Output
    parser.add_argument('--output-prefix', type=str, 
                       default='sensitivity_results/tau10_tau01_agreement',
                       help='Output file prefix')
    
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
    
    logger.log(f"2D Agreement Proportion Analysis Log")
    logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_file}")
    logger.log("=" * 70)
    logger.log("")
    
    try:
        # Create the heatmap
        create_agreement_heatmap(
            beta=args.beta,
            alpha=args.alpha,
            gamma=args.gamma,
            tau11=args.tau11,
            tau00=args.tau00,
            p11=args.p11,
            p10=args.p10,
            p01=args.p01,
            p00=args.p00,
            tau10_min=args.tau10_min,
            tau10_max=args.tau10_max,
            tau10_steps=args.tau10_steps,
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
        logger.log(f"Results saved with prefix: {args.output_prefix}")
        
    finally:
        logger.close()


if __name__ == '__main__':
    main()

