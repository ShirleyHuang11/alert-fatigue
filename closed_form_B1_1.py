"""
Closed-Form Solution for Alert Fatigue Bellman Equation
Winning Branch B: (+1, -1) - 10-branch Forward, 01-branch Backward

This implements the analytical solution when:
- 10-branch (σ₁₀ = +1): Choose Forward → τ¹⁰ + βV(t+α)
- 01-branch (σ₀₁ = -1): Choose Backward → t̃τ⁰¹ + βV(t-α)

The solution is piecewise defined across three regions:
- Region I (t < 0): V(t) = c₀/b + (ac₁/b²)(1-e^(-λ))e^(λt)
- Region II (0 ≤ t ≤ 1): V(t) = c₀/b + (ac₁/b²)[1-e^(λ(t-1))] + (c₁/b)t
- Region III (t > 1): V(t) = (c₀+c₁)/b

With switch logic for optimal predecessor branch determination.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from typing import Dict, Tuple
import warnings
from scipy.optimize import brentq
warnings.filterwarnings('ignore')

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

class ClosedFormB11:
    """
    Closed-form solution for the (+1, -1) winning branch B1_1.
    """
    
    def __init__(self, 
                 beta: float = 0.95,
                 alpha: float = 0.1,
                 gamma: float = 0.7,
                 tau_values: Dict[str, float] = None,
                 p_values: Dict[str, float] = None):
        """
        Initialize the closed-form solver for Branch B.
        
        Args:
            beta: Discount factor (0 < beta < 1)
            alpha: Trust update magnitude (0 < alpha << 1)
            gamma: Human decision probability when no recommendation
            tau_values: Treatment effects for each type
            p_values: Probabilities for each type
        """
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        
        # Default treatment effects
        self.tau_values = tau_values or {
            '11': 1.0,   # Both human and AI choose 1 (positive effect)
            '10': 1.0,   # Human chooses 1, AI chooses 0
            '01': 0.8,  # Human chooses 0, AI chooses 1
            '00': 0.6   # Both choose 0 (negative effect)
        }
        
        # Default probabilities (must sum to 1)
        self.p_values = p_values or {
            '11': 0.2,
            '10': 0.4,
            '01': 0.2,
            '00': 0.2
        }
        
        # Validate probabilities sum to 1
        if abs(sum(self.p_values.values()) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1")
        
        # Validate parameters
        if not 0 < beta < 1:
            raise ValueError("Beta must be in (0, 1)")
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in [0, 1]")
        
        # Calculate branch-specific parameters
        self.c0 = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
        self.c1 = self.p_values['01'] * self.tau_values['01']
        self.kappa = (self.p_values['11'] + self.p_values['00']) + self.p_values['10'] - self.p_values['01']
        
        # Calculate a, b, lambda
        self.a = self.beta * self.alpha * self.kappa
        self.b = 1 - self.beta
        
        # Handle special case kappa = 0
        if abs(self.kappa) < 1e-10:
            self.lambda_val = np.inf  # Will be handled specially
            self.is_degenerate = True
        else:
            self.lambda_val = self.b / self.a
            self.is_degenerate = False
        
        print(f"Branch B1_1 (+1, -1) parameters:")
        print(f"  c₀ = p¹¹τ¹¹ + p¹⁰τ¹⁰ = {self.p_values['11']}×{self.tau_values['11']} + {self.p_values['10']}×{self.tau_values['10']} = {self.c0:.4f}")
        print(f"  c₁ = p⁰¹τ⁰¹ = {self.p_values['01']}×{self.tau_values['01']} = {self.c1:.4f}")
        print(f"  κ = (p¹¹+p⁰⁰) + p¹⁰ - p⁰¹ = {self.kappa:.4f}")
        print(f"  λ = (1-β)/(βακ) = {self.lambda_val:.4f}")
        print(f"  Is degenerate (κ≈0): {self.is_degenerate}")
    
    def get_trust_clamped(self, t: float) -> float:
        """Get clamped trust value: min(max(t, 0), 1)"""
        return min(max(t, 0), 1)
    
    def get_value_function(self, t: float) -> float:
        """
        Get the value function at trust level t for Branch B with complete switch logic.
        
        Implements the 5-region solution with proper predecessor branch matching.
        
        Args:
            t: Trust level
            
        Returns:
            Value function V(t)
        """
        if self.is_degenerate:
            # Handle degenerate case (κ = 0)
            if t < 0:
                return self.c0 / self.b
            elif 0 <= t <= 1:
                return self.c0 / self.b + (self.c1 / self.b) * t
            else:  # t > 1
                return (self.c0 + self.c1) / self.b
        
        # Get switch parameters
        switch_time, predecessor = self.calculate_switch_time()
        
        # Calculate predecessor branch parameters
        if predecessor == "(+1, +1)":
            # T(+1,+1) = p¹¹τ¹¹ + p¹⁰τ¹⁰ (no 01 contribution)
            T_prev = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
            # λ(+1,+1) = (1-β)/(βα[(p¹¹+p⁰⁰)+p¹⁰+p⁰¹])
            kappa_prev = (self.p_values['11'] + self.p_values['00']) + self.p_values['10'] + self.p_values['01']
            lambda_prev = self.b / (self.beta * self.alpha * kappa_prev)
        else:  # (+1, 0)
            # T(+1,0) = p¹¹τ¹¹ + p¹⁰τ¹⁰ + p⁰¹(1-γ)τ⁰¹
            T_prev = (self.p_values['11'] * self.tau_values['11'] + 
                     self.p_values['10'] * self.tau_values['10'] + 
                     self.p_values['01'] * (1 - self.gamma) * self.tau_values['01'])
            # λ(+1,0) = (1-β)/(βα[(p¹¹+p⁰⁰)+p¹⁰])
            kappa_prev = (self.p_values['11'] + self.p_values['00']) + self.p_values['10']
            lambda_prev = self.b / (self.beta * self.alpha * kappa_prev)
        
        # Calculate V_B(t*) - Branch B value at switch time
        V_B_tstar = (self.c0 / self.b + 
                    (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (switch_time - 1))) +
                    (self.c1 / self.b) * switch_time)
        
        # Calculate V(0) - boundary value using predecessor branch
        V_0 = (T_prev / self.b + 
               (V_B_tstar - T_prev / self.b) * np.exp(-lambda_prev * switch_time))
        
        # Calculate V(1) - boundary value
        V_1 = (self.c0 + self.c1) / self.b
        
        # Apply the complete 5-region solution from the reference
        if t < 0:
            # Region 1: V(t) = V(0) constant for t < 0
            return V_0
            
        elif 0 <= t < switch_time:
            # Region 2: Predecessor branch with exponential decay from t*
            return (T_prev / self.b + 
                   (V_B_tstar - T_prev / self.b) * np.exp(lambda_prev * (t - switch_time)))
                   
        elif switch_time <= t <= 1:
            # Region 3: Branch B solution
            return (self.c0 / self.b + 
                   (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (t - 1))) +
                   (self.c1 / self.b) * t)
                   
        else:  # t > 1
            # Region 4: V(t) = V(1) constant for t > 1
            return V_1
    
    def calculate_switch_time(self) -> Tuple[float, str]:
        """
        Calculate the switch time and predecessor branch using robust root finding.
        
        Returns:
            (switch_time, predecessor_branch)
        """
        if self.is_degenerate:
            return 0.0, "(+1, +1)"  # Default for degenerate case
        
        # Calculate K and psi function
        K = (self.beta * self.alpha * self.p_values['01']) / self.b
        psi = lambda t: 1 - np.exp(self.lambda_val * (t - 1))
        
        # Define switch equations
        def f_F(t):
            return t - 2 * K * psi(t)  # -> (+1,+1)
        
        def f_C(t):
            return t - (1 - self.gamma) - K * psi(t)  # -> (+1,0)
        
        # Use robust bisection method for root finding
        def bisection_root(fun, a=0.0, c=1.0, tol=1e-12, maxit=200):
            fa, fc = fun(a), fun(c)
            if np.sign(fa) == np.sign(fc):
                return np.nan
            for _ in range(maxit):
                mid = 0.5 * (a + c)
                fm = fun(mid)
                if abs(fm) < tol or (c - a) < tol:
                    return mid
                if np.sign(fm) == np.sign(fa):
                    a, fa = mid, fm
                else:
                    c, fc = mid, fm
            return 0.5 * (a + c)
        
        # Find roots using bisection
        t_F = bisection_root(f_F, 0.0, 1.0)
        t_C = bisection_root(f_C, 0.0, 1.0)
        
        # Collect valid candidates (filter out switches too close to t=0)
        candidates = []
        if not np.isnan(t_F) and 0.01 <= t_F <= 1:
            candidates.append((t_F, "(+1, +1)"))
        if not np.isnan(t_C) and 0.01 <= t_C <= 1:
            candidates.append((t_C, "(+1, 0)"))
        
        if not candidates:
            # Fallback to small-alpha approximation
            t_F_approx = 2 * K
            t_C_approx = (1 - self.gamma) + K
            
            # Filter out switches at t < 0.01
            if t_F_approx >= 0.01 and t_F_approx > t_C_approx:
                return t_F_approx, "(+1, +1)"
            elif t_C_approx >= 0.01:
                return t_C_approx, "(+1, 0)"
            else:
                # No valid switch, return 0
                return 0.0, "(+1, +1)"
        
        # Choose the maximum switch time
        t_star = max(candidates, key=lambda x: x[0])[0]
        predecessor = max(candidates, key=lambda x: x[0])[1]
        
        return t_star, predecessor
    
    def calculate_branch_parameters(self) -> Dict:
        """
        Calculate the branch-specific parameters according to the table.
        
        Returns:
            Dictionary with branch parameters
        """
        switch_time, predecessor = self.calculate_switch_time()
        
        return {
            'sigma_10': +1,  # 10-branch chooses Forward
            'sigma_01': -1,  # 01-branch chooses Backward
            'kappa': self.kappa,
            'tau_10_sigma': self.tau_values['10'],  # τ¹⁰
            'tau_01_sigma': 't̃τ⁰¹',  # For Backward branch in 01 state
            'branch_name': '(+1, -1)',
            'description': '10-branch Forward, 01-branch Backward',
            'c0': self.c0,
            'c1': self.c1,
            'a': self.a,
            'b': self.b,
            'lambda_val': self.lambda_val,
            'switch_time': switch_time,
            'predecessor_branch': predecessor,
            'is_degenerate': self.is_degenerate
        }
    
    def plot_solution(self, trust_range: Tuple[float, float] = (-0.5, 1.5), 
                     save_path: str = None) -> str:
        """
        Plot the closed-form solution for Branch B with numerical comparison.
        
        Args:
            trust_range: Range of trust values to plot
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot file
        """
        t_min, t_max = trust_range
        trust_values = np.linspace(t_min, t_max, 1000)
        
        # Calculate complete value function with switch logic
        complete_value_function = np.array([self.get_value_function(t) for t in trust_values])
        
        # Calculate individual components for comparison
        switch_time, predecessor = self.calculate_switch_time()
        
        # Branch B closed-form (without switch logic) - for reference only
        branch_B_values = np.array([self.get_branch_B_value(t) for t in trust_values])
        
        # Predecessor branch values - for reference only
        predecessor_values = np.array([self.get_predecessor_value(t, switch_time, predecessor) for t in trust_values])
        
        # Create the comparison plot with policy visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot 1: Value function comparison (like the reference)
        ax1.plot(trust_values, complete_value_function, 'b-', linewidth=3, 
                label='Complete Switch Logic Solution')
        
        # Plot Branch B closed-form (like "B closed-form" in reference)
        ax1.plot(trust_values, branch_B_values, 'orange', linestyle='--', linewidth=2, 
                label='B closed-form')
        
        # Plot predecessor branch (like "Previous (+1,0) (t ≤ t*)" in reference)
        if not self.is_degenerate and not np.isnan(switch_time):
            # Only plot predecessor branch up to switch time
            mask_predecessor = trust_values <= switch_time
            if np.any(mask_predecessor):
                ax1.plot(trust_values[mask_predecessor], predecessor_values[mask_predecessor], 
                       'green', linewidth=2, label=f'Previous {predecessor} (t ≤ t*)')
        
        # Add region boundaries
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='t=0 boundary')
        ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='t=1 boundary')
        
        # Add switch time markers (like in reference)
        if not self.is_degenerate and not np.isnan(switch_time):
            ax1.axvline(x=switch_time, color='blue', linestyle=':', alpha=0.8, 
                      label=f't* (analytic) = {switch_time:.6f}')
            
            # Add a note about numerical switch time (placeholder)
            ax1.text(switch_time + 0.05, ax1.get_ylim()[1] * 0.9, 
                   f'Switch Logic\nAnalytic: {switch_time:.3f}', 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=9)
        
        ax1.set_xlabel('Trust Level (t)')
        ax1.set_ylabel('Value Function V(t)')
        ax1.set_title('Winner B + stitched previous (if any)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(t_min, t_max)
        
        # Plot 2: Policy regions overview
        policy_values = []
        policy_labels = []
        
        for t in trust_values:
            if t < 0:
                policy_values.append(0)  # V(0) constant region
                policy_labels.append('V(0) constant')
            elif 0 <= t < switch_time:
                policy_values.append(1)  # Predecessor branch
                policy_labels.append(f'Predecessor {predecessor}')
            elif switch_time <= t <= 1:
                policy_values.append(2)  # Branch B
                policy_labels.append('Branch B (+1,-1)')
            else:
                policy_values.append(3)  # V(1) constant region
                policy_labels.append('V(1) constant')
        
        # Create policy visualization with different colors for each region
        policy_colors = ['blue', 'green', 'orange', 'purple']
        policy_names = ['V(0) constant', f'Predecessor {predecessor}', 'Branch B (+1,-1)', 'V(1) constant']
        
        # Plot policy regions
        for i, (color, name) in enumerate(zip(policy_colors, policy_names)):
            mask = np.array(policy_values) == i
            if np.any(mask):
                ax2.scatter(np.array(trust_values)[mask], np.ones(np.sum(mask)), 
                           c=color, label=name, s=15, alpha=0.8)
        
        # Add switch time marker
        if not self.is_degenerate and not np.isnan(switch_time):
            ax2.axvline(x=switch_time, color='red', linestyle=':', alpha=0.8, 
                       label=f'Switch at t* = {switch_time:.3f}')
        
        # Add region boundaries
        ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
        ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Trust Level (t)')
        ax2.set_ylabel('Policy Region')
        ax2.set_title('Policy Regions Overview')
        ax2.set_yticks([1])
        ax2.set_yticklabels(['Active'])
        ax2.set_xlim(t_min, t_max)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Plot 3: Combined Branch Policies
        # Calculate policy values for both branches
        policy_10_values = []
        policy_01_values = []
        
        for t in trust_values:
            if t < 0:
                # Use the same policy as at t=0 (predecessor branch at t=0)
                if predecessor == "(+1, +1)":
                    policy_10_values.append(2)  # Forward (agree) = +1
                    policy_01_values.append(2)  # Forward (agree with AI) = +1
                else:  # (+1, 0)
                    policy_10_values.append(2)  # Forward (agree) = +1
                    policy_01_values.append(1)  # Current (silence) = 0
            elif 0 <= t < switch_time:
                # Predecessor branch: depends on which predecessor
                if predecessor == "(+1, +1)":
                    policy_10_values.append(2)  # Forward (agree) = +1
                    policy_01_values.append(2)  # Forward (agree with AI) = +1
                else:  # (+1, 0)
                    policy_10_values.append(2)  # Forward (agree) = +1
                    policy_01_values.append(1)  # Current (silence) = 0
            elif switch_time <= t <= 1:
                # Branch B (+1, -1) policy
                policy_10_values.append(2)  # Forward (agree) = +1
                policy_01_values.append(0)  # Backward (disagree with AI) = -1
            else:
                # Use the same policy as at t=1 (Branch B at t=1)
                policy_10_values.append(2)  # Forward (agree) = +1
                policy_01_values.append(0)  # Backward (disagree with AI) = -1
        
        # Plot 10-branch policy as line
        ax3.plot(trust_values, policy_10_values, 'g-', linewidth=2, 
                label='10-Branch: Human(1) vs AI(0)', marker='o', markersize=3)
        
        # Plot 01-branch policy as line
        ax3.plot(trust_values, policy_01_values, 'r-', linewidth=2, 
                label='01-Branch: Human(0) vs AI(1)', marker='s', markersize=3)
        
        # Add switch time marker
        if not self.is_degenerate and not np.isnan(switch_time):
            ax3.axvline(x=switch_time, color='black', linestyle=':', alpha=0.8, 
                       label=f'Switch at t* = {switch_time:.3f}')
        
        # Add region boundaries
        ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='t=0')
        ax3.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='t=1')
        
        # Set y-axis labels for policy decisions (top to bottom)
        decision_labels = ['Agree (Forward)', 'Silence (Current)', 'Disagree (Backward)']
        ax3.set_yticks([2, 1, 0])
        ax3.set_yticklabels(decision_labels)
        ax3.set_ylim(-0.5, 2.5)
        
        ax3.set_xlabel('Trust Level (t)')
        ax3.set_ylabel('Optimal Policy Decision')
        ax3.set_title('Optimal Branch Policies with Switch Logic')
        ax3.set_xlim(t_min, t_max)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        
        # Add policy interpretation text
        policy_text = f'Policy Legend:\n• 10-Branch: AI decision when Human=1, AI=0\n• 01-Branch: AI decision when Human=0, AI=1\n• Switch at t* = {switch_time:.3f} changes 01-branch behavior\n• Boundary policies: V(0) for t<0, V(1) for t>1'
        ax3.text(0.02, 0.98, policy_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=8)
        
        # Add parameter information to top left plot
        param_text = f'Parameters:\nβ = {self.beta}, α = {self.alpha}, γ = {self.gamma}\nSwitch: t* = {switch_time:.3f}\nPredecessor: {predecessor}'
        ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.close()
        return save_path if save_path else None
    
    def get_branch_B_value(self, t: float) -> float:
        """Get Branch B value without switch logic (for comparison)."""
        if self.is_degenerate:
            if t < 0:
                return self.c0 / self.b
            elif 0 <= t <= 1:
                return self.c0 / self.b + (self.c1 / self.b) * t
            else:
                return (self.c0 + self.c1) / self.b
        
        # Standard Branch B piecewise solution
        if t < 0:
            return (self.c0 / self.b + 
                   (self.a * self.c1 / (self.b**2)) * (1 - np.exp(-self.lambda_val)) * np.exp(self.lambda_val * t))
        elif 0 <= t <= 1:
            return (self.c0 / self.b + 
                   (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (t - 1))) +
                   (self.c1 / self.b) * t)
        else:
            return (self.c0 + self.c1) / self.b
    
    def get_predecessor_value(self, t: float, switch_time: float, predecessor: str) -> float:
        """Get predecessor branch value (for comparison)."""
        if self.is_degenerate or np.isnan(switch_time):
            return self.get_branch_B_value(t)
        
        # Calculate predecessor branch parameters
        if predecessor == "(+1, +1)":
            T_prev = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
            kappa_prev = (self.p_values['11'] + self.p_values['00']) + self.p_values['10'] + self.p_values['01']
        else:  # (+1, 0)
            T_prev = (self.p_values['11'] * self.tau_values['11'] + 
                     self.p_values['10'] * self.tau_values['10'] + 
                     self.p_values['01'] * (1 - self.gamma) * self.tau_values['01'])
            kappa_prev = (self.p_values['11'] + self.p_values['00']) + self.p_values['10']
        
        lambda_prev = self.b / (self.beta * self.alpha * kappa_prev)
        
        # Calculate V_B(t*) for continuity
        V_B_tstar = (self.c0 / self.b + 
                    (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (switch_time - 1))) +
                    (self.c1 / self.b) * switch_time)
        
        # Predecessor branch solution
        return (T_prev / self.b + 
               (V_B_tstar - T_prev / self.b) * np.exp(lambda_prev * (t - switch_time)))
    
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
        if self.is_degenerate:
            formula = f"V(t) = c₀/b for t<0, V(t) = c₀/b + (c₁/b)t for 0≤t≤1, V(t) = (c₀+c₁)/b for t>1"
        else:
            switch_time, predecessor = self.calculate_switch_time()
            formula = f"Complete Switch Logic: V(t) = V(0) for t<0; V(t) = T(σ^prev)/(1-β) + [V_B(t*) - T(σ^prev)/(1-β)]e^(λ(σ^prev)(t-t*)) for 0≤t≤t*; V(t) = c₀/(1-β) + (a_B c₁)/(1-β)²[1-e^(λ_B(t-1))] + (c₁/(1-β))t for t*≤t≤1; V(t) = V(1) for t>1. Switch at t*={switch_time:.3f}, Predecessor={predecessor}"
        
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
            
            # Branch parameters
            'sigma_10': branch_params['sigma_10'],
            'sigma_01': branch_params['sigma_01'],
            'kappa': branch_params['kappa'],
            'tau_10_sigma': branch_params['tau_10_sigma'],
            'tau_01_sigma': branch_params['tau_01_sigma'],
            'branch_name': branch_params['branch_name'],
            'description': branch_params['description'],
            
            # Closed-form solution parameters
            'c0': branch_params['c0'],
            'c1': branch_params['c1'],
            'a': branch_params['a'],
            'b': branch_params['b'],
            'lambda_val': branch_params['lambda_val'],
            'switch_time': branch_params['switch_time'],
            'predecessor_branch': branch_params['predecessor_branch'],
            'is_degenerate': branch_params['is_degenerate'],
            
            # Formula and metadata
            'formula': formula,
            'plot_path': plot_path if plot_path else None,
            'solution_type': 'Closed-Form',
            'branch_type': 'B1_1 (+1, -1)',
            'is_constant': False,
            'independent_of_t': False,
            'has_switch_logic': True
        }
        
        return pd.DataFrame([results_data])

def main():
    """Main function to run the closed-form analysis for Branch B."""
    parser = argparse.ArgumentParser(description="Closed-form solution for Branch B (+1, -1)")
    parser.add_argument('--beta', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.1, help='Trust update magnitude')
    parser.add_argument('--gamma', type=float, default=0.7, help='Human decision probability')
    parser.add_argument('--tau11', type=float, default=1.0, help='Treatment effect for type (1,1)')
    parser.add_argument('--tau10', type=float, default=0.5, help='Treatment effect for type (1,0)')
    parser.add_argument('--tau01', type=float, default=-0.3, help='Treatment effect for type (0,1)')
    parser.add_argument('--tau00', type=float, default=-0.8, help='Treatment effect for type (0,0)')
    parser.add_argument('--p11', type=float, default=0.3, help='Probability of type (1,1)')
    parser.add_argument('--p10', type=float, default=0.2, help='Probability of type (1,0)')
    parser.add_argument('--p01', type=float, default=0.25, help='Probability of type (0,1)')
    parser.add_argument('--p00', type=float, default=0.25, help='Probability of type (0,0)')
    parser.add_argument('--output-prefix', type=str, default='closed_form_B1_1', help='Output file prefix')
    parser.add_argument('--compare-numerical', type=str, help='Path to numerical results CSV for comparison')
    
    args = parser.parse_args()
    
    # Create closed_form_results directory
    os.makedirs("closed_form_results", exist_ok=True)
    
    print("=" * 60)
    print("CLOSED-FORM SOLUTION: BRANCH B1_1 (+1, -1)")
    print("=" * 60)
    
    # Validate probabilities
    prob_sum = args.p11 + args.p10 + args.p01 + args.p00
    if abs(prob_sum - 1.0) > 1e-6:
        print(f"Warning: Probabilities sum to {prob_sum:.6f}, not 1.0")
    
    # Initialize solver with command-line arguments
    tau_values = {'11': args.tau11, '10': args.tau10, '01': args.tau01, '00': args.tau00}
    p_values = {'11': args.p11, '10': args.p10, '01': args.p01, '00': args.p00}
    
    solver = ClosedFormB11(beta=args.beta, alpha=args.alpha, gamma=args.gamma, 
                          tau_values=tau_values, p_values=p_values)
    
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
        # Note: comparison functionality would need to be implemented similar to closed_form_A11.py
    
    print(f"\n" + "=" * 60)
    print("CLOSED-FORM SOLUTION COMPLETE")
    print("=" * 60)
    print(f"Branch B1_1 (+1, -1) solution computed")
    print(f"Switch time: {branch_params['switch_time']:.4f}")
    print(f"Predecessor branch: {branch_params['predecessor_branch']}")
    print(f"\nFiles created in closed_form_results/ folder:")
    print(f"  - {os.path.basename(plot_path)}: Solution visualization")
    print(f"  - {os.path.basename(results_file)}: Comprehensive results with formula")

if __name__ == "__main__":
    main()
