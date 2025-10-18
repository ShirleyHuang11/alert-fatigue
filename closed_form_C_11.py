"""
Closed-form solution for Branch C: (-1, +1) winning branch.

This module implements the analytical solution for the trust dynamics model
when the optimal policy at the end (t=1) is to disagree in the 10-state 
and agree in the 01-state.

Branch C: (-1, +1)
- 10-branch: Backward (disagree)
- 01-branch: Forward (agree)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import argparse
import os
from typing import Tuple, Dict

class ClosedFormC11:
    """
    Closed-form solution for Branch C: (-1, +1).
    
    This class computes the analytical solution for the value function
    when the policy chooses Backward in 10-state and Forward in 01-state.
    """
    
    def __init__(self, beta: float = 0.9, alpha: float = 0.05, gamma: float = 0.5,
                 tau_values: Dict[str, float] = None, p_values: Dict[str, float] = None):
        """
        Initialize the closed-form solver.
        
        Args:
            beta: Discount factor (0 < beta < 1)
            alpha: Trust update step size
            gamma: Silence reward factor
            tau_values: Dictionary of outcome values {'11', '10', '01', '00'}
            p_values: Dictionary of state probabilities {'11', '10', '01', '00'}
        """
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        
        # Default tau values (no tau00 needed for Branch C)
        self.tau_values = tau_values if tau_values is not None else {
            '11': 1.0,
            '10': -1.0,
            '01': 0.05,
            '00': 0.0  # Not used in Branch C
        }
        
        # Default probability values
        self.p_values = p_values if p_values is not None else {
            '11': 0.2,
            '10': 0.2,
            '01': 0.4,
            '00': 0.2
        }
        
        # Branch C parameters
        self.sigma_10 = -1  # Backward in 10-state
        self.sigma_01 = 1   # Forward in 01-state
        
        # Calculate basic parameters
        self.b = 1 - beta
        self.kappa = (self.p_values['11'] + self.p_values['00']) - self.p_values['10'] + self.p_values['01']
        # Note: kappa_C = 1 - 2*p^{10}
        
        # Check for degeneracy
        self.is_degenerate = abs(self.kappa) < 1e-10
        
        if not self.is_degenerate:
            self.a = self.beta * self.alpha * self.kappa
            self.lambda_val = self.b / self.a
        else:
            self.a = 0.0
            self.lambda_val = 0.0
        
        # Calculate c0 and c1
        self.c0 = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
        self.c1 = -self.p_values['10'] * self.tau_values['10']
    
    def clip_trust(self, t: float) -> float:
        """Clip trust to [0, 1] range."""
        return min(max(t, 0.0), 1.0)
    
    def get_branch_C_value(self, t: float) -> float:
        """
        Get the raw Branch C value (without switch logic).
        This is the piecewise solution for regions I, II, III.
        """
        if self.is_degenerate:
            # Linear solution when kappa = 0
            t_tilde = self.clip_trust(t)
            if t < 0:
                return self.c0 / self.b
            elif t <= 1:
                return self.c0 / self.b + (self.c1 / self.b) * t
            else:
                return (self.c0 + self.c1) / self.b
        
        # Non-degenerate case
        if t < 0:
            # Region I: t < 0
            C_minus = (self.a * self.c1 / (self.b**2)) * (1 - np.exp(-self.lambda_val))
            return self.c0 / self.b + C_minus * np.exp(self.lambda_val * t)
        elif t <= 1:
            # Region II: 0 <= t <= 1
            C_0 = -(self.a * self.c1 / (self.b**2)) * np.exp(-self.lambda_val)
            return (self.c0 / self.b + 
                   (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (t - 1))) +
                   (self.c1 / self.b) * t)
        else:
            # Region III: t > 1
            c_plus = self.p_values['11'] * self.tau_values['11']
            return c_plus / self.b
    
    def calculate_switch_time(self) -> Tuple[float, str]:
        """
        Calculate the switch time and predecessor branch using robust root finding.
        
        Returns:
            (switch_time, predecessor_branch)
        """
        if self.is_degenerate:
            return 0.0, "(+1, +1)"  # Default for degenerate case
        
        # Calculate K_10 and psi function
        K_10 = (self.beta * self.alpha * self.p_values['10']) / self.b
        psi = lambda t: 1 - np.exp(self.lambda_val * (t - 1))
        
        # Define switch equations
        def g_F(t):
            return t - 2 * K_10 * psi(t)  # -> (+1,+1)
        
        def g_C(t):
            return t - (1 - self.gamma) - K_10 * psi(t)  # -> (0,+1)
        
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
        t_F = bisection_root(g_F, 0.0, 1.0)
        t_C = bisection_root(g_C, 0.0, 1.0)
        
        # Collect valid candidates (filter out switches too close to t=0)
        candidates = []
        if not np.isnan(t_F) and 0.01 <= t_F <= 1:
            candidates.append((t_F, "(+1, +1)"))
        if not np.isnan(t_C) and 0.01 <= t_C <= 1:
            candidates.append((t_C, "(0, +1)"))
        
        if not candidates:
            return 0.0, "(+1, +1)"
        
        # Choose the maximum (rightmost) switch time
        switch_time, predecessor = max(candidates, key=lambda x: x[0])
        return switch_time, predecessor
    
    def get_value_function(self, t: float) -> float:
        """
        Get the complete value function with switch logic.
        
        This implements the full 5-region solution:
        - t < 0: constant V(0)
        - 0 <= t <= t*: predecessor branch
        - t* <= t <= 1: Branch C
        - t > 1: constant V(1)
        """
        switch_time, predecessor = self.calculate_switch_time()
        
        # Calculate boundary values
        V_1 = self.p_values['11'] * self.tau_values['11'] / self.b
        
        # V_C(t*)
        if self.is_degenerate:
            V_C_tstar = self.c0 / self.b + (self.c1 / self.b) * switch_time
        else:
            V_C_tstar = (self.c0 / self.b + 
                        (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (switch_time - 1))) +
                        (self.c1 / self.b) * switch_time)
        
        # Get predecessor parameters
        if predecessor == "(+1, +1)":
            T_prev = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
            lambda_prev = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                               self.p_values['10'] + self.p_values['01']))
        else:  # (0, +1)
            T_prev = (self.p_values['11'] * self.tau_values['11'] + 
                     self.p_values['10'] * self.gamma * self.tau_values['10'])
            lambda_prev = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                               self.p_values['10']))
        
        # V(0)
        V_0 = T_prev / self.b + (V_C_tstar - T_prev / self.b) * np.exp(-lambda_prev * switch_time)
        
        # Apply the complete switch logic
        if t < 0:
            return V_0
        elif t <= switch_time:
            # Predecessor branch region
            return T_prev / self.b + (V_C_tstar - T_prev / self.b) * np.exp(lambda_prev * (t - switch_time))
        elif t <= 1:
            # Branch C region
            if self.is_degenerate:
                return self.c0 / self.b + (self.c1 / self.b) * t
            else:
                return (self.c0 / self.b + 
                       (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (t - 1))) +
                       (self.c1 / self.b) * t)
        else:
            return V_1
    
    def get_predecessor_value(self, t: float, switch_time: float, predecessor: str) -> float:
        """
        Get the predecessor branch value at trust level t.
        """
        # Get predecessor parameters
        if predecessor == "(+1, +1)":
            T_prev = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
            lambda_prev = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                               self.p_values['10'] + self.p_values['01']))
        else:  # (0, +1)
            T_prev = (self.p_values['11'] * self.tau_values['11'] + 
                     self.p_values['10'] * self.gamma * self.tau_values['10'])
            lambda_prev = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                               self.p_values['10']))
        
        # V_C(t*)
        if self.is_degenerate:
            V_C_tstar = self.c0 / self.b + (self.c1 / self.b) * switch_time
        else:
            V_C_tstar = (self.c0 / self.b + 
                        (self.a * self.c1 / (self.b**2)) * (1 - np.exp(self.lambda_val * (switch_time - 1))) +
                        (self.c1 / self.b) * switch_time)
        
        # Predecessor branch solution
        return (T_prev / self.b + 
               (V_C_tstar - T_prev / self.b) * np.exp(lambda_prev * (t - switch_time)))
    
    def calculate_branch_parameters(self) -> Dict:
        """Calculate and return all branch parameters."""
        switch_time, predecessor = self.calculate_switch_time()
        
        return {
            'sigma_10': self.sigma_10,
            'sigma_01': self.sigma_01,
            'kappa': self.kappa,
            'tau_10_sigma': '(1-t̃)τ¹⁰',
            'tau_01_sigma': '0',
            'branch_name': '(-1, +1)',
            'description': '10-branch Backward, 01-branch Forward',
            'c0': self.c0,
            'c1': self.c1,
            'a': self.a,
            'b': self.b,
            'lambda_val': self.lambda_val,
            'switch_time': switch_time,
            'predecessor_branch': predecessor,
            'is_degenerate': self.is_degenerate
        }
    
    def plot_solution(self, t_min: float = -0.5, t_max: float = 1.5, 
                     save_path: str = None) -> None:
        """
        Plot the closed-form solution with switch logic visualization.
        
        Args:
            t_min: Minimum trust value to plot
            t_max: Maximum trust value to plot
            save_path: Path to save the plot (if None, displays interactively)
        """
        trust_values = np.linspace(t_min, t_max, 1000)
        
        # Calculate complete value function with switch logic
        complete_value_function = np.array([self.get_value_function(t) for t in trust_values])
        
        # Calculate individual components for comparison
        switch_time, predecessor = self.calculate_switch_time()
        
        # Branch C closed-form (without switch logic) - for reference only
        branch_C_values = np.array([self.get_branch_C_value(t) for t in trust_values])
        
        # Predecessor branch values - for reference only
        predecessor_values = np.array([self.get_predecessor_value(t, switch_time, predecessor) for t in trust_values])
        
        # Create the comparison plot with policy visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot 1: Value function comparison (like the reference)
        ax1.plot(trust_values, complete_value_function, 'b-', linewidth=3, 
                label='Complete Switch Logic Solution')
        
        # Plot Branch C closed-form (like "C closed-form" in reference)
        ax1.plot(trust_values, branch_C_values, 'orange', linestyle='--', linewidth=2, 
                label='C closed-form')
        
        # Plot predecessor branch (like "Previous (+1,+1) or (0,+1) (t ≤ t*)" in reference)
        if not self.is_degenerate and not np.isnan(switch_time):
            # Only plot predecessor branch up to switch time
            mask_predecessor = trust_values <= switch_time
            if np.any(mask_predecessor):
                ax1.plot(trust_values[mask_predecessor], predecessor_values[mask_predecessor], 
                       'green', linewidth=2, label=f'Previous {predecessor} (t ≤ t*)')
        
        # Add region boundaries
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='t=0 boundary')
        ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.7, label='t=1 boundary')
        
        if not self.is_degenerate and not np.isnan(switch_time):
            ax1.axvline(x=switch_time, color='red', linestyle=':', linewidth=2, 
                       label=f't* = {switch_time:.3f}')
        
        ax1.set_xlabel('Trust Level (t)', fontsize=12)
        ax1.set_ylabel('Value Function V(t)', fontsize=12)
        ax1.set_title('Closed-Form Solution: Branch C (-1, +1)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Policy Regions Overview
        ax2.axhspan(-0.1, 0.1, xmin=0, xmax=(0 - t_min)/(t_max - t_min), 
                   alpha=0.3, color='lightblue', label='V(0) constant')
        
        if not self.is_degenerate and not np.isnan(switch_time):
            ax2.axhspan(-0.1, 0.1, xmin=(0 - t_min)/(t_max - t_min), 
                       xmax=(switch_time - t_min)/(t_max - t_min),
                       alpha=0.3, color='lightgreen', label=f'Predecessor {predecessor}')
            ax2.axhspan(-0.1, 0.1, xmin=(switch_time - t_min)/(t_max - t_min), 
                       xmax=(1 - t_min)/(t_max - t_min),
                       alpha=0.3, color='lightyellow', label='Branch C (-1,+1)')
        else:
            ax2.axhspan(-0.1, 0.1, xmin=(0 - t_min)/(t_max - t_min), 
                       xmax=(1 - t_min)/(t_max - t_min),
                       alpha=0.3, color='lightyellow', label='Branch C (-1,+1)')
        
        ax2.axhspan(-0.1, 0.1, xmin=(1 - t_min)/(t_max - t_min), xmax=1, 
                   alpha=0.3, color='lightcoral', label='V(1) constant')
        
        ax2.set_xlim(t_min, t_max)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_xlabel('Trust Level (t)', fontsize=12)
        ax2.set_title('Policy Regions', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_yticks([])
        
        # Plot 3: Combined Branch Policies (10-branch and 01-branch)
        # Calculate policies for both branches
        policy_10_values = []
        policy_01_values = []
        
        for t in trust_values:
            if t < 0:
                # Use predecessor policy at t=0
                if predecessor == "(+1, +1)":
                    policy_10_values.append(2)  # Forward
                    policy_01_values.append(2)  # Forward
                else:  # (0, +1)
                    policy_10_values.append(1)  # Current
                    policy_01_values.append(2)  # Forward
            elif t <= switch_time:
                # Predecessor branch
                if predecessor == "(+1, +1)":
                    policy_10_values.append(2)  # Forward
                    policy_01_values.append(2)  # Forward
                else:  # (0, +1)
                    policy_10_values.append(1)  # Current
                    policy_01_values.append(2)  # Forward
            elif t <= 1:
                # Branch C (-1, +1)
                policy_10_values.append(0)  # Backward (disagree)
                policy_01_values.append(2)  # Forward (agree)
            else:
                # Use Branch C policy at t=1
                policy_10_values.append(0)  # Backward
                policy_01_values.append(2)  # Forward
        
        policy_10_values = np.array(policy_10_values)
        policy_01_values = np.array(policy_01_values)
        
        # Plot both policies on the same axes
        ax3.plot(trust_values, policy_10_values, 'b-', linewidth=2, label='10-branch policy', alpha=0.7)
        ax3.plot(trust_values, policy_01_values, 'r--', linewidth=2, label='01-branch policy', alpha=0.7)
        
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Disagree (Backward)', 'Silence (Current)', 'Agree (Forward)'])
        ax3.set_xlabel('Trust Level (t)', fontsize=12)
        ax3.set_ylabel('Optimal Policy', fontsize=12)
        ax3.set_title('Combined Branch Policies', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
        ax3.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
        if not self.is_degenerate and not np.isnan(switch_time):
            ax3.axvline(x=switch_time, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def create_results_dataframe(self, plot_path: str = None) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all parameters and results.
        
        Args:
            plot_path: Path to the generated plot file
            
        Returns:
            DataFrame with all parameters, results, and metadata
        """
        branch_params = self.calculate_branch_parameters()
        switch_time, predecessor = self.calculate_switch_time()
        
        # Calculate all the additional parameters from the mathematical formulation
        # Basic parameters
        c0 = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
        c1 = -self.p_values['10'] * self.tau_values['10']
        a_C = self.beta * self.alpha * (1 - 2 * self.p_values['10'])
        lambda_C = self.b / a_C if abs(a_C) > 1e-10 else 0.0
        
        # Predecessor branch parameters
        T_plus1_plus1 = self.p_values['11'] * self.tau_values['11'] + self.p_values['10'] * self.tau_values['10']
        lambda_plus1_plus1 = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                                   self.p_values['10'] + self.p_values['01']))
        
        T_0_plus1 = (self.p_values['11'] * self.tau_values['11'] + 
                    self.p_values['10'] * self.gamma * self.tau_values['10'])
        lambda_0_plus1 = self.b / (self.beta * self.alpha * ((self.p_values['11'] + self.p_values['00']) + 
                                                              self.p_values['10']))
        
        # Switch time determination
        sigma_prev = predecessor
        
        # V_C(t*) calculation
        V_C_tstar = (c0 / self.b + 
                    (a_C * c1 / (self.b**2)) * (1 - np.exp(lambda_C * (switch_time - 1))) +
                    (c1 / self.b) * switch_time)
        
        # Boundary values
        if predecessor == "(+1, +1)":
            T_prev = T_plus1_plus1
            lambda_prev = lambda_plus1_plus1
        else:  # (0, +1)
            T_prev = T_0_plus1
            lambda_prev = lambda_0_plus1
        
        V_0 = (T_prev / self.b + 
               (V_C_tstar - T_prev / self.b) * np.exp(-lambda_prev * switch_time))
        V_1 = self.p_values['11'] * self.tau_values['11'] / self.b
        
        # Create trust level grid and corresponding V values
        trust_grid = np.linspace(-0.5, 1.5, 1000)
        V_values = np.array([self.get_value_function(t) for t in trust_grid])
        
        # Calculate optimal policies for each trust level
        policy_10_branch = []
        policy_01_branch = []
        
        for t in trust_grid:
            if t < 0:
                # Use predecessor policy at t=0
                if predecessor == "(+1, +1)":
                    policy_10_branch.append('F')  # Forward
                    policy_01_branch.append('F')  # Forward
                else:  # (0, +1)
                    policy_10_branch.append('C')  # Current
                    policy_01_branch.append('F')  # Forward
            elif t <= switch_time:
                # Predecessor branch
                if predecessor == "(+1, +1)":
                    policy_10_branch.append('F')  # Forward
                    policy_01_branch.append('F')  # Forward
                else:  # (0, +1)
                    policy_10_branch.append('C')  # Current
                    policy_01_branch.append('F')  # Forward
            elif t <= 1:
                # Branch C (-1, +1)
                policy_10_branch.append('B')  # Backward (disagree)
                policy_01_branch.append('F')  # Forward (agree)
            else:
                # Use Branch C policy at t=1
                policy_10_branch.append('B')  # Backward
                policy_01_branch.append('F')  # Forward
        
        # Create the mathematical formula string
        if self.is_degenerate:
            formula = f"V(t) = c₀/b for t<0, V(t) = c₀/b + (c₁/b)t for 0≤t≤1, V(t) = c₊/b for t>1"
        else:
            formula = f"Complete Switch Logic: V(t) = V(0) for t<0; V(t) = T(σ^prev)/(1-β) + [V_C(t*) - T(σ^prev)/(1-β)]e^(λ(σ^prev)(t-t*)) for 0≤t≤t*; V(t) = c₀/(1-β) + (a_C c₁)/(1-β)²[1-e^(λ_C(t-1))] + (c₁/(1-β))t for t*≤t≤1; V(t) = V(1) for t>1. Switch at t*={switch_time:.3f}, Predecessor={predecessor}"
        
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
            
            # Additional mathematical parameters
            'c0_calc': c0,
            'c1_calc': c1,
            'a_C': a_C,
            'lambda_C': lambda_C,
            'T_plus1_plus1': T_plus1_plus1,
            'lambda_plus1_plus1': lambda_plus1_plus1,
            'T_0_plus1': T_0_plus1,
            'lambda_0_plus1': lambda_0_plus1,
            'sigma_prev': sigma_prev,
            'V_C_tstar': V_C_tstar,
            'V_0': V_0,
            'V_1': V_1,
            
            # Trust grid and V values as arrays
            'trust_grid': trust_grid.tolist(),  # Convert to list for CSV storage
            'V_values': V_values.tolist(),      # Convert to list for CSV storage
            
            # Optimal policies as arrays
            'policy_10_branch': policy_10_branch,  # B=Backward, C=Current, F=Forward
            'policy_01_branch': policy_01_branch,  # B=Backward, C=Current, F=Forward
            
            # Formula and metadata
            'formula': formula,
            'plot_path': plot_path if plot_path else None,
            'solution_type': 'Closed-Form',
            'branch_type': 'C_11 (-1, +1)',
            'is_constant': False,
            'independent_of_t': False,
            'has_switch_logic': True
        }
        
        return pd.DataFrame([results_data])

def main():
    """Main function to run the closed-form analysis for Branch C."""
    parser = argparse.ArgumentParser(description="Closed-form solution for Branch C (-1, +1)")
    parser.add_argument('--beta', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='Trust update step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Silence reward factor')
    parser.add_argument('--tau11', type=float, default=1.0, help='Outcome value for (1,1)')
    parser.add_argument('--tau10', type=float, default=-1.0, help='Outcome value for (1,0)')
    parser.add_argument('--tau01', type=float, default=0.05, help='Outcome value for (0,1)')
    parser.add_argument('--tau00', type=float, default=0.0, help='Outcome value for (0,0) (not used)')
    parser.add_argument('--p11', type=float, default=0.2, help='Probability of state (1,1)')
    parser.add_argument('--p10', type=float, default=0.2, help='Probability of state (1,0)')
    parser.add_argument('--p01', type=float, default=0.4, help='Probability of state (0,1)')
    parser.add_argument('--p00', type=float, default=0.2, help='Probability of state (0,0)')
    
    args = parser.parse_args()
    
    # Create tau and p dictionaries
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
    
    # Create solver instance
    solver = ClosedFormC11(
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        tau_values=tau_values,
        p_values=p_values
    )
    
    # Print branch information
    print("=" * 60)
    print("CLOSED-FORM SOLUTION: BRANCH C_11 (-1, +1)")
    print("=" * 60)
    
    params = solver.calculate_branch_parameters()
    print(f"Branch C_11 (-1, +1) parameters:")
    print(f"  c₀ = p¹¹τ¹¹ + p¹⁰τ¹⁰ = {solver.p_values['11']:.1f}×{solver.tau_values['11']:.1f} + {solver.p_values['10']:.1f}×{solver.tau_values['10']:.1f} = {solver.c0:.4f}")
    print(f"  c₁ = -p¹⁰τ¹⁰ = -{solver.p_values['10']:.1f}×{solver.tau_values['10']:.1f} = {solver.c1:.4f}")
    print(f"  κ = 1 - 2p¹⁰ = {params['kappa']:.4f}")
    print(f"  λ = (1-β)/(βακ) = {params['lambda_val']:.4f}")
    print(f"  Is degenerate (κ≈0): {params['is_degenerate']}")
    print()
    
    # Print all branch parameters
    print("Branch Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = "closed_form_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot
    plot_path = os.path.join(output_dir, "closed_form_C_11_solution.png")
    solver.plot_solution(t_min=-0.5, t_max=1.5, save_path=plot_path)
    
    # Create and save results DataFrame
    print("\n" + "=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    results_df = solver.create_results_dataframe(plot_path=plot_path)
    results_path = os.path.join(output_dir, "closed_form_C_11_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✓ Comprehensive results saved: {results_path}")
    
    print("\n" + "=" * 60)
    print("CLOSED-FORM SOLUTION COMPLETE")
    print("=" * 60)
    print(f"Branch C_11 (-1, +1) solution computed")
    print(f"Switch time: {params['switch_time']:.4f}")
    print(f"Predecessor branch: {params['predecessor_branch']}")
    print(f"\nFiles created in {output_dir}/ folder:")
    print(f"  - closed_form_C_11_solution.png: Solution visualization")
    print(f"  - closed_form_C_11_results.csv: Comprehensive results with formula")

if __name__ == "__main__":
    main()

