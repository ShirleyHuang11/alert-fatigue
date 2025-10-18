#!/usr/bin/env python3
"""
Closed-form solution for Branch D: (-1,-1) winning branch.
This branch has the most complex switch logic, potentially requiring two switches.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from typing import Dict, Tuple, Optional
from scipy.optimize import brentq


class ClosedFormD11:
    """Closed-form solution for Branch D: (-1,-1)"""
    
    def __init__(self, beta: float, alpha: float, gamma: float,
                 tau_values: Dict[str, float], p_values: Dict[str, float]):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau_values = tau_values
        self.p_values = p_values
        
        # Basic constants
        self.b = 1 - beta
        
        # Branch D parameters
        self.kappa_D = (p_values['11'] + p_values['00']) - p_values['10'] - p_values['01']
        self.a_D = beta * alpha * self.kappa_D if abs(self.kappa_D) > 1e-10 else 0
        self.lambda_D = self.b / self.a_D if abs(self.a_D) > 1e-10 else 0
        
        # Convenient constants
        self.c0 = p_values['11'] * tau_values['11'] + p_values['10'] * tau_values['10']
        self.c1 = -p_values['10'] * tau_values['10'] + p_values['01'] * tau_values['01']
        self.c_plus = self.c0 + self.c1
        
        # Switch constants
        self.K_10 = (beta * alpha * p_values['10']) / self.b
        self.K_01 = (beta * alpha * p_values['01']) / self.b
        
        # Store switch information
        self.t1_star = None
        self.sigma1 = None
        self.t2_star = None
        self.sigma2 = None
        
        # Calculate switches
        self._calculate_switches()
    
    def _psi_D(self, t: np.ndarray) -> np.ndarray:
        """Helper function for Branch D derivative term."""
        if abs(self.kappa_D) < 1e-10:
            return np.ones_like(t)
        
        if self.kappa_D > 0:
            return 1 - np.exp(self.lambda_D * (t - 1))
        else:  # kappa_D < 0
            return 1 - np.exp(self.lambda_D * t)
    
    def get_branch_D_value(self, t: np.ndarray) -> np.ndarray:
        """Calculate Branch D value function."""
        V = np.zeros_like(t)
        
        if abs(self.kappa_D) < 1e-10:
            # Degenerate case
            mask_below = t < 0
            mask_middle = (t >= 0) & (t <= 1)
            mask_above = t > 1
            
            V[mask_below] = self.c0 / self.b
            V[mask_middle] = (self.c0 + self.c1 * t[mask_middle]) / self.b
            V[mask_above] = self.c_plus / self.b
        else:
            mask_below = t < 0
            mask_middle = (t >= 0) & (t <= 1)
            mask_above = t > 1
            
            # Region I (t < 0)
            if self.kappa_D > 0:
                C_minus = (self.a_D * self.c1 / (self.b ** 2)) * (1 - np.exp(-self.lambda_D))
                V[mask_below] = self.c0 / self.b + C_minus * np.exp(self.lambda_D * t[mask_below])
            else:  # kappa_D < 0, C_minus = 0
                V[mask_below] = self.c0 / self.b
            
            # Region II (0 <= t <= 1)
            psi_middle = self._psi_D(t[mask_middle])
            V[mask_middle] = (self.c0 / self.b + 
                             (self.a_D * self.c1 / (self.b ** 2)) * psi_middle +
                             (self.c1 / self.b) * t[mask_middle])
            
            # Region III (t > 1)
            if self.kappa_D < 0:
                C_plus = (self.a_D * self.c1 / (self.b ** 2)) * (1 - np.exp(self.lambda_D))
                V[mask_above] = self.c_plus / self.b + C_plus * np.exp(self.lambda_D * t[mask_above])
            else:  # kappa_D > 0, C_plus = 0
                V[mask_above] = self.c_plus / self.b
        
        return V
    
    def _get_candidate_params(self, sigma: Tuple[int, int]) -> Tuple[float, float]:
        """Get T and lambda for a given branch sigma."""
        s10, s01 = sigma
        
        # Calculate T(sigma)
        T_sigma = self.p_values['11'] * self.tau_values['11']
        if s10 == 1:  # Forward
            T_sigma += self.p_values['10'] * self.tau_values['10']
        elif s10 == 0:  # Current
            T_sigma += self.p_values['10'] * self.gamma * self.tau_values['10']
        # s10 == -1 (Backward) contributes (1-t)*tau10, handled separately
        
        if s01 == 1:  # Forward
            pass  # contributes 0
        elif s01 == 0:  # Current
            T_sigma += self.p_values['01'] * (1 - self.gamma) * self.tau_values['01']
        elif s01 == -1:  # Backward
            pass  # contributes t*tau01, handled separately
        
        # Calculate kappa(sigma)
        kappa_sigma = (self.p_values['11'] + self.p_values['00'] + 
                      s10 * self.p_values['10'] + s01 * self.p_values['01'])
        
        # Calculate lambda(sigma)
        if abs(kappa_sigma) < 1e-10:
            lambda_sigma = 0
        else:
            lambda_sigma = self.b / (self.beta * self.alpha * kappa_sigma)
        
        return T_sigma, lambda_sigma
    
    def _check_optimality_interval(self, sigma: Tuple[int, int], t_start: float, t_end: float) -> Tuple[bool, Optional[float], Optional[Tuple[int, int]]]:
        """
        Check if a branch sigma remains optimal in [t_start, t_end].
        Returns: (is_optimal, violation_time, new_sigma)
        """
        if sigma is None or t_end <= t_start:
            return True, None, None
        
        # Sample the interval to check optimality
        t_check = np.linspace(t_start, t_end, 100)
        
        # Get X^sigma(t) = beta * alpha * V'^sigma(t)
        # This is the derivative coefficient that must stay within feasibility bounds
        s10, s01 = sigma
        
        # For each conflict state, check if the chosen action remains optimal
        # The feasibility conditions depend on the branch type
        
        # For 10-branch:
        if s10 == 1:  # Forward chosen
            # Must satisfy: X >= 2K_10 * psi_D and X >= (1-gamma) + K_10 * psi_D
            pass  # Forward is often stable near t=1
        elif s10 == 0:  # Current chosen
            # Must satisfy: X is between Backward and Forward bounds
            # Can switch to either Forward or Backward
            pass
        elif s10 == -1:  # Backward chosen
            # Must satisfy: X <= bounds for Current and Forward
            pass
        
        # Simplified check: look for where sigma1 loses to other candidates
        # by checking if any other branch would give higher value
        
        # This is a placeholder for the full optimality check
        # In practice, we'd compute X^sigma(t) and check against feasibility intervals
        
        return True, None, None  # Placeholder - assume optimal for now
    
    def _calculate_switches(self):
        """Calculate switch times and successors with full second-switch logic."""
        # Define switch equations for four candidates
        def h_plus1_minus1(t):
            return t - 2 * self.K_10 * self._psi_D(np.array([t]))[0]
        
        def h_minus1_plus1(t):
            return t - 2 * self.K_01 * self._psi_D(np.array([t]))[0]
        
        def h_0_minus1(t):
            return t - (1 - self.gamma) - self.K_10 * self._psi_D(np.array([t]))[0]
        
        def h_minus1_0(t):
            return t - (1 - self.gamma) - self.K_01 * self._psi_D(np.array([t]))[0]
        
        # Find roots for each candidate
        candidates = []
        equations = [
            (h_plus1_minus1, (1, -1), "(+1,-1)"),
            (h_minus1_plus1, (-1, 1), "(-1,+1)"),
            (h_0_minus1, (0, -1), "(0,-1)"),
            (h_minus1_0, (-1, 0), "(-1,0)")
        ]
        
        for eq_func, sigma, name in equations:
            try:
                # Check if there's a sign change in (0, 1)
                val_start = eq_func(0.01)
                val_end = eq_func(0.99)
                
                if val_start * val_end < 0:
                    t_root = brentq(eq_func, 0.01, 0.99)
                    candidates.append((t_root, sigma, name))
            except:
                pass
        
        # First switch is the maximum
        if not candidates:
            # No switch found, Branch D dominates everywhere
            self.t1_star = 0.0
            self.sigma1 = None
            self.sigma1_name = "None"
            self.t2_star = None
            self.sigma2 = None
            self.sigma2_name = "None"
            return
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        self.t1_star, self.sigma1, self.sigma1_name = candidates[0]
        
        # Now check if sigma1 remains optimal in [0, t1_star]
        # We need to check if sigma1 can lose optimality to another branch
        
        # Get parameters for sigma1
        T_sigma1, lambda_sigma1 = self._get_candidate_params(self.sigma1)
        V_D_t1 = self.get_branch_D_value(np.array([self.t1_star]))[0]
        C1 = V_D_t1 - T_sigma1 / self.b
        
        # Compute V^sigma1(t) for t in [0, t1_star]
        t_check = np.linspace(0.001, self.t1_star, 200)
        V_sigma1 = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (t_check - self.t1_star))
        
        # Compute X^sigma1(t) = beta * alpha * lambda_sigma1 * C1 * exp(lambda_sigma1 * (t - t1_star))
        X_sigma1 = self.beta * self.alpha * lambda_sigma1 * C1 * np.exp(lambda_sigma1 * (t_check - self.t1_star))
        
        # Check if sigma1 loses optimality by testing against other candidates
        # For each point, check if any other candidate branch would be better
        
        # Define feasibility conditions based on sigma1's choices
        s10_1, s01_1 = self.sigma1
        
        # Check where sigma1 might violate optimality
        # We need to check BOTH 10-branch and 01-branch optimality conditions
        violation_idx = None
        violating_branch = None
        
        psi_check = self._psi_D(t_check)
        
        # === Check 10-branch optimality ===
        # For 10-branch, the feasibility conditions depend on what was chosen
        
        if s10_1 == 1:  # Forward chosen in 10-branch
            # Forward is optimal if X >= max(2*K_10*psi, (1-gamma) + K_10*psi)
            # Can lose to:
            #   - Current (0): if X < (1-gamma) + K_10*psi
            #   - Backward (-1): if X < 0 (very rare from Forward)
            
            bound_current_10 = (1 - self.gamma) + self.K_10 * psi_check
            bound_backward_10 = np.zeros_like(t_check)  # Backward vs Forward at X=0
            
            for i in range(len(t_check)):
                if X_sigma1[i] < bound_current_10[i]:
                    # Would switch to Current or Backward
                    # Check which is better: if X < 0, Backward; else Current
                    violation_idx = i
                    violating_branch = (0, s01_1) if X_sigma1[i] >= 0 else (-1, s01_1)
                    break
                    
        elif s10_1 == 0:  # Current chosen in 10-branch
            # Current is optimal if:
            #   Forward not better: X < max(2*K_10*psi, (1-gamma) + K_10*psi)
            #   Backward not better: X > (1-gamma) - K_10*psi or X > 0
            
            bound_forward_10 = np.maximum(2 * self.K_10 * psi_check, 
                                         (1 - self.gamma) + self.K_10 * psi_check)
            bound_backward_10 = np.maximum((1 - self.gamma) - self.K_10 * psi_check, 0)
            
            for i in range(len(t_check)):
                if X_sigma1[i] >= bound_forward_10[i]:
                    # Switch to Forward
                    violation_idx = i
                    violating_branch = (1, s01_1)
                    break
                elif X_sigma1[i] <= bound_backward_10[i]:
                    # Switch to Backward
                    violation_idx = i
                    violating_branch = (-1, s01_1)
                    break
                    
        elif s10_1 == -1:  # Backward chosen in 10-branch
            # Backward is optimal if X <= min((1-gamma) - K_10*psi, 0)
            # Can lose to:
            #   - Current (0): if X > (1-gamma) - K_10*psi
            #   - Forward (1): if X > max(2*K_10*psi, (1-gamma) + K_10*psi)
            
            bound_current_10 = (1 - self.gamma) - self.K_10 * psi_check
            bound_forward_10 = np.maximum(2 * self.K_10 * psi_check,
                                         (1 - self.gamma) + self.K_10 * psi_check)
            
            for i in range(len(t_check)):
                if X_sigma1[i] >= bound_forward_10[i]:
                    # Switch to Forward
                    violation_idx = i
                    violating_branch = (1, s01_1)
                    break
                elif X_sigma1[i] >= bound_current_10[i]:
                    # Switch to Current
                    violation_idx = i
                    violating_branch = (0, s01_1)
                    break
        
        # === Check 01-branch optimality (if no violation found yet) ===
        if violation_idx is None:
            if s01_1 == 1:  # Forward chosen in 01-branch
                # Forward is optimal if X >= max(2*K_01*psi, (1-gamma) + K_01*psi)
                # Can lose to Current or Backward
                
                bound_current_01 = (1 - self.gamma) + self.K_01 * psi_check
                
                for i in range(len(t_check)):
                    if X_sigma1[i] < bound_current_01[i]:
                        violation_idx = i
                        violating_branch = (s10_1, 0) if X_sigma1[i] >= 0 else (s10_1, -1)
                        break
                        
            elif s01_1 == 0:  # Current chosen in 01-branch
                # Current is optimal if between Forward and Backward bounds
                
                bound_forward_01 = np.maximum(2 * self.K_01 * psi_check,
                                             (1 - self.gamma) + self.K_01 * psi_check)
                bound_backward_01 = np.maximum((1 - self.gamma) - self.K_01 * psi_check, 0)
                
                for i in range(len(t_check)):
                    if X_sigma1[i] >= bound_forward_01[i]:
                        violation_idx = i
                        violating_branch = (s10_1, 1)
                        break
                    elif X_sigma1[i] <= bound_backward_01[i]:
                        violation_idx = i
                        violating_branch = (s10_1, -1)
                        break
                        
            elif s01_1 == -1:  # Backward chosen in 01-branch
                # Backward is optimal if X <= min((1-gamma) - K_01*psi, 0)
                # Can lose to Current or Forward
                
                bound_current_01 = (1 - self.gamma) - self.K_01 * psi_check
                bound_forward_01 = np.maximum(2 * self.K_01 * psi_check,
                                             (1 - self.gamma) + self.K_01 * psi_check)
                
                for i in range(len(t_check)):
                    if X_sigma1[i] >= bound_forward_01[i]:
                        violation_idx = i
                        violating_branch = (s10_1, 1)
                        break
                    elif X_sigma1[i] >= bound_current_01[i]:
                        violation_idx = i
                        violating_branch = (s10_1, 0)
                        break
        
        if violation_idx is not None:
            # Found a violation - need second switch
            self.t2_star = t_check[violation_idx]
            
            # If second switch is too close to t=0, ignore it (likely numerical noise)
            if self.t2_star < 0.01:
                self.t2_star = None
                self.sigma2 = None
                self.sigma2_name = "None"
            else:
                # Use the violating_branch we identified from optimality check
                if violating_branch is not None:
                    self.sigma2 = violating_branch
                    # Format the name
                    s10_2, s01_2 = violating_branch
                    s10_str = "+" + str(s10_2) if s10_2 >= 0 else str(s10_2)
                    s01_str = "+" + str(s01_2) if s01_2 >= 0 else str(s01_2)
                    self.sigma2_name = f"({s10_str},{s01_str})"
                else:
                    # Fallback: try to find among original candidates
                    best_sigma2 = None
                    best_name2 = None
                    
                    for t_cand, sigma_cand, name_cand in candidates:
                        if sigma_cand != self.sigma1:
                            if best_sigma2 is None or t_cand < self.t1_star:
                                best_sigma2 = sigma_cand
                                best_name2 = name_cand
                    
                    # If still no candidate, use a sensible default
                    if best_sigma2 is None:
                        if s10_1 == 1:
                            best_sigma2 = (0, -1)
                            best_name2 = "(0,-1)"
                        elif s10_1 == 0:
                            best_sigma2 = (1, -1) if s01_1 == -1 else (-1, 1)
                            best_name2 = "(+1,-1)" if s01_1 == -1 else "(-1,+1)"
                        else:
                            best_sigma2 = (-1, 0)
                            best_name2 = "(-1,0)"
                    
                    self.sigma2 = best_sigma2
                    self.sigma2_name = best_name2
        else:
            # No violation - only one switch
            self.t2_star = None
            self.sigma2 = None
            self.sigma2_name = "None"
    
    def get_value_function(self, t: np.ndarray) -> np.ndarray:
        """Calculate the complete value function with switches."""
        V = np.zeros_like(t)
        
        # If no valid switch, just return Branch D
        if self.sigma1 is None or self.t1_star <= 0:
            V_D = self.get_branch_D_value(t)
            # Apply boundary conditions
            V_0 = self.get_branch_D_value(np.array([0.0]))[0]
            V_1 = self.get_branch_D_value(np.array([1.0]))[0]
            V[t < 0] = V_0
            V[(t >= 0) & (t <= 1)] = V_D[(t >= 0) & (t <= 1)]
            V[t > 1] = V_1
            return V
        
        # Get parameters for sigma1
        T_sigma1, lambda_sigma1 = self._get_candidate_params(self.sigma1)
        
        # Value at switch point
        V_D_t1 = self.get_branch_D_value(np.array([self.t1_star]))[0]
        
        # Calculate V for sigma1
        C1 = V_D_t1 - T_sigma1 / self.b
        
        # Different regions
        if self.t2_star is not None and self.sigma2 is not None:
            # Case 2: Two switches
            T_sigma2, lambda_sigma2 = self._get_candidate_params(self.sigma2)
            
            # Value at second switch point
            V_sigma1_t2 = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (self.t2_star - self.t1_star))
            C2 = V_sigma1_t2 - T_sigma2 / self.b
            
            # Build the solution
            mask_0 = t < 0
            mask_1 = (t >= 0) & (t <= self.t2_star)
            mask_2 = (t > self.t2_star) & (t <= self.t1_star)
            mask_3 = (t > self.t1_star) & (t <= 1)
            mask_4 = t > 1
            
            # V(0) is the limit as t -> 0+
            V_0 = T_sigma2 / self.b + C2 * np.exp(lambda_sigma2 * (0 - self.t2_star))
            V_1 = self.get_branch_D_value(np.array([1.0]))[0]
            
            V[mask_0] = V_0
            V[mask_1] = T_sigma2 / self.b + C2 * np.exp(lambda_sigma2 * (t[mask_1] - self.t2_star))
            V[mask_2] = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (t[mask_2] - self.t1_star))
            V[mask_3] = self.get_branch_D_value(t[mask_3])
            V[mask_4] = V_1
        else:
            # Case 1: One switch
            mask_0 = t < 0
            mask_1 = (t >= 0) & (t <= self.t1_star)
            mask_2 = (t > self.t1_star) & (t <= 1)
            mask_3 = t > 1
            
            # V(0) is the limit as t -> 0+
            V_0 = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (0 - self.t1_star))
            V_1 = self.get_branch_D_value(np.array([1.0]))[0]
            
            V[mask_0] = V_0
            V[mask_1] = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (t[mask_1] - self.t1_star))
            V[mask_2] = self.get_branch_D_value(t[mask_2])
            V[mask_3] = V_1
        
        return V
    
    def get_policy(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get optimal policy for both 10-branch and 01-branch."""
        policy_10 = np.zeros_like(t, dtype=int)
        policy_01 = np.zeros_like(t, dtype=int)
        
        # For Branch D: (-1,-1) means Backward for both
        # 10-branch: Backward (0)
        # 01-branch: Backward (0)
        
        if self.sigma1 is None or self.t1_star <= 0:
            # Branch D everywhere
            policy_10[:] = 0  # Backward
            policy_01[:] = 0  # Backward
        else:
            if self.t2_star is not None and self.sigma2 is not None:
                # Two switches
                mask_sigma2 = (t >= 0) & (t <= self.t2_star)
                mask_sigma1 = (t > self.t2_star) & (t <= self.t1_star)
                mask_D = (t > self.t1_star) & (t <= 1)
                
                # Set policies for sigma2
                s10_2, s01_2 = self.sigma2
                policy_10[mask_sigma2] = s10_2 + 1  # Convert -1,0,1 to 0,1,2
                policy_01[mask_sigma2] = s01_2 + 1
                
                # Set policies for sigma1
                s10_1, s01_1 = self.sigma1
                policy_10[mask_sigma1] = s10_1 + 1
                policy_01[mask_sigma1] = s01_1 + 1
                
                # Set policies for Branch D
                policy_10[mask_D] = 0  # Backward
                policy_01[mask_D] = 0  # Backward
                
                # Boundary conditions
                policy_10[t < 0] = policy_10[np.argmin(np.abs(t - 0))]
                policy_01[t < 0] = policy_01[np.argmin(np.abs(t - 0))]
                policy_10[t > 1] = 0  # Branch D at t=1
                policy_01[t > 1] = 0
            else:
                # One switch
                mask_sigma1 = (t >= 0) & (t <= self.t1_star)
                mask_D = (t > self.t1_star) & (t <= 1)
                
                # Set policies for sigma1
                s10_1, s01_1 = self.sigma1
                policy_10[mask_sigma1] = s10_1 + 1
                policy_01[mask_sigma1] = s01_1 + 1
                
                # Set policies for Branch D
                policy_10[mask_D] = 0  # Backward
                policy_01[mask_D] = 0  # Backward
                
                # Boundary conditions
                policy_10[t < 0] = policy_10[np.argmin(np.abs(t - 0))]
                policy_01[t < 0] = policy_01[np.argmin(np.abs(t - 0))]
                policy_10[t > 1] = 0  # Branch D at t=1
                policy_01[t > 1] = 0
        
        return policy_10, policy_01
    
    def plot_solution(self, save_dir: str = "closed_form_results"):
        """Create comprehensive 3-panel plot."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create trust grid
        trust_grid = np.linspace(0, 1, 1000)
        V_values = self.get_value_function(trust_grid)
        policy_10, policy_01 = self.get_policy(trust_grid)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot 1: Value Function
        ax1.plot(trust_grid, V_values, 'b-', linewidth=3, label='Complete Solution')
        
        # Mark switch times
        if self.t1_star is not None and self.t1_star > 0:
            V_t1 = self.get_value_function(np.array([self.t1_star]))[0]
            ax1.axvline(x=self.t1_star, color='orange', linestyle='--', alpha=0.7)
            ax1.plot(self.t1_star, V_t1, 'ro', markersize=8, label=f't₁* = {self.t1_star:.3f} → {self.sigma1_name}')
        
        if self.t2_star is not None:
            V_t2 = self.get_value_function(np.array([self.t2_star]))[0]
            ax1.axvline(x=self.t2_star, color='green', linestyle='--', alpha=0.7)
            ax1.plot(self.t2_star, V_t2, 'go', markersize=8, label=f't₂* = {self.t2_star:.3f} → {self.sigma2_name}')
        
        ax1.set_xlabel('Trust Level (t)', fontsize=12)
        ax1.set_ylabel('Value Function V(t)', fontsize=12)
        ax1.set_title('Branch D (-1,-1): Closed-Form Solution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Policy Regions
        color_map = {0: 'lightcoral', 1: 'lightyellow', 2: 'lightgreen'}
        
        # 10-branch regions (upper half)
        for i in range(len(trust_grid) - 1):
            t = trust_grid[i]
            t_next = trust_grid[i + 1]
            policy = policy_10[i]
            ax2.fill_between([t, t_next], [0.05, 0.05], [0.5, 0.5],
                           color=color_map.get(policy, 'lightgray'), alpha=0.5)
        
        # 01-branch regions (lower half)
        for i in range(len(trust_grid) - 1):
            t = trust_grid[i]
            t_next = trust_grid[i + 1]
            policy = policy_01[i]
            ax2.fill_between([t, t_next], [-0.5, -0.5], [-0.05, -0.05],
                           color=color_map.get(policy, 'lightgray'), alpha=0.5)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.55, 0.55)
        ax2.set_xlabel('Trust Level (t)', fontsize=12)
        ax2.set_title('Policy Regions: 10-branch (top) and 01-branch (bottom)', fontsize=12, fontweight='bold')
        ax2.set_yticks([0.275, -0.275])
        ax2.set_yticklabels(['10-branch', '01-branch'])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Mark switch times
        if self.t1_star is not None and self.t1_star > 0:
            ax2.axvline(x=self.t1_star, color='orange', linestyle='--', alpha=0.7)
        if self.t2_star is not None:
            ax2.axvline(x=self.t2_star, color='green', linestyle='--', alpha=0.7)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.5, label='Backward (B)'),
            Patch(facecolor='lightyellow', alpha=0.5, label='Current (C)'),
            Patch(facecolor='lightgreen', alpha=0.5, label='Forward (F)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Plot 3: Combined Branch Policies
        ax3.plot(trust_grid, policy_10, 'b-', linewidth=2, label='10-branch policy', alpha=0.7)
        ax3.plot(trust_grid, policy_01, 'r--', linewidth=2, label='01-branch policy', alpha=0.7)
        
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Disagree (Backward)', 'Silence (Current)', 'Agree (Forward)'])
        ax3.set_xlabel('Trust Level (t)', fontsize=12)
        ax3.set_ylabel('Optimal Policy', fontsize=12)
        ax3.set_title('Combined Branch Policies', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Mark switch times
        if self.t1_star is not None and self.t1_star > 0:
            ax3.axvline(x=self.t1_star, color='orange', linestyle='--', alpha=0.5)
        if self.t2_star is not None:
            ax3.axvline(x=self.t2_star, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(save_dir, 'closed_form_D_1_1_solution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_results_dataframe(self, plot_path: str) -> pd.DataFrame:
        """Create comprehensive results DataFrame."""
        # Create trust grid
        trust_grid = np.linspace(0, 1, 1000)
        V_values = self.get_value_function(trust_grid)
        policy_10, policy_01 = self.get_policy(trust_grid)
        
        # Convert policies to letters
        policy_names = ['B', 'C', 'F']
        policy_10_letters = [policy_names[p] for p in policy_10]
        policy_01_letters = [policy_names[p] for p in policy_01]
        
        # Calculate boundary values
        V_0 = self.get_value_function(np.array([0.0]))[0]
        V_1 = self.get_value_function(np.array([1.0]))[0]
        
        # Get parameters for sigma1
        if self.sigma1 is not None:
            T_sigma1, lambda_sigma1 = self._get_candidate_params(self.sigma1)
        else:
            T_sigma1, lambda_sigma1 = 0, 0
        
        # Get parameters for sigma2 if exists
        if self.sigma2 is not None:
            T_sigma2, lambda_sigma2 = self._get_candidate_params(self.sigma2)
        else:
            T_sigma2, lambda_sigma2 = 0, 0
        
        # Calculate V at switch points
        V_D_t1 = self.get_branch_D_value(np.array([self.t1_star]))[0] if self.t1_star else 0
        
        # Calculate V^sigma1(t2*) for Case 2 (two switches)
        V_sigma1_t2 = np.nan
        if self.t2_star is not None and self.sigma1 is not None:
            # V^sigma1(t2*) = T(sigma1)/(1-beta) + [V^D(t1*) - T(sigma1)/(1-beta)] * exp(lambda(sigma1)*(t2*-t1*))
            C1 = V_D_t1 - T_sigma1 / self.b
            V_sigma1_t2 = T_sigma1 / self.b + C1 * np.exp(lambda_sigma1 * (self.t2_star - self.t1_star))
        
        results_data = {
            # Input parameters
            'beta': self.beta,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau_11': self.tau_values['11'],
            'tau_10': self.tau_values['10'],
            'tau_01': self.tau_values['01'],
            'tau_00': self.tau_values.get('00', 0),
            'p_11': self.p_values['11'],
            'p_10': self.p_values['10'],
            'p_01': self.p_values['01'],
            'p_00': self.p_values['00'],
            
            # Derived mathematical parameters
            'c0': self.c0,
            'c1': self.c1,
            'c_plus': self.c_plus,
            'a_D': self.a_D,
            'lambda_D': self.lambda_D,
            'kappa_D': self.kappa_D,
            'K_10': self.K_10,
            'K_01': self.K_01,
            
            # Switch information for Case 1 & 2
            't1_star': self.t1_star if self.t1_star else np.nan,
            'sigma1': str(self.sigma1) if self.sigma1 else "None",
            'T_sigma1': T_sigma1,
            'lambda_sigma1': lambda_sigma1,
            'V_D_t1_star': V_D_t1,
            
            # Additional info for Case 2 (two switches)
            't2_star': self.t2_star if self.t2_star else np.nan,
            'sigma2': str(self.sigma2) if self.sigma2 else "None",
            'T_sigma2': T_sigma2,
            'lambda_sigma2': lambda_sigma2,
            'V_sigma1_t2_star': V_sigma1_t2,
            
            # Boundary values
            'V_0': V_0,
            'V_1': V_1,
            
            # Arrays
            'trust_grid': [trust_grid.tolist()],
            'V_values': [V_values.tolist()],
            'policy_10_branch': [policy_10_letters],
            'policy_01_branch': [policy_01_letters],
            
            # Plot path
            'plot_path': plot_path,
            
            # Formula
            'formula': 'Branch D (-1,-1) with potential two switches - see documentation'
        }
        
        return pd.DataFrame([results_data])


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Closed-form solution for Branch D (-1,-1)')
    
    parser.add_argument('--beta', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='Trust update rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='Current option parameter')
    
    parser.add_argument('--tau11', type=float, default=1.0, help='Reward for (1,1)')
    parser.add_argument('--tau10', type=float, default=-0.8, help='Reward for (1,0)')
    parser.add_argument('--tau01', type=float, default=0.6, help='Reward for (0,1)')
    parser.add_argument('--tau00', type=float, default=0.0, help='Reward for (0,0)')
    
    parser.add_argument('--p11', type=float, default=0.15, help='Probability of (1,1)')
    parser.add_argument('--p10', type=float, default=0.35, help='Probability of (1,0)')
    parser.add_argument('--p01', type=float, default=0.35, help='Probability of (0,1)')
    parser.add_argument('--p00', type=float, default=0.15, help='Probability of (0,0)')
    
    parser.add_argument('--output-dir', type=str, default='closed_form_results',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Prepare parameters
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
    
    # Create solver
    print("=" * 60)
    print("Branch D (-1,-1): Closed-Form Solution")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  β={args.beta}, α={args.alpha}, γ={args.gamma}")
    print(f"  τ={tau_values}")
    print(f"  p={p_values}")
    print()
    
    solver = ClosedFormD11(args.beta, args.alpha, args.gamma, tau_values, p_values)
    
    # Print switch information
    print(f"Branch D parameters:")
    print(f"  κ_D = {solver.kappa_D:.4f}")
    print(f"  λ_D = {solver.lambda_D:.4f}")
    print(f"  c0 = {solver.c0:.4f}, c1 = {solver.c1:.4f}")
    print()
    
    print(f"Switch analysis:")
    if solver.t1_star is not None and solver.t1_star > 0:
        print(f"  First switch: t₁* = {solver.t1_star:.4f} → {solver.sigma1_name}")
    else:
        print(f"  No first switch found")
    
    if solver.t2_star is not None:
        print(f"  Second switch: t₂* = {solver.t2_star:.4f} → {solver.sigma2_name}")
    else:
        print(f"  No second switch")
    print()
    
    # Plot solution
    print("Generating plot...")
    plot_path = solver.plot_solution(args.output_dir)
    print(f"✓ Plot saved: {plot_path}")
    
    # Create and save results
    print("Creating results DataFrame...")
    results_df = solver.create_results_dataframe(plot_path)
    
    csv_path = os.path.join(args.output_dir, 'closed_form_D_1_1_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved: {csv_path}")
    
    print()
    print("=" * 60)
    print("Branch D Solution Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

