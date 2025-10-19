# Comparison Results Column Structure

## Overview
The `comparison_results.csv` file has a **variable number of columns** depending on which branch is selected as the best match. All rows share a common set of columns, and then branch-specific columns are added with the `cf_` prefix.

## Column Categories

### 1. Core Comparison Info (3 columns)
Always present for all rows:
- `row_index` - Parameter set index from input CSV
- `best_branch` - Selected branch (A, B, C, or D)
- `comparison_plot_path` - Path to the comparison plot

### 2. Branch Selection Metrics (12 columns)
Always present for all rows, showing performance of all 4 branches:
- `A_V_at_1`, `A_non_decreasing`, `A_error`
- `B_V_at_1`, `B_non_decreasing`, `B_error`
- `C_V_at_1`, `C_non_decreasing`, `C_error`
- `D_V_at_1`, `D_non_decreasing`, `D_error`

### 3. Simulation Results (20 columns, prefixed with `sim_`)
Always present for all rows:
- **Parameters**: `sim_beta`, `sim_alpha`, `sim_gamma`, `sim_tau11`, `sim_tau10`, `sim_tau01`, `sim_tau00`, `sim_p11`, `sim_p10`, `sim_p01`, `sim_p00`
- **Settings**: `sim_trust_min`, `sim_trust_max`, `sim_grid_size`, `sim_tolerance`, `sim_max_iterations`
- **Results**: `sim_trust_grid`, `sim_value_function`, `sim_policy_10_branch`, `sim_policy_01_branch`

### 4. Closed-Form Results (variable columns, prefixed with `cf_`)
Only the **selected branch's** columns are included. Number of columns varies by branch:

#### Branch A (14 cf_ columns) - Constant Solution
- `cf_sigma_10`, `cf_sigma_01`, `cf_kappa`
- `cf_tau_10_sigma`, `cf_tau_01_sigma`
- `cf_branch_name`, `cf_description`
- `cf_C0`, `cf_V_constant`
- `cf_formula`
- `cf_solution_type`, `cf_branch_type`, `cf_is_constant`, `cf_independent_of_t`

#### Branch B (21 cf_ columns) - One Switch Solution
All Branch A columns plus:
- `cf_c0`, `cf_c1`, `cf_a`, `cf_b`, `cf_lambda_val`
- `cf_switch_time`, `cf_predecessor_branch`, `cf_is_degenerate`
- Additional parameters and computed values

#### Branch C (37 cf_ columns) - One Switch Solution (Different Parameters)
All Branch B columns plus:
- `cf_a_C`, `cf_lambda_C`
- `cf_T_plus1_plus1`, `cf_lambda_plus1_plus1`
- `cf_T_0_plus1`, `cf_lambda_0_plus1`
- `cf_sigma_prev`, `cf_V_C_tstar`, `cf_V_0`, `cf_V_1`
- `cf_trust_grid`, `cf_V_values`, `cf_policy_10_branch`, `cf_policy_01_branch`
- `cf_has_switch_logic`

#### Branch D (33 cf_ columns) - One or Two Switch Solution
Includes all one-switch columns plus two-switch specific columns:

**One-switch columns:**
- `cf_c0`, `cf_c1`, `cf_c_plus`
- `cf_a_D`, `cf_lambda_D`, `cf_kappa_D`
- `cf_K_10`, `cf_K_01`
- `cf_t1_star`, `cf_sigma1`, `cf_T_sigma1`, `cf_lambda_sigma1`, `cf_V_D_t1_star`

**Two-switch columns** (present even if NaN):
- `cf_t2_star` - Second switch time (NaN if no second switch)
- `cf_sigma2` - Second switch branch (None if no second switch)
- `cf_T_sigma2` - T value for second switch (0 if no second switch)
- `cf_lambda_sigma2` - Lambda value for second switch (0 if no second switch)
- `cf_V_sigma1_t2_star` - Value of sigma1 branch at t2_star (NaN if no second switch)

**Common columns:**
- `cf_V_0`, `cf_V_1` - Boundary values
- `cf_trust_grid`, `cf_V_values`, `cf_policy_10_branch`, `cf_policy_01_branch`
- `cf_formula` - Complete mathematical formulation

## Total Column Counts by Branch

| Selected Branch | Total Columns |
|----------------|---------------|
| Branch A       | ~49           |
| Branch B       | ~56           |
| Branch C       | ~72           |
| Branch D       | ~68           |

**Note:** Branch D always includes all two-switch columns, even when only one switch occurs (in which case `t2_star` is NaN, `sigma2` is None, and related values are 0 or NaN).

## Key Points

1. **No Duplicate Parameters**: Common parameters (beta, alpha, gamma, tau, p) only appear once with the `sim_` prefix
2. **Branch-Specific Data**: Only the selected branch's closed-form columns are populated
3. **Arrays Preserved**: Trust grids, value functions, and policies are stored as arrays
4. **Complete Traceability**: All data needed to reproduce both simulation and closed-form results is included
5. **Two-Switch Support**: Branch D includes columns for both one-switch and two-switch scenarios

## Example Usage

```python
import pandas as pd

# Read results
df = pd.read_csv('comparison_results/comparison_results.csv')

# Filter to rows where Branch D was selected
branch_d_rows = df[df['best_branch'] == 'D']

# Check if any had two switches
two_switch_cases = branch_d_rows[branch_d_rows['cf_t2_star'].notna()]
print(f"Found {len(two_switch_cases)} cases with two switches")

# Access closed-form formula for first Branch D case
if len(branch_d_rows) > 0:
    formula = branch_d_rows.iloc[0]['cf_formula']
    print(formula)
```

