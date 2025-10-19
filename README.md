# Alert Fatigue: Bellman Equation Simulation and Closed-Form Solutions

This project implements both numerical simulation and closed-form analytical solutions for the alert fatigue problem using dynamic programming and the Bellman equation.

## Problem Description

We consider a dynamic environment where an AI system assists a human decision-maker in making repeated binary decisions. The system provides recommendations that can be:
- Truthful: $R_t = A_t$
- Counter: $R_t = 1 - A_t$ 
- No recommendation: $R_t = -1$

The Bellman equation for the optimal value function is:

$$V(t) = p^{11}(\tau^{11} + \beta V(t+\alpha)) + p^{00}(\beta V(t+\alpha))$$
$$+ p^{10} \max\{\tau^{10} + \beta V(t+\alpha), (1-\tilde{t})\tau^{10} + \beta V(t-\alpha), \gamma\tau^{10} + \beta V(t)\}$$
$$+ p^{01} \max\{\beta V(t+\alpha), \tilde{t}\tau^{01} + \beta V(t-\alpha), (1-\gamma)\tau^{01} + \beta V(t)\}$$

where $\tilde{t} = \min\{\max\{t,0\},1\}$.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Recommended: Comprehensive Comparison
Run numerical simulation with all closed-form solutions and automatic branch selection:

```bash
# Use default parameter_set.csv
python run_simulation_w_cf.py

# Use custom parameter file
python run_simulation_w_cf.py --input my_params.csv --output my_results/
```

This will:
- Run numerical simulation for each parameter set
- Compute all four closed-form solutions (Branches A, B, C, D)
- Select the best matching closed-form based on non-decreasing curve and V(1) error
- Generate comparison plots showing all solutions
- Save comprehensive results with all metrics

## Main Scripts

### 1. `run_simulation_w_cf.py` (Recommended)
**Purpose**: Comprehensive analysis combining numerical simulation and all closed-form solutions.

**Usage**:
```bash
# Use default input file (parameter_set.csv) and output directory (comparison_results/)
python run_simulation_w_cf.py

# Specify custom input file and output directory
python run_simulation_w_cf.py --input my_params.csv --output my_results/

# Get help
python run_simulation_w_cf.py --help
```

**Input CSV Format**:
```csv
beta,alpha,gamma,tau11,tau10,tau01,tau00,p11,p10,p01,p00
0.9,0.02,0.9,1.0,1.0,0.8,-1.0,0.2,0.4,0.2,0.2
0.95,0.1,0.5,1.0,0.5,-0.3,0.0,0.3,0.2,0.25,0.25
```

**Outputs**:
- `comparison_results.csv` - Comprehensive results with all metrics (see `COLUMN_STRUCTURE.md`)
- `comparison_row_N.png` - Comparison plots for each parameter set

### 2. `bellman_simulation.py`
**Purpose**: Numerical simulation using value iteration (standalone).

**Usage**:
```bash
# Run with command-line arguments
python bellman_simulation.py --beta 0.9 --alpha 0.02 --gamma 0.9 \
    --tau11 1.0 --tau10 1.0 --tau01 0.8 --tau00 -1.0 \
    --p11 0.2 --p10 0.4 --p01 0.2 --p00 0.2

# Use parameter CSV file
python bellman_simulation.py --parameter-csv parameter_set.csv --run-index 0
```

### 3. Closed-Form Solution Scripts
Run individual closed-form solutions:

#### Branch A (+1, +1) - Constant Solution
Both conflict states choose Forward.
```bash
python closed_form_A11.py --beta 0.95 \
    --tau11 1.0 --tau10 0.5 --tau01 -0.3 --tau00 0.0 \
    --p11 0.3 --p10 0.2 --p01 0.25 --p00 0.25
```

#### Branch B (+1, -1) - One-Switch Solution
10-branch Forward, 01-branch Backward.
```bash
python closed_form_B1_1.py --beta 0.9 --alpha 0.02 --gamma 0.9 \
    --tau11 1.0 --tau10 1.0 --tau01 0.8 --tau00 -1.0 \
    --p11 0.2 --p10 0.4 --p01 0.2 --p00 0.2
```

#### Branch C (-1, +1) - One-Switch Solution
10-branch Backward, 01-branch Forward.
```bash
python closed_form_C_11.py --beta 0.95 --alpha 0.1 --gamma 0.5 \
    --tau11 1.0 --tau10 0.5 --tau01 -0.3 --tau00 0.0 \
    --p11 0.3 --p10 0.2 --p01 0.25 --p00 0.25
```

#### Branch D (-1, -1) - One or Two-Switch Solution
Both branches Backward initially.
```bash
python closed_form_D_1_1.py --beta 0.9 --alpha 0.05 --gamma 0.5 \
    --tau11 1.0 --tau10 -0.8 --tau01 0.6 --tau00 0.0 \
    --p11 0.15 --p10 0.35 --p01 0.35 --p00 0.15
```

**All closed-form scripts output**:
- `closed_form_{branch}_results.csv` - Full results with mathematical formula
- `closed_form_{branch}_solution.png` - 3-panel visualization

### 4. `plot_tau10_sensitivity.py`
**Purpose**: Sensitivity analysis showing how optimal policies vary with tau10 across trust levels.

**Usage**:
```bash
python plot_tau10_sensitivity.py \
    --beta 0.9 --alpha 0.02 --gamma 0.9 \
    --tau11 1.0 --tau01 0.8 --tau00 -1.0 \
    --p11 0.2 --p10 0.4 --p01 0.2 --p00 0.2 \
    --tau10-min -1.0 --tau10-max 2.0 --tau10-steps 20 \
    --trust-grid-size 100 \
    --output-prefix sensitivity_results/tau10
```

**Outputs**:
- `{prefix}_simulation.png` - Policies from numerical simulation
- `{prefix}_closedform.png` - Policies from closed-form solutions (with branch selection info)

**What each plot shows**:
- Two heatmaps (10-branch and 01-branch)
- X-axis: Trust level (t) from 0 to 1
- Y-axis: τ¹⁰ value (varying parameter)
- Colors: Red=Backward, Yellow=Current, Green=Forward
- Closed-form plot shows which branches (A, B, C, D) were selected for each τ¹⁰ value

**Options**:
- `--tau10-min`, `--tau10-max`: Range of τ¹⁰ values to test
- `--tau10-steps`: Number of τ¹⁰ values (resolution)
- `--trust-grid-size`: Number of trust levels (resolution)
- `--output-prefix`: Prefix for output files (default: sensitivity_results/tau10)

## Typical Workflows

### Single Parameter Analysis
```bash
# Create a CSV with your parameters
cat > my_params.csv << EOF
beta,alpha,gamma,tau11,tau10,tau01,tau00,p11,p10,p01,p00
0.9,0.02,0.9,1.0,1.0,0.8,-1.0,0.2,0.4,0.2,0.2
EOF

# Run comprehensive comparison
python run_simulation_w_cf.py --input my_params.csv --output my_results/

# View results
cat my_results/comparison_results.csv
open my_results/comparison_row_0.png
```

### Batch Analysis
```bash
# Create CSV with multiple parameter sets
cat > batch_params.csv << EOF
beta,alpha,gamma,tau11,tau10,tau01,tau00,p11,p10,p01,p00
0.9,0.02,0.9,1.0,1.0,0.8,-1.0,0.2,0.4,0.2,0.2
0.95,0.1,0.5,1.0,0.5,-0.3,0.0,0.3,0.2,0.25,0.25
0.92,0.08,0.6,0.8,0.4,-0.2,0.0,0.25,0.3,0.25,0.2
EOF

# Run comprehensive comparison on all sets
python run_simulation_w_cf.py --input batch_params.csv --output batch_results/

# Analyze results
python -c "
import pandas as pd
df = pd.read_csv('batch_results/comparison_results.csv')
print('Branch Selection Summary:')
print(df['best_branch'].value_counts())
print('\nAverage errors by branch:')
for branch in ['A', 'B', 'C', 'D']:
    avg_error = df[f'{branch}_error'].mean()
    print(f'  Branch {branch}: {avg_error:.6f}')
"
```

## Parameters

### Core Parameters
- `--beta`: Discount factor (0 < β < 1)
- `--alpha`: Trust update magnitude (0 < α << 1)
- `--gamma`: Human decision probability when no recommendation (0 < γ < 1)

### Treatment Effects
- `--tau11`, `--tau10`, `--tau01`, `--tau00`: Treatment effects for each type

### Type Probabilities
- `--p11`, `--p10`, `--p01`, `--p00`: Probabilities for each type (must sum to 1)

### Grid Parameters (for simulation)
- `--trust-min`, `--trust-max`: Trust level range (default: 0.0 to 1.0)
- `--grid-size`: Number of discretization points (default: 500)

### Convergence (for simulation)
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--max-iterations`: Maximum iterations (default: 10000)

### Parameter Constraints
- `0 < beta < 1` (discount factor)
- `alpha > 0` (trust evolution rate)
- `0 < gamma < 1` (trust threshold)
- Probabilities must sum to 1: `p11 + p10 + p01 + p00 = 1`

## Output Structure

### Comparison Results (`comparison_results.csv`)
See `COLUMN_STRUCTURE.md` for detailed column descriptions.

**Key columns**:
- `row_index` - Parameter set index
- `best_branch` - Selected branch (A, B, C, or D)
- `{branch}_V_at_1` - Value at t=1 for each branch
- `{branch}_non_decreasing` - Whether curve is non-decreasing
- `{branch}_error` - Error vs numerical simulation
- `sim_*` - All simulation results (20 columns)
- `cf_*` - Closed-form results for selected branch (14-37 columns depending on branch)

### Comparison Plots
Each plot shows:
- **Black line**: Numerical simulation (ground truth)
- **Colored dashed lines**: All four closed-form solutions
- **Orange thick line**: Selected best branch
- Branches marked as "decreasing ✗" are eliminated

### Simulation-Only Output
**Comprehensive Results** (`simulation_results/{prefix}_comprehensive_results.csv`):
- Single row with all parameters and results as arrays
- Columns: All parameters + `trust_grid`, `value_function`, `policy_10_branch`, `policy_01_branch` arrays
- Policies encoded as: `B` (Backward/Disagree), `C` (Current/Silence), `F` (Forward/Agree)

**Plot** (`simulation_results/{prefix}_results.png`):
- 3-panel visualization: value function, policy regions overview, and combined branch policies

## Mathematical Background

The problem models the evolution of human trust in AI recommendations and finds the optimal recommendation strategy to maximize expected discounted outcomes. The value function represents the maximum expected utility achievable from any given trust level.

### Policy Options
The three policy options for each branch correspond to:
- **Forward (F)**: Agree - Increase trust (recommendation that agrees with human)
- **Backward (B)**: Disagree - Decrease trust (recommendation that disagrees with human)  
- **Current (C)**: Silence - Maintain trust (no recommendation)

### Branches
Two branches are analyzed:
- **10-branch**: Policy for cases where human chooses 1 (says yes) and algorithm chooses 0 (says no)
- **01-branch**: Policy for cases where human chooses 0 (says no) and AI chooses 1 (says yes)

### Closed-Form Solutions
Four branches represent different optimal policy structures:
- **Branch A (+1, +1)**: Constant solution - both branches choose Forward
- **Branch B (+1, -1)**: One-switch solution - 10-branch Forward, 01-branch Backward
- **Branch C (-1, +1)**: One-switch solution - 10-branch Backward, 01-branch Forward
- **Branch D (-1, -1)**: One or two-switch solution - both branches initially Backward

## Key Features

- **Value Iteration**: Numerical solution of the infinite-horizon Bellman equation
- **Closed-Form Solutions**: Analytical solutions for four branch types with switch logic
- **Automatic Branch Selection**: Selects best closed-form based on non-decreasing curve and V(1) error
- **Policy Analysis**: Identification of optimal recommendations (Forward/Backward/Current)
- **Comprehensive Results**: Single DataFrame with all parameters and results arrays
- **Visualization**: Plots of value functions and optimal policies
- **Convergence Monitoring**: Automatic detection of solution convergence

## Tips

### Interpreting Results
- Lower `{branch}_error` = better match to numerical simulation
- `non_decreasing=False` eliminates that branch from selection
- Branch D may have two switches (check `cf_t2_star` column)
- Check `cf_formula` column for mathematical details of selected solution

### Performance
- Each parameter set takes ~3-8 seconds to process
- Batch processing uses progress bars
- Results are saved incrementally (safe to interrupt)

### Troubleshooting
- If no branch is selected, check if any have `non_decreasing=True`
- Large errors suggest parameter regime may not fit any closed-form
- Verify probabilities sum to 1 and parameters are in valid ranges

## Examples

See `parameter_set.csv` for example parameter sets that have been tested.

## Documentation

- `COLUMN_STRUCTURE.md` - Detailed description of output CSV columns
- `parameter_set.csv` - Example parameter sets
