# Alert Fatigue: Bellman Equation Simulation

This project implements numerical simulation for the alert fatigue problem using dynamic programming and the Bellman equation.

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

## Usage

### Batch Simulation (Recommended)
Run all parameter sets from CSV file:
```bash
python run_batch_simulation.py
```

This will:
- Run simulations for all parameter sets in `parameter_set.csv`
- Generate individual plots for each parameter set
- Combine all results into single CSV files
- Clean up individual result files

### Individual Simulation
Run single parameter set:
```bash
# Run first parameter set from CSV
python bellman_simulation.py --parameter-csv parameter_set.csv --run-index 0

# Run with custom parameters
python bellman_simulation.py --beta 0.98 --alpha 0.05 --gamma 0.6
```

### Configuration Options

**JSON Configuration File:**
```bash
python bellman_simulation.py --config sample_config.json
```

### Available Parameters

**Core Parameters:**
- `--beta`: Discount factor (0 < β < 1)
- `--alpha`: Trust update magnitude (0 < α << 1)
- `--gamma`: Human decision probability when no recommendation

**Treatment Effects:**
- `--tau11`, `--tau10`, `--tau01`, `--tau00`: Treatment effects for each type

**Type Probabilities:**
- `--p11`, `--p10`, `--p01`, `--p00`: Probabilities for each type (must sum to 1)

**Grid Parameters:**
- `--trust-min`, `--trust-max`: Trust level range (default: 0.0 to 1.0)
- `--grid-size`: Number of discretization points (default: 500)

**Convergence:**
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--max-iterations`: Maximum iterations (default: 10000)

**Output Options:**
- `--output-prefix`: Prefix for output files
- `--no-plots`: Skip generating plots
- `--verbose`: Enable verbose output

### Example Usage

```bash
# Quick run with defaults
python bellman_simulation.py

# Custom parameters
python bellman_simulation.py --beta 0.98 --alpha 0.05 --gamma 0.6 --verbose

# Using configuration file
python bellman_simulation.py --config my_config.json --output-prefix experiment1

# Using CSV parameter file
python bellman_simulation.py --parameter-csv parameter_set.csv --run-index 0

# High precision run
python bellman_simulation.py --grid-size 2000 --tolerance 1e-10 --max-iterations 50000
```

The simulation will:
1. Solve the Bellman equation using value iteration
2. Analyze the optimal policy across trust levels
3. Generate visualizations of the value function and policy
4. Save all results to CSV files and plots in the `simulation_results/` folder

## Key Features

- **Value Iteration**: Numerical solution of the infinite-horizon Bellman equation
- **Policy Analysis**: Identification of optimal recommendations (Forward/Backward/Current)
- **Comprehensive Results**: Single DataFrame with all parameters and results arrays
- **Visualization**: Plots of value functions and optimal policies
- **Convergence Monitoring**: Automatic detection of solution convergence

## Parameters

- `β` (beta): Discount factor (0 < β < 1)
- `α` (alpha): Trust update magnitude (0 < α << 1)  
- `γ` (gamma): Human decision probability when no recommendation
- `τ` (tau): Treatment effects for each type of individual
- `p`: Probabilities for each type of individual

## Output Files

All output files are automatically saved in the `simulation_results/` folder:

### Batch Simulation Output

**Comprehensive Results (`simulation_results/comprehensive_results.csv`)**
- Contains all parameter sets and results in one file
- Each row represents one parameter set with `row_index` column
- **Columns**: All parameters + `trust_grid`, `value_function`, `policy_10_branch`, `policy_01_branch` arrays + `row_index`
- Policies are encoded as: `B` (Backward/Disagree), `C` (Current/Silence), `F` (Forward/Agree)

**Individual Plots (`simulation_results/row_X_results.png`)**
- Separate plot file for each parameter set
- Shows value function and optimal policy regions (3 panels: value function, policy regions overview, combined branch policies)

### Individual Simulation Output

**Comprehensive Results (`simulation_results/{prefix}_comprehensive_results.csv`)**
- Single row with all parameters and results as arrays
- **Columns**: All parameters + `trust_grid`, `value_function`, `policy_10_branch`, `policy_01_branch` arrays + `plot_path`
- Policies are encoded as: `B` (Backward/Disagree), `C` (Current/Silence), `F` (Forward/Agree)

**Plot (`simulation_results/{prefix}_results.png`)**
- 3-panel visualization: value function, policy regions overview, and combined branch policies

## Mathematical Background

The problem models the evolution of human trust in AI recommendations and finds the optimal recommendation strategy to maximize expected discounted outcomes. The value function represents the maximum expected utility achievable from any given trust level.

The three policy options for each branch correspond to:
- **Forward (F)**: Agree - Increase trust (recommend what AI suggests)
- **Backward (B)**: Disagree - Decrease trust (recommend opposite of AI)  
- **Current (C)**: Silence - Maintain trust (no recommendation)

Two branches are analyzed:
- **10-branch**: Policy for cases where human chooses 1 and AI chooses 0
- **01-branch**: Policy for cases where human chooses 0 and AI chooses 1
