# Workflow Surrogate - Automated CFD Surrogate Model System

**Official PyFluent Repository:** [https://github.com/ansys/pyfluent](https://github.com/ansys/pyfluent)
**Official PyFluent Documentation:** [https://fluent.docs.pyansys.com/version/stable/](https://fluent.docs.pyansys.com/version/stable/)

---

## Table of Contents
1. [Overview](#overview)
2. [Technical Overview](#technical-overview)
3. [Architecture](#architecture)
4. [Key Features](#key-features)
5. [PyFluent Implementation](#pyfluent-implementation-details)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [Requirements](#version-requirements)

---

## Overview

The Workflow Surrogate system is an integrated platform for creating neural network surrogate models of Ansys Fluent CFD simulations. It automates the entire workflow from DOE generation through model training and validation, enabling rapid exploration of design spaces without running expensive CFD simulations.

### What This System Does:

1. **Design of Experiments (DOE)** - Automated full-factorial parameter space sampling
2. **Batch Simulation** - Parallel execution of Fluent simulations with varying BCs
3. **Data Extraction** - Automated extraction of scalar, 2D surface, and 3D volume fields
4. **Multi-Model Training** - Automatic training of specialized neural networks for each output type
5. **Validation & Visualization** - Real-time comparison of surrogate predictions vs. Fluent CFD

---

## Technical Overview

### Objectives

The primary goal is to create **fast, accurate surrogate models** that can replace expensive CFD simulations for design space exploration, optimization, and uncertainty quantification. The system targets:

- **Speed:** 1000-10000× faster than CFD for predictions
- **Accuracy:** >90% R² on test data
- **Flexibility:** Handles scalar, 2D field, and 3D volume outputs simultaneously
- **Automation:** Minimal user intervention after initial setup

### Methods

#### 1. Proper Orthogonal Decomposition (POD)

For high-dimensional field data (2D surfaces, 3D volumes), direct neural network regression is impractical due to:
- Computational cost (thousands to millions of output points)
- Risk of overfitting
- Memory constraints

**POD Solution:**
```
Original field: Y(x,t) ∈ ℝⁿ (n = thousands to millions)
↓ POD Decomposition
Y(x,t) ≈ Σᵢ αᵢ(t)φᵢ(x)
↓ Dimension Reduction
POD coefficients: α(t) ∈ ℝᵐ (m = 8-20 modes)
```

The neural network learns the mapping: **Parameters → POD Coefficients**

Then reconstructs full fields via: **Full Field = Σ(coefficients × POD modes)**

**Variance Captured:**
- 2D Fields (8-10 modes): typically 95-99.9% variance
- 3D Volumes (15-20 modes): typically 90-98% variance

#### 2. Neural Network Architectures

**Scalar Model** (1D outputs ≤100 points):
```
Input(n_params) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Output(n_scalars)
```
- Direct regression, no POD needed
- Used for: Report definitions, average values, single points

**Field Model** (2D outputs 100-10,000 points):
```
Input(n_params) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(n_modes)
```
- POD-based compression
- Used for: Surface temperature/pressure/velocity distributions
- Typical modes: 8-10

**Volume Model** (3D outputs >10,000 points):
```
Input(n_params) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(64, ReLU) → Output(n_modes)
```
- Deeper architecture for complex 3D fields
- POD-based compression
- Used for: Full 3D cell zone data
- Typical modes: 15-20

#### 3. Training Strategy

**Data Split:**
- 80% training / 20% testing (random split)
- Minimum 20 samples recommended (80% of 25 = 20 train samples)

**Optimization:**
- Adam optimizer with default learning rate (0.001)
- Adaptive batch sizing: `max(8, n_samples // 4)`
- Early stopping (patience=15 epochs) to prevent overfitting

**Loss Function:**
- Mean Squared Error (MSE)
- Evaluated on both raw outputs (scalars) and POD coefficients (fields)

### Model Capabilities

**What the Models Can Do:**
- **Interpolation:** Predict outputs for parameter combinations within the training range
- **Fast Evaluation:** 1000-10000× faster than CFD
- **Uncertainty Estimation:** Comparing Fluent validation runs to NN predictions
- **Multi-Output:** Simultaneous prediction of multiple fields/scalars

**What the Models Cannot Do:**
- **Extrapolation:** Unreliable for parameters outside training range
- **Geometry Changes:** Models are tied to specific mesh/geometry
- **Physics Changes:** Cannot handle different fluids, turbulence models, etc. without retraining
- **Transient Dynamics:** Currently only steady-state or time-averaged fields

### Limitations

1. **Data Requirements:**
   - Minimum ~20-25 simulations for reliable training
   - More parameters → more data needed (curse of dimensionality)
   - Full factorial DOE with 5 levels × 2 params = 25 simulations minimum

2. **Computational Constraints:**
   - POD modes limited by number of training samples (n_modes < n_samples)
   - 3D volume data can be memory-intensive (visualization downsampled to 2000 points)

3. **Accuracy Trade-offs:**
   - POD introduces reconstruction error (~1-10% for 8-20 modes)
   - Neural network introduces regression error (~5-15% test MAE typical)
   - Combined error: 5-20% depending on complexity

4. **Scope:**
   - Single geometry per model
   - Steady-state or ensemble-averaged results only
   - Boundary condition parameters only (no material property variations currently)

### Validation Workflow

The system includes **Fluent Validation** mode where:
1. User specifies parameter values
2. System runs **both** NN prediction AND Fluent simulation
3. Visualizations show 3-panel comparison:
   - **Left:** Fluent CFD result (ground truth)
   - **Middle:** Neural Network prediction
   - **Right:** Absolute error field

This enables:
- Quantitative error assessment (MAE, RMSE, R²)
- Spatial error distribution analysis
- Model confidence evaluation

---

## Architecture

### Project Structure
```
Workflow Surrogate/
├── front_end.py                 # Main CLI application
├── modules/
│   ├── fluent_interface.py      # PyFluent session management & case loading
│   ├── simulation_runner.py     # DOE execution, BC application, data extraction
│   ├── doe_setup.py              # Design of Experiments configuration
│   ├── output_parameters.py     # Field variable selection
│   ├── multi_model_trainer.py   # Automatic multi-output model training
│   ├── multi_model_visualizer.py # Prediction, validation, visualization
│   ├── scalar_nn_model.py       # 1D scalar surrogate model
│   ├── field_nn_model.py        # 2D field surrogate with POD
│   ├── volume_nn_model.py       # 3D volume surrogate with POD
│   ├── project_system.py        # Project & case management
│   └── ui_helpers.py            # CLI utilities
├── user_settings.json           # User preferences & recent files
└── README.md

Projects/
└── {project_name}/
    ├── project_info.json
    └── cases/
        └── {case_name}/
            ├── model_setup.json         # Input/output configuration
            ├── output_parameters.json   # Field variables to extract
            ├── dataset/                 # Simulation outputs (sim_*.npz)
            │   ├── sim_0001.npz
            │   ├── sim_0002.npz
            │   └── ...
            └── models/                  # Trained surrogate models
                ├── 1D_avg-outlet-temp_1.h5
                ├── 2D_temperature_1.h5
                ├── 2D_temperature_1_pod_components.npz
                ├── 2D_temperature_1_metadata.json
                └── training_summary.json
```

---

## Key Features

### Recent Improvements (2025-11-12)

1. **Fluent Validation Integration**
   - Added real-time Fluent comparison during predictions
   - 3-panel visualization (Fluent | NN | Error)
   - Keeps Fluent console visible for iteration monitoring

2. **Visualization Enhancements**
   - Fixed key format mismatch for Fluent data lookup
   - Added proper 3D aspect ratio based on geometry
   - Downsampling for large 3D datasets (2000 points) for performance
   - Synchronized color scales between Fluent and NN plots

3. **Report Definition Auto-Configuration**
   - Automatic initialization of scalar outputs (Report Definitions)
   - Fixed bug where 1D models weren't being trained
   - No manual configuration needed for pre-configured Fluent reports

4. **Code Quality**
   - Removed deprecated modules and debug scripts
   - Consolidated duplicate code by reusing `simulation_runner` functions
   - Improved error handling and user feedback

---

## PyFluent Implementation Details

### ⚠️ MANDATORY REQUIREMENTS FOR CODING AGENTS ⚠️

**ALL code modifications MUST follow the official PyFluent API syntax from the repository above.**

### Critical PyFluent Syntax Rules:

1. **Setting Boundary Condition Values:**
   ```python
   # ✅ CORRECT - Use .value attribute
   solver_session.setup.boundary_conditions.velocity_inlet["inlet-name"].momentum.velocity_magnitude.value = 1.0

   # ❌ WRONG - Do NOT use set_state() or setattr()
   param_obj.set_state(value)  # INCORRECT
   setattr(target_obj, param_name, value)  # INCORRECT
   ```

2. **Accessing Boundary Conditions:**
   ```python
   # Access pattern:
   bc = solver_session.setup.boundary_conditions.{bc_type}["{bc_name}"]

   # Examples:
   cold_inlet = solver_session.setup.boundary_conditions.velocity_inlet["cold-inlet"]
   outlet = solver_session.setup.boundary_conditions.pressure_outlet["outlet"]
   ```

3. **Parameter Navigation:**
   ```python
   # For nested parameters like momentum.velocity_magnitude:
   cold_inlet.momentum.velocity_magnitude.value = 0.4
   cold_inlet.thermal.temperature.value = 293.15
   cold_inlet.turbulence.turbulent_intensity = 0.05
   ```

### Boundary Condition Application

**Location:** `simulation_runner.py:apply_boundary_conditions()`

```python
def apply_boundary_conditions(solver, bc_values):
    """Apply BCs using official PyFluent syntax."""
    boundary_conditions = solver.setup.boundary_conditions

    for bc_key, bc_info in bc_values.items():
        bc_name = bc_info['bc_name']
        bc_type = bc_info['bc_type']  # e.g., 'velocity_inlet'
        param_path = bc_info['param_path']  # e.g., 'momentum.velocity_magnitude'
        value = bc_info['value']

        # Get BC object by type and name
        bc_container = getattr(boundary_conditions, bc_type)
        bc_obj = bc_container[bc_name]

        # Navigate to parameter and set .value
        target = bc_obj
        for part in param_path.split('.')[:-1]:
            target = getattr(target, part)

        param_name = param_path.split('.')[-1]
        param_obj = getattr(target, param_name)
        param_obj.value = value  # ✅ CORRECT SYNTAX
```

### Data Extraction Methods

**2D Surface Data:**
```python
# Correct syntax for consistent point counts
field_data.get_scalar_field_data(
    field_name='temperature',
    surfaces=['mid-plane'],
    node_value=False  # ✅ Use face centers for consistency
)
```

**3D Volume Data:**
```python
# Extract cell zone data using solution variables
solver.fields.solution_variable_data.get_data(
    zone_names=['fluid'],
    variable_name='SV_T',  # Temperature (SV_ prefix for solution variables)
    domain_name='mixture'
)
```

**Report Definitions (Scalars):**
```python
# Report definitions are pre-configured in Fluent
# Extract using standard field data API
report_value = solver.fields.report_definitions[report_name]()
```

---

## Usage Guide

### 1. Create New Project & Case

```
Main Menu → [1] Create New Project
  → Enter project name
  → [1] Create New Case
  → Enter case name
  → [1] Open Fluent Case File
  → Select .cas or .cas.h5 file
```

### 2. Configure Inputs & Outputs

```
Setup Menu → [1] Configure Model Inputs
  → Select boundary conditions (velocity inlets, etc.)

Setup Menu → [2] Configure Model Outputs
  → Select surfaces for 2D field data
  → Select cell zones for 3D volume data
  → Select report definitions for scalar data

Setup Menu → [3] Configure Output Parameters
  → Select field variables (temperature, pressure, velocity, etc.)
  → Report Definitions auto-configured ✓

Setup Menu → [4] Design of Experiment Setup
  → Enter min/max values for each BC parameter
  → Enter number of levels (5 recommended)
  → System generates full factorial DOE
```

### 3. Run Batch Simulations

```
Setup Menu → [5] Save Setup & Finish
  → Choose number of iterations per simulation
  → System runs all DOE combinations automatically
  → Progress: [■■■■■░░░░░] 12/25 complete
```

### 4. Train Surrogate Models

```
Case Menu → [2] Train Surrogate Models
  → System auto-detects output types (1D/2D/3D)
  → Trains specialized model for each output
  → Displays training metrics (R², MAE, RMSE)
```

### 5. Make Predictions

```
Case Menu → [3] Visualize & Predict
  → [1] Run Prediction (NN Only) - Fast predictions
  → [2] Run Prediction with Fluent Validation - Compare to CFD

  → Enter parameter values (or use random)
  → View 3-panel plots: Fluent | NN | Error
  → Check scalar results summary table
```

---

## Data Format

### NPZ Files (`sim_XXXX.npz`)

Each simulation produces an NPZ file with keys following: `"{location}|{field_name}"`

```python
# Example structure:
{
    "mid-plane|temperature": array([2337]),         # 2D surface field
    "mid-plane|coordinates": array([[x,y,z], ...]),  # Spatial coordinates
    "fluid|velocity-magnitude": array([45678]),     # 3D volume field
    "fluid|coordinates": array([[x,y,z], ...]),
    "avg-outlet-temp|temperature": array([val])     # Scalar report
}
```

### Model Metadata

**Training Summary (`training_summary.json`):**
```json
{
  "case_name": "5x5",
  "trained_date": "2025-11-11T18:56:13",
  "n_models": 7,
  "models": [
    {
      "model_name": "2D_temperature_1",
      "output_key": "mid-plane_temperature",
      "npz_key": "mid-plane|temperature",
      "output_type": "2D",
      "n_modes": 8,
      "variance_explained": 0.9977,
      "test_metrics": {
        "r2": 0.9646,
        "mae": 0.6407,
        "rmse": 1.4773
      }
    }
  ]
}
```

---

## Troubleshooting

### Issue: Dimension Mismatch Between Training and Prediction

**Cause:** Using `node_value=True` causes variable point counts between simulations
**Solution:** Always use `node_value=False` for consistent face-center extraction

### Issue: Perfect R² (1.000) on Training

**Causes:**
1. Dataset too small (<20 samples)
2. All training simulations identical (BC application failed)
3. Overfitting due to early stopping disabled

**Solutions:**
1. Generate larger DOE (25+ samples minimum)
2. Verify BC application in Fluent logs shows different values
3. Check early stopping is enabled (patience=15)

### Issue: Negative R² on Test/Custom Predictions

**Cause:** Model never learned input-output relationship (training data has no variance)
**Solution:**
1. Verify DOE has different parameter values
2. Check BC application logs for errors
3. Re-run batch simulations after fixing BC syntax

### Issue: "Output key not found in Fluent data"

**Cause:** Key format mismatch between model metadata and Fluent extraction
**Solution:** System now uses `npz_key` (pipe format) for lookups - FIXED in v2025-11-12

### Issue: 3D Plots Slow/Laggy

**Cause:** Too many points being rendered
**Solution:** System now auto-downsamples to 2000 points - FIXED in v2025-11-12

### Issue: Report Definition Not Trained

**Cause:** Report Definitions weren't auto-configured in output_parameters.json
**Solution:** Auto-initialization added - FIXED in v2025-11-12

---

## Version Requirements

### Core Dependencies
- **Python:** 3.9+
- **PyFluent:** 0.35.0+ ([Latest](https://github.com/ansys/pyfluent))
- **TensorFlow/Keras:** 2.13+
- **scikit-learn:** 1.3+
- **NumPy:** 1.24+
- **Matplotlib:** 3.7+

### Ansys Fluent
- **Fluent:** 2023 R1 or newer
- **License:** Valid Ansys license with Fluent solver
- **VPN:** Required if license server is remote

### Installation

```bash
pip install ansys-fluent-core tensorflow scikit-learn numpy matplotlib
```

---

## References

- [PyFluent Official Repository](https://github.com/ansys/pyfluent)
- [PyFluent Documentation](https://fluent.docs.pyansys.com/version/stable/)
- [PyFluent Examples](https://fluent.docs.pyansys.com/version/stable/examples/index.html)
- [Mixing Elbow Settings API Example](https://fluent.docs.pyansys.com/version/stable/examples/00-fluent/mixing_elbow_settings_api.html)
- Berkooz, G., Holmes, P., & Lumley, J. L. (1993). "The Proper Orthogonal Decomposition in the Analysis of Turbulent Flows." *Annual Review of Fluid Mechanics*, 25(1), 539-575.

---

## Citation

If using this system for research, please cite:
- Ansys PyFluent: [https://github.com/ansys/pyfluent](https://github.com/ansys/pyfluent)
- Your institution's surrogate modeling methodology

---

## License

Ensure compliance with Ansys Fluent and PyFluent licensing requirements.

---

**Last Updated:** 2025-11-12
**Version:** 2.0
**Verified Against:** PyFluent v0.35.0
