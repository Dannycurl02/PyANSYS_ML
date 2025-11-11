# Workflow Surrogate - Automated CFD Surrogate Model System

**Official PyFluent Repository:** [https://github.com/ansys/pyfluent](https://github.com/ansys/pyfluent)
**Official PyFluent Documentation:** [https://fluent.docs.pyansys.com/version/stable/](https://fluent.docs.pyansys.com/version/stable/)

---

## ⚠️ MANDATORY REQUIREMENTS FOR CODING AGENTS ⚠️

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

4. **Before Making Changes:**
   - **Always verify** syntax against [official examples](https://fluent.docs.pyansys.com/version/stable/examples/00-fluent/mixing_elbow_settings_api.html)
   - **Never assume** API methods without checking documentation
   - **Test with a simple case** before modifying batch processing code

---

## Overview

The Workflow Surrogate system is an integrated platform for creating surrogate models of Ansys Fluent CFD simulations. It automates:

1. **Design of Experiments (DOE)** - Automated generation of simulation parameter matrices
2. **Batch Simulation** - Running multiple Fluent simulations with varying boundary conditions
3. **Data Extraction** - Extracting scalar, 2D surface, and 3D volume field data
4. **Model Training** - Training neural network surrogates using POD (Proper Orthogonal Decomposition)
5. **Visualization & Validation** - Comparing surrogate predictions against Fluent simulations

---

## Architecture

### Project Structure
```
Workflow Surrogate/
├── front_end.py                 # Main CLI application
├── modules/
│   ├── fluent_interface.py      # PyFluent session management
│   ├── simulation_runner.py     # DOE execution and data extraction
│   ├── doe_setup.py              # DOE configuration
│   ├── output_parameters.py     # Field variable selection
│   ├── multi_model_trainer.py   # Automatic model training
│   ├── multi_model_visualizer.py # Visualization and comparison
│   ├── scalar_nn_model.py       # 1D scalar surrogate model
│   ├── field_nn_model.py        # 2D field surrogate with POD
│   ├── volume_nn_model.py       # 3D volume surrogate with POD
│   └── project_system.py        # Project management
└── user_settings.json           # User preferences

Projects/
└── {project_name}/
    ├── project_info.json
    └── cases/
        └── {case_name}/
            ├── model_setup.json
            ├── output_parameters.json
            ├── dataset/           # Simulation outputs (sim_*.npz)
            └── models/            # Trained models (.h5, .npz, metadata)
```

---

## Key Components

### 1. Fluent Interface (`fluent_interface.py`)
- Launches PyFluent sessions
- Manages case file loading
- Handles Fluent logs and output redirection

### 2. Simulation Runner (`simulation_runner.py`)
- **DOE Generation**: Creates full factorial designs from BC parameters
- **BC Application**: Sets boundary condition values using proper PyFluent syntax
- **Data Extraction**:
  - Scalars: Report definitions (single values)
  - 2D Surfaces: Field data using `get_scalar_field_data(node_value=False)`
  - 3D Volumes: Cell zone data using `get_data()` with solution variables
- **Batch Processing**: Automated simulation execution with progress tracking

### 3. Surrogate Models

#### Scalar Model (`scalar_nn_model.py`)
- For: Single values or small arrays (≤100 points)
- Architecture: Dense(64) → Dense(32) → Dense(16) → Dense(n_outputs)
- No POD required

#### Field Model (`field_nn_model.py`)
- For: 2D surface data (100-10,000 points)
- Method: POD + Neural Network
- Typical modes: 10
- Architecture: Input → Dense(64) → Dense(64) → Dense(32) → Dense(n_modes)

#### Volume Model (`volume_nn_model.py`)
- For: 3D volume data (>10,000 points)
- Method: POD + Deeper Neural Network
- Typical modes: 20
- Architecture: Input → Dense(128) → Dense(128) → Dense(64) → Dense(64) → Dense(n_modes)

---

## PyFluent Implementation Details

### Boundary Condition Application

**Location:** `simulation_runner.py:apply_boundary_conditions()`

The system uses the official PyFluent API to set BC values:

```python
def apply_boundary_conditions(solver, bc_values):
    """Apply BCs using official PyFluent syntax."""
    boundary_conditions = solver.setup.boundary_conditions

    for bc_info in bc_values:
        bc_name = bc_info['bc_name']
        bc_type = bc_info['bc_type']  # e.g., 'velocity_inlet'
        param_path = bc_info['param_path']  # e.g., 'momentum.velocity_magnitude'
        value = bc_info['value']

        # Get BC object
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
# Extract cell zone data
solver.fields.solution_variable_data.get_data(
    zone_names=['fluid'],
    variable_name='SV_T',  # Temperature
    domain_name='mixture'
)
```

---

## Training Configuration

### Model Training Parameters

**Batch Size:** Adaptive with minimum floor
```python
batch_size = max(8, len(parameters) // 4)
# Ensures: min 8 for stability, scales up for large datasets
```

**Early Stopping:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,      # Reduced from 50 to prevent overfitting
    restore_best_weights=True,
    min_delta=1e-6
)
```

**POD Modes:**
- 2D Fields: 10 modes (captures ~95%+ variance)
- 3D Volumes: 17-20 modes (higher complexity)
- Auto-adjusted if dataset is too small

---

## Data Format

### NPZ Files (`sim_XXXX.npz`)

Each simulation produces an NPZ file with keys following the pattern: `"{location}|{field_name}"`

```python
# Example structure:
{
    "mid-plane|temperature": array([...]),      # 2D surface field
    "mid-plane|coordinates": array([[x,y,z]...]), # Spatial coordinates
    "fluid|velocity-magnitude": array([...]),   # 3D volume field
    "fluid|coordinates": array([[x,y,z]...]),
    "avg-outlet-temp|temperature": array([val]) # Scalar report
}
```

### Model Metadata (`{model_name}_metadata.json`)

```json
{
  "model_name": "2D_temperature_1",
  "output_type": "2D",
  "n_points": 2337,
  "n_modes": 10,
  "variance_explained": 0.999998,
  "train_metrics": {
    "r2": 0.985,
    "mae": 0.45,
    "rmse": 0.62
  },
  "test_metrics": { ... }
}
```

---

## Troubleshooting

### Issue: Dimension Mismatch Between Training and Prediction

**Cause:** Using `node_value=True` causes variable point counts
**Solution:** Always use `node_value=False` for consistent face-center extraction

### Issue: Perfect R² (1.000) on Training

**Causes:**
1. Dataset too small (<50 samples)
2. Batch size too small (causing noisy gradients)
3. All training simulations identical (BC application failed)

**Solutions:**
1. Generate larger DOE (≥100 samples for 2D inputs)
2. Check batch size is ≥8
3. Verify BC application messages show different values

### Issue: Negative R² on Custom Predictions

**Cause:** Training data has identical BCs (model never learned input-output relationship)
**Solution:** Re-run batch simulations after verifying BC application syntax

---

## Validation Workflow

1. **Run Batch Simulations** (16+ samples minimum)
2. **Verify Dataset Variance:**
   ```python
   # All sims should have DIFFERENT statistics
   import numpy as np
   data = np.load("sim_0001.npz")
   print(data["mid-plane|temperature"].mean())
   # Should differ across sim files!
   ```
3. **Train Models** (watch for realistic R² values: 0.85-0.98)
4. **Test Custom Prediction:**
   - Use parameter values **within** training range
   - Check diagnostic report for errors
   - Verify R² > 0.8 for good surrogate

---

## Version Requirements

- **Python:** 3.9+
- **PyFluent:** Latest stable (check [repo](https://github.com/ansys/pyfluent))
- **TensorFlow/Keras:** 2.x
- **scikit-learn:** 1.x
- **NumPy, Matplotlib:** Latest stable

---

## References

- [PyFluent Official Repository](https://github.com/ansys/pyfluent)
- [PyFluent Documentation](https://fluent.docs.pyansys.com/version/stable/)
- [PyFluent Examples](https://fluent.docs.pyansys.com/version/stable/examples/index.html)
- [Mixing Elbow Settings API Example](https://fluent.docs.pyansys.com/version/stable/examples/00-fluent/mixing_elbow_settings_api.html)

---

## Citation

If using this system for research, please cite:
- Ansys PyFluent: [https://github.com/ansys/pyfluent](https://github.com/ansys/pyfluent)
- Your institution's surrogate modeling methodology

---

## License

Ensure compliance with Ansys Fluent and PyFluent licensing requirements.

---

**Last Updated:** 2025-11-11
**Verified Against:** PyFluent v0.35.0
