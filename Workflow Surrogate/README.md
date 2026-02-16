# Workflow Surrogate - Automated CFD Surrogate Model System
---

## Overview

The Workflow Surrogate system is an integrated platform for creating neural network surrogate models of Ansys Fluent CFD simulations. It automates the entire workflow from DOE generation through model training and validation, enabling rapid exploration of design spaces without running expensive CFD simulations.

### What This System Does:

1. **Design of Experiments (DOE)** - Automated LHS parameter space sampling
2. **Batch Simulation** - Parallel execution of Fluent simulations with varying BCs
3. **Data Extraction** - Automated extraction of scalar, 2D surface, and 3D volume fields
4. **Multi-Model Training** - Automatic training of specialized neural networks for each output type
5. **Validation & Visualization** - Real-time comparison of surrogate predictions vs. Fluent CFD

---

## Technical Overview

### Objectives

The primary goal is to create fast, accurate surrogate models that can replace expensive CFD simulations for design space exploration and optimization. The system targets:

- **Speed:** 1000-10000× faster than CFD for predictions
- **Accuracy:** >98% R² on test data
- **Flexibility:** Handles scalar, 2D field, and 3D volume outputs simultaneously
- **Automation:** Minimal user intervention after initial setup

### Methods

#### 1. Proper Orthogonal Decomposition (POD)

For high-dimensional field data (2D surfaces, 3D volumes), direct neural network regression is impractical due to:
- Computational cost (thousands to millions of output points)
- Risk of overfitting
- Memory constraints

The neural network learns the mapping: **Parameters → POD Coefficients**

Then reconstructs full fields via: **Full Field = Σ(coefficients × POD modes)**


#### 2. Neural Network Architectures

**1D Model** :
```
Input(n_params) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Output(n_scalars)
```
- Direct regression, no POD needed
- Used for: Report definitions, average values, single points

**2D Model** :
```
Input(n_params) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(n_modes)
```
- POD-based compression
- Used for: Surface temperature/pressure/velocity distributions

**3D Model** :
```
Input(n_params) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(64, ReLU) → Output(n_modes)
```
- Deeper architecture for complex 3D fields
- POD-based compression
- Used for: Full 3D cell zone data

#### 3. Training Strategy

**Data Split:**
- 80% training / 20% testing (random split)

**Optimization:**
- Adam optimizer with default learning rate (0.001)
- Adaptive batch sizing: `max(8, n_samples // 4)`
- Early stopping (patience=15 epochs) to prevent overfitting

**Loss Function:**
- Mean Squared Error (MSE)
- Evaluated on both raw outputs (scalars) and POD coefficients (fields)

### Model Capabilities

- **Interpolation:** Predict outputs for parameter combinations within the training range
- **Fast Evaluation:** 1000-10000× faster than CFD
- **Uncertainty Estimation:** Comparing Fluent validation runs to NN predictions
- **Multi-Output:** Simultaneous prediction of multiple fields/scalars


### Limitations

   - Single geometry per model
   - Steady-state or ensemble-averaged results only
   - Boundary condition parameters only (no material property variations currently)
   - Unreliable for parameters outside training range
   - Models are tied to specific mesh/geometry


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
            ├── model1/                  # Named model folder (NEW in Dec 2025)
            │   ├── 1D_avg-outlet-temp_1.h5
            │   ├── 2D_temperature_1.h5
            │   ├── 2D_temperature_1.npz
            │   ├── 2D_temperature_1_metadata.json
            │   └── training_summary.json
            └── baseline/                # Another model configuration
                └── (trained model files)
```

---
