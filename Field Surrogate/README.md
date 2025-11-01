# Field Surrogate - Mixing Elbow DOE System

Automated design of experiments (DOE) system for generating field surrogate training data from Ansys Fluent simulations.

## Project Structure

```
Field Surrogate/
├── front_end.py                # Interactive TUI menu - START HERE
├── elbow.cas.h5                # Fluent case file
├── *.npz                       # Dataset files (generated)
├── modules/                    # Core programs (called by front_end)
│   ├── auto_fl_matrix.py      # DOE simulation engine
│   ├── fluent_output_check.py # Visualization and data inspection
│   ├── train_surrogate.py     # Train POD-NN surrogate models
│   ├── predict_with_surrogate.py # Validation with Fluent comparison
│   └── fluent_cleanup.py      # Cleanup utilities
├── surrogate_models/           # Trained models directory (generated)
│   ├── model1/                 # Model-specific folder
│   │   ├── *.npz              # Copy of dataset used for training
│   │   ├── surrogate_*.npz    # Trained POD components
│   │   ├── surrogate_*.h5     # Trained neural networks
│   │   ├── comparison_*.png   # Training comparison plots
│   │   └── validation/        # Validation results
│   │       └── validation_*.png
│   └── model2/                 # Another model (different dataset/parameters)
│       └── ...
└── README.md                   # This file
```

## Quick Start

**Simply run the front end:**

```bash
python front_end.py
```

The interactive TUI menu provides access to all workflows:
- Generate datasets with custom DOE settings
- Visualize and inspect data
- Train surrogate models
- Validate against Fluent
- Edit all configuration parameters

### Old Method (Manual Configuration)

If you prefer to configure parameters directly in code:

Edit `runner.py` to set your test matrix and Fluent settings:

```python
# DOE Test Matrix
COLD_VEL_ARRAY = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # m/s
HOT_VEL_ARRAY = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])   # m/s

# Fluent Settings
PROCESSOR_COUNT = 6
ITERATIONS = 200
```

### 2. Run the Complete Workflow

```bash
python runner.py
```

This will:
1. Run all 49 DOE simulations (7×7 matrix)
2. Extract field data (temperature, pressure, velocity) from mid-plane
3. Save everything to compressed NPZ file
4. Generate verification plots

**Expected runtime**: ~2-3 hours for 49 simulations (depends on hardware)

### 3. Inspect Results

```bash
python fluent_output_check.py
```

Options:
- Visualize random simulation
- Visualize specific simulation by index
- Create grid of all simulations

## Dataset Format

The output `field_surrogate_dataset.npz` contains:

```python
data = np.load('field_surrogate_dataset.npz')

# Arrays:
data['parameters']      # Shape: (49, 2) - [cold_vel, hot_vel] for each sim
data['coordinates']     # Shape: (4235, 3) - [x, y, z] coords (same for all)
data['temperature']     # Shape: (49, 4235) - Temperature field for each sim
data['pressure']        # Shape: (49, 4235) - Pressure field for each sim
data['velocity_x']      # Shape: (49, 4235) - X-velocity for each sim
data['velocity_y']      # Shape: (49, 4235) - Y-velocity for each sim
data['velocity_z']      # Shape: (49, 4235) - Z-velocity for each sim
data['metadata']        # Dictionary with case info and timestamps
```

## File Descriptions

### `runner.py`
- **Purpose**: Main entry point for the entire workflow
- **Configurable**: All DOE parameters, Fluent settings, file paths
- **Usage**: `python runner.py`
- **When to edit**: When changing test matrix, processor count, or iterations

### `auto_fl_matrix.py`
- **Purpose**: Automated DOE execution engine
- **Features**:
  - Launches Fluent once for all simulations
  - Extracts field data from specified surface
  - Saves to compressed NPZ format
  - Progress tracking with time estimates
- **Usage**: Called by runner.py (or standalone for testing)
- **When to edit**: When adding new field variables to extract

### `fluent_output_check.py`
- **Purpose**: Standalone visualization and data inspection
- **Features**:
  - Plots temperature, pressure, velocity magnitude
  - Random or specific simulation selection
  - Grid visualization of all simulations
  - Calculates velocity magnitude from components
- **Usage**: `python fluent_output_check.py`
- **When to edit**: When adding new visualization types

## Usage Examples

### Example 1: Standard Full DOE
```bash
# Edit runner.py if needed, then:
python runner.py
```

### Example 2: Quick Test (2×2 Matrix)
```python
# In runner.py, change:
COLD_VEL_ARRAY = np.array([0.1, 0.4])
HOT_VEL_ARRAY = np.array([0.8, 1.4])
# Then run:
python runner.py
```

### Example 3: Visualize Specific Simulation
```bash
python fluent_output_check.py
# Select option 2, enter simulation index
```

### Example 4: Inspect Dataset Programmatically
```python
import numpy as np

# Load dataset
data = np.load('field_surrogate_dataset.npz', allow_pickle=True)

# Get simulation 10 data
params = data['parameters'][10]  # [cold_vel, hot_vel]
temp = data['temperature'][10]    # Temperature field
coords = data['coordinates']      # XYZ coordinates

print(f"Sim 10: Cold={params[0]:.2f}, Hot={params[1]:.2f}")
print(f"Mean temperature: {temp.mean():.2f} K")
```

## Data Flow

```
runner.py (configure)
    ↓
auto_fl_matrix.py (execute DOE)
    ↓
    → Launch Fluent
    → For each (cold_vel, hot_vel) pair:
        → Set BCs
        → Solve
        → Extract fields from mid-plane
    → Save to NPZ
    ↓
field_surrogate_dataset.npz (output)
    ↓
fluent_output_check.py (visualize)
```

## Requirements

```
numpy
matplotlib
ansys-fluent-core
```

## Notes

- **Coordinates**: Extracted once (first simulation) since they're identical for all runs
- **Compression**: Using `np.savez_compressed` reduces file size by ~50-70%
- **Memory**: All data stored in memory during DOE, then saved at once
- **Resuming**: No automatic resume - if interrupted, restart from beginning
- **CGNS**: No longer needed - data extracted directly from Fluent

## Surrogate Model Training

After generating the dataset, train the POD-NN surrogate model:

### Train Models

```bash
python train_surrogate.py
```

**Interactive prompts:**
1. Select which dataset (.npz file) to train on
2. Choose a model name (or use default based on dataset name)

**What it does:**
- Apply POD to reduce dimensionality (4235 → 10 modes per field)
- Train neural networks (2 parameters → 10 modes)
- Evaluate on train/test split (80/20)
- Generate comparison plots (ground truth vs prediction)
- Create model-specific folder in `surrogate_models/<model_name>/`
- Copy dataset to model folder for reference
- Save all outputs to model folder

**Training time**: ~5-10 minutes for all 5 fields

**Outputs** (saved to `surrogate_models/<model_name>/`):
- `*.npz` - Copy of training dataset
- `surrogate_*.npz` - Trained POD components (5 files)
- `surrogate_*.h5` - Trained neural networks (5 files)
- `comparison_*.png` - Training comparison plots
- Performance metrics printed to console (R², RMSE, MAE)

### Validate Models

Validate trained models against actual Fluent simulations:

```bash
python predict_with_surrogate.py
```

**Interactive prompts:**
1. Select which trained model to use
2. Enter cold and hot inlet velocities for validation

**What it does:**
- Load selected model and its dataset
- Run surrogate prediction (fast, <1 second)
- Run Fluent simulation for ground truth (~2-4 minutes)
- Create 3×3 comparison plot (Fluent | Prediction | Error)
- Save validation plots to `surrogate_models/<model_name>/validation/`

**Outputs** (saved to `surrogate_models/<model_name>/validation/`):
- `validation_cold{X}_hot{Y}.png` - Comparison plots with error analysis
- Shows R², MAE, max error for each field
- Error maps with diverging colormap (white=0, blue=over-predict, red=under-predict)

### `train_surrogate.py`
- **Purpose**: Train POD-based neural network surrogates
- **Method**: POD dimensionality reduction + shallow neural networks
- **Features**:
  - Interactive dataset selection from available .npz files
  - Custom model naming for organization
  - Automatic train/test split (80/20)
  - Performance metrics (R², RMSE, MAE per field)
  - Side-by-side comparison plots (ground truth vs prediction)
  - Model checkpointing and early stopping
  - Dataset copied to model folder for traceability
- **Output**: Complete model package in `surrogate_models/<model_name>/`

### `predict_with_surrogate.py`
- **Purpose**: Validate surrogate models against Fluent ground truth
- **Usage**: Test model accuracy for any parameter combination
- **Features**:
  - Interactive model selection from trained models
  - Loads corresponding dataset automatically
  - Runs actual Fluent simulation for comparison
  - 3×3 comparison visualization (Fluent | Prediction | Error)
  - Comprehensive error metrics (R², MAE, max error)
  - Diverging error colormap centered at zero
  - Organized output in model-specific validation folder

## Working with Multiple Models

The system supports training multiple models from different datasets or with different parameters:

### Example Workflow

```bash
# Generate first dataset with 7×7 matrix
python runner.py
# Creates: field_surrogate_dataset.npz

# Train first model
python train_surrogate.py
# Select dataset: field_surrogate_dataset.npz
# Name it: "baseline_model"
# Creates: surrogate_models/baseline_model/

# Generate second dataset with finer resolution (10×10 matrix)
# Edit runner.py to change COLD_VEL_ARRAY and HOT_VEL_ARRAY
python runner.py
# Creates: field_surrogate_dataset_fine.npz (or rename it)

# Train second model
python train_surrogate.py
# Select dataset: field_surrogate_dataset_fine.npz
# Name it: "fine_resolution_model"
# Creates: surrogate_models/fine_resolution_model/

# Validate either model
python predict_with_surrogate.py
# Select model: [1] baseline_model or [2] fine_resolution_model
```

### Model Organization Benefits

- **Traceability**: Each model folder contains its training dataset
- **Comparison**: Easily compare different models side-by-side
- **Versioning**: Keep multiple model versions without conflicts
- **Validation**: Separate validation folders for each model
- **Reproducibility**: Complete record of what data trained each model
