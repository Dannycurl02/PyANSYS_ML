# Cold Plate Digital Twin Interface

Real-time visualization and prediction interface for the cold plate thermal analysis surrogate model.

## Features

### Input Controls (Right Panel)
- **6 Input Parameters** with dual control:
  - Slider for quick adjustment within range
  - Text box for precise value entry
  - Automatic range clamping
  - Real-time prediction on change

Parameters:
- `chip1.heat_flux`: [800000, 1000000] W/m²
- `chip2.heat_flux`: [800000, 1000000] W/m²
- `inlet1.mass_flow_rate`: [0.005, 0.01] kg/s
- `inlet1.total_temperature`: [300, 340] K
- `inlet2.mass_flow_rate`: [0.005, 0.01] kg/s
- `inlet2.total_temperature`: [300, 340] K

### Visualization Panel (Left Side)

#### Field Outputs (Top - Tabs)
Three 3D surface plots showing temperature distribution:
- **yz-mid**: YZ plane cross-section
- **zx-mid**: ZX plane cross-section
- **bottom**: Bottom surface

Features:
- Interactive 3D rotation and zoom
- Color-mapped temperature field
- Instant updates when inputs change

#### Scalar Outputs (Bottom)
Real-time line plot with dual y-axes:
- **Left axis (blue)**: Temperature outputs
  - chip1_tmax, chip1_tavg
  - chip2_tmax, chip2_tavg
  - outlet_tavg, outlet_tmax
- **Right axis (red)**: Pressure drops
  - pdrop_1
  - pdrop_2
- **X-axis**: Prediction index (increments with each change)

Current values displayed below plot with units.

## Usage

### Running the Interface

```bash
cd "C:\Users\danny\OneDrive - University of Arkansas\.Machine Learning\Project\CP"
python DT_sim.py
```

### Controls

- **Reset to Defaults**: Restore all inputs to middle values
- **Clear History**: Erase scalar output history and reset prediction counter
- **Load Trained Model**: Load ML models (to be implemented)

### Workflow

1. Launch the interface
2. Adjust input parameters using sliders or text boxes
3. Observe instant prediction updates in all visualizations
4. Scalar plot tracks history of predictions over time
5. Switch between field output tabs to view different surfaces

## Current Status

### ✓ Implemented
- Full GUI with all controls and visualizations
- Input parameter management
- Real-time plot updates
- 3D surface visualization
- Dual-axis scalar plotting
- Configuration loading from model_setup.json

### ⏳ Pending (When Models Are Trained)
- Load trained scalar model (ScalarNNModel)
- Load trained field models (FieldNNModel for each surface)
- Actual prediction instead of placeholder values

## Model Integration (To Be Added)

When ML models are ready, update these functions in `DT_sim.py`:

### `load_model()`
```python
# Load scalar model
from modules.scalar_nn_model import ScalarNNModel
self.models['scalars'] = ScalarNNModel()
self.models['scalars'].load('cases/operation_conditions_1/trained_models/scalar_model.keras')

# Load field models
from modules.field_nn_model import FieldNNModel
for field_name in self.field_outputs:
    model = FieldNNModel(n_modes=10, field_name=field_name)
    model.load(f'cases/operation_conditions_1/trained_models/{field_name}_model.keras')
    self.models['fields'][field_name] = model
```

### `predict_scalars()`
```python
# Scale inputs
scaled_inputs = self.models['scalars'].param_scaler.transform(inputs)

# Predict
scaled_predictions = self.models['scalars'].model.predict(scaled_inputs, verbose=0)

# Inverse transform
predictions = self.models['scalars'].output_scaler.inverse_transform(scaled_predictions)[0]

# Map to output names
return {name: predictions[i] for i, name in enumerate(self.scalar_outputs)}
```

### `predict_fields()`
```python
predictions = {}
for field_name in self.field_outputs:
    model = self.models['fields'][field_name]

    # Scale inputs
    scaled_inputs = model.param_scaler.transform(inputs)

    # Predict POD modes
    scaled_modes = model.model.predict(scaled_inputs, verbose=0)
    modes = model.mode_scaler.inverse_transform(scaled_modes)

    # Reconstruct field
    field_1d = model.pca.inverse_transform(modes)[0]
    field_2d = field_1d.reshape(model.field_shape)  # Need to store shape during training

    predictions[field_name] = field_2d

return predictions
```

## Technical Details

- **Framework**: tkinter with matplotlib embedded
- **3D Plots**: matplotlib mplot3d
- **Real-time Updates**: Triggered by slider/text changes
- **History**: Last 50 predictions tracked for scalar plot
- **Resolution**: Field plots use 50×50 grid for performance

## File Structure

```
CP/
├── DT_sim.py                          # Main interface
├── README_DT.md                       # This file
├── cases/
│   └── operation_conditions_1/
│       ├── model_setup.json           # Model configuration (loaded)
│       ├── output_parameters.json     # Output parameters
│       ├── dataset/                   # Training data
│       │   └── sim_*.npz
│       └── trained_models/            # Models (to be added)
│           ├── scalar_model.keras
│           ├── yz-mid_model.keras
│           ├── zx-mid_model.keras
│           └── bottom_model.keras
```

## Notes

- Currently shows placeholder values (zeros for scalars, sinusoidal pattern for fields)
- Status message indicates "No model loaded"
- All UI features are fully functional
- Ready for model integration once training is complete
