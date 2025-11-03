# Volume Surrogate - 3D Integration Summary

## Overview
Successfully integrated 3D volume field extraction into the POD-NN surrogate system. The Volume Surrogate now extracts data from the entire 3D fluid volume (17,865 cells) instead of a 2D surface plane (~4,000 points).

## Key Changes

### 1. Data Extraction API Change
**File**: `modules/auto_fl_matrix.py`

**Critical Change**: Switched from `field_data` API to `solution_variable_data` API
- **OLD (2D Surface)**: `solver.fields.field_data.get_scalar_field_data(field_name='temperature', surfaces=['mid-plane'])`
- **NEW (3D Volume)**: `solver.fields.solution_variable_data.get_data(variable_name='SV_T', zone_names=['fluid'], domain_name='mixture')`

**Reason**: PyFluent has two separate APIs:
- `field_data`: For surfaces (boundaries, planes, named face zones)
- `solution_variable_data`: For cell zones (3D volumes)

**Solution Variable Names**:
- Temperature: `SV_T`
- Pressure: `SV_P`
- Velocity X: `SV_U`
- Velocity Y: `SV_V`
- Velocity Z: `SV_W`
- Coordinates: `SV_CENTROID` (returns flattened array, reshape to (n_cells, 3))

### 2. Configuration Updates
**File**: `front_end.py`

```python
# OLD
PLANE_NAME = "mid-plane"
OUTPUT_NPZ = "field_simulation_dataset.npz"

# NEW
PLANE_NAME = "fluid"  # 3D cell zone name
OUTPUT_NPZ = "volume_simulation_dataset.npz"
```

### 3. 3D Visualization System
**File**: `modules/fluent_output_check.py`

**Changes**:
- Converted all 2D scatter plots to 3D scatter plots
- Added interactive 3D rotation capability
- Subsampling for performance (5000 points from 17,865 cells)
- Equal aspect ratio for all 3 axes
- Updated menu to include interactive options

**New Features**:
- Option 1: Static 3D plots (3 fields side-by-side)
- Option 2: Specific simulation index
- Option 3: Interactive 3D view (random)
- Option 4: Interactive 3D view (specific index)
- Options 5-7: Grid visualizations

### 4. Interactive 3D Viewer
**New File**: `modules/interactive_3d_viewer.py`

**Features**:
- Full 3D volume visualization with mouse rotation
- Multiple viewing angles (isometric, front, top, bottom)
- 2D slicing along X, Y, or Z axes
- Field selection (temperature, pressure, velocity)
- Interactive menu system

**Usage**:
```bash
cd "Volume Surrogate/modules"
python interactive_3d_viewer.py
```

## Data Format Changes

### NPZ Structure
```python
# Same structure, different dimensions
coordinates     # (n_cells, 3)      - was (n_points, 3)
temperature     # (n_sims, n_cells) - was (n_sims, n_points)
pressure        # (n_sims, n_cells) - was (n_sims, n_points)
velocity_x      # (n_sims, n_cells) - was (n_sims, n_points)
velocity_y      # (n_sims, n_cells) - was (n_sims, n_points)
velocity_z      # (n_sims, n_cells) - was (n_sims, n_points)
parameters      # (n_sims, 2)       - unchanged
```

### Size Comparison
| Aspect | Field (2D) | Volume (3D) |
|--------|------------|-------------|
| **Points/Cells** | ~4,000 | ~17,865 |
| **Dataset Size** | 10-50 MB | 100-500 MB |
| **Extraction Time** | 1-2 sec/sim | 3-5 sec/sim |

## Testing

### Test Script
**File**: `test_volume_extraction.py`

Successfully tested:
- Launches Fluent
- Runs simulation (converges at iteration 69)
- Extracts 17,865 cells from "fluid" zone
- Saves NPZ file (test_volume_data.npz)
- Creates 3D visualization (test_volume_temperature_3d.png)

**Results**:
- Volume cells: 17,865
- Temperature: 293.10 - 313.27 K
- Pressure: -389.76 - 196.11 Pa
- Velocity X: -0.3256 - 0.5190 m/s
- Velocity Y: -0.3644 - 1.5656 m/s
- Velocity Z: -0.5634 - 0.1752 m/s

## Workflow

### Running the System

1. **Test Extraction** (verify 3D extraction works):
   ```bash
   python test_volume_extraction.py
   ```

2. **Generate Full DOE Dataset**:
   ```bash
   python front_end.py
   # Select Option 1: Generate Dataset
   ```

3. **Visualize Data**:
   ```bash
   # Static 3D plots
   cd modules
   python fluent_output_check.py

   # Interactive 3D viewer
   python interactive_3d_viewer.py
   ```

4. **Train Surrogate** (same as before):
   ```bash
   python front_end.py
   # Select Option 2: Train Surrogate
   ```

## Important Notes

### POD Considerations
- **2D**: Typically needs 10 POD modes
- **3D**: May need 15-20 POD modes due to more complex field structure
- Monitor POD variance - aim for >99.5% explained

### Memory Requirements
- 3D volumes use ~10x more memory than 2D surfaces
- Consider smaller DOE for initial testing (5x5 instead of 7x7)
- Subsampling recommended for visualization

### Visualization Performance
- Matplotlib 3D scatter plots can be slow with >10,000 points
- Subsampling to 5,000 points provides good balance
- Interactive rotation works smoothly with subsampled data

## Files Modified

1. `modules/auto_fl_matrix.py` - Changed to solution_variable_data API
2. `modules/fluent_output_check.py` - Updated for 3D visualization
3. `front_end.py` - Changed PLANE_NAME to "fluid"
4. `.gitignore` - Added Volume Surrogate paths

## Files Created

1. `test_volume_extraction.py` - Test script for 3D extraction
2. `modules/interactive_3d_viewer.py` - Interactive 3D viewer
3. `README.md` - Documentation for 3D system
4. `CHANGES.md` - This file

## Next Steps

1. Run full DOE workflow to generate volume_surrogate_dataset.npz
2. Train POD-NN surrogate on 3D volume data
3. Validate predictions against Fluent
4. Compare accuracy to 2D Field Surrogate
5. Consider adaptive POD mode selection based on variance

## Troubleshooting

### Common Issues

**Issue**: `fluid is not an allowed surface`
- **Cause**: Using `field_data` API instead of `solution_variable_data`
- **Fix**: Use `SV_*` variable names with `solution_variable_data.get_data()`

**Issue**: `IndexError: too many indices for array`
- **Cause**: `SV_CENTROID` returns flattened array
- **Fix**: Reshape with `coords_flat.reshape((n_cells, 3))`

**Issue**: Visualization is slow
- **Cause**: Too many points to render
- **Fix**: Increase subsampling step or reduce dataset size

**Issue**: Out of memory during POD
- **Cause**: 3D data is 10x larger than 2D
- **Fix**: Use smaller DOE or increase POD modes incrementally
