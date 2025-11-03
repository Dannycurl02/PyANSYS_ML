# Volume Surrogate - 3D Mixing Elbow DOE System

Automated design of experiments (DOE) system for generating **3D volume** surrogate training data from Ansys Fluent simulations.

**Key Difference from Field Surrogate**: Extracts data from the entire 3D fluid volume (cell zones) instead of a 2D mid-plane surface.

## Project Overview

This system creates machine learning surrogates that predict complete 3D field distributions (temperature, pressure, velocity) throughout the entire fluid volume based on inlet boundary conditions.

### Data Extraction
- **Source**: `fluid` cell zone (entire water volume)
- **Mesh**: Full 3D computational mesh (~10,000-50,000+ cells depending on resolution)
- **Fields**: Temperature, Pressure, Velocity (X, Y, Z components)
- **Coordinates**: 3D spatial coordinates (X, Y, Z) for each cell centroid

## Quick Start

### Test Volume Data Extraction

Before running the full DOE, test that 3D volume extraction works:

```bash
python test_volume_extraction.py
```

This will:
1. Launch Fluent and load the elbow case
2. Run a single simulation
3. Extract all field data from the 3D fluid volume
4. Save to `test_volume_data.npz`
5. Create a 3D scatter plot of temperature distribution

**Expected output**:
- NPZ file with ~10k-50k cells
- 3D visualization showing temperature throughout the volume

### Run Full Workflow

```bash
python front_end.py
```

The interactive TUI menu provides access to:
- Generate 3D volume datasets with custom DOE
- Visualize 3D field data
- Train volume surrogate models
- Validate against Fluent

## Dataset Format

The output NPZ files contain:

```python
data = np.load('volume_surrogate_dataset.npz')

# Arrays:
data['parameters']      # Shape: (n_sims, 2) - [cold_vel, hot_vel]
data['x']               # Shape: (n_cells,) - X coordinates
data['y']               # Shape: (n_cells,) - Y coordinates
data['z']               # Shape: (n_cells,) - Z coordinates
data['temperature']     # Shape: (n_sims, n_cells) - Temperature field
data['pressure']        # Shape: (n_sims, n_cells) - Pressure field
data['velocity_x']      # Shape: (n_sims, n_cells) - X-velocity
data['velocity_y']      # Shape: (n_sims, n_cells) - Y-velocity
data['velocity_z']      # Shape: (n_sims, n_cells) - Z-velocity
```

**Typical sizes**:
- 2D mid-plane (Field Surrogate): ~4,000 points
- 3D volume (Volume Surrogate): ~10,000-50,000 cells

## Key Differences from Field Surrogate

| Aspect | Field Surrogate (2D) | Volume Surrogate (3D) |
|--------|---------------------|---------------------|
| **Data Source** | Mid-plane surface | Entire fluid volume |
| **PyFluent API** | `field_data.get_scalar_field_data()` | `solution_variable_data.get_data()` |
| **Extraction Syntax** | `surfaces=["mid-plane"]` | `zone_names=["fluid"]` |
| **Variable Names** | `'temperature'`, `'pressure'` | `'SV_T'`, `'SV_P'`, `'SV_U'`, etc. |
| **Point Count** | ~4,000-5,000 | ~17,865 cells (elbow case) |
| **Coordinates** | (X, Y, Z) on plane | (X, Y, Z) throughout volume |
| **Visualization** | 2D scatter plots | 3D scatter/isosurface plots |
| **Training Time** | 5-10 minutes | 15-30 minutes (more data) |
| **POD Modes** | 10 modes (default) | 20 modes (default) |
| **NN Architecture** | 64-64-32 layers | 128-128-64-64 layers |
| **Use Case** | Fast, planar analysis | Full 3D field prediction |

## Computational Considerations

### Dataset Size
- **Field (2D)**: ~10-50 MB for 49 simulations
- **Volume (3D)**: ~100-500 MB for 49 simulations (10x larger)

### Training Implications
1. **Memory**: Need more RAM for POD decomposition
2. **POD Modes**: May need to increase from 10 → 15-20 to capture variance
3. **Training Time**: 3-5x longer due to larger matrices
4. **Model Size**: Larger H5 files for neural networks

### Recommendations
- Start with smaller DOE (5x5 instead of 7x7) for testing
- Monitor POD variance explained - aim for >99.5%
- Consider subsampling for visualization (plot every Nth point)
- Use compression: `np.savez_compressed()` reduces size by 50-70%

## Visualization

### 2D Slice Views (Fast)
For quick inspection, extract 2D slices:
```python
# X=0 plane
mask = np.abs(x) < 0.01
plot_2d(y[mask], z[mask], temperature[mask])
```

### 3D Scatter Plots (Medium)
Subsample for performance:
```python
step = len(temperature) // 5000  # Plot 5000 points
scatter3d(x[::step], y[::step], z[::step], temperature[::step])
```

### Isosurfaces (Advanced)
Use Mayavi or ParaView for production-quality visualizations.

## File Structure

```
Volume Surrogate/
├── front_end.py                  # Interactive TUI menu
├── test_volume_extraction.py     # Test script for 3D extraction
├── elbow.cas.h5                  # Fluent case file
├── CHANGES.md                    # Detailed 3D integration summary
├── README.md                     # This file
├── context.md                    # Comprehensive project context
├── modules/
│   ├── auto_fl_matrix.py         # DOE engine (uses solution_variable_data API)
│   ├── train_surrogate.py        # POD-NN training (20 modes, 128-128-64-64 NN)
│   ├── predict_with_surrogate.py # Validation (3D comparison plots)
│   ├── fluent_output_check.py    # 3D visualization with subsampling
│   ├── interactive_3d_viewer.py  # Dynamic 3D viewer with rotation/slicing
│   ├── fluent_cleanup.py         # Cleanup utilities
│   └── fluent_logger.py          # Separate window logging
└── volume_surrogate_*/           # Output folders (gitignored)
    ├── *.npz                     # Dataset and models
    ├── training_results/         # Training curves, metrics
    ├── validation/               # Comparison plots vs Fluent
    └── sim_visualization/        # 3D plots
```

## Requirements

```
numpy
matplotlib
scipy
scikit-learn
tensorflow
ansys-fluent-core>=0.18.0
```

For advanced 3D visualization (optional):
```
mayavi
vtk
pyvista
```

## Workflow

1. **Test Extraction**: `python test_volume_extraction.py`
2. **Generate DOE**: Use front_end.py → Option 1
3. **Inspect Data**: Use front_end.py → Option 4
4. **Train Model**: Use front_end.py → Option 2
5. **Validate**: Use front_end.py → Option 3

## Performance Expectations

### Extraction Speed
- 2D surface: ~1-2 seconds per simulation
- 3D volume: ~3-5 seconds per simulation

### Surrogate Accuracy
Target metrics (similar to 2D):
- R² > 0.98 for temperature, pressure
- R² > 0.95 for velocity components
- POD variance > 99.5%

## Interactive 3D Visualization

The Volume Surrogate includes a powerful interactive 3D viewer:

```bash
cd "Volume Surrogate/modules"
python interactive_3d_viewer.py
```

**Features**:
- Full 3D volume visualization with mouse rotation
- Multiple viewing angles (isometric, front, top, bottom)
- 2D slicing along X, Y, or Z axes
- Field selection (temperature, pressure, velocity magnitude)
- Interactive menu system

## Troubleshooting

### Error: "fluid is not an allowed surface"
**Cause**: Using `field_data` API instead of `solution_variable_data`
**Fix**: Use `solution_variable_data.get_data()` with `zone_names=['fluid']` and variable names like `'SV_T'`, `'SV_P'`

### Error: "IndexError: too many indices for array"
**Cause**: `SV_CENTROID` returns flattened 1D array [x1,y1,z1,x2,y2,z2,...]
**Fix**: Reshape with `coords.reshape((n_cells, 3))`

### Memory Issues
```python
# Reduce dataset size
COLD_VEL_ARRAY = np.array([0.1, 0.4, 0.7])  # 3x3 instead of 7x7
```

### Visualization Too Slow
```python
# Increase subsampling (already implemented)
step = max(1, len(temperature) // 3000)  # Show ~3000 points
```

### POD Not Converging
```python
# Default is already 20 modes for 3D
# Increase further if needed
n_modes = 25  # For very complex fields
```

## Future Enhancements

- [ ] Adaptive mesh refinement tracking
- [ ] Parallel POD computation
- [ ] GPU-accelerated training
- [ ] Real-time 3D rendering with PyVista
- [ ] Compressed sparse storage for large volumes
- [ ] Multi-resolution POD (coarse + fine)

## References

- PyFluent Documentation: https://fluent.docs.pyansys.com/
- POD Theory: Berkooz et al. (1993)
- Field Surrogate (2D version): `../Field Surrogate/`
