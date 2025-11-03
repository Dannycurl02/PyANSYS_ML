#!/usr/bin/env python
"""
Fluent Output Checker - 3D Volume Visualization
================================================
Standalone script to visualize 3D volume field data from the NPZ dataset.
Plots temperature, pressure, and velocity magnitude for a random simulation.
Supports both static 3D plots and interactive visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys


def load_dataset(dataset_file):
    """Load the NPZ dataset and return data dictionary."""
    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    print(f"Loading dataset: {dataset_file}")
    data = np.load(dataset_file, allow_pickle=True)

    print(f"\nDataset contents:")
    for key in data.keys():
        if key != 'metadata':
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  metadata: {data[key].item()}")

    return data


def calculate_velocity_magnitude(vx, vy, vz):
    """Calculate velocity magnitude from components."""
    return np.sqrt(vx**2 + vy**2 + vz**2)


def visualize_random_simulation(dataset_file, sim_index=None, interactive=False):
    """
    Visualize 3D volume field data for a random (or specified) simulation.

    Parameters
    ----------
    dataset_file : str or Path
        Path to the NPZ dataset file
    sim_index : int, optional
        Specific simulation index to visualize. If None, selects randomly.
    interactive : bool, optional
        If True, show interactive 3D plot. If False, show static 3D views.
    """

    # Load dataset
    data = load_dataset(dataset_file)

    # Select simulation
    n_sims = data['parameters'].shape[0]
    if sim_index is None:
        sim_index = np.random.randint(0, n_sims)
    else:
        if sim_index < 0 or sim_index >= n_sims:
            raise ValueError(f"sim_index must be between 0 and {n_sims-1}")

    print(f"\nVisualizing simulation {sim_index}/{n_sims-1}")

    # Extract data for this simulation
    params = data['parameters'][sim_index]
    cold_vel, hot_vel = params
    print(f"  Parameters: Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s")

    # Get coordinates (same for all simulations)
    coords = data['coordinates']
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]

    # Get field data for this simulation
    temperature = data['temperature'][sim_index]
    pressure = data['pressure'][sim_index]
    vx = data['velocity_x'][sim_index]
    vy = data['velocity_y'][sim_index]
    vz = data['velocity_z'][sim_index]

    # Calculate velocity magnitude
    vel_mag = calculate_velocity_magnitude(vx, vy, vz)

    # Print statistics
    print(f"\n  Temperature: {temperature.min():.2f} - {temperature.max():.2f} K (mean: {temperature.mean():.2f} K)")
    print(f"  Pressure: {pressure.min():.2f} - {pressure.max():.2f} Pa (mean: {pressure.mean():.2f} Pa)")
    print(f"  Velocity Magnitude: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s (mean: {vel_mag.mean():.4f} m/s)")

    # Subsample for visualization (to avoid cluttering)
    step = max(1, len(temperature) // 5000)
    x_plot = x_coords[::step]
    y_plot = y_coords[::step]
    z_plot = z_coords[::step]
    temp_plot = temperature[::step]
    press_plot = pressure[::step]
    vel_plot = vel_mag[::step]

    print(f"\n  Creating 3D plots (showing {len(temp_plot)} of {len(temperature)} cells)...")

    # Create 3D visualization
    fig = plt.figure(figsize=(18, 6))

    # Temperature plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(x_plot, y_plot, z_plot, c=temp_plot,
                          cmap='hot', s=10, alpha=0.6, edgecolors='none')
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title(f'Temperature\n(Cold={cold_vel:.2f}, Hot={hot_vel:.2f} m/s)',
                 fontsize=11, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
    cbar1.set_label('T (K)', fontsize=9)

    # Pressure plot
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(x_plot, y_plot, z_plot, c=press_plot,
                          cmap='viridis', s=10, alpha=0.6, edgecolors='none')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.set_title(f'Pressure\n(Cold={cold_vel:.2f}, Hot={hot_vel:.2f} m/s)',
                 fontsize=11, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label('P (Pa)', fontsize=9)

    # Velocity magnitude plot
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(x_plot, y_plot, z_plot, c=vel_plot,
                          cmap='plasma', s=10, alpha=0.6, edgecolors='none')
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Z (m)', fontsize=10)
    ax3.set_title(f'Velocity Magnitude\n(Cold={cold_vel:.2f}, Hot={hot_vel:.2f} m/s)',
                 fontsize=11, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3, pad=0.1, shrink=0.8)
    cbar3.set_label('|V| (m/s)', fontsize=9)

    # Set equal aspect ratio for all 3D plots
    for ax in [ax1, ax2, ax3]:
        max_range = np.array([x_plot.max()-x_plot.min(),
                             y_plot.max()-y_plot.min(),
                             z_plot.max()-z_plot.min()]).max() / 2.0
        mid_x = (x_plot.max()+x_plot.min()) * 0.5
        mid_y = (y_plot.max()+y_plot.min()) * 0.5
        mid_z = (z_plot.max()+z_plot.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    # Save plot to sim_visualization subfolder
    dataset_file = Path(dataset_file)
    if dataset_file.parent.name == dataset_file.stem:
        output_dir = dataset_file.parent / "sim_visualization"
    else:
        output_dir = dataset_file.parent / dataset_file.stem / "sim_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f"visualization_3d_sim{sim_index:03d}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] 3D plot saved: {plot_file}")

    if interactive:
        print(f"\n  Opening interactive 3D view (rotate with mouse, close to continue)...")

    plt.show()

    return fig, (ax1, ax2, ax3)


def visualize_all_simulations_grid(dataset_file, field='temperature'):
    """
    Create a grid visualization of a field for all simulations.

    Parameters
    ----------
    dataset_file : str or Path
        Path to the NPZ dataset file
    field : str
        Field to visualize: 'temperature', 'pressure', or 'velocity_magnitude'
    """

    data = load_dataset(dataset_file)

    n_cold = len(data['cold_vel_array'])
    n_hot = len(data['hot_vel_array'])

    coords = data['coordinates']
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    fig, axes = plt.subplots(n_cold, n_hot, figsize=(3*n_hot, 3*n_cold))

    sim_idx = 0
    for i, cold_vel in enumerate(data['cold_vel_array']):
        for j, hot_vel in enumerate(data['hot_vel_array']):
            ax = axes[i, j] if n_cold > 1 else axes[j]

            # Get field data
            if field == 'temperature':
                field_data = data['temperature'][sim_idx]
                cmap = 'hot'
                label = 'T (K)'
            elif field == 'pressure':
                field_data = data['pressure'][sim_idx]
                cmap = 'viridis'
                label = 'P (Pa)'
            elif field == 'velocity_magnitude':
                vx = data['velocity_x'][sim_idx]
                vy = data['velocity_y'][sim_idx]
                vz = data['velocity_z'][sim_idx]
                field_data = calculate_velocity_magnitude(vx, vy, vz)
                cmap = 'plasma'
                label = '|V| (m/s)'
            else:
                raise ValueError(f"Unknown field: {field}")

            scatter = ax.scatter(x_coords, y_coords, c=field_data,
                                cmap=cmap, s=5, alpha=0.8, edgecolors='none')
            ax.set_title(f'C={cold_vel:.1f}, H={hot_vel:.1f}', fontsize=8)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(scatter, ax=ax, label=label)

            sim_idx += 1

    plt.tight_layout()

    # Save to sim_visualization subfolder
    dataset_file = Path(dataset_file)
    # If parent folder already matches dataset name, use it directly
    if dataset_file.parent.name == dataset_file.stem:
        output_dir = dataset_file.parent / "sim_visualization"
    else:
        output_dir = dataset_file.parent / dataset_file.stem / "sim_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"all_sims_{field}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved: {output_file}")
    plt.show()


if __name__ == "__main__":
    # Standalone execution
    print("="*70)
    print("3D VOLUME VISUALIZATION")
    print("="*70)

    # Default dataset location
    default_dataset = Path(__file__).parent / "volume_surrogate_dataset.npz"

    # Check if dataset exists
    if not default_dataset.exists():
        print(f"\nX Dataset not found: {default_dataset}")
        print(f"\nPlease run the DOE workflow first to generate the dataset.")
        sys.exit(1)

    # Menu
    print(f"\nDataset found: {default_dataset}")
    print(f"\nOptions:")
    print(f"  1. Visualize random simulation (3D)")
    print(f"  2. Visualize specific simulation (3D)")
    print(f"  3. Interactive 3D view (random simulation)")
    print(f"  4. Interactive 3D view (specific simulation)")
    print(f"  5. Create grid of all simulations (temperature)")
    print(f"  6. Create grid of all simulations (pressure)")
    print(f"  7. Create grid of all simulations (velocity magnitude)")

    choice = input(f"\nSelect option [1-7] (default=1): ").strip()

    if choice == '2':
        sim_num = int(input("Enter simulation index: ").strip())
        visualize_random_simulation(default_dataset, sim_index=sim_num, interactive=False)
    elif choice == '3':
        visualize_random_simulation(default_dataset, interactive=True)
    elif choice == '4':
        sim_num = int(input("Enter simulation index: ").strip())
        visualize_random_simulation(default_dataset, sim_index=sim_num, interactive=True)
    elif choice == '5':
        visualize_all_simulations_grid(default_dataset, field='temperature')
    elif choice == '6':
        visualize_all_simulations_grid(default_dataset, field='pressure')
    elif choice == '7':
        visualize_all_simulations_grid(default_dataset, field='velocity_magnitude')
    else:
        visualize_random_simulation(default_dataset, interactive=False)
