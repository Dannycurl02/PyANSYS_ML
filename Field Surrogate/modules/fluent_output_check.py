#!/usr/bin/env python
"""
Fluent Output Checker
=====================
Standalone script to visualize field data from the NPZ dataset.
Plots temperature, pressure, and velocity magnitude for a random simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def visualize_random_simulation(dataset_file, sim_index=None):
    """
    Visualize field data for a random (or specified) simulation.

    Parameters
    ----------
    dataset_file : str or Path
        Path to the NPZ dataset file
    sim_index : int, optional
        Specific simulation index to visualize. If None, selects randomly.
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

    # Create visualization
    print(f"\n  Creating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Temperature plot
    scatter1 = axes[0].scatter(x_coords, y_coords, c=temperature,
                               cmap='hot', s=15, alpha=0.8, edgecolors='none')
    axes[0].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[0].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[0].set_title(
        f'Temperature Distribution (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
        fontsize=14, fontweight='bold'
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Temperature (K)', fontsize=10)

    # Pressure plot
    scatter2 = axes[1].scatter(x_coords, y_coords, c=pressure,
                               cmap='viridis', s=15, alpha=0.8, edgecolors='none')
    axes[1].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[1].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[1].set_title(
        f'Pressure Distribution (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
        fontsize=14, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Pressure (Pa)', fontsize=10)

    # Velocity magnitude plot
    scatter3 = axes[2].scatter(x_coords, y_coords, c=vel_mag,
                               cmap='plasma', s=15, alpha=0.8, edgecolors='none')
    axes[2].set_xlabel('X Coordinate (m)', fontsize=12)
    axes[2].set_ylabel('Y Coordinate (m)', fontsize=12)
    axes[2].set_title(
        f'Velocity Magnitude (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
        fontsize=14, fontweight='bold'
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')
    cbar3 = plt.colorbar(scatter3, ax=axes[2])
    cbar3.set_label('Velocity Magnitude (m/s)', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_dir = Path(dataset_file).parent
    plot_file = output_dir / f"visualization_sim{sim_index:03d}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot saved: {plot_file}")

    plt.show()

    return fig, axes


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
    output_file = Path(dataset_file).parent / f"all_sims_{field}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved: {output_file}")
    plt.show()


if __name__ == "__main__":
    # Standalone execution
    print("="*70)
    print("FLUENT OUTPUT CHECKER")
    print("="*70)

    # Default dataset location
    default_dataset = Path(__file__).parent / "field_surrogate_dataset.npz"

    # Check if dataset exists
    if not default_dataset.exists():
        print(f"\n✗ Dataset not found: {default_dataset}")
        print(f"\nPlease run 'python runner.py' first to generate the dataset.")
        sys.exit(1)

    # Menu
    print(f"\nDataset found: {default_dataset}")
    print(f"\nOptions:")
    print(f"  1. Visualize random simulation")
    print(f"  2. Visualize specific simulation")
    print(f"  3. Create grid of all simulations (temperature)")
    print(f"  4. Create grid of all simulations (pressure)")
    print(f"  5. Create grid of all simulations (velocity magnitude)")

    choice = input(f"\nSelect option [1-5] (default=1): ").strip()

    if choice == '2':
        sim_num = int(input("Enter simulation index: ").strip())
        visualize_random_simulation(default_dataset, sim_index=sim_num)
    elif choice == '3':
        visualize_all_simulations_grid(default_dataset, field='temperature')
    elif choice == '4':
        visualize_all_simulations_grid(default_dataset, field='pressure')
    elif choice == '5':
        visualize_all_simulations_grid(default_dataset, field='velocity_magnitude')
    else:
        visualize_random_simulation(default_dataset)
