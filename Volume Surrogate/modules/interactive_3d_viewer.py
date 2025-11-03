#!/usr/bin/env python
"""
Interactive 3D Volume Viewer
=============================
Dynamic 3D visualization with rotation, slicing, and field selection.
Uses matplotlib interactive mode for real-time manipulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys


def calculate_velocity_magnitude(vx, vy, vz):
    """Calculate velocity magnitude from components."""
    return np.sqrt(vx**2 + vy**2 + vz**2)


class Interactive3DViewer:
    """Interactive 3D volume field viewer with slicing and rotation."""

    def __init__(self, dataset_file, sim_index=0):
        """
        Initialize the interactive viewer.

        Parameters
        ----------
        dataset_file : str or Path
            Path to NPZ dataset file
        sim_index : int
            Simulation index to visualize
        """
        self.dataset_file = Path(dataset_file)
        self.sim_index = sim_index

        # Load data
        print(f"Loading dataset: {self.dataset_file}")
        data = np.load(dataset_file, allow_pickle=True)

        # Extract simulation data
        n_sims = data['parameters'].shape[0]
        if sim_index < 0 or sim_index >= n_sims:
            raise ValueError(f"sim_index must be between 0 and {n_sims-1}")

        params = data['parameters'][sim_index]
        self.cold_vel, self.hot_vel = params

        # Get coordinates
        coords = data['coordinates']
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]

        # Get field data
        self.temperature = data['temperature'][sim_index]
        self.pressure = data['pressure'][sim_index]
        vx = data['velocity_x'][sim_index]
        vy = data['velocity_y'][sim_index]
        vz = data['velocity_z'][sim_index]
        self.velocity_mag = calculate_velocity_magnitude(vx, vy, vz)

        # Current settings
        self.current_field = 'temperature'
        self.subsample_step = max(1, len(self.temperature) // 5000)

        print(f"Loaded simulation {sim_index}/{n_sims-1}")
        print(f"  Cold={self.cold_vel:.2f} m/s, Hot={self.hot_vel:.2f} m/s")
        print(f"  Cells: {len(self.temperature)} (showing {len(self.temperature)//self.subsample_step})")

    def get_field_data(self, field_name):
        """Get field data and colormap for a field."""
        if field_name == 'temperature':
            return self.temperature, 'hot', 'Temperature (K)'
        elif field_name == 'pressure':
            return self.pressure, 'viridis', 'Pressure (Pa)'
        elif field_name == 'velocity':
            return self.velocity_mag, 'plasma', 'Velocity Magnitude (m/s)'
        else:
            raise ValueError(f"Unknown field: {field_name}")

    def plot_full_volume(self, field='temperature', azim=45, elev=30):
        """
        Plot full 3D volume with specified field.

        Parameters
        ----------
        field : str
            Field to visualize: 'temperature', 'pressure', or 'velocity'
        azim : float
            Azimuthal viewing angle (degrees)
        elev : float
            Elevation viewing angle (degrees)
        """
        # Get field data
        field_data, cmap, label = self.get_field_data(field)

        # Subsample
        step = self.subsample_step
        x_plot = self.x[::step]
        y_plot = self.y[::step]
        z_plot = self.z[::step]
        field_plot = field_data[::step]

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        scatter = ax.scatter(x_plot, y_plot, z_plot, c=field_plot,
                           cmap=cmap, s=15, alpha=0.7, edgecolors='none')

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'{label}\nCold={self.cold_vel:.2f} m/s, Hot={self.hot_vel:.2f} m/s',
                    fontsize=13, fontweight='bold')

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Equal aspect ratio
        max_range = np.array([x_plot.max()-x_plot.min(),
                             y_plot.max()-y_plot.min(),
                             z_plot.max()-z_plot.min()]).max() / 2.0
        mid_x = (x_plot.max()+x_plot.min()) * 0.5
        mid_y = (y_plot.max()+y_plot.min()) * 0.5
        mid_z = (z_plot.max()+z_plot.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(label, fontsize=10)

        plt.tight_layout()
        return fig, ax

    def plot_slice(self, field='temperature', axis='z', position=0.0):
        """
        Plot 2D slice at specified position.

        Parameters
        ----------
        field : str
            Field to visualize
        axis : str
            Axis to slice: 'x', 'y', or 'z'
        position : float
            Position along axis to slice
        """
        # Get field data
        field_data, cmap, label = self.get_field_data(field)

        # Create mask for slice
        tolerance = 0.01  # 1cm tolerance
        if axis == 'x':
            mask = np.abs(self.x - position) < tolerance
            x_coords = self.y[mask]
            y_coords = self.z[mask]
            xlabel, ylabel = 'Y (m)', 'Z (m)'
        elif axis == 'y':
            mask = np.abs(self.y - position) < tolerance
            x_coords = self.x[mask]
            y_coords = self.z[mask]
            xlabel, ylabel = 'X (m)', 'Z (m)'
        elif axis == 'z':
            mask = np.abs(self.z - position) < tolerance
            x_coords = self.x[mask]
            y_coords = self.y[mask]
            xlabel, ylabel = 'X (m)', 'Y (m)'
        else:
            raise ValueError(f"Unknown axis: {axis}")

        field_slice = field_data[mask]

        if len(field_slice) == 0:
            print(f"No data found at {axis}={position:.3f} m")
            return None, None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(x_coords, y_coords, c=field_slice,
                           cmap=cmap, s=30, alpha=0.8, edgecolors='none')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{label} - Slice at {axis.upper()}={position:.3f} m\n'
                    f'Cold={self.cold_vel:.2f} m/s, Hot={self.hot_vel:.2f} m/s',
                    fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(label, fontsize=11)

        plt.tight_layout()
        return fig, ax

    def multi_angle_view(self, field='temperature'):
        """Create multiple viewing angles in one figure."""
        field_data, cmap, label = self.get_field_data(field)

        # Subsample
        step = self.subsample_step
        x_plot = self.x[::step]
        y_plot = self.y[::step]
        z_plot = self.z[::step]
        field_plot = field_data[::step]

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(16, 12))

        angles = [
            (30, 45, 'Isometric'),
            (0, 0, 'Front View'),
            (0, 90, 'Top View'),
            (0, -90, 'Bottom View')
        ]

        for idx, (elev, azim, title) in enumerate(angles, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')

            scatter = ax.scatter(x_plot, y_plot, z_plot, c=field_plot,
                               cmap=cmap, s=8, alpha=0.6, edgecolors='none')

            ax.set_xlabel('X (m)', fontsize=9)
            ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_zlabel('Z (m)', fontsize=9)
            ax.set_title(f'{title}\n{label}', fontsize=10, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)

            # Equal aspect ratio
            max_range = np.array([x_plot.max()-x_plot.min(),
                                 y_plot.max()-y_plot.min(),
                                 z_plot.max()-z_plot.min()]).max() / 2.0
            mid_x = (x_plot.max()+x_plot.min()) * 0.5
            mid_y = (y_plot.max()+y_plot.min()) * 0.5
            mid_z = (z_plot.max()+z_plot.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.suptitle(f'Multi-Angle View: Cold={self.cold_vel:.2f} m/s, Hot={self.hot_vel:.2f} m/s',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def interactive_menu(dataset_file):
    """Interactive menu for 3D visualization."""
    print("="*70)
    print("INTERACTIVE 3D VOLUME VIEWER")
    print("="*70)

    # Load dataset and select simulation
    data = np.load(dataset_file, allow_pickle=True)
    n_sims = data['parameters'].shape[0]

    print(f"\nDataset: {dataset_file}")
    print(f"Simulations available: {n_sims}")

    sim_idx = int(input(f"\nEnter simulation index [0-{n_sims-1}]: ").strip())
    viewer = Interactive3DViewer(dataset_file, sim_index=sim_idx)

    while True:
        print("\n" + "="*70)
        print("VISUALIZATION OPTIONS")
        print("="*70)
        print("  [1] Full 3D volume (interactive)")
        print("  [2] Multiple viewing angles")
        print("  [3] 2D slice (X-axis)")
        print("  [4] 2D slice (Y-axis)")
        print("  [5] 2D slice (Z-axis)")
        print("  [6] Change field (temp/pressure/velocity)")
        print("  [0] Exit")
        print("="*70)

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            field = input("Field [temperature/pressure/velocity] (default=temperature): ").strip() or 'temperature'
            fig, ax = viewer.plot_full_volume(field=field)
            print("\nInteractive mode: Rotate with mouse, close window to continue")
            plt.show()

        elif choice == '2':
            field = input("Field [temperature/pressure/velocity] (default=temperature): ").strip() or 'temperature'
            fig = viewer.multi_angle_view(field=field)
            plt.show()

        elif choice in ['3', '4', '5']:
            axis_map = {'3': 'x', '4': 'y', '5': 'z'}
            axis = axis_map[choice]
            field = input("Field [temperature/pressure/velocity] (default=temperature): ").strip() or 'temperature'
            position = float(input(f"Position along {axis.upper()} axis (m): ").strip())
            fig, ax = viewer.plot_slice(field=field, axis=axis, position=position)
            if fig:
                plt.show()

        elif choice == '6':
            print("\nAvailable fields: temperature, pressure, velocity")

        elif choice == '0':
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    # Standalone execution
    default_dataset = Path(__file__).parent / "volume_surrogate_dataset.npz"

    if not default_dataset.exists():
        print(f"Dataset not found: {default_dataset}")
        print("Please generate the dataset first.")
        sys.exit(1)

    interactive_menu(default_dataset)
