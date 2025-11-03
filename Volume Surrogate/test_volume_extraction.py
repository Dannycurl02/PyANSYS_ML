#!/usr/bin/env python
"""
Volume Data Extraction Test
============================
Test script to extract 3D volume field data from Fluent simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import ansys.fluent.core as pyfluent

# Configuration
CASE_FILE = Path(__file__).parent / "elbow.cas.h5"
COLD_VEL = 0.4  # m/s
HOT_VEL = 1.4   # m/s
ITERATIONS = 200
VOLUME_NAME = "fluid"  # Cell zone name for water volume

def main():
    print("="*70)
    print("3D VOLUME DATA EXTRACTION TEST")
    print("="*70)

    # Launch Fluent
    print("\nLaunching Fluent...")
    solver = pyfluent.launch_fluent(
        precision="single",
        processor_count=11,
        dimension=3,
        mode="solver"
    )

    print(f"Fluent version: {solver.get_fluent_version()}")

    # Load case
    print(f"\nLoading case: {CASE_FILE}")
    solver.settings.file.read_case(file_name=str(CASE_FILE))

    # Set boundary conditions
    print(f"\nSetting boundary conditions:")
    print(f"  Cold inlet: {COLD_VEL} m/s")
    print(f"  Hot inlet: {HOT_VEL} m/s")

    cold_inlet = solver.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"]
    cold_inlet.momentum.velocity.value = COLD_VEL

    hot_inlet = solver.settings.setup.boundary_conditions.velocity_inlet["hot-inlet"]
    hot_inlet.momentum.velocity.value = HOT_VEL

    # Initialize and run
    print(f"\nInitializing solution...")
    solver.settings.solution.initialization.initialization_type = "standard"
    solver.settings.solution.initialization.standard_initialize()

    print(f"Running {ITERATIONS} iterations...")
    solver.settings.solution.run_calculation.iterate(iter_count=ITERATIONS)

    # Try to extract volume data
    print("\n" + "="*70)
    print("Extracting volume field data...")
    print("="*70)

    try:
        # Get solution variable data service (for cell zones, not surfaces)
        print(f"\nAttempting to extract from cell zone: '{VOLUME_NAME}'")
        sv_data = solver.fields.solution_variable_data

        # Temperature (SV_T)
        print("\n[1/6] Temperature...")
        temp_dict = sv_data.get_data(
            variable_name="SV_T",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        temperature = np.array(temp_dict[VOLUME_NAME])

        # Pressure (SV_P)
        print("[2/6] Pressure...")
        press_dict = sv_data.get_data(
            variable_name="SV_P",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        pressure = np.array(press_dict[VOLUME_NAME])

        # Velocity X (SV_U)
        print("[3/6] Velocity X...")
        vx_dict = sv_data.get_data(
            variable_name="SV_U",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        vx = np.array(vx_dict[VOLUME_NAME])

        # Velocity Y (SV_V)
        print("[4/6] Velocity Y...")
        vy_dict = sv_data.get_data(
            variable_name="SV_V",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        vy = np.array(vy_dict[VOLUME_NAME])

        # Velocity Z (SV_W)
        print("[5/6] Velocity Z...")
        vz_dict = sv_data.get_data(
            variable_name="SV_W",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        vz = np.array(vz_dict[VOLUME_NAME])

        # Coordinates (SV_CENTROID) - returns flattened array [x1,y1,z1,x2,y2,z2,...]
        print("[6/6] Coordinates...")
        coord_dict = sv_data.get_data(
            variable_name="SV_CENTROID",
            zone_names=[VOLUME_NAME],
            domain_name="mixture"
        )
        coords_flat = np.array(coord_dict[VOLUME_NAME])
        # Reshape from flat [x1,y1,z1,x2,y2,z2,...] to (n_cells, 3)
        n_cells = len(temperature)
        coords = coords_flat.reshape((n_cells, 3))
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        print("\n" + "="*70)
        print("Data extracted successfully!")
        print("="*70)
        print(f"\nVolume cells: {len(temperature)}")
        print(f"\nTemperature: {temperature.min():.2f} - {temperature.max():.2f} K")
        print(f"Pressure: {pressure.min():.2f} - {pressure.max():.2f} Pa")
        print(f"Velocity X: {vx.min():.4f} - {vx.max():.4f} m/s")
        print(f"Velocity Y: {vy.min():.4f} - {vy.max():.4f} m/s")
        print(f"Velocity Z: {vz.min():.4f} - {vz.max():.4f} m/s")

        # Save to NPZ
        output_file = Path(__file__).parent / "test_volume_data.npz"
        np.savez_compressed(
            output_file,
            temperature=temperature,
            pressure=pressure,
            velocity_x=vx,
            velocity_y=vy,
            velocity_z=vz,
            x=x,
            y=y,
            z=z,
            parameters=np.array([COLD_VEL, HOT_VEL])
        )
        print(f"\n[OK] Data saved to: {output_file}")

        # Create 3D scatter plot of temperature
        print("\n" + "="*70)
        print("Creating 3D visualization...")
        print("="*70)

        # Subsample for visualization (every 10th point to avoid cluttering)
        step = max(1, len(temperature) // 5000)
        x_plot = x[::step]
        y_plot = y[::step]
        z_plot = z[::step]
        temp_plot = temperature[::step]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x_plot, y_plot, z_plot, c=temp_plot,
                            cmap='hot', s=5, alpha=0.6, edgecolors='none')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'3D Temperature Distribution\n(Cold={COLD_VEL} m/s, Hot={HOT_VEL} m/s, {len(temp_plot)} points)',
                    fontsize=14, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Temperature (K)', fontsize=11)

        # Set equal aspect ratio
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

        # Save plot
        plot_file = Path(__file__).parent / "test_volume_temperature_3d.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n[OK] 3D plot saved: {plot_file}")

        plt.show()

    except Exception as e:
        print(f"\n[ERROR] Error extracting volume data: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        print("\nClosing Fluent...")
        solver.exit()
        print("Done!")


if __name__ == "__main__":
    main()
