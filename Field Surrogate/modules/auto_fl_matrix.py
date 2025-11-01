#!/usr/bin/env python
"""
Automated Fluent DOE Matrix Runner
==================================
Runs a design of experiments (DOE) matrix for the mixing elbow case
and saves all field data to a compressed NumPy archive.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import ansys.fluent.core as pyfluent
from fluent_logger import FluentLogger


def run_doe(cold_vel_array, hot_vel_array, case_file, plane_name,
            fluent_precision, processor_count, iterations, output_file,
            separate_fluent_window=True):
    """
    Run DOE simulations and save results to NPZ file.

    Parameters
    ----------
    cold_vel_array : np.ndarray
        Array of cold inlet velocities (m/s)
    hot_vel_array : np.ndarray
        Array of hot inlet velocities (m/s)
    case_file : str or Path
        Path to Fluent case file
    plane_name : str
        Name of the surface to extract data from
    fluent_precision : str
        "single" or "double"
    processor_count : int
        Number of processors for Fluent
    iterations : int
        Number of iterations per simulation
    output_file : str or Path
        Output NPZ file path
    separate_fluent_window : bool, optional
        If True, redirect Fluent output to separate console window (default: True)
    """

    n_cold = len(cold_vel_array)
    n_hot = len(hot_vel_array)
    n_sims = n_cold * n_hot

    print(f"\nStarting DOE with {n_sims} simulations...")
    print(f"  Cold velocities: {n_cold} levels")
    print(f"  Hot velocities: {n_hot} levels")

    # Initialize Fluent output redirection if separate window requested
    fluent_log_file = None
    original_stdout = None
    original_stderr = None

    if separate_fluent_window:
        import tempfile
        import subprocess
        # Create temporary log file for Fluent output
        fluent_log_file = tempfile.NamedTemporaryFile(
            mode='w+',
            suffix='.log',
            prefix='fluent_output_',
            delete=False,
            buffering=1  # Line buffered
        )
        fluent_log_path = fluent_log_file.name

        # Write header to Fluent log
        fluent_log_file.write("="*70 + "\n")
        fluent_log_file.write("FLUENT DOE SIMULATIONS OUTPUT\n")
        fluent_log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fluent_log_file.write("="*70 + "\n\n")
        fluent_log_file.flush()

        print(f"\n[Setup] Fluent output will be redirected to separate window")
        print(f"  Log file: {fluent_log_path}")

        # Launch PowerShell window to tail the log file
        ps_cmd = f'Get-Content -Path "{fluent_log_path}" -Wait'
        subprocess.Popen(
            ['powershell', '-NoExit', '-Command', ps_cmd],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

    # Launch Fluent once for all simulations
    print(f"\n[Setup] Launching Fluent...")

    # Redirect stdout/stderr to Fluent window during Fluent operations
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if separate_fluent_window:
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

    solver_session = pyfluent.launch_fluent(
        precision=fluent_precision,
        processor_count=processor_count,
        dimension=pyfluent.Dimension.THREE,
        ui_mode=pyfluent.UIMode.HIDDEN_GUI
    )

    # Restore stdout/stderr for our progress messages
    if separate_fluent_window:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    version = solver_session.get_fluent_version()
    print(f"  Fluent version: {version}")

    # Read case file
    print(f"\n[Setup] Loading case file: {case_file}")

    # Redirect to Fluent window for case loading
    if separate_fluent_window:
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

    try:
        solver_session.settings.file.read_case_data(file_name=str(case_file))
    except Exception:
        solver_session.settings.file.read_case(file_name=str(case_file))

    # Restore for progress messages
    if separate_fluent_window:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    print(f"  Case loaded successfully")

    # Storage arrays - will collect data from all simulations
    parameters_list = []  # (cold_vel, hot_vel) for each sim
    coordinates_data = None  # (x, y, z) - same for all sims, only store once
    temperature_list = []
    pressure_list = []
    velocity_x_list = []
    velocity_y_list = []
    velocity_z_list = []

    # Field data accessor
    fd = solver_session.fields.field_data

    # Run DOE matrix
    sim_count = 0
    start_time = datetime.now()

    for i, cold_vel in enumerate(cold_vel_array):
        for j, hot_vel in enumerate(hot_vel_array):
            sim_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / sim_count if sim_count > 0 else 0
            remaining = avg_time * (n_sims - sim_count)

            # Main console: brief progress only
            print(f"\n[Sim {sim_count}/{n_sims}] Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s")
            print(f"  Progress: {100*sim_count/n_sims:.1f}% | "
                  f"Elapsed: {elapsed/60:.1f} min | "
                  f"Remaining: ~{remaining/60:.1f} min")

            # Redirect to Fluent window for all Fluent operations
            if separate_fluent_window:
                sys.stdout = fluent_log_file
                sys.stderr = fluent_log_file
                print(f"\n{'='*70}")
                print(f"SIMULATION {sim_count}/{n_sims}")
                print(f"{'='*70}")
                print(f"Cold inlet velocity: {cold_vel:.3f} m/s")
                print(f"Hot inlet velocity: {hot_vel:.3f} m/s")
                print(f"Iterations: {iterations}")
                fluent_log_file.flush()

            # Set boundary conditions (using newer syntax)
            solver_session.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"].momentum.velocity_magnitude.value = float(cold_vel)
            solver_session.settings.setup.boundary_conditions.velocity_inlet["hot-inlet"].momentum.velocity_magnitude.value = float(hot_vel)

            # Initialize and solve (Fluent output goes to separate window)
            solver_session.settings.solution.initialization.initialization_type = "standard"
            solver_session.settings.solution.initialization.standard_initialize()
            solver_session.settings.solution.run_calculation.iterate(iter_count=iterations)

            # Restore stdout for progress messages
            if separate_fluent_window:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            # Extract field data
            print(f"  Extracting field data from '{plane_name}'...")

            try:
                # Get temperature
                temp_dict = fd.get_scalar_field_data(
                    field_name='temperature',
                    surfaces=[plane_name],
                    node_value=True
                )
                temp_data = temp_dict[plane_name]

                # Get pressure
                press_dict = fd.get_scalar_field_data(
                    field_name='absolute-pressure',
                    surfaces=[plane_name],
                    node_value=True
                )
                press_data = press_dict[plane_name]

                # Get velocity components
                vx_dict = fd.get_scalar_field_data(
                    field_name='x-velocity',
                    surfaces=[plane_name],
                    node_value=True
                )
                vx_data = vx_dict[plane_name]

                vy_dict = fd.get_scalar_field_data(
                    field_name='y-velocity',
                    surfaces=[plane_name],
                    node_value=True
                )
                vy_data = vy_dict[plane_name]

                vz_dict = fd.get_scalar_field_data(
                    field_name='z-velocity',
                    surfaces=[plane_name],
                    node_value=True
                )
                vz_data = vz_dict[plane_name]

                # Get coordinates (only once - same for all sims)
                if coordinates_data is None:
                    print(f"  Extracting coordinates (first simulation)...")
                    x_dict = fd.get_scalar_field_data(
                        field_name='x-coordinate',
                        surfaces=[plane_name],
                        node_value=True
                    )
                    y_dict = fd.get_scalar_field_data(
                        field_name='y-coordinate',
                        surfaces=[plane_name],
                        node_value=True
                    )
                    z_dict = fd.get_scalar_field_data(
                        field_name='z-coordinate',
                        surfaces=[plane_name],
                        node_value=True
                    )

                    x_coords = x_dict[plane_name]
                    y_coords = y_dict[plane_name]
                    z_coords = z_dict[plane_name]

                    # Stack into (n_points, 3) array
                    coordinates_data = np.stack([x_coords, y_coords, z_coords], axis=1)
                    print(f"  Coordinates shape: {coordinates_data.shape}")

                # Store results
                parameters_list.append([cold_vel, hot_vel])
                temperature_list.append(temp_data)
                pressure_list.append(press_data)
                velocity_x_list.append(vx_data)
                velocity_y_list.append(vy_data)
                velocity_z_list.append(vz_data)

                print(f"  ✓ Data extracted: {len(temp_data)} points")

            except Exception as e:
                error_msg = f"Error extracting data: {e}"
                print(f"  ✗ {error_msg}")
                raise

    # Close Fluent
    from fluent_cleanup import end_fluent_session
    end_fluent_session(solver_session, verbose=False)

    if separate_fluent_window:
        fluent_log_file.write("\n" + "="*70 + "\n")
        fluent_log_file.write("ALL SIMULATIONS COMPLETE\n")
        fluent_log_file.write("="*70 + "\n")
        fluent_log_file.write(f"Total simulations: {n_sims}\n")
        fluent_log_file.write(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes\n")
        fluent_log_file.write(f"\nFluent session closed\n")
        fluent_log_file.flush()

    # Convert lists to arrays
    print(f"\n[Saving] Converting data to NumPy arrays...")
    parameters_array = np.array(parameters_list)  # Shape: (n_sims, 2)
    temperature_array = np.array(temperature_list)  # Shape: (n_sims, n_points)
    pressure_array = np.array(pressure_list)  # Shape: (n_sims, n_points)
    velocity_x_array = np.array(velocity_x_list)  # Shape: (n_sims, n_points)
    velocity_y_array = np.array(velocity_y_list)  # Shape: (n_sims, n_points)
    velocity_z_array = np.array(velocity_z_list)  # Shape: (n_sims, n_points)

    print(f"  Parameters shape: {parameters_array.shape}")
    print(f"  Coordinates shape: {coordinates_data.shape}")
    print(f"  Temperature shape: {temperature_array.shape}")
    print(f"  Pressure shape: {pressure_array.shape}")
    print(f"  Velocity X shape: {velocity_x_array.shape}")
    print(f"  Velocity Y shape: {velocity_y_array.shape}")
    print(f"  Velocity Z shape: {velocity_z_array.shape}")

    # Save to compressed NPZ file
    print(f"\n[Saving] Writing to {output_file}...")
    np.savez_compressed(
        output_file,
        parameters=parameters_array,
        coordinates=coordinates_data,
        temperature=temperature_array,
        pressure=pressure_array,
        velocity_x=velocity_x_array,
        velocity_y=velocity_y_array,
        velocity_z=velocity_z_array,
        cold_vel_array=cold_vel_array,
        hot_vel_array=hot_vel_array,
        metadata={
            'case_file': str(case_file),
            'plane_name': plane_name,
            'iterations': iterations,
            'n_simulations': n_sims,
            'n_points': coordinates_data.shape[0],
            'timestamp': datetime.now().isoformat()
        }
    )

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  ✓ Dataset saved: {file_size_mb:.2f} MB")

    # Clean up .trn files
    from fluent_cleanup import cleanup_trn_files
    project_dir = Path(output_file).parent
    cleanup_trn_files(project_dir, verbose=True)

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*70}")
    print(f"DOE COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"  Total simulations: {n_sims}")
    print(f"  Total time: {total_time:.1f} minutes")
    print(f"  Average time per simulation: {total_time/n_sims:.2f} minutes")
    print(f"  Dataset: {output_file}")

    # Close Fluent log file
    if separate_fluent_window:
        fluent_log_file.write("\n" + "="*70 + "\n")
        fluent_log_file.write("DOE WORKFLOW COMPLETE\n")
        fluent_log_file.write("="*70 + "\n")
        fluent_log_file.write(f"Dataset saved: {file_size_mb:.2f} MB\n")
        fluent_log_file.write(f"Average time per simulation: {total_time/n_sims:.2f} minutes\n")
        fluent_log_file.write("\nThis window will remain open for review.\n")
        fluent_log_file.write("You can close it manually when done.\n")
        fluent_log_file.close()
        print(f"\n  Fluent log saved: {fluent_log_path}")


if __name__ == "__main__":
    # Standalone execution with default parameters
    print("This module should be run via runner.py")
    print("For standalone testing, modify parameters below:")

    # Example standalone execution
    COLD_VEL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    HOT_VEL = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    CASE = Path(__file__).parent / "elbow.cas.h5"

    run_doe(
        cold_vel_array=COLD_VEL,
        hot_vel_array=HOT_VEL,
        case_file=CASE,
        plane_name="mid-plane",
        fluent_precision="single",
        processor_count=6,
        iterations=200,
        output_file=Path(__file__).parent / "field_surrogate_dataset.npz"
    )
