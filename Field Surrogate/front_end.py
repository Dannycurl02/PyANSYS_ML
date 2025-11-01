#!/usr/bin/env python
"""
Field Surrogate - Front End Interface
======================================
Interactive TUI menu system for the complete field surrogate workflow.
"""

import sys
import numpy as np
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

class Config:
    """Default configuration settings."""
    # Project directory
    PROJECT_DIR = Path(__file__).parent

    # DOE Test Matrix
    COLD_VEL_ARRAY = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # m/s
    HOT_VEL_ARRAY = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])   # m/s

    # Fluent Settings
    FLUENT_PRECISION = "single"  # "single" or "double"
    PROCESSOR_COUNT = 11
    DIMENSION = 3  # 3D simulation
    ITERATIONS = 200

    # Case File Path
    CASE_FILE = PROJECT_DIR / "elbow.cas.h5"

    # Surface to Extract
    PLANE_NAME = "mid-plane"

    # Output Settings
    OUTPUT_NPZ = "field_surrogate_dataset.npz"


# Global config instance
config = Config()


# ============================================================
# TUI MENU FUNCTIONS
# ============================================================

def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def print_menu(title, options):
    """
    Print a menu with options.

    Parameters
    ----------
    title : str
        Menu title
    options : list of str
        Menu options
    """
    print_header(title)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print(f"  [0] {'Back' if title != 'MAIN MENU' else 'Exit'}")
    print("="*70)


def get_choice(max_choice):
    """Get user choice with validation."""
    while True:
        try:
            choice = int(input("\nEnter choice: ").strip())
            if 0 <= choice <= max_choice:
                return choice
            print(f"Invalid choice. Please enter 0-{max_choice}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def pause():
    """Pause and wait for user input."""
    input("\nPress Enter to continue...")


# ============================================================
# CONFIGURATION EDITOR
# ============================================================

def edit_settings():
    """Interactive settings editor for DOE configuration."""
    global config

    while True:
        print_header("DOE CONFIGURATION SETTINGS")
        print(f"\n  [1] Test Matrix")
        print(f"      Cold velocities: {len(config.COLD_VEL_ARRAY)} points")
        print(f"      Hot velocities: {len(config.HOT_VEL_ARRAY)} points")
        print(f"      Total simulations: {len(config.COLD_VEL_ARRAY) * len(config.HOT_VEL_ARRAY)}")

        print(f"\n  [2] Fluent Settings")
        print(f"      Precision: {config.FLUENT_PRECISION}")
        print(f"      Processors: {config.PROCESSOR_COUNT}")
        print(f"      Iterations: {config.ITERATIONS}")

        print(f"\n  [3] Case File")
        print(f"      Path: {config.CASE_FILE}")
        print(f"      Exists: {'Yes' if config.CASE_FILE.exists() else 'No'}")

        print(f"\n  [4] Surface Name")
        print(f"      Plane: {config.PLANE_NAME}")

        print(f"\n  [5] Output Filename")
        print(f"      File: {config.OUTPUT_NPZ}")

        print(f"\n  [6] Reset to Defaults")
        print(f"\n  [0] Back to Main Menu")
        print("="*70)

        choice = get_choice(6)

        if choice == 0:
            break
        elif choice == 1:
            edit_test_matrix()
        elif choice == 2:
            edit_fluent_settings()
        elif choice == 3:
            edit_case_file()
        elif choice == 4:
            edit_surface_name()
        elif choice == 5:
            edit_output_filename()
        elif choice == 6:
            config = Config()
            print("\n✓ Settings reset to defaults")
            pause()


def edit_test_matrix():
    """Edit test matrix velocities."""
    global config

    print_header("EDIT TEST MATRIX")
    print(f"\nCurrent cold velocities: {config.COLD_VEL_ARRAY.tolist()}")
    print(f"Current hot velocities: {config.HOT_VEL_ARRAY.tolist()}")

    print(f"\n[1] Edit cold velocities")
    print(f"[2] Edit hot velocities")
    print(f"[3] Quick presets")
    print(f"[0] Back")

    choice = get_choice(3)

    if choice == 1:
        try:
            vals = input("\nEnter cold velocities (space-separated): ").strip()
            config.COLD_VEL_ARRAY = np.array([float(x) for x in vals.split()])
            print(f"✓ Updated: {config.COLD_VEL_ARRAY.tolist()}")
        except:
            print("✗ Invalid input")
        pause()

    elif choice == 2:
        try:
            vals = input("\nEnter hot velocities (space-separated): ").strip()
            config.HOT_VEL_ARRAY = np.array([float(x) for x in vals.split()])
            print(f"✓ Updated: {config.HOT_VEL_ARRAY.tolist()}")
        except:
            print("✗ Invalid input")
        pause()

    elif choice == 3:
        print(f"\n[1] 3×3 (9 simulations) - Quick test")
        print(f"[2] 5×5 (25 simulations) - Medium")
        print(f"[3] 7×7 (49 simulations) - Default")
        print(f"[4] 10×10 (100 simulations) - High resolution")

        preset = get_choice(4)
        if preset == 1:
            config.COLD_VEL_ARRAY = np.linspace(0.1, 0.7, 3)
            config.HOT_VEL_ARRAY = np.linspace(0.8, 2.0, 3)
        elif preset == 2:
            config.COLD_VEL_ARRAY = np.linspace(0.1, 0.7, 5)
            config.HOT_VEL_ARRAY = np.linspace(0.8, 2.0, 5)
        elif preset == 3:
            config.COLD_VEL_ARRAY = np.linspace(0.1, 0.7, 7)
            config.HOT_VEL_ARRAY = np.linspace(0.8, 2.0, 7)
        elif preset == 4:
            config.COLD_VEL_ARRAY = np.linspace(0.1, 0.7, 10)
            config.HOT_VEL_ARRAY = np.linspace(0.8, 2.0, 10)
        if preset > 0:
            print(f"✓ Applied preset {preset}")
        pause()


def edit_fluent_settings():
    """Edit Fluent solver settings."""
    global config

    print_header("EDIT FLUENT SETTINGS")
    print(f"\nCurrent settings:")
    print(f"  Precision: {config.FLUENT_PRECISION}")
    print(f"  Processors: {config.PROCESSOR_COUNT}")
    print(f"  Iterations: {config.ITERATIONS}")

    print(f"\n[1] Change precision (single/double)")
    print(f"[2] Change processor count")
    print(f"[3] Change iteration count")
    print(f"[0] Back")

    choice = get_choice(3)

    if choice == 1:
        prec = input("\nEnter precision (single/double): ").strip().lower()
        if prec in ['single', 'double']:
            config.FLUENT_PRECISION = prec
            print(f"✓ Updated to {prec}")
        else:
            print("✗ Invalid precision")
        pause()

    elif choice == 2:
        try:
            count = int(input("\nEnter processor count: ").strip())
            if count > 0:
                config.PROCESSOR_COUNT = count
                print(f"✓ Updated to {count}")
            else:
                print("✗ Must be positive")
        except:
            print("✗ Invalid input")
        pause()

    elif choice == 3:
        try:
            iters = int(input("\nEnter iteration count: ").strip())
            if iters > 0:
                config.ITERATIONS = iters
                print(f"✓ Updated to {iters}")
            else:
                print("✗ Must be positive")
        except:
            print("✗ Invalid input")
        pause()


def edit_case_file():
    """Edit case file path."""
    global config

    print_header("EDIT CASE FILE")
    print(f"\nCurrent: {config.CASE_FILE}")
    print(f"Exists: {'Yes' if config.CASE_FILE.exists() else 'No'}")

    new_path = input("\nEnter new case file path (or Enter to cancel): ").strip()
    if new_path:
        config.CASE_FILE = Path(new_path)
        print(f"✓ Updated to {config.CASE_FILE}")
        if not config.CASE_FILE.exists():
            print("  Warning: File does not exist!")
    pause()


def edit_surface_name():
    """Edit surface/plane name."""
    global config

    print_header("EDIT SURFACE NAME")
    print(f"\nCurrent: {config.PLANE_NAME}")

    new_name = input("\nEnter new surface name (or Enter to cancel): ").strip()
    if new_name:
        config.PLANE_NAME = new_name
        print(f"✓ Updated to {config.PLANE_NAME}")
    pause()


def edit_output_filename():
    """Edit output NPZ filename."""
    global config

    print_header("EDIT OUTPUT FILENAME")
    print(f"\nCurrent: {config.OUTPUT_NPZ}")

    new_name = input("\nEnter new output filename (or Enter to cancel): ").strip()
    if new_name:
        if not new_name.endswith('.npz'):
            new_name += '.npz'
        config.OUTPUT_NPZ = new_name
        print(f"✓ Updated to {config.OUTPUT_NPZ}")
    pause()


# ============================================================
# WORKFLOW FUNCTIONS
# ============================================================

def run_doe_workflow():
    """Run the complete DOE workflow with current settings."""

    while True:
        print_header("DOE WORKFLOW")

        # Display configuration
        print("\n[Configuration Summary]")
        print(f"  Test matrix: {len(config.COLD_VEL_ARRAY)} × {len(config.HOT_VEL_ARRAY)} = {len(config.COLD_VEL_ARRAY) * len(config.HOT_VEL_ARRAY)} simulations")
        print(f"  Cold inlet velocities: {config.COLD_VEL_ARRAY.tolist()}")
        print(f"  Hot inlet velocities: {config.HOT_VEL_ARRAY.tolist()}")
        print(f"  Fluent: {config.FLUENT_PRECISION} precision, {config.PROCESSOR_COUNT} processors")
        print(f"  Iterations per simulation: {config.ITERATIONS}")
        print(f"  Case file: {config.CASE_FILE}")

        # DOE Menu
        print(f"\n{'='*70}")
        print(f"[1] Run with current settings")
        print(f"[2] Edit DOE settings")
        print(f"[0] Back to main menu")
        print(f"{'='*70}")

        choice = get_choice(2)

        if choice == 0:
            return
        elif choice == 2:
            edit_settings()
            continue
        elif choice == 1:
            break

    # Get output filename
    print(f"\n{'='*70}")
    print("Output File")
    print(f"{'='*70}")

    while True:
        print(f"\nCurrent output: {config.OUTPUT_NPZ}")
        filename = input("Enter output filename (or press Enter to use current): ").strip()

        if not filename:
            filename = config.OUTPUT_NPZ
        elif not filename.endswith('.npz'):
            filename += '.npz'

        output_path = config.PROJECT_DIR / filename

        # Check if file exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n⚠️  File already exists: {filename} ({size_mb:.2f} MB)")
            overwrite = input("Overwrite existing file? [y/N]: ").strip().lower()
            if overwrite != 'y':
                print("Please enter a different filename.")
                continue
            else:
                print(f"✓ Will overwrite {filename}")
                break
        else:
            print(f"✓ Will create new file: {filename}")
            break

    config.OUTPUT_NPZ = filename

    # Run DOE
    print("\n" + "="*70)
    print("[STEP 1] Running DOE Simulations")
    print("="*70)

    try:
        import auto_fl_matrix
        auto_fl_matrix.run_doe(
            cold_vel_array=config.COLD_VEL_ARRAY,
            hot_vel_array=config.HOT_VEL_ARRAY,
            case_file=config.CASE_FILE,
            plane_name=config.PLANE_NAME,
            fluent_precision=config.FLUENT_PRECISION,
            processor_count=config.PROCESSOR_COUNT,
            iterations=config.ITERATIONS,
            output_file=config.PROJECT_DIR / config.OUTPUT_NPZ,
            separate_fluent_window=True  # Redirect Fluent output to separate window
        )
        print("\n✓ DOE simulations completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during DOE simulations: {e}")
        pause()
        return

    # Verify dataset
    print("\n" + "="*70)
    print("[STEP 2] Verifying Dataset")
    print("="*70)

    dataset_path = config.PROJECT_DIR / config.OUTPUT_NPZ
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  Dataset file: {dataset_path}")
        print(f"  Size: {size_mb:.2f} MB")

        data = np.load(dataset_path, allow_pickle=True)
        print(f"  Arrays in dataset: {list(data.keys())}")
        print(f"  Number of simulations: {data['parameters'].shape[0]}")
        print(f"  Points per simulation: {data['coordinates'].shape[0]}")
        print("\n✓ Dataset verified successfully!")
    else:
        print(f"  ✗ Dataset file not found: {dataset_path}")
        pause()
        return

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nDataset ready for surrogate model training:")
    print(f"  {dataset_path}")

    pause()


def inspect_dataset():
    """Inspect and visualize existing datasets."""
    import fluent_output_check

    datasets = list(config.PROJECT_DIR.glob("*.npz"))

    if not datasets:
        print("\n✗ No datasets found in project directory!")
        pause()
        return

    print_header("DATASET INSPECTION")
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        size_mb = ds.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {ds.name} ({size_mb:.2f} MB)")

    print(f"  [0] Back")

    choice = get_choice(len(datasets))
    if choice == 0:
        return

    dataset_file = datasets[choice - 1]

    print(f"\n[1] Visualize random simulation")
    print(f"[2] Visualize specific simulation")
    print(f"[3] View all simulations grid")
    print(f"[0] Back")

    viz_choice = get_choice(3)

    try:
        if viz_choice == 1:
            fluent_output_check.visualize_random_simulation(dataset_file)
        elif viz_choice == 2:
            sim_idx = int(input("\nEnter simulation index: ").strip())
            fluent_output_check.visualize_random_simulation(dataset_file, sim_index=sim_idx)
        elif viz_choice == 3:
            print("\nSelect field to visualize:")
            print("  [1] Temperature")
            print("  [2] Pressure")
            print("  [3] Velocity Magnitude")
            field_choice = get_choice(3)
            if field_choice == 0:
                return
            field_map = {1: 'temperature', 2: 'pressure', 3: 'velocity_magnitude'}
            fluent_output_check.visualize_all_simulations_grid(dataset_file, field=field_map[field_choice])
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")

    pause()


def train_model():
    """Train surrogate model."""
    import train_surrogate

    print_header("TRAIN SURROGATE MODEL")
    print("\nLaunching training program...")
    print("(Follow prompts in training program)")
    print("="*70)

    try:
        # This will run the interactive training
        import subprocess
        subprocess.run([sys.executable, str(config.PROJECT_DIR / "modules" / "train_surrogate.py")])
    except Exception as e:
        print(f"\n✗ Error launching training: {e}")

    pause()


def validate_model():
    """Validate surrogate model."""
    print_header("VALIDATE SURROGATE MODEL")
    print("\nLaunching validation program...")
    print("(Follow prompts in validation program)")
    print("="*70)

    try:
        import subprocess
        subprocess.run([sys.executable, str(config.PROJECT_DIR / "modules" / "predict_with_surrogate.py")])
    except Exception as e:
        print(f"\n✗ Error launching validation: {e}")

    pause()


def cleanup_files():
    """Clean up temporary files."""
    import fluent_cleanup

    print_header("CLEANUP UTILITIES")

    trn_count = len(list(config.PROJECT_DIR.glob("*.trn")))
    pycache_exists = (config.PROJECT_DIR / "__pycache__").exists()
    modules_pycache = (config.PROJECT_DIR / "modules" / "__pycache__").exists()

    print(f"\nFiles to clean:")
    print(f"  .trn files: {trn_count}")
    print(f"  __pycache__ (root): {'Yes' if pycache_exists else 'No'}")
    print(f"  __pycache__ (modules): {'Yes' if modules_pycache else 'No'}")

    if trn_count == 0 and not pycache_exists and not modules_pycache:
        print("\nNothing to clean!")
        pause()
        return

    response = input("\nProceed with cleanup? [y/N]: ").strip().lower()
    if response == 'y':
        fluent_cleanup.cleanup_trn_files(config.PROJECT_DIR, verbose=True)
        fluent_cleanup.cleanup_pycache(config.PROJECT_DIR, verbose=True)
        fluent_cleanup.cleanup_pycache(config.PROJECT_DIR / "modules", verbose=True)
        print("\n✓ Cleanup completed!")
    else:
        print("\nCleanup cancelled.")

    pause()


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    """Main TUI menu."""
    while True:
        clear_screen()
        print_menu("FIELD SURROGATE - MAIN MENU", [
            "Run DOE Workflow (Generate Dataset)",
            "Inspect & Visualize Dataset",
            "Train Surrogate Model",
            "Validate Surrogate Model",
            "Cleanup Temporary Files"
        ])

        choice = get_choice(5)

        if choice == 0:
            print("\nExiting Field Surrogate front end. Goodbye!")
            sys.exit(0)
        elif choice == 1:
            run_doe_workflow()
        elif choice == 2:
            inspect_dataset()
        elif choice == 3:
            train_model()
        elif choice == 4:
            validate_model()
        elif choice == 5:
            cleanup_files()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
