"""
Data Visualizer Module
=======================
Visualizes training results, comparison plots, and model performance.
"""

from pathlib import Path
import json


def data_visualization_menu(ui_helpers):
    """
    Main menu for data visualization.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module
    """
    selected_training = None

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DATA VISUALIZATION")

        if selected_training:
            print(f"\nSelected Training: {selected_training.name}")
        else:
            print("\nNo training data selected")

        print(f"\n{'='*70}")
        print("  [1] Select Training Data Folder")
        print("  [2] View Training/Validation Curves")
        print("  [3] View Model Performance Metrics")
        print("  [4] Compare Predictions vs Actual")
        print("  [5] Visualize POD Modes")
        print("  [6] Error Distribution Analysis")
        print("  [0] Back to Main Menu")
        print("="*70)

        choice = ui_helpers.get_choice(6)

        if choice == 0:
            return
        elif choice == 1:
            selected_training = select_training_folder(ui_helpers)
        elif choice == 2:
            if selected_training:
                view_training_curves(selected_training, ui_helpers)
            else:
                print("\n✗ Please select a training folder first.")
                ui_helpers.pause()
        elif choice == 3:
            if selected_training:
                view_performance_metrics(selected_training, ui_helpers)
            else:
                print("\n✗ Please select a training folder first.")
                ui_helpers.pause()
        elif choice == 4:
            if selected_training:
                view_comparison_plots(selected_training, ui_helpers)
            else:
                print("\n✗ Please select a training folder first.")
                ui_helpers.pause()
        elif choice == 5:
            if selected_training:
                visualize_pod_modes(selected_training, ui_helpers)
            else:
                print("\n✗ Please select a training folder first.")
                ui_helpers.pause()
        elif choice == 6:
            if selected_training:
                view_error_distribution(selected_training, ui_helpers)
            else:
                print("\n✗ Please select a training folder first.")
                ui_helpers.pause()


def select_training_folder(ui_helpers):
    """
    Browse for and select a training data folder.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    Path or None
        Selected training folder
    """
    from tkinter import Tk, filedialog

    ui_helpers.clear_screen()
    ui_helpers.print_header("SELECT TRAINING FOLDER")

    print("\nOpening folder dialog...")
    Tk().withdraw()

    folder = filedialog.askdirectory(
        title="Select Training Data Folder",
        mustexist=True
    )

    if not folder:
        print("\n✗ No folder selected")
        ui_helpers.pause()
        return None

    folder = Path(folder)

    # Verify it's a valid training folder
    if not (folder / "metadata.json").exists():
        print(f"\n✗ Invalid training folder: metadata.json not found")
        ui_helpers.pause()
        return None

    print(f"\n✓ Selected: {folder.name}")
    ui_helpers.pause()

    return folder


def view_training_curves(training_dir, ui_helpers):
    """
    Display training and validation loss curves.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("TRAINING/VALIDATION CURVES")

    print(f"\nTraining Directory: {training_dir.name}")

    history_file = training_dir / "training_history.json"

    if not history_file.exists():
        print("\n✗ Training history not found")
        ui_helpers.pause()
        return

    print("\n[PLACEHOLDER] This will display:")
    print("  - Training loss vs epochs")
    print("  - Validation loss vs epochs")
    print("  - Learning rate schedule")
    print("  - Early stopping marker (if applicable)")
    print("\n  Matplotlib visualization will be shown here.")

    ui_helpers.pause()


def view_performance_metrics(training_dir, ui_helpers):
    """
    Display model performance metrics.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL PERFORMANCE METRICS")

    metadata_file = training_dir / "metadata.json"

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print(f"\nModel: {training_dir.name}")
        print("\n" + "="*70)
        print("TRAINING METRICS:")
        print("="*70)
        print(f"  Final Training Loss: [PLACEHOLDER]")
        print(f"  Final Validation Loss: [PLACEHOLDER]")
        print(f"  R² Score: [PLACEHOLDER]")
        print(f"  Mean Absolute Error: [PLACEHOLDER]")
        print(f"  Root Mean Square Error: [PLACEHOLDER]")

        print("\n" + "="*70)
        print("MODEL CONFIGURATION:")
        print("="*70)
        if 'model_config' in metadata:
            config = metadata['model_config']
            print(f"  POD Modes: {config.get('pod_modes', 'N/A')}")
            print(f"  Architecture: {config.get('encoder', 'N/A')}")
            print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
            print(f"  Epochs Trained: {config.get('epochs', 'N/A')}")
    else:
        print("\n✗ Metadata file not found")

    ui_helpers.pause()


def view_comparison_plots(training_dir, ui_helpers):
    """
    Display prediction vs actual comparison plots.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("PREDICTIONS VS ACTUAL")

    print(f"\nTraining Directory: {training_dir.name}")

    print("\n[PLACEHOLDER] This will display:")
    print("  - Scatter plot: Predicted vs Actual values")
    print("  - Residual plots")
    print("  - Output field comparisons (side-by-side)")
    print("  - Error heatmaps on geometry")
    print("\n  Interactive matplotlib visualizations")

    ui_helpers.pause()


def visualize_pod_modes(training_dir, ui_helpers):
    """
    Visualize POD mode shapes.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("POD MODE VISUALIZATION")

    print(f"\nTraining Directory: {training_dir.name}")

    pod_file = training_dir / "pod_modes.npz"

    if not pod_file.exists():
        print("\n✗ POD modes file not found")
        ui_helpers.pause()
        return

    print("\n[PLACEHOLDER] This will display:")
    print("  - First 5-10 POD modes")
    print("  - Explained variance per mode")
    print("  - Cumulative variance explained")
    print("  - Mode shapes on geometry")

    ui_helpers.pause()


def view_error_distribution(training_dir, ui_helpers):
    """
    Display error distribution analysis.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("ERROR DISTRIBUTION ANALYSIS")

    print(f"\nTraining Directory: {training_dir.name}")

    print("\n[PLACEHOLDER] This will display:")
    print("  - Error histogram")
    print("  - Error vs input parameters")
    print("  - Spatial error distribution")
    print("  - Worst-case prediction examples")

    ui_helpers.pause()
