"""
POD (Proper Orthogonal Decomposition) Visualization Script

This standalone script demonstrates how POD works by:
1. Loading 2D temperature field data from NPZ files
2. Decomposing fields using different numbers of modes
3. Reconstructing fields from mode subsets
4. Visualizing reconstruction quality and individual modes

Usage:
    python visualize_pod.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


def load_npz_field(file_path, field_location):
    """
    Load 2D field data from NPZ file.

    Parameters:
    -----------
    file_path : str
        Path to the NPZ file
    field_location : str
        Field location identifier (e.g., 'yz-mid', 'zx-mid', 'bottom')

    Returns:
    --------
    coordinates : ndarray
        Array of shape (n_points, 3) with spatial coordinates
    field_values : ndarray
        Array of shape (n_points,) with field values
    """
    data = np.load(file_path)

    # NPZ keys use pipe-delimited format: "location|field_name"
    field_key = f"{field_location}|temperature"
    coord_key = f"{field_location}|coordinates"

    if field_key not in data or coord_key not in data:
        raise ValueError(f"Field '{field_location}' not found in NPZ file")

    coordinates = data[coord_key]
    field_values = data[field_key]

    return coordinates, field_values


def detect_2d_plane(coordinates):
    """
    Auto-detect 2D plane from coordinate data by finding axes with highest variance.

    Parameters:
    -----------
    coordinates : ndarray
        Array of shape (n_points, 3)

    Returns:
    --------
    varying_dims : list
        Indices [0-2] of the two axes with highest variance
    axis_labels : list
        Labels ['X', 'Y'] for the varying dimensions
    """
    variances = [np.var(coordinates[:, i]) for i in range(3)]
    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]

    axis_names = ['X', 'Y', 'Z']
    axis_labels = [axis_names[varying_dims[0]], axis_names[varying_dims[1]]]

    return varying_dims, axis_labels


def perform_pod(field_data, max_modes=100):
    """
    Perform POD decomposition using PCA.

    Parameters:
    -----------
    field_data : ndarray
        Array of shape (n_samples, n_points) with multiple field snapshots
    max_modes : int
        Maximum number of POD modes to compute

    Returns:
    --------
    pca : PCA object
        Fitted PCA object containing modes and variance information
    """
    # Ensure 2D shape (n_samples, n_features)
    if field_data.ndim == 1:
        field_data = field_data.reshape(1, -1)

    n_samples, n_points = field_data.shape

    # Limit max_modes to reasonable value
    max_modes = min(max_modes, n_samples, n_points // 10, 100)

    print(f"  Fitting POD with up to {max_modes} modes on {n_samples} snapshots...")

    pca = PCA(n_components=max_modes)
    pca.fit(field_data)

    return pca


def reconstruct_from_modes(pca, field_data, n_modes):
    """
    Reconstruct field using only first n_modes.

    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    field_data : ndarray
        Original field data (n_points,)
    n_modes : int
        Number of modes to use for reconstruction

    Returns:
    --------
    reconstructed : ndarray
        Reconstructed field (n_points,)
    r2 : float
        R² score measuring reconstruction quality
    """
    # Transform to mode space
    if field_data.ndim == 1:
        field_data_2d = field_data.reshape(1, -1)
    else:
        field_data_2d = field_data

    coefficients = pca.transform(field_data_2d)

    # Zero out modes beyond n_modes
    coefficients_truncated = coefficients.copy()
    coefficients_truncated[:, n_modes:] = 0

    # Inverse transform
    reconstructed = pca.inverse_transform(coefficients_truncated)

    if field_data.ndim == 1:
        reconstructed = reconstructed.flatten()

    # Calculate R² score
    r2 = r2_score(field_data.flatten(), reconstructed.flatten())

    return reconstructed, r2


def plot_field_2d(ax, coordinates, values, title, varying_dims, vmin=None, vmax=None):
    """
    Plot 2D field as scatter plot.

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    coordinates : ndarray
        Array of shape (n_points, 3)
    values : ndarray
        Field values (n_points,)
    title : str
        Plot title
    varying_dims : list
        Indices of varying dimensions
    vmin, vmax : float
        Colormap limits
    """
    axis_names = ['X', 'Y', 'Z']

    scatter = ax.scatter(
        coordinates[:, varying_dims[0]],
        coordinates[:, varying_dims[1]],
        c=values,
        cmap='jet',
        s=10,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel(f'{axis_names[varying_dims[0]]} (m)', fontsize=10)
    ax.set_ylabel(f'{axis_names[varying_dims[1]]} (m)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    return scatter


def plot_reconstruction_comparison(original, coords, pca, mode_counts, field_name):
    """
    Create comprehensive comparison figure showing reconstruction with different mode counts.

    Parameters:
    -----------
    original : ndarray
        Original field values (n_points,)
    coords : ndarray
        Coordinate array (n_points, 3)
    pca : PCA object
        Fitted PCA object
    mode_counts : list
        List of mode counts to visualize [1, 2, 3, 5, 20]
    field_name : str
        Name of the field for figure title
    """
    varying_dims, axis_labels = detect_2d_plane(coords)

    # Set consistent colormap limits
    vmin, vmax = np.min(original), np.max(original)

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 10))

    # Plot original field
    ax1 = plt.subplot(2, 3, 1)
    scatter = plot_field_2d(ax1, coords, original, 'Original Field', varying_dims, vmin, vmax)
    plt.colorbar(scatter, ax=ax1, label='Temperature (K)', fraction=0.046)

    # Plot reconstructions with different mode counts
    positions = [2, 3, 4, 5, 6]  # Subplot positions
    for idx, n_modes in enumerate(mode_counts):
        ax = plt.subplot(2, 3, positions[idx])

        reconstructed, r2 = reconstruct_from_modes(pca, original, n_modes)

        scatter = plot_field_2d(
            ax, coords, reconstructed,
            f'{n_modes} Mode{"s" if n_modes > 1 else ""} (R² = {r2:.4f})',
            varying_dims, vmin, vmax
        )
        plt.colorbar(scatter, ax=ax, label='Temperature (K)', fraction=0.046)

    plt.suptitle(f'POD Reconstruction Comparison - {field_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_individual_modes(pca, coords, n_modes_to_show=6, field_name=''):
    """
    Visualize individual POD mode spatial patterns.

    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    coords : ndarray
        Coordinate array (n_points, 3)
    n_modes_to_show : int
        Number of modes to visualize
    field_name : str
        Name of the field
    """
    varying_dims, axis_labels = detect_2d_plane(coords)

    n_modes_available = len(pca.explained_variance_ratio_)
    n_modes_to_show = min(n_modes_to_show, n_modes_available)

    # Determine grid size
    n_cols = 3
    n_rows = (n_modes_to_show + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, 6 * n_rows))

    for i in range(n_modes_to_show):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        mode_pattern = pca.components_[i]
        variance_pct = pca.explained_variance_ratio_[i] * 100

        # Plot mode pattern
        scatter = plot_field_2d(
            ax, coords, mode_pattern,
            f'Mode {i+1} ({variance_pct:.2f}% variance)',
            varying_dims
        )
        plt.colorbar(scatter, ax=ax, label='Mode Amplitude', fraction=0.046)

    plt.suptitle(f'Individual POD Mode Patterns - {field_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def list_available_fields(file_path):
    """
    List all 2D fields available in the NPZ file.

    Parameters:
    -----------
    file_path : str
        Path to NPZ file

    Returns:
    --------
    fields : list
        List of available field location names
    """
    data = np.load(file_path)
    keys = list(data.keys())

    # Extract unique field locations (everything before the pipe)
    locations = set()
    for key in keys:
        if '|temperature' in key:
            location = key.split('|')[0]
            # Check if it's a 2D field (has coordinates)
            coord_key = f"{location}|coordinates"
            if coord_key in keys:
                n_points = len(data[key])
                if 1000 < n_points < 100000:  # Likely 2D field
                    locations.add((location, n_points))

    return sorted(list(locations))


def load_multiple_snapshots(file_pattern, field_location, n_snapshots=2000, skip_first=0):
    """
    Load multiple snapshots from a dataset for proper POD.

    Parameters:
    -----------
    file_pattern : str
        Pattern for NPZ files (e.g., "CP/cases/*/dataset/sim_*.npz")
    field_location : str
        Field location to load (e.g., 'yz-mid')
    n_snapshots : int
        Number of snapshots to load (default: 2000 for last clean samples)
    skip_first : int
        Number of initial files to skip (default: 0, use 2500 to skip FF data)

    Returns:
    --------
    snapshots : ndarray
        Array of shape (n_snapshots, n_points) with field data
    coordinates : ndarray
        Coordinates array (same for all snapshots)
    """
    import glob

    # Get list of files
    files = sorted(glob.glob(file_pattern))

    if len(files) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    # Use last n_snapshots (skip corrupted FF data if skip_first > 0)
    if skip_first > 0:
        files = files[skip_first:skip_first + n_snapshots]
        print(f"Skipping first {skip_first} files (corrupted FF data)")
    else:
        # Take last n_snapshots
        files = files[-n_snapshots:]

    print(f"Loading {len(files)} snapshots for POD training...")

    snapshots = []
    coordinates = None

    for i, file_path in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(files)} snapshots...")

        try:
            coords, field = load_npz_field(file_path, field_location)

            if coordinates is None:
                coordinates = coords

            snapshots.append(field)
        except Exception as e:
            print(f"  Warning: Failed to load {file_path}: {e}")
            continue

    if len(snapshots) == 0:
        raise ValueError("No snapshots were successfully loaded")

    print(f"Successfully loaded {len(snapshots)} snapshots")

    return np.array(snapshots), coordinates


def main():
    """
    Main interactive interface for POD visualization.
    """
    print("="*70)
    print(" POD (Proper Orthogonal Decomposition) Visualization Tool")
    print("="*70)
    print()

    # Get dataset directory
    default_pattern = "CP/cases/operation_conditions_1/dataset/sim_*.npz"
    file_pattern = input(f"Enter NPZ file pattern (or press Enter for default):\n[{default_pattern}]: ").strip()

    if not file_pattern:
        file_pattern = default_pattern

    # Check if pattern exists
    import glob
    test_files = glob.glob(file_pattern)
    if len(test_files) == 0:
        print(f"\nError: No files found matching pattern: {file_pattern}")
        return

    print(f"\nFound {len(test_files)} files matching pattern")

    # List available fields from first file
    try:
        available_fields = list_available_fields(test_files[0])
        print("\nAvailable 2D fields:")
        for idx, (location, n_points) in enumerate(available_fields, 1):
            print(f"  {idx}. {location} ({n_points:,} points)")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not available_fields:
        print("No 2D fields found in files.")
        return

    # Select field
    print()
    selection = input(f"Select field (1-{len(available_fields)}): ").strip()

    try:
        idx = int(selection) - 1
        if idx < 0 or idx >= len(available_fields):
            raise ValueError()
        field_location = available_fields[idx][0]
    except:
        print("Invalid selection.")
        return

    # Load multiple snapshots for POD training (last 2000 = clean LHS data)
    print(f"\nLoading dataset for POD training...")
    print("Using last 2000 snapshots (clean LHS-sampled data)")

    try:
        snapshots, coords = load_multiple_snapshots(file_pattern, field_location, n_snapshots=2000)
        print(f"  Snapshots shape: {snapshots.shape}")
        print(f"  Temperature range: {np.min(snapshots):.2f} - {np.max(snapshots):.2f} K")
    except Exception as e:
        print(f"Error loading snapshots: {e}")
        return

    # Perform POD on all snapshots
    print("\nPerforming POD decomposition on full dataset...")
    pca = perform_pod(snapshots, max_modes=100)
    n_modes_computed = len(pca.explained_variance_ratio_)
    print(f"  Computed {n_modes_computed} modes")

    # Display variance summary
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
    print("\n  Variance captured:")
    for n in [1, 3, 5, 10, 20]:
        if n <= n_modes_computed:
            print(f"    First {n:2d} modes: {cumulative_variance[n-1]:6.2f}%")

    # Select which snapshot to visualize
    print()
    snapshot_input = input(f"Which snapshot to visualize? (1-{len(snapshots)}, default: 1): ").strip()

    if snapshot_input:
        try:
            viz_idx = int(snapshot_input) - 1
            if viz_idx < 0 or viz_idx >= len(snapshots):
                print(f"Invalid selection. Using snapshot #1.")
                viz_idx = 0
        except:
            print(f"Invalid input. Using snapshot #1.")
            viz_idx = 0
    else:
        viz_idx = 0

    field_values = snapshots[viz_idx]
    print(f"\nVisualizing reconstruction for snapshot #{viz_idx + 1}...")

    # Plot reconstruction comparison
    print("\nGenerating reconstruction comparison plot...")
    mode_counts = [1, 2, 3, 5, 20]
    fig1 = plot_reconstruction_comparison(field_values, coords, pca, mode_counts, field_location)

    # Plot variance explained
    print("Generating variance explained plot...")
    fig2 = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)

    n_modes_to_show = min(30, len(pca.explained_variance_ratio_))
    variance_ratios = pca.explained_variance_ratio_[:n_modes_to_show] * 100
    cumulative_variance = np.cumsum(variance_ratios)

    x = np.arange(1, n_modes_to_show + 1)
    ax.bar(x, variance_ratios, alpha=0.7, color='steelblue', label='Individual Mode')
    ax.plot(x, cumulative_variance, 'ro-', linewidth=2, markersize=5, label='Cumulative')
    ax.set_xlabel('Mode Number', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title(f'Variance Explained by POD Modes - {field_location}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_modes_to_show + 1)
    plt.tight_layout()

    # Ask about individual modes
    print()
    show_modes = input("View individual mode patterns? (y/n): ").strip().lower()

    if show_modes == 'y':
        n_modes_input = input(f"How many modes to show? (default: 6, max: {n_modes_computed}): ").strip()

        if n_modes_input:
            try:
                n_modes_to_show = int(n_modes_input)
                n_modes_to_show = min(n_modes_to_show, n_modes_computed)
            except:
                n_modes_to_show = 6
        else:
            n_modes_to_show = 6

        print(f"\nGenerating individual mode plots ({n_modes_to_show} modes)...")
        fig3 = plot_individual_modes(pca, coords, n_modes_to_show, field_location)

    print("\nDisplaying plots...")
    plt.show()

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
