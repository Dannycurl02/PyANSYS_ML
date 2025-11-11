"""
Quick script to check simulation results and see if outlet temps vary
"""
import numpy as np
from pathlib import Path

dataset_dir = Path("proj1/cases/4x4/dataset")
npz_files = sorted(dataset_dir.glob("sim_*.npz"))

print("="*70)
print("SIMULATION RESULTS COMPARISON")
print("="*70)

if not npz_files:
    print("\nNo simulation files found!")
    print(f"Looking in: {dataset_dir.absolute()}")
else:
    print(f"\nFound {len(npz_files)} simulation files")
    print()

    for npz_file in npz_files:
        print(f"\n{npz_file.name}:")
        print("-" * 50)

        try:
            data = np.load(npz_file, allow_pickle=True)

            # List all keys in the file
            print(f"  Keys: {list(data.keys())}")

            # Look for report definition (scalar output like avg-outlet-temp)
            for key in data.keys():
                if 'avg-outlet-temp' in key or 'outlet' in key.lower():
                    value = data[key]
                    if value.size == 1:
                        print(f"  {key}: {value[0]:.6f}")
                    else:
                        print(f"  {key}: shape={value.shape}, mean={np.mean(value):.6f}, std={np.std(value):.6f}")

            # Also show any scalar (single-value) outputs
            print("\n  Scalar outputs:")
            for key in data.keys():
                value = data[key]
                if value.size == 1:
                    print(f"    {key}: {value[0]:.6f}")

        except Exception as e:
            print(f"  Error loading file: {e}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if len(npz_files) >= 2:
    print("\nComparing first two simulations:")
    try:
        data1 = np.load(npz_files[0], allow_pickle=True)
        data2 = np.load(npz_files[1], allow_pickle=True)

        # Compare all keys
        for key in data1.keys():
            if key in data2:
                val1 = data1[key]
                val2 = data2[key]

                if val1.size == 1 and val2.size == 1:
                    diff = abs(val1[0] - val2[0])
                    if diff < 1e-6:
                        print(f"  ⚠ {key}: IDENTICAL ({val1[0]:.6f})")
                    else:
                        print(f"  ✓ {key}: DIFFERENT (Δ = {diff:.6f})")
                elif val1.shape == val2.shape:
                    max_diff = np.max(np.abs(val1 - val2))
                    mean_diff = np.mean(np.abs(val1 - val2))
                    if max_diff < 1e-6:
                        print(f"  ⚠ {key}: IDENTICAL arrays")
                    else:
                        print(f"  ✓ {key}: DIFFERENT arrays (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")

    except Exception as e:
        print(f"  Error comparing: {e}")
