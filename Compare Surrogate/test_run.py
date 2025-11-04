"""
Test script to run comparison non-interactively.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Workflow Surrogate"))

from modules.project_system import WorkflowProject
from compare_architectures import (
    load_dataset_data, calculate_hyperparameters,
    train_bottleneck_nn, train_direct_nn,
    evaluate_model, create_comparison_plots,
    save_comparison_report
)
import numpy as np
from sklearn.model_selection import train_test_split

# Load project
project_path = Path(__file__).parent.parent / "Workflow Surrogate" / "newproject1"
project = WorkflowProject(project_path)
if not project.load():
    print(f"ERROR: Could not load project at {project_path}")
    sys.exit(1)

print(f"Project loaded: {project.info['project_name']}")
print(f"Available datasets: {[d['name'] for d in project.datasets]}")

# Select velo-temp_10x10 dataset
dataset = None
for d in project.datasets:
    if d['name'] == 'velo-temp_10x10':
        dataset = d
        break

if not dataset:
    print("ERROR: velo-temp_10x10 dataset not found!")
    sys.exit(1)

print(f"\nSelected dataset: {dataset['name']}")
print(f"  Inputs: {dataset['num_inputs']}")
print(f"  Outputs: {dataset['num_outputs']}")
print(f"  Simulations: {dataset['num_simulations']}")

# Load data
print("\nLoading dataset...")
X_data, Y_data, setup_data, output_params = load_dataset_data(dataset)

# Split data
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_data, Y_data, test_size=0.3, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

print(f"\nData split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val:   {X_val.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# Calculate hyperparameters
hyperparams = calculate_hyperparameters(len(X_train), Y_train.shape[1])

print(f"\nHyperparameters:")
for key, value in hyperparams.items():
    print(f"  {key:25s}: {value}")

# Train Bottleneck NN
print("\n" + "="*70)
print("TRAINING BOTTLENECK NN")
print("="*70)
bottleneck_model, bottleneck_time, bottleneck_history = train_bottleneck_nn(
    X_train, Y_train, X_val, Y_val, hyperparams
)
print(f"\n[OK] Bottleneck NN training completed in {bottleneck_time:.2f} seconds")

# Train Direct NN
print("\n" + "="*70)
print("TRAINING DIRECT NN")
print("="*70)
direct_model, direct_time, direct_history = train_direct_nn(
    X_train, Y_train, X_val, Y_val, hyperparams
)
print(f"\n[OK] Direct NN training completed in {direct_time:.2f} seconds")

# Evaluate both models
print("\n" + "="*70)
print("EVALUATION")
print("="*70)
bottleneck_eval = evaluate_model(bottleneck_model, X_test, Y_test, "Bottleneck NN")
direct_eval = evaluate_model(direct_model, X_test, Y_test, "Direct NN")

# Print comparison
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"\n{'Metric':<20s} {'Bottleneck NN':>15s} {'Direct NN':>15s} {'Winner':>15s}")
print("-"*70)

metrics = ['mse', 'rmse', 'mae', 'r2']
for metric in metrics:
    bn_val = bottleneck_eval[metric]
    dn_val = direct_eval[metric]

    if metric == 'r2':
        winner = "Bottleneck" if bn_val > dn_val else "Direct"
    else:
        winner = "Bottleneck" if bn_val < dn_val else "Direct"

    print(f"{metric:<20s} {bn_val:>15.6f} {dn_val:>15.6f} {winner:>15s}")

print(f"\n{'Training Time (s)':<20s} {bottleneck_time:>15.2f} {direct_time:>15.2f}")

# Save results
results_dir = Path("Compare Surrogate/comparison_results") / dataset['name']
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\n\nSaving results to: {results_dir}")

# Create plots
create_comparison_plots(
    bottleneck_history, direct_history,
    bottleneck_eval, direct_eval,
    bottleneck_time, direct_time,
    results_dir
)

# Save report
save_comparison_report(
    dataset, bottleneck_eval, direct_eval,
    bottleneck_time, direct_time,
    bottleneck_history, direct_history,
    hyperparams, results_dir
)

print(f"\n[OK] All results saved to: {results_dir.absolute()}")
print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
