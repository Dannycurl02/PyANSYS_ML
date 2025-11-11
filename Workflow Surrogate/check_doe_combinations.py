"""
Check what DOE combinations should be run
"""
import json
from pathlib import Path
import itertools

setup_file = Path("proj1/cases/4x4/model_setup.json")

with open(setup_file, 'r') as f:
    setup = json.load(f)

print("="*70)
print("DOE CONFIGURATION")
print("="*70)

doe_config = setup['doe_configuration']

print("\nBoundary Conditions:")
for bc_name, params in doe_config.items():
    print(f"\n  {bc_name}:")
    for param_name, values in params.items():
        print(f"    {param_name}: {values}")

# Generate combinations
param_arrays = []
param_names = []

for bc_name, params in doe_config.items():
    for param_name, values in params.items():
        param_arrays.append(values)
        param_names.append(f"{bc_name}.{param_name}")

combinations = list(itertools.product(*param_arrays))

print("\n" + "="*70)
print(f"TOTAL COMBINATIONS: {len(combinations)}")
print("="*70)

print("\nFirst 5 combinations:")
for i, combo in enumerate(combinations[:5], 1):
    print(f"\nSimulation {i}:")
    for param_name, value in zip(param_names, combo):
        print(f"  {param_name} = {value}")

print("\n" + "="*70)
print("WHAT SHOULD HAPPEN")
print("="*70)
print("\nFor the mixing elbow problem:")
print("  - Different inlet velocities should cause different mixing")
print("  - Higher velocities = more kinetic energy = different temp distribution")
print("  - Outlet temperature should vary based on hot/cold inlet velocities")
print("\nExpected outlet temp range: ~295K to ~300K (depending on velocities)")
