"""
POD Neural Network Builder Module
==================================
Handles dataset creation, labeling, and POD-based neural network configuration.
"""

import json
from pathlib import Path
import numpy as np


def format_number(num):
    """
    Format a number by rounding to 2 decimals after the last zero.

    Examples:
        1.0 -> "1.0"
        300.0 -> "300.0"
        1.234567 -> "1.23"
        0.00123 -> "0.0012"
    """
    # Convert to string with enough precision
    s = f"{num:.10f}"

    # Remove trailing zeros but keep at least one decimal
    s = s.rstrip('0')
    if s.endswith('.'):
        s += '0'

    # Find position of decimal point
    if '.' in s:
        integer_part, decimal_part = s.split('.')

        # If decimal part has more than 2 non-zero digits, round to 2 decimals
        if len(decimal_part) > 2:
            return f"{num:.2f}".rstrip('0').rstrip('.') or f"{num:.1f}"

    return s


def validate_model_setup_file(setup_data):
    """
    Validate that the loaded JSON is a proper model setup file.

    Parameters
    ----------
    setup_data : dict
        Loaded JSON data

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    required_keys = ['model_inputs', 'model_outputs', 'doe_configuration', 'timestamp']

    # Check required keys
    for key in required_keys:
        if key not in setup_data:
            return False, f"Missing required key: '{key}'. This does not appear to be a model setup file."

    # Check if model_inputs is a list
    if not isinstance(setup_data['model_inputs'], list):
        return False, "'model_inputs' must be a list."

    # Check if model_outputs is a list
    if not isinstance(setup_data['model_outputs'], list):
        return False, "'model_outputs' must be a list."

    # Check if there are any inputs
    if len(setup_data['model_inputs']) == 0:
        return False, "No model inputs found. Please configure inputs in the Fluent Setup menu."

    # Check if there are any outputs
    if len(setup_data['model_outputs']) == 0:
        return False, "No model outputs found. Please configure outputs in the Fluent Setup menu."

    # Check if DOE configuration exists and has parameters
    doe_config = setup_data.get('doe_configuration', {})
    has_doe_params = any(
        any(values for values in params.values())
        for params in doe_config.values()
    )

    if not has_doe_params:
        return False, "No DOE parameters configured. Please set up design of experiments in the Fluent Setup menu."

    return True, "Valid model setup file."


def analyze_setup_dimensions(setup_data):
    """
    Analyze model setup to determine input and output dimensionality.

    Parameters
    ----------
    setup_data : dict
        Model setup dictionary from JSON

    Returns
    -------
    dict
        Dictionary containing dimensional analysis:
        - num_inputs: Number of input variables
        - num_outputs: Number of output variables
        - input_details: List of input configurations
        - output_details: List of output configurations
        - total_input_combinations: Total number of input combinations in DOE
    """
    analysis = {
        'num_inputs': 0,
        'num_outputs': 0,
        'input_details': [],
        'output_details': [],
        'total_input_combinations': 1
    }

    # Analyze inputs
    for input_item in setup_data.get('model_inputs', []):
        doe_params = input_item.get('doe_parameters', {})

        # Count parameters with configured values
        for param_name, param_values in doe_params.items():
            if param_values:  # Non-empty list
                analysis['num_inputs'] += 1
                analysis['input_details'].append({
                    'bc_name': input_item['name'],
                    'bc_type': input_item['type'],
                    'parameter': param_name,
                    'num_values': len(param_values),
                    'range': [min(param_values), max(param_values)] if param_values else None
                })
                # Multiply for total combinations
                analysis['total_input_combinations'] *= len(param_values)

    # Analyze outputs
    for output_item in setup_data.get('model_outputs', []):
        analysis['num_outputs'] += 1
        analysis['output_details'].append({
            'name': output_item['name'],
            'type': output_item['type'],
            'category': output_item.get('category', 'Unknown')
        })

    return analysis


def recommend_pod_architecture(analysis):
    """
    Recommend POD-NN architecture based on dimensional analysis.

    Parameters
    ----------
    analysis : dict
        Dimensional analysis from analyze_setup_dimensions()

    Returns
    -------
    dict
        Recommended architecture configuration
    """
    num_inputs = analysis['num_inputs']
    num_outputs = analysis['num_outputs']
    total_combinations = analysis['total_input_combinations']

    # Recommend POD modes based on output complexity
    if num_outputs <= 3:
        recommended_modes = min(10, total_combinations // 2)
    elif num_outputs <= 10:
        recommended_modes = min(20, total_combinations // 2)
    else:
        recommended_modes = min(50, total_combinations // 2)

    # Recommend hidden layer sizes
    # Rule of thumb: neurons = 2/3 * (input + output) or 2 * input
    hidden_layer_1 = max(16, int(2 * num_inputs))
    hidden_layer_2 = max(8, hidden_layer_1 // 2)

    architecture = {
        'pod_modes': recommended_modes,
        'encoder_architecture': [num_inputs, hidden_layer_1, hidden_layer_2, recommended_modes],
        'decoder_architecture': [recommended_modes, hidden_layer_2, hidden_layer_1, num_outputs],
        'activation': 'relu',
        'output_activation': 'linear',
        'recommended_epochs': 1000,
        'batch_size': min(32, total_combinations // 4),
        'learning_rate': 0.001
    }

    return architecture


def create_dataset_structure(dataset_dir, analysis, setup_data, ui_helpers):
    """Create the directory structure for dataset storage."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("CREATE DATASET STRUCTURE")

    print(f"\nCreating dataset directory: {dataset_dir}")

    try:
        # Create main dataset directory
        dataset_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (dataset_dir / "inputs").mkdir(exist_ok=True)
        (dataset_dir / "outputs").mkdir(exist_ok=True)
        (dataset_dir / "raw_fluent_data").mkdir(exist_ok=True)
        (dataset_dir / "pod_modes").mkdir(exist_ok=True)
        (dataset_dir / "trained_models").mkdir(exist_ok=True)

        # Create README
        readme_path = dataset_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("Dataset Directory Structure\n")
            f.write("="*70 + "\n\n")
            f.write(f"Created: {setup_data['timestamp']}\n")
            f.write(f"Required Simulations: {analysis['total_input_combinations']}\n")
            f.write(f"Input Variables: {analysis['num_inputs']}\n")
            f.write(f"Output Locations: {analysis['num_outputs']}\n\n")
            f.write("Directories:\n")
            f.write("  - inputs/          : DOE input parameters (CSV)\n")
            f.write("  - outputs/         : Simulation output data (NPZ)\n")
            f.write("  - raw_fluent_data/ : Raw Fluent export files\n")
            f.write("  - pod_modes/       : POD basis functions\n")
            f.write("  - trained_models/  : Trained neural networks\n")

        print("\n✓ Dataset structure created successfully!")
        print(f"\n  {dataset_dir}/")
        print(f"  ├── inputs/")
        print(f"  ├── outputs/")
        print(f"  ├── raw_fluent_data/")
        print(f"  ├── pod_modes/")
        print(f"  ├── trained_models/")
        print(f"  └── README.txt")

    except Exception as e:
        print(f"\n✗ Error creating dataset structure: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()
