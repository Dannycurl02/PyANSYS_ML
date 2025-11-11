"""
Design of Experiments Setup Module
===================================
Handles DOE configuration including parameter detection and value setup.
"""

import numpy as np


def get_bc_parameters(solver, bc_name, bc_type):
    """Get available parameters for a given boundary condition by querying Fluent.

    Args:
        solver: PyFluent solver object
        bc_name: Name of the boundary condition (e.g., 'inlet')
        bc_type: Type of boundary condition (e.g., 'velocity_inlet')

    Returns:
        List of parameter dictionaries with 'name', 'path', and 'type' keys
    """
    parameters = []

    try:
        # Normalize the bc_type to match API naming
        bc_type_key = bc_type.lower().replace(' ', '_')

        # Access the boundary condition object
        boundary_conditions = solver.settings.setup.boundary_conditions

        if not hasattr(boundary_conditions, bc_type_key):
            # Fallback to generic parameters if type not found
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_container = getattr(boundary_conditions, bc_type_key)

        if bc_name not in bc_container:
            return [{'name': 'value', 'path': None, 'type': 'unknown'}]

        bc_obj = bc_container[bc_name]

        # Recursively explore the BC object to find settable parameters
        def explore_object(obj, path="", max_depth=3):
            """Recursively explore object structure to find parameters."""
            if max_depth <= 0:
                return

            try:
                # Get child names to explore nested structures
                if hasattr(obj, 'child_names'):
                    child_names = obj.child_names

                    for child_name in child_names:
                        if child_name in ['child_names', 'command_names']:
                            continue

                        try:
                            child_obj = getattr(obj, child_name)
                            child_path = f"{path}.{child_name}" if path else child_name

                            # Check if this is a settable value (has 'value' attribute or is numeric)
                            if hasattr(child_obj, 'value'):
                                # This is a parameter we can set
                                param_info = {
                                    'name': child_path.replace('.', ' > '),
                                    'path': child_path,
                                    'type': type(child_obj).__name__,
                                    'object': child_obj
                                }

                                # Try to get min/max if available
                                if hasattr(child_obj, 'min') and hasattr(child_obj, 'max'):
                                    try:
                                        param_info['min'] = child_obj.min()
                                        param_info['max'] = child_obj.max()
                                    except:
                                        pass

                                # Try to get allowed values if available
                                if hasattr(child_obj, 'allowed_values'):
                                    try:
                                        param_info['allowed_values'] = child_obj.allowed_values()
                                    except:
                                        pass

                                parameters.append(param_info)
                            else:
                                # Recurse into this child object
                                explore_object(child_obj, child_path, max_depth - 1)
                        except Exception as e:
                            # Skip parameters that cause errors
                            pass
            except Exception as e:
                pass

        # Start exploration
        explore_object(bc_obj)

        # If no parameters found, return generic fallback
        if not parameters:
            parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    except Exception as e:
        # Fallback to generic parameter on error
        parameters = [{'name': 'value', 'path': None, 'type': 'unknown'}]

    return parameters


def setup_parameter_values(param_name, current_values, ui_helpers):
    """Configure test values for a single parameter."""
    if current_values is None:
        current_values = []

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header(f"CONFIGURE PARAMETER: {param_name.upper()}")

        # Show current values
        if current_values:
            print("\n" + "="*70)
            print("CURRENT VALUES:")
            print("="*70)
            for i, val in enumerate(current_values, 1):
                print(f"  [{i:2d}] {val}")
            print("="*70)
        else:
            print("\nNo values configured yet.")

        print(f"\n{'='*70}")
        print("  [1] Add Value Manually")
        print("  [2] Fill Range (Evenly Spaced)")
        print("  [3] Fill Range (Edge-Biased)")
        print("  [4] Clear All Values")
        print("  [5] Remove Specific Value")
        print("  [D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return current_values
        elif choice == '1':
            # Manual entry
            try:
                val = float(input(f"\nEnter value for {param_name}: ").strip())
                current_values.append(val)
                current_values.sort()
                print(f"✓ Added value: {val}")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid number")
                ui_helpers.pause()
        elif choice == '2':
            # Evenly spaced
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    ui_helpers.pause()
                    continue

                new_values = np.linspace(min_val, max_val, num_points).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} evenly spaced values")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '3':
            # Edge-biased (higher resolution near edges)
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    ui_helpers.pause()
                    continue

                # Use cosine spacing for edge bias
                t = np.linspace(0, np.pi, num_points)
                normalized = (1 - np.cos(t)) / 2  # Maps to [0, 1] with edge bias
                new_values = (min_val + normalized * (max_val - min_val)).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} edge-biased values")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()
        elif choice == '4':
            # Clear all
            current_values.clear()
            print("✓ Cleared all values")
            ui_helpers.pause()
        elif choice == '5':
            # Remove specific value
            if not current_values:
                print("✗ No values to remove")
                ui_helpers.pause()
                continue
            try:
                idx = int(input(f"\nEnter index to remove [1-{len(current_values)}]: ").strip())
                if 1 <= idx <= len(current_values):
                    removed = current_values.pop(idx - 1)
                    print(f"✓ Removed value: {removed}")
                else:
                    print("✗ Invalid index")
                ui_helpers.pause()
            except ValueError:
                print("✗ Invalid input")
                ui_helpers.pause()


def setup_doe(solver, selected_inputs, doe_parameters, ui_helpers):
    """Configure Design of Experiment parameters for selected inputs."""

    if not selected_inputs:
        ui_helpers.print_header("DESIGN OF EXPERIMENT SETUP")
        print("\n✗ No model inputs selected! Please select inputs first (Option 1).")
        ui_helpers.pause()
        return doe_parameters

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DESIGN OF EXPERIMENT SETUP")

        print("\nSELECTED INPUTS:")
        print("="*70)
        for i, item in enumerate(selected_inputs, 1):
            # Count only parameters that have values configured (non-empty lists)
            params_dict = doe_parameters.get(item['name'], {})
            num_params = sum(1 for values in params_dict.values() if values)
            status = f"({num_params} parameters configured)" if num_params > 0 else "(not configured)"
            print(f"  [{i:2d}] {item['name']:30s} {status}")
        print("="*70)

        print(f"\n{'='*70}")
        print("[Number] Configure DOE for input")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return doe_parameters
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(selected_inputs):
                item = selected_inputs[idx]

                print(f"\nDetecting available parameters for {item['name']}...")

                # Get available parameters for this BC/zone type by querying Fluent
                available_params = get_bc_parameters(solver, item['name'], item['type'])

                if not available_params or (len(available_params) == 1 and available_params[0]['name'] == 'value'):
                    print(f"\n✗ Could not detect parameters for {item['type']}")
                    print("   This BC/zone type may not be supported or has no settable parameters.")
                    ui_helpers.pause()
                    continue

                # Initialize DOE config for this item if not exists
                if item['name'] not in doe_parameters:
                    doe_parameters[item['name']] = {}

                # Configure each parameter
                while True:
                    ui_helpers.clear_screen()
                    ui_helpers.print_header(f"DOE: {item['name']} ({item['type']})")

                    # Ensure the BC entry exists (it might have been cleaned up if empty)
                    if item['name'] not in doe_parameters:
                        doe_parameters[item['name']] = {}

                    print("\nAVAILABLE PARAMETERS:")
                    print("="*70)
                    for i, param in enumerate(available_params, 1):
                        param_key = param['path'] if param['path'] else param['name']
                        num_values = len(doe_parameters[item['name']].get(param_key, []))
                        status = f"({num_values} values)" if num_values > 0 else "(not configured)"

                        # Show additional info if available
                        extra_info = ""
                        if 'min' in param and 'max' in param:
                            extra_info = f" [{param['min']}, {param['max']}]"
                        elif 'allowed_values' in param:
                            extra_info = f" (options: {len(param['allowed_values'])})"

                        print(f"  [{i:2d}] {param['name']:40s} {status}{extra_info}")
                    print("="*70)

                    print(f"\n{'='*70}")
                    print("[Number] Configure parameter")
                    print("[B] Back to input list")
                    print("="*70)

                    param_choice = input("\nEnter choice: ").strip().upper()

                    if param_choice == 'B':
                        break
                    elif param_choice.isdigit():
                        param_idx = int(param_choice) - 1
                        if 0 <= param_idx < len(available_params):
                            param = available_params[param_idx]
                            param_key = param['path'] if param['path'] else param['name']
                            current_values = doe_parameters[item['name']].get(param_key, [])
                            new_values = setup_parameter_values(param['name'], current_values.copy(), ui_helpers)

                            # Only save if there are actual values, otherwise remove the key
                            if new_values:
                                doe_parameters[item['name']][param_key] = new_values
                            elif param_key in doe_parameters[item['name']]:
                                # Remove empty parameter entry
                                del doe_parameters[item['name']][param_key]

                            # Clean up empty BC entries
                            if not doe_parameters[item['name']]:
                                del doe_parameters[item['name']]

    return doe_parameters


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


def create_dataset_structure(dataset_dir, analysis, setup_data, ui_helpers):
    """Create the directory structure for dataset storage."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("CREATE DATASET STRUCTURE")

    print(f"\nCreating dataset directory: {dataset_dir}")

    try:
        # Create main dataset directory
        dataset_dir.mkdir(exist_ok=True)

        # Create dataset directory for simulation outputs
        (dataset_dir / "dataset").mkdir(exist_ok=True)


        # Create README
        readme_path = dataset_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("Dataset Directory Structure\n")
            f.write("="*70 + "\n\n")
            f.write(f"Created: {setup_data['timestamp']}\n")
            f.write(f"Required Simulations: {analysis['total_input_combinations']}\n")
            f.write(f"Input Variables: {analysis['num_inputs']}\n")
            f.write(f"Output Locations: {analysis['num_outputs']}\n\n")


    except Exception as e:
        print(f"\n✗ Error creating dataset structure: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()
