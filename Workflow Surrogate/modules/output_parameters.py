"""
Output Parameters Module
=========================
Handles selection of output parameters (temp, pressure, velocity, etc.)
for each surface/cell zone.
"""


def get_available_field_variables():
    """
    Get list of common field variables available in Fluent.

    Returns
    -------
    dict
        Dictionary of variable categories and their variables
    """
    return {
        'Pressure': ['absolute-pressure', 'pressure-coefficient', 'dynamic-pressure', 'total-pressure'],
        'Velocity': ['velocity-magnitude', 'x-velocity', 'y-velocity', 'z-velocity', 'radial-velocity', 'axial-velocity'],
        'Temperature': ['temperature', 'total-temperature'],
        'Density': ['density'],
        'Turbulence': ['k', 'epsilon', 'omega', 'turb-kinetic-energy', 'turb-diss-rate', 'turbulent-viscosity'],
        'Wall': ['wall-shear', 'y-plus', 'wall-temperature', 'heat-transfer-coef'],
        'Species': ['mass-fraction', 'mole-fraction'],
        'Vorticity': ['vorticity-magnitude', 'x-vorticity', 'y-vorticity', 'z-vorticity']
    }


def setup_output_parameters(selected_outputs, output_params, ui_helpers):
    """
    Configure which field variables to extract from each output location.

    Parameters
    ----------
    selected_outputs : list
        List of selected output surfaces/zones
    output_params : dict
        Dictionary mapping output names to selected parameters
    ui_helpers : module
        UI helpers module

    Returns
    -------
    dict
        Updated output_params dictionary
    """
    if not selected_outputs:
        ui_helpers.print_header("CONFIGURE OUTPUT PARAMETERS")
        print("\n✗ No outputs selected! Please select outputs first.")
        ui_helpers.pause()
        return output_params

    # Get available field variables
    field_vars = get_available_field_variables()
    all_vars = []
    for category, vars_list in field_vars.items():
        for var in vars_list:
            all_vars.append({'name': var, 'category': category})

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("CONFIGURE OUTPUT PARAMETERS")

        print("\nOUTPUT LOCATIONS:")
        print("="*70)
        for i, output in enumerate(selected_outputs, 1):
            num_params = len(output_params.get(output['name'], []))
            status = f"({num_params} parameters)" if num_params > 0 else "(not configured)"
            print(f"  [{i:2d}] {output['name']:30s} {status}")
        print("="*70)

        print(f"\n{'='*70}")
        print("[Number] Configure parameters for output")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return output_params
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(selected_outputs):
                output = selected_outputs[idx]

                # Initialize if not exists
                if output['name'] not in output_params:
                    output_params[output['name']] = []

                # Configure parameters for this output
                while True:
                    ui_helpers.clear_screen()
                    ui_helpers.print_header(f"OUTPUT PARAMETERS: {output['name']}")

                    # Show selected parameters
                    selected_params = output_params.get(output['name'], [])
                    if selected_params:
                        print("\n" + "="*70)
                        print("SELECTED PARAMETERS:")
                        print("="*70)
                        for param in selected_params:
                            print(f"  [X] {param}")
                        print("="*70)

                    # Show available parameters by category
                    print(f"\nAVAILABLE FIELD VARIABLES:\n")
                    var_index = 1
                    var_map = {}

                    for category, vars_list in field_vars.items():
                        print(f"{category}:")
                        for var in vars_list:
                            marker = "[X]" if var in selected_params else "[ ]"
                            print(f"  {marker} [{var_index:2d}] {var}")
                            var_map[var_index] = var
                            var_index += 1
                        print()

                    print("="*70)
                    print("[Number] Toggle parameter")
                    print("[C] Clear all")
                    print("[B] Back to output list")
                    print("="*70)

                    param_choice = input("\nEnter choice: ").strip().upper()

                    if param_choice == 'B':
                        break
                    elif param_choice == 'C':
                        output_params[output['name']] = []
                    elif param_choice.isdigit():
                        var_idx = int(param_choice)
                        if var_idx in var_map:
                            var_name = var_map[var_idx]
                            if var_name in selected_params:
                                selected_params.remove(var_name)
                            else:
                                selected_params.append(var_name)
                            output_params[output['name']] = selected_params

    return output_params
