#!/usr/bin/env python
"""
Workflow Surrogate - Interactive Fluent Integration
====================================================
Customizable workflow for connecting to Fluent, selecting surfaces/volumes,
and setting up POD-based surrogate models.
"""

import sys
import json
from pathlib import Path
from tkinter import Tk, filedialog

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).parent
CONFIG_FILE = PROJECT_DIR / "user_settings.json"


class UserSettings:
    """Manager for user settings and preferences."""

    def __init__(self, config_file):
        self.config_file = config_file
        self.data = self.load()

    def load(self):
        """Load settings from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return self._default_settings()
        return self._default_settings()

    def _default_settings(self):
        """Return default settings structure."""
        return {
            'recent_projects': [],
            'solver_settings': {
                'precision': 'single',
                'processor_count': 2,
                'dimension': 3,
                'use_gui': True
            }
        }

    def save(self):
        """Save settings to config file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_recent_project(self, project_path):
        """Add a project to recent list (most recent first)."""
        project_path = str(project_path)

        # Remove if already exists
        if project_path in self.data['recent_projects']:
            self.data['recent_projects'].remove(project_path)

        # Add to front
        self.data['recent_projects'].insert(0, project_path)

        # Keep only 3 most recent
        self.data['recent_projects'] = self.data['recent_projects'][:3]

        self.save()

    def get_recent_projects(self):
        """Get list of 3 most recent projects."""
        return [p for p in self.data['recent_projects'] if Path(p).exists()]

    def get_solver_settings(self):
        """Get saved solver settings."""
        return self.data['solver_settings'].copy()

    def save_solver_settings(self, settings):
        """Save solver settings."""
        self.data['solver_settings'] = settings
        self.save()


user_settings = UserSettings(CONFIG_FILE)


# ============================================================
# TUI UTILITIES
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
    """Print a menu with options."""
    print_header(title)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print(f"  [0] {'Back' if title != 'WORKFLOW SURROGATE - MAIN MENU' else 'Exit'}")
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
# FLUENT CONNECTION
# ============================================================

def configure_solver_settings():
    """
    Configure Fluent solver settings before launch.

    Returns
    -------
    dict or None
        Solver settings dict, or None if cancelled
    """
    # Load saved settings or use defaults
    settings = user_settings.get_solver_settings()

    while True:
        print("\n" + "="*70)
        print("FLUENT SOLVER SETTINGS")
        print("="*70)
        print(f"\n  [1] Precision: {settings['precision']}")
        print(f"  [2] Processor Count: {settings['processor_count']}")
        print(f"  [3] Dimension: {settings['dimension']}D")
        print(f"  [4] GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")
        print("\n  [P] Proceed with these settings")
        print("  [C] Cancel")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'P':
            # Save settings before returning
            user_settings.save_solver_settings(settings)
            return settings
        elif choice == 'C':
            return None
        elif choice == '1':
            print("\n[1] Single precision")
            print("[2] Double precision")
            prec_choice = input("Select precision [1-2]: ").strip()
            if prec_choice == '1':
                settings['precision'] = 'single'
            elif prec_choice == '2':
                settings['precision'] = 'double'
        elif choice == '2':
            try:
                count = int(input("\nEnter processor count: ").strip())
                if count > 0:
                    settings['processor_count'] = count
                else:
                    print("Must be positive")
                    pause()
            except:
                print("Invalid input")
                pause()
        elif choice == '3':
            print("\n[1] 2D")
            print("[2] 3D")
            dim_choice = input("Select dimension [1-2]: ").strip()
            if dim_choice == '1':
                settings['dimension'] = 2
            elif dim_choice == '2':
                settings['dimension'] = 3
        elif choice == '4':
            settings['use_gui'] = not settings['use_gui']


def open_case_file():
    """Open a Fluent case file with GUI."""
    print_header("OPEN FLUENT CASE FILE")

    # Use file dialog to select case file
    print("\nOpening file dialog...")
    Tk().withdraw()  # Hide tkinter root window
    case_file = filedialog.askopenfilename(
        title="Select Fluent Case File",
        filetypes=[
            ("Fluent Case Files", "*.cas *.cas.h5 *.cas.gz"),
            ("All Files", "*.*")
        ],
        initialdir=str(PROJECT_DIR)
    )

    if not case_file:
        print("\n✗ No file selected")
        pause()
        return None

    case_file = Path(case_file)
    print(f"\n✓ Selected: {case_file.name}")
    print(f"  Full path: {case_file}")

    # Add to recent projects
    user_settings.add_recent_project(case_file)

    # Configure solver settings
    settings = configure_solver_settings()
    if settings is None:
        print("\n✗ Launch cancelled by user")
        pause()
        return None

    # Launch Fluent
    print("\nLaunching Fluent...")
    print(f"  Precision: {settings['precision']}")
    print(f"  Processors: {settings['processor_count']}")
    print(f"  Dimension: {settings['dimension']}D")
    print(f"  Mode: solver")
    print(f"  GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")

    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Create log file for Fluent output
        log_dir = PROJECT_DIR / "fluent_logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"fluent_launch_{case_file.stem}.log"
        fluent_log_file = open(log_file_path, 'w', buffering=1)

        # Redirect stdout/stderr to suppress Fluent TUI output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

        # Determine UI mode
        ui_mode = UIMode.GUI if settings['use_gui'] else UIMode.NO_GUI_OR_GRAPHICS

        solver = pyfluent.launch_fluent(
            precision=settings['precision'],
            processor_count=settings['processor_count'],
            dimension=settings['dimension'],
            mode="solver",
            ui_mode=ui_mode
        )

        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Fluent launched (version {solver.get_fluent_version()})")
        print(f"  Loading case file: {case_file.name}")
        print(f"  Fluent output redirected to: {log_file_path.name}")

        # Redirect during case loading
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file
        solver.settings.file.read_case(file_name=str(case_file))
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Case file loaded successfully")

        fluent_log_file.close()

        return solver

    except Exception as e:
        # Restore stdout/stderr if error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\n✗ Error launching Fluent: {e}")
        import traceback
        traceback.print_exc()
        try:
            fluent_log_file.close()
        except:
            pass
        pause()
        return None


def connect_to_fluent():
    """Connect to an existing Fluent process."""
    print_header("CONNECT TO EXISTING FLUENT SESSION")

    try:
        import ansys.fluent.core as pyfluent

        print("\nSearching for running Fluent sessions...")

        # This will attempt to connect to a running Fluent instance
        # User needs to have Fluent already running with server mode enabled
        print("\nNote: Fluent must be running with server mode enabled.")
        print("      Start Fluent with: fluent 3d -server")

        host = input("\nEnter Fluent server host [localhost]: ").strip() or "localhost"
        port_str = input("Enter Fluent server port [empty for auto-detect]: ").strip()

        if port_str:
            port = int(port_str)
            print(f"\nConnecting to Fluent at {host}:{port}...")
            solver = pyfluent.connect_to_fluent(host=host, port=port)
        else:
            print(f"\nConnecting to Fluent at {host} (auto-detect port)...")
            solver = pyfluent.connect_to_fluent(host=host)

        print(f"\n✓ Connected to Fluent (version {solver.get_fluent_version()})")

        return solver

    except Exception as e:
        print(f"\n✗ Error connecting to Fluent: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Fluent is running with: fluent 3d -server")
        print("  2. Check that the port is not blocked by firewall")
        print("  3. Verify the host and port are correct")
        pause()
        return None


def open_recent_project(project_path):
    """Open a recent project."""
    print_header("OPEN RECENT PROJECT")

    project_path = Path(project_path)
    print(f"\n✓ Selected: {project_path.name}")
    print(f"  Full path: {project_path}")

    if not project_path.exists():
        print(f"\n✗ File not found: {project_path}")
        pause()
        return None

    # Configure solver settings
    settings = configure_solver_settings()
    if settings is None:
        print("\n✗ Launch cancelled by user")
        pause()
        return None

    # Launch Fluent
    print("\nLaunching Fluent...")
    print(f"  Precision: {settings['precision']}")
    print(f"  Processors: {settings['processor_count']}")
    print(f"  Dimension: {settings['dimension']}D")
    print(f"  Mode: solver")
    print(f"  GUI: {'Enabled' if settings['use_gui'] else 'Disabled'}")

    try:
        import ansys.fluent.core as pyfluent
        from ansys.fluent.core.launcher.launcher import UIMode

        # Create log file for Fluent output
        log_dir = PROJECT_DIR / "fluent_logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"fluent_launch_{project_path.stem}.log"
        fluent_log_file = open(log_file_path, 'w', buffering=1)

        # Redirect stdout/stderr to suppress Fluent TUI output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file

        # Determine UI mode
        ui_mode = UIMode.GUI if settings['use_gui'] else UIMode.NO_GUI_OR_GRAPHICS

        solver = pyfluent.launch_fluent(
            precision=settings['precision'],
            processor_count=settings['processor_count'],
            dimension=settings['dimension'],
            mode="solver",
            ui_mode=ui_mode
        )

        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Fluent launched (version {solver.get_fluent_version()})")
        print(f"  Loading case file: {project_path.name}")
        print(f"  Fluent output redirected to: {log_file_path.name}")

        # Redirect during case loading
        sys.stdout = fluent_log_file
        sys.stderr = fluent_log_file
        solver.settings.file.read_case(file_name=str(project_path))
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"\n✓ Case file loaded successfully")

        fluent_log_file.close()

        # Update recent projects (move to top)
        user_settings.add_recent_project(project_path)

        return solver

    except Exception as e:
        # Restore stdout/stderr if error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\n✗ Error opening project: {e}")
        import traceback
        traceback.print_exc()
        try:
            fluent_log_file.close()
        except:
            pass
        pause()
        return None


# ============================================================
# FLUENT INSPECTION
# ============================================================

def list_surfaces(solver):
    """List all available surfaces in the Fluent case."""
    print("\n" + "="*70)
    print("Available Surfaces:")
    print("="*70)

    try:
        # Get all boundary zones (surfaces)
        boundary_conditions = solver.settings.setup.boundary_conditions

        surfaces = []

        # Iterate through all boundary types
        for bc_type in dir(boundary_conditions):
            if bc_type.startswith('_'):
                continue

            # Skip child_names and command_names types entirely
            if bc_type in ['child_names', 'command_names']:
                continue

            bc_obj = getattr(boundary_conditions, bc_type)

            # Check if it's a boundary condition container
            if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                try:
                    # Filter out child_names and command_names attributes
                    for name in bc_obj:
                        if name not in ['child_names', 'command_names']:
                            surfaces.append({
                                'name': name,
                                'type': bc_type.replace('_', ' ').title()
                            })
                except:
                    pass

        # Try to get ALL surfaces (including created surfaces like planes, iso-surfaces, etc.)
        try:
            if hasattr(solver, 'fields') and hasattr(solver.fields, 'field_data'):
                # Get all accessible surface names using allowed_values()
                all_surface_names = solver.fields.field_data.surfaces.allowed_values()

                for surf_name in all_surface_names:
                    # Skip if already in list (avoid duplicates)
                    if not any(s['name'] == surf_name for s in surfaces):
                        surfaces.append({
                            'name': surf_name,
                            'type': 'Created Surface'
                        })
        except Exception as e:
            # If field_data not available or error, just skip
            pass

        if surfaces:
            print(f"\nFound {len(surfaces)} surface(s):\n")
            for i, surf in enumerate(surfaces, 1):
                print(f"  [{i:2d}] {surf['name']:30s} (Type: {surf['type']})")
        else:
            print("\n✗ No surfaces found in case file")

        return surfaces

    except Exception as e:
        print(f"\n✗ Error listing surfaces: {e}")
        import traceback
        traceback.print_exc()
        return []


def list_cell_zones(solver):
    """List all available cell zones (volumes) in the Fluent case."""
    print("\n" + "="*70)
    print("Available Cell Zones (Volumes):")
    print("="*70)

    try:
        # Get all cell zones
        cell_zones = solver.settings.setup.cell_zone_conditions

        zones = []

        # Iterate through zone types
        for zone_type in dir(cell_zones):
            if zone_type.startswith('_'):
                continue

            # Skip child_names and command_names types entirely
            if zone_type in ['child_names', 'command_names']:
                continue

            zone_obj = getattr(cell_zones, zone_type)

            # Check if it's a cell zone container
            if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                try:
                    # Filter out child_names and command_names attributes
                    for name in zone_obj:
                        if name not in ['child_names', 'command_names']:
                            zones.append({
                                'name': name,
                                'type': zone_type.replace('_', ' ').title()
                            })
                except:
                    pass

        if zones:
            print(f"\nFound {len(zones)} cell zone(s):\n")
            for i, zone in enumerate(zones, 1):
                print(f"  [{i:2d}] {zone['name']:30s} (Type: {zone['type']})")
        else:
            print("\n✗ No cell zones found in case file")

        return zones

    except Exception as e:
        print(f"\n✗ Error listing cell zones: {e}")
        import traceback
        traceback.print_exc()
        return []


def list_report_definitions(solver):
    """List all available report definitions in the Fluent case."""
    print("\n" + "="*70)
    print("Available Report Definitions:")
    print("="*70)

    try:
        # Access report definitions
        report_defs = solver.settings.solution.report_definitions

        all_reports = []

        # Report definition types to check
        report_types = ['surface', 'volume', 'flux', 'force', 'lift', 'drag',
                       'moment', 'expression', 'user_defined']

        for report_type in report_types:
            if hasattr(report_defs, report_type):
                report_obj = getattr(report_defs, report_type)

                # Check if it's iterable
                if hasattr(report_obj, '__iter__') and not isinstance(report_obj, str):
                    try:
                        for name in report_obj:
                            if name not in ['child_names', 'command_names']:
                                all_reports.append({
                                    'name': name,
                                    'type': report_type.replace('_', ' ').title()
                                })
                    except:
                        pass

        if all_reports:
            print(f"\nFound {len(all_reports)} report definition(s):\n")
            for i, rep in enumerate(all_reports, 1):
                print(f"  [{i:2d}] {rep['name']:30s} (Type: {rep['type']})")
        else:
            print("\n  No report definitions found in case file")

        return all_reports

    except Exception as e:
        print(f"\n✗ Error listing report definitions: {e}")
        import traceback
        traceback.print_exc()
        return []


def inspect_fluent_case(solver):
    """Inspect Fluent case and display available surfaces and volumes."""
    print_header("FLUENT CASE INSPECTION")

    # List surfaces
    surfaces = list_surfaces(solver)

    # List cell zones
    cell_zones = list_cell_zones(solver)

    # List report definitions
    report_defs = list_report_definitions(solver)

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"  Total Surfaces: {len(surfaces)}")
    print(f"  Total Cell Zones: {len(cell_zones)}")
    print(f"  Total Report Definitions: {len(report_defs)}")
    print("\nThese surfaces/zones can be used for POD extraction.")

    pause()


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

def setup_model_inputs(solver, selected_inputs):
    """Configure model inputs (boundary conditions and cell zones)."""

    while True:
        clear_screen()
        print_header("CONFIGURE MODEL INPUTS")

        # Show selected items at top
        if selected_inputs:
            print("\n" + "="*70)
            print("SELECTED INPUTS:")
            print("="*70)
            for i, item in enumerate(selected_inputs, 1):
                print(f"  [{i:2d}] {item['name']:30s} (Type: {item['type']})")
            print("="*70)

        print("\nLoading boundary conditions and cell zones...")

        # Get boundary conditions
        try:
            boundary_conditions = solver.settings.setup.boundary_conditions
            surfaces = []

            for bc_type in dir(boundary_conditions):
                if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                    continue

                bc_obj = getattr(boundary_conditions, bc_type)
                if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                    try:
                        for name in bc_obj:
                            if name not in ['child_names', 'command_names']:
                                surfaces.append({
                                    'name': name,
                                    'type': bc_type.replace('_', ' ').title(),
                                    'category': 'Boundary Condition'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading boundary conditions: {e}")
            surfaces = []

        # Get cell zones
        try:
            cell_zones_obj = solver.settings.setup.cell_zone_conditions
            cell_zones = []

            for zone_type in dir(cell_zones_obj):
                if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                    continue

                zone_obj = getattr(cell_zones_obj, zone_type)
                if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                    try:
                        for name in zone_obj:
                            if name not in ['child_names', 'command_names']:
                                cell_zones.append({
                                    'name': name,
                                    'type': zone_type.replace('_', ' ').title(),
                                    'category': 'Cell Zone'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading cell zones: {e}")
            cell_zones = []

        # Combine all available items
        all_items = surfaces + cell_zones

        # Display available items
        print(f"\nAVAILABLE INPUTS ({len(all_items)} total):\n")
        for i, item in enumerate(all_items, 1):
            marker = "[X]" if item in selected_inputs else "[ ]"
            print(f"  {marker} [{i:2d}] {item['name']:30s} ({item['category']} - {item['type']})")

        print(f"\n{'='*70}")
        print("[Number] Toggle selection")
        print("[R] Refresh list")
        print("[C] Clear all selections")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return selected_inputs
        elif choice == 'R':
            continue  # Refresh - loop will re-fetch data
        elif choice == 'C':
            selected_inputs.clear()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_items):
                item = all_items[idx]
                if item in selected_inputs:
                    selected_inputs.remove(item)
                else:
                    selected_inputs.append(item)


def setup_model_outputs(solver, selected_outputs):
    """Configure model outputs (surfaces, cell zones, report definitions)."""

    while True:
        clear_screen()
        print_header("CONFIGURE MODEL OUTPUTS")

        # Show selected items at top
        if selected_outputs:
            print("\n" + "="*70)
            print("SELECTED OUTPUTS:")
            print("="*70)
            for i, item in enumerate(selected_outputs, 1):
                print(f"  [{i:2d}] {item['name']:30s} (Type: {item['type']})")
            print("="*70)

        print("\nLoading surfaces, cell zones, and report definitions...")

        # Get surfaces
        try:
            boundary_conditions = solver.settings.setup.boundary_conditions
            surfaces = []

            for bc_type in dir(boundary_conditions):
                if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                    continue

                bc_obj = getattr(boundary_conditions, bc_type)
                if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                    try:
                        for name in bc_obj:
                            if name not in ['child_names', 'command_names']:
                                surfaces.append({
                                    'name': name,
                                    'type': bc_type.replace('_', ' ').title(),
                                    'category': 'Surface'
                                })
                    except Exception as e:
                        pass

            # Try to get ALL surfaces (including created surfaces like planes, iso-surfaces, etc.)
            try:
                if hasattr(solver, 'fields') and hasattr(solver.fields, 'field_data'):
                    # Get all accessible surface names using allowed_values()
                    all_surface_names = solver.fields.field_data.surfaces.allowed_values()

                    for surf_name in all_surface_names:
                        # Skip if already in list (avoid duplicates)
                        if not any(s['name'] == surf_name for s in surfaces):
                            surfaces.append({
                                'name': surf_name,
                                'type': 'Created Surface',
                                'category': 'Surface'
                            })
            except:
                pass

        except Exception as e:
            print(f"Warning: Error loading surfaces: {e}")
            surfaces = []

        # Get cell zones
        try:
            cell_zones_obj = solver.settings.setup.cell_zone_conditions
            cell_zones = []

            for zone_type in dir(cell_zones_obj):
                if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                    continue

                zone_obj = getattr(cell_zones_obj, zone_type)
                if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                    try:
                        for name in zone_obj:
                            if name not in ['child_names', 'command_names']:
                                cell_zones.append({
                                    'name': name,
                                    'type': zone_type.replace('_', ' ').title(),
                                    'category': 'Cell Zone'
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"Warning: Error loading cell zones: {e}")
            cell_zones = []

        # Get report definitions
        try:
            report_defs_obj = solver.settings.solution.report_definitions
            report_defs = []

            report_types = ['surface', 'volume', 'flux', 'force', 'lift', 'drag',
                           'moment', 'expression', 'user_defined']

            for report_type in report_types:
                if hasattr(report_defs_obj, report_type):
                    report_obj = getattr(report_defs_obj, report_type)
                    if hasattr(report_obj, '__iter__') and not isinstance(report_obj, str):
                        try:
                            for name in report_obj:
                                if name not in ['child_names', 'command_names']:
                                    report_defs.append({
                                        'name': name,
                                        'type': report_type.replace('_', ' ').title(),
                                        'category': 'Report Definition'
                                    })
                        except Exception as e:
                            pass
        except Exception as e:
            print(f"Warning: Error loading report definitions: {e}")
            report_defs = []

        # Combine all available items
        all_items = surfaces + cell_zones + report_defs

        # Display available items
        print(f"\nAVAILABLE OUTPUTS ({len(all_items)} total):\n")
        for i, item in enumerate(all_items, 1):
            marker = "[X]" if item in selected_outputs else "[ ]"
            print(f"  {marker} [{i:2d}] {item['name']:30s} ({item['category']} - {item['type']})")

        print(f"\n{'='*70}")
        print("[Number] Toggle selection")
        print("[R] Refresh list")
        print("[C] Clear all selections")
        print("[D] Done")
        print("="*70)

        choice = input("\nEnter choice: ").strip().upper()

        if choice == 'D':
            return selected_outputs
        elif choice == 'R':
            continue  # Refresh - loop will re-fetch data
        elif choice == 'C':
            selected_outputs.clear()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_items):
                item = all_items[idx]
                if item in selected_outputs:
                    selected_outputs.remove(item)
                else:
                    selected_outputs.append(item)


def get_bc_parameters(bc_type):
    """Get available parameters for a given boundary condition type."""
    # Map BC types to their configurable parameters
    bc_params = {
        'velocity_inlet': ['velocity', 'temperature'],
        'pressure_inlet': ['pressure', 'temperature'],
        'pressure_outlet': ['pressure', 'temperature'],
        'mass_flow_inlet': ['mass_flow', 'temperature'],
        'wall': ['temperature', 'heat_flux'],
        'fluid': ['temperature', 'density', 'viscosity'],
        'solid': ['temperature', 'thermal_conductivity']
    }

    # Normalize the bc_type
    bc_key = bc_type.lower().replace(' ', '_')

    # Return parameters if found, otherwise return generic options
    return bc_params.get(bc_key, ['value'])


def setup_parameter_values(param_name, current_values=None):
    """Configure test values for a single parameter."""
    if current_values is None:
        current_values = []

    while True:
        clear_screen()
        print_header(f"CONFIGURE PARAMETER: {param_name.upper()}")

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
                pause()
            except ValueError:
                print("✗ Invalid number")
                pause()
        elif choice == '2':
            # Evenly spaced
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    pause()
                    continue

                import numpy as np
                new_values = np.linspace(min_val, max_val, num_points).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} evenly spaced values")
                pause()
            except ValueError:
                print("✗ Invalid input")
                pause()
        elif choice == '3':
            # Edge-biased (higher resolution near edges)
            try:
                min_val = float(input("\nEnter minimum value: ").strip())
                max_val = float(input("Enter maximum value: ").strip())
                num_points = int(input("Enter number of points: ").strip())

                if num_points < 2:
                    print("✗ Need at least 2 points")
                    pause()
                    continue

                import numpy as np
                # Use cosine spacing for edge bias
                t = np.linspace(0, np.pi, num_points)
                normalized = (1 - np.cos(t)) / 2  # Maps to [0, 1] with edge bias
                new_values = (min_val + normalized * (max_val - min_val)).tolist()
                current_values.extend(new_values)
                current_values = sorted(list(set(current_values)))  # Remove duplicates and sort
                print(f"✓ Added {len(new_values)} edge-biased values")
                pause()
            except ValueError:
                print("✗ Invalid input")
                pause()
        elif choice == '4':
            # Clear all
            current_values.clear()
            print("✓ Cleared all values")
            pause()
        elif choice == '5':
            # Remove specific value
            if not current_values:
                print("✗ No values to remove")
                pause()
                continue
            try:
                idx = int(input(f"\nEnter index to remove [1-{len(current_values)}]: ").strip())
                if 1 <= idx <= len(current_values):
                    removed = current_values.pop(idx - 1)
                    print(f"✓ Removed value: {removed}")
                else:
                    print("✗ Invalid index")
                pause()
            except ValueError:
                print("✗ Invalid input")
                pause()


def setup_doe(solver, selected_inputs, doe_parameters):
    """Configure Design of Experiment parameters for selected inputs."""

    if not selected_inputs:
        print_header("DESIGN OF EXPERIMENT SETUP")
        print("\n✗ No model inputs selected! Please select inputs first (Option 1).")
        pause()
        return doe_parameters

    while True:
        clear_screen()
        print_header("DESIGN OF EXPERIMENT SETUP")

        print("\nSELECTED INPUTS:")
        print("="*70)
        for i, item in enumerate(selected_inputs, 1):
            num_params = len(doe_parameters.get(item['name'], {}))
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

                # Get available parameters for this BC/zone type
                available_params = get_bc_parameters(item['type'])

                # Initialize DOE config for this item if not exists
                if item['name'] not in doe_parameters:
                    doe_parameters[item['name']] = {}

                # Configure each parameter
                while True:
                    clear_screen()
                    print_header(f"DOE: {item['name']} ({item['type']})")

                    print("\nAVAILABLE PARAMETERS:")
                    print("="*70)
                    for i, param in enumerate(available_params, 1):
                        num_values = len(doe_parameters[item['name']].get(param, []))
                        status = f"({num_values} values)" if num_values > 0 else "(not configured)"
                        print(f"  [{i:2d}] {param:20s} {status}")
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
                            param_name = available_params[param_idx]
                            current_values = doe_parameters[item['name']].get(param_name, [])
                            new_values = setup_parameter_values(param_name, current_values.copy())
                            doe_parameters[item['name']][param_name] = new_values


def save_model_setup(solver, selected_inputs, selected_outputs, doe_parameters):
    """Save model setup to a JSON file."""
    print_header("SAVE MODEL SETUP")

    if not selected_inputs and not selected_outputs:
        print("\n✗ No inputs or outputs selected! Nothing to save.")
        pause()
        return

    # Get the case file path from solver if possible
    try:
        # Try to determine the case file location
        # Default to current directory
        default_dir = PROJECT_DIR

        # Ask user for folder name
        print(f"\nDefault save location: {default_dir}")
        folder_name = input("\nEnter folder name [new_project]: ").strip() or "new_project"

        # Create the save directory
        save_dir = default_dir / folder_name
        save_dir.mkdir(exist_ok=True)

        # Prepare the setup data
        setup_data = {
            'timestamp': str(Path(__file__).parent),  # Placeholder for actual timestamp
            'model_inputs': [
                {
                    'name': item['name'],
                    'type': item['type'],
                    'category': item.get('category', 'Unknown'),
                    'doe_parameters': doe_parameters.get(item['name'], {})
                }
                for item in selected_inputs
            ],
            'model_outputs': [
                {
                    'name': item['name'],
                    'type': item['type'],
                    'category': item.get('category', 'Unknown')
                }
                for item in selected_outputs
            ],
            'doe_configuration': doe_parameters
        }

        # Add timestamp
        from datetime import datetime
        setup_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to JSON file
        setup_file = save_dir / "model_setup.json"
        with open(setup_file, 'w') as f:
            json.dump(setup_data, f, indent=2)

        print(f"\n✓ Model setup saved successfully!")
        print(f"  Location: {setup_file}")
        print(f"\n  Inputs: {len(selected_inputs)}")
        print(f"  Outputs: {len(selected_outputs)}")
        print(f"  DOE Configured: {len([k for k, v in doe_parameters.items() if v])}")

    except Exception as e:
        print(f"\n✗ Error saving model setup: {e}")
        import traceback
        traceback.print_exc()

    pause()


def configure_project(solver):
    """Main project configuration menu."""

    selected_inputs = []
    selected_outputs = []
    doe_parameters = {}  # Store DOE configuration for each input

    while True:
        clear_screen()
        print_header("PROJECT CONFIGURATION")

        print(f"\n  Model Inputs Selected: {len(selected_inputs)}")
        print(f"  Model Outputs Selected: {len(selected_outputs)}")
        print(f"  DOE Parameters Configured: {len(doe_parameters)}")

        print(f"\n{'='*70}")
        print("  [1] Set Up Model Inputs (Boundary Conditions & Cell Zones)")
        print("  [2] Set Up Model Outputs (Surfaces, Zones, Report Definitions)")
        print("  [3] Design of Experiment Setup")
        print("  [4] Save Model Setup")
        print("  [5] Unload Project, Close Fluent, Back to Main")
        print("  [0] Back")
        print("="*70)

        choice = get_choice(5)

        if choice == 0:
            return
        elif choice == 1:
            selected_inputs = setup_model_inputs(solver, selected_inputs)
        elif choice == 2:
            selected_outputs = setup_model_outputs(solver, selected_outputs)
        elif choice == 3:
            doe_parameters = setup_doe(solver, selected_inputs, doe_parameters)
        elif choice == 4:
            save_model_setup(solver, selected_inputs, selected_outputs, doe_parameters)
        elif choice == 5:
            print("\nClosing Fluent session...")
            try:
                solver.exit()
                print("✓ Fluent session closed")
            except Exception as e:
                print(f"✗ Error closing Fluent: {e}")
            pause()
            return


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    """Main menu for Workflow Surrogate."""

    while True:
        clear_screen()

        # Build menu options
        options = [
            "Open Fluent Case File"
        ]

        # Add recent projects (up to 3)
        recent = user_settings.get_recent_projects()
        for i, proj in enumerate(recent[:3], 1):
            proj_path = Path(proj)
            # Show filename and parent directory
            location = proj_path.parent.name if proj_path.parent.name else proj_path.parent
            options.append(f"Recent Project {i}: {proj_path.name} ({location})")

        print_menu("WORKFLOW SURROGATE - MAIN MENU", options)

        choice = get_choice(len(options))

        if choice == 0:
            # Exit
            print("\nExiting Workflow Surrogate. Goodbye!")
            sys.exit(0)

        elif choice == 1:
            # Open case file
            clear_screen()
            solver = open_case_file()
            if solver:
                clear_screen()
                configure_project(solver)

        elif choice in [2, 3, 4]:
            # Recent projects
            clear_screen()
            recent = user_settings.get_recent_projects()
            proj_index = choice - 2
            if proj_index < len(recent):
                solver = open_recent_project(recent[proj_index])
                if solver:
                    clear_screen()
                    configure_project(solver)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
