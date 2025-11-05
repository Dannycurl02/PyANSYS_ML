"""
Project System Module
=====================
Handles project creation, opening, scanning, and management.
"""

import json
from pathlib import Path
from datetime import datetime


class WorkflowProject:
    """
    Represents a Workflow Surrogate project.

    Project Structure:
    ------------------
    project_folder/
    ├── project_info.json          # Project metadata
    ├── simulation_datasets/       # Simulation setups (DOE configurations)
    │   ├── setup_name_1/
    │   │   ├── model_setup.json
    │   │   ├── output_parameters.json
    │   │   ├── inputs/
    │   │   ├── outputs/
    │   │   └── ...
    │   └── setup_name_2/
    └── trained_models/            # Trained models
        ├── model_name_1/
        │   ├── metadata.json
        │   ├── model.pth
        │   └── ...
        └── model_name_2/
    """

    def __init__(self, project_path):
        """
        Initialize project.

        Parameters
        ----------
        project_path : Path
            Path to project folder
        """
        self.project_path = Path(project_path)
        self.project_info_file = self.project_path / "project_info.json"
        self.sim_datasets_dir = self.project_path / "simulation_datasets"
        self.trained_models_dir = self.project_path / "trained_models"

        self.info = None
        self.datasets = []
        self.models = []

    def create(self, project_name):
        """
        Create a new project.

        Parameters
        ----------
        project_name : str
            Name of the project

        Returns
        -------
        bool
            True if successful
        """
        try:
            # Create directories
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.sim_datasets_dir.mkdir(exist_ok=True)
            self.trained_models_dir.mkdir(exist_ok=True)

            # Create project info
            self.info = {
                'project_name': project_name,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_opened': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'version': '1.0'
            }

            # Save project info
            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            return True

        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def load(self):
        """
        Load an existing project.

        Returns
        -------
        bool
            True if successful
        """
        try:
            if not self.project_info_file.exists():
                return False

            # Load project info
            with open(self.project_info_file, 'r') as f:
                self.info = json.load(f)

            # Update last opened
            self.info['last_opened'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.project_info_file, 'w') as f:
                json.dump(self.info, f, indent=2)

            # Scan for datasets and models
            self.scan()

            return True

        except Exception as e:
            print(f"Error loading project: {e}")
            return False

    def scan(self):
        """
        Scan project for simulation setups and trained models.
        """
        self.datasets = []
        self.models = []

        # Scan simulation setups
        if self.sim_datasets_dir.exists():
            for dataset_dir in self.sim_datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_info = self._scan_dataset(dataset_dir)
                    if dataset_info:
                        self.datasets.append(dataset_info)

        # Scan trained models
        if self.trained_models_dir.exists():
            for model_dir in self.trained_models_dir.iterdir():
                if model_dir.is_dir():
                    model_info = self._scan_model(model_dir)
                    if model_info:
                        self.models.append(model_info)

    def _scan_dataset(self, dataset_dir):
        """
        Scan a simulation setup directory.

        Parameters
        ----------
        dataset_dir : Path
            Setup directory

        Returns
        -------
        dict or None
            Setup information
        """
        setup_file = dataset_dir / "model_setup.json"

        if not setup_file.exists():
            return None

        try:
            with open(setup_file, 'r') as f:
                setup_data = json.load(f)

            # Count simulation files
            outputs_dir = dataset_dir / "outputs"
            num_sims = 0
            if outputs_dir.exists():
                num_sims = len(list(outputs_dir.glob("sim_*.npz")))

            # Count total required
            from modules import doe_setup as doe
            analysis = doe.analyze_setup_dimensions(setup_data)
            total_required = analysis['total_input_combinations']

            completeness = (num_sims / total_required * 100) if total_required > 0 else 0

            return {
                'name': dataset_dir.name,
                'path': dataset_dir,
                'setup_file': setup_file,
                'num_inputs': len(setup_data.get('model_inputs', [])),
                'num_outputs': len(setup_data.get('model_outputs', [])),
                'num_simulations': num_sims,
                'total_required': total_required,
                'completeness': completeness,
                'created': setup_data.get('timestamp', 'Unknown')
            }

        except Exception as e:
            print(f"Warning: Error scanning dataset {dataset_dir.name}: {e}")
            return None

    def _scan_model(self, model_dir):
        """
        Scan a trained model directory.

        Parameters
        ----------
        model_dir : Path
            Model directory

        Returns
        -------
        dict or None
            Model information
        """
        # Check for model_info.json (new autoencoder format)
        model_info_file = model_dir / "model_info.json"

        # Fallback to metadata.json (old format)
        metadata_file = model_dir / "metadata.json"

        if model_info_file.exists():
            # New autoencoder format
            try:
                with open(model_info_file, 'r') as f:
                    model_info = json.load(f)

                config = model_info.get('config', {})

                # Detect architecture type
                is_direct_nn = (model_dir / "direct_nn.pth").exists()
                arch_type = "Direct NN" if is_direct_nn else "Bottleneck NN"

                # Try to get source dataset from evaluation_results.json
                eval_file = model_dir / "evaluation_results.json"
                source_dataset = 'Unknown'
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        source_dataset = eval_data.get('dataset_name', 'Unknown')
                    except:
                        pass

                # Build architecture string based on type
                if is_direct_nn:
                    architecture_str = f"Direct NN: {config.get('input_dim', '?')} -> ... -> {config.get('output_dim', '?')}"
                else:
                    architecture_str = f"Bottleneck: {config.get('input_dim', '?')} -> {config.get('latent_dim', '?')} -> {config.get('output_dim', '?')}"

                return {
                    'name': model_dir.name,
                    'path': model_dir,
                    'metadata_file': model_info_file,
                    'source_dataset': source_dataset,
                    'trained': model_info.get('timestamp', 'Unknown'),
                    'architecture_type': arch_type,
                    'latent_size': config.get('latent_dim', 'N/A') if not is_direct_nn else 'N/A',
                    'architecture': architecture_str
                }

            except Exception as e:
                print(f"Warning: Error scanning model {model_dir.name}: {e}")
                return None

        elif metadata_file.exists():
            # Old POD-NN format
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                return {
                    'name': model_dir.name,
                    'path': model_dir,
                    'metadata_file': metadata_file,
                    'source_dataset': metadata.get('source_dataset', 'Unknown'),
                    'trained': metadata.get('training_completed', 'Unknown'),
                    'latent_size': metadata.get('model_config', {}).get('pod_modes', 'N/A'),
                    'architecture': metadata.get('model_config', {}).get('encoder', 'N/A')
                }

            except Exception as e:
                print(f"Warning: Error scanning model {model_dir.name}: {e}")
                return None
        else:
            return None

    def get_dataset(self, dataset_name):
        """
        Get dataset by name.

        Parameters
        ----------
        dataset_name : str
            Dataset name

        Returns
        -------
        dict or None
            Dataset information
        """
        for dataset in self.datasets:
            if dataset['name'] == dataset_name:
                return dataset
        return None

    def delete_dataset(self, dataset_name):
        """
        Delete a simulation setup.

        Parameters
        ----------
        dataset_name : str
            Setup name

        Returns
        -------
        bool
            True if successful
        """
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            return False

        try:
            import shutil
            import time

            # On Windows, sometimes files are locked. Try with onerror handler
            def handle_remove_readonly(func, path, exc_info):
                """Error handler for Windows readonly files."""
                import os
                import stat
                # Try to remove readonly flag and retry
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except:
                    pass

            # First attempt with error handler
            try:
                shutil.rmtree(dataset['path'], onerror=handle_remove_readonly)
            except PermissionError:
                # If still fails, try manual deletion with retry
                print("\n⚠ Permission error. Attempting to close any open files...")
                time.sleep(1)  # Give OS time to release handles

                # Try again
                shutil.rmtree(dataset['path'], onerror=handle_remove_readonly)

            self.scan()  # Refresh
            return True
        except Exception as e:
            print(f"Error deleting dataset: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Close any programs that might have files open (Excel, editors, etc.)")
            print("  2. Close the Fluent case if it's still running")
            print("  3. Try deleting the folder manually in File Explorer")
            return False

    def delete_model(self, model_name):
        """
        Delete a trained model.

        Parameters
        ----------
        model_name : str
            Model name

        Returns
        -------
        bool
            True if successful
        """
        for model in self.models:
            if model['name'] == model_name:
                try:
                    import shutil
                    import time

                    model_path = model['path']

                    # Try to delete with retry logic for Windows file locking issues
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            shutil.rmtree(model_path)
                            self.scan()  # Refresh
                            return True
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                # Wait a bit and retry
                                time.sleep(0.5)
                                continue
                            else:
                                # Final attempt failed
                                raise e

                except PermissionError as e:
                    print(f"Error deleting model: {e}")
                    print("\nTroubleshooting tips:")
                    print("  1. Close any Python processes or Jupyter notebooks using this model")
                    print("  2. Close any programs viewing model files (text editors, etc.)")
                    print("  3. Check if another instance of this application is running")
                    print("  4. Try deleting the folder manually in File Explorer")
                    print(f"     Location: {model_path}")
                    return False
                except Exception as e:
                    print(f"Error deleting model: {e}")
                    return False
        return False


def create_new_project(ui_helpers):
    """
    Create a new project interactively.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Created project
    """
    from tkinter import Tk, filedialog

    ui_helpers.clear_screen()
    ui_helpers.print_header("CREATE NEW PROJECT")

    # Get project name
    project_name = input("\nEnter project name: ").strip()
    if not project_name:
        print("\n[X] Project name cannot be empty")
        ui_helpers.pause()
        return None

    # Select parent directory
    print("\nSelect parent directory for project...")
    Tk().withdraw()

    parent_dir = filedialog.askdirectory(
        title="Select Parent Directory for Project"
    )

    if not parent_dir:
        print("\n[X] No directory selected")
        ui_helpers.pause()
        return None

    # Create project folder
    project_folder = Path(parent_dir) / project_name

    if project_folder.exists():
        overwrite = input(f"\n[WARNING] Folder '{project_name}' already exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != 'y':
            print("\n[X] Project creation cancelled")
            ui_helpers.pause()
            return None

    # Create project
    project = WorkflowProject(project_folder)

    if project.create(project_name):
        print(f"\n✓ Project created successfully!")
        print(f"  Location: {project_folder}")
        ui_helpers.pause()
        return project
    else:
        print(f"\n✗ Failed to create project")
        ui_helpers.pause()
        return None


def open_existing_project(ui_helpers):
    """
    Open an existing project from file explorer.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Opened project
    """
    from tkinter import Tk, filedialog

    ui_helpers.clear_screen()
    ui_helpers.print_header("OPEN EXISTING PROJECT")

    print("\nSelect project folder...")
    Tk().withdraw()

    project_folder = filedialog.askdirectory(
        title="Select Project Folder"
    )

    if not project_folder:
        print("\n✗ No folder selected")
        ui_helpers.pause()
        return None

    project_folder = Path(project_folder)

    # Check if it's a valid project
    project = WorkflowProject(project_folder)

    if project.load():
        print(f"\n✓ Project opened successfully!")
        print(f"  Name: {project.info['project_name']}")
        print(f"  Location: {project_folder}")
        print(f"  Created: {project.info['created']}")
        ui_helpers.pause()
        return project
    else:
        print(f"\n✗ Invalid project folder (project_info.json not found)")
        ui_helpers.pause()
        return None


def open_recent_project(project_path, ui_helpers):
    """
    Open a recent project.

    Parameters
    ----------
    project_path : Path or str
        Project path
    ui_helpers : module
        UI helpers module

    Returns
    -------
    WorkflowProject or None
        Opened project
    """
    project = WorkflowProject(project_path)

    if project.load():
        return project
    else:
        print(f"\n✗ Could not open project: {project_path}")
        ui_helpers.pause()
        return None
