"""
Cold Plate Digital Twin - Real-Time Interface
==============================================
Interactive visualization and prediction interface for cold plate thermal analysis.

Features:
- Real-time input adjustment with sliders and text boxes
- 2D heatmap plots for field outputs (yz-mid, zx-mid, bottom)
- Real-time line plot for scalar outputs (temperatures and pressures)
- Instant prediction updates when inputs change
"""

import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class DigitalTwinInterface:
    """Main application for the digital twin interface."""

    def __init__(self, root):
        self.root = root
        self.root.title("Cold Plate Digital Twin")
        self.root.geometry("1600x900")

        # Load project configuration
        self.project_dir = Path(__file__).parent
        self.case_dir = self.project_dir / "cases" / "operation_conditions_1"
        self.load_configuration()

        # Initialize model storage
        self.models = {
            'scalars': {},  # Dict of scalar model names to model data
            'fields': {}    # Dict of field model names to model data
        }

        # Load coordinate data for field plots
        self.field_coordinates = {}

        # Temperature unit toggle (True = Kelvin, False = Celsius)
        self.use_kelvin = True

        # Colorbar range settings (in Kelvin)
        self.colorbar_min = 300.0
        self.colorbar_max = 500.0

        # Prediction counter for x-axis
        self.prediction_counter = 0
        self.scalar_history = {name: [] for name in self.scalar_outputs}
        self.counter_history = []

        # Create UI
        self.create_ui()

        # Select which model to use
        self.selected_model_folder = self.select_model_folder()

        if self.selected_model_folder is None:
            messagebox.showerror("No Model Selected", "No model was selected. The application will now close.")
            self.root.destroy()
            return

        # Load trained models automatically
        self.load_models()

        # Set initial values and make first prediction
        self.reset_to_defaults()

    def load_configuration(self):
        """Load model configuration from JSON files."""
        # Load model setup
        with open(self.case_dir / "model_setup.json", 'r') as f:
            setup = json.load(f)

        # Extract input parameters with ranges from DOE configuration
        # IMPORTANT: Preserve order from model_inputs to match training data
        self.inputs = {}
        self.input_order = []  # Track original order

        for model_input in setup['model_inputs']:
            bc_name = model_input['name']
            if 'doe_parameters' in model_input:
                # Process parameters in the order they appear in doe_parameters
                # The dict maintains insertion order in Python 3.7+
                for param_name, values in model_input['doe_parameters'].items():
                    if values:
                        # Extract the last part of the parameter path (e.g., "mass_flow_rate" from "momentum.mass_flow_rate")
                        param_short_name = param_name.split('.')[-1]
                        key = f"{bc_name}.{param_short_name}"
                        self.inputs[key] = {
                            'min': float(min(values)),
                            'max': float(max(values)),
                            'default': float(np.mean(values)),
                            'bc_name': bc_name,
                            'param_path': param_name
                        }
                        self.input_order.append(key)

        # Extract outputs
        self.field_outputs = []
        self.scalar_outputs = []

        for output in setup['model_outputs']:
            if output['category'] == 'Surface' and output['name'] in ['yz-mid', 'zx-mid', 'bottom']:
                self.field_outputs.append(output['name'])
            elif output['category'] == 'Report Definition':
                self.scalar_outputs.append(output['name'])

        print(f"Loaded configuration:")
        print(f"  Inputs: {len(self.inputs)}")
        print(f"  Field outputs: {self.field_outputs}")
        print(f"  Scalar outputs: {self.scalar_outputs}")

    def select_model_folder(self):
        """Show dialog to select which model folder to use."""
        # Find all model folders (directories that contain model files)
        model_folders = []
        for item in self.case_dir.iterdir():
            if item.is_dir() and (list(item.glob("*_metadata.json")) or list(item.glob("*.h5"))):
                model_folders.append(item.name)

        if not model_folders:
            messagebox.showerror("No Models Found",
                               f"No trained models found in {self.case_dir}\n\n"
                               "Please train models first using the training interface.")
            return None

        # If only one model folder exists, use it automatically
        if len(model_folders) == 1:
            selected = model_folders[0]
            print(f"Using model folder: {selected}")
            return selected

        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        selected_model = [None]  # Use list to store selection in nested function

        # Title
        title_label = ttk.Label(dialog, text="Select Model to Load", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Info label
        info_label = ttk.Label(dialog, text=f"Case: {self.case_dir.name}", font=('Arial', 10))
        info_label.pack(pady=5)

        # Frame for listbox and scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Courier', 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate listbox with model folders
        model_info = []
        for folder_name in sorted(model_folders):
            summary_file = self.case_dir / folder_name / "training_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    timestamp = summary.get('training_info', {}).get('timestamp', 'Unknown')
                    num_models = len(summary.get('models', []))
                    display_text = f"{folder_name:25s} ({num_models} models, {timestamp})"
                except:
                    display_text = folder_name
            else:
                display_text = folder_name

            model_info.append((folder_name, display_text))
            listbox.insert(tk.END, display_text)

        # Select first item by default
        if model_info:
            listbox.selection_set(0)

        def on_select():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                selected_model[0] = model_info[idx][0]
                dialog.destroy()

        def on_cancel():
            selected_model[0] = None
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        select_btn = ttk.Button(button_frame, text="Select", command=on_select, width=15)
        select_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel, width=15)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Bind double-click to select
        listbox.bind('<Double-Button-1>', lambda e: on_select())

        # Wait for dialog to close
        self.root.wait_window(dialog)

        if selected_model[0]:
            print(f"Selected model folder: {selected_model[0]}")

        return selected_model[0]

    def create_ui(self):
        """Create the user interface."""
        # Main paned window (left=viz, right=controls)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Visualizations
        left_frame = ttk.Frame(main_paned, relief=tk.RIDGE, borderwidth=2)
        main_paned.add(left_frame, weight=7)
        self.create_visualization_panel(left_frame)

        # Right panel: Controls
        right_frame = ttk.Frame(main_paned, relief=tk.RIDGE, borderwidth=2)
        main_paned.add(right_frame, weight=3)
        self.create_control_panel(right_frame)

    def create_visualization_panel(self, parent):
        """Create the visualization panel with plots."""
        # Use paned window for top/bottom split
        viz_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        viz_paned.pack(fill=tk.BOTH, expand=True)

        # Top: Field outputs (2D heatmaps)
        field_frame = ttk.LabelFrame(viz_paned, text="Field Outputs (Temperature Distribution)", padding=10)
        viz_paned.add(field_frame, weight=6)

        # Create notebook for tabs
        self.field_notebook = ttk.Notebook(field_frame)
        self.field_notebook.pack(fill=tk.BOTH, expand=True)

        self.field_plots = {}

        for field_name in self.field_outputs:
            # Create frame for this tab
            tab_frame = ttk.Frame(self.field_notebook)
            self.field_notebook.add(tab_frame, text=field_name)

            # Create matplotlib figure
            fig = Figure(figsize=(8, 6))
            canvas = FigureCanvasTkAgg(fig, master=tab_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, tab_frame)
            toolbar.update()

            ax = fig.add_subplot(111)

            self.field_plots[field_name] = {
                'figure': fig,
                'canvas': canvas,
                'ax': ax,
                'colorbar': None
            }

        # Bottom: Scalar outputs (real-time plot)
        scalar_frame = ttk.LabelFrame(viz_paned, text="Scalar Outputs (Real-Time)", padding=10)
        viz_paned.add(scalar_frame, weight=4)

        # Container for plot and checkboxes
        plot_and_controls = ttk.Frame(scalar_frame)
        plot_and_controls.pack(fill=tk.BOTH, expand=True)

        # Left: Scalar plot
        plot_container = ttk.Frame(plot_and_controls)
        plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scalar_fig = Figure(figsize=(8, 3))
        self.scalar_canvas = FigureCanvasTkAgg(self.scalar_fig, master=plot_container)
        self.scalar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create axes with two y-axes
        self.ax_temp = self.scalar_fig.add_subplot(111)
        self.ax_pressure = self.ax_temp.twinx()

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.scalar_canvas, plot_container)
        toolbar.update()

        # Right: Checkboxes for toggling plots
        checkbox_frame = ttk.LabelFrame(plot_and_controls, text="Visible Outputs", padding=5)
        checkbox_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.scalar_visibility = {}
        for name in self.scalar_outputs:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(checkbox_frame, text=name, variable=var,
                               command=self.update_scalar_plot_visibility)
            cb.pack(anchor=tk.W, pady=2)
            self.scalar_visibility[name] = var

        # Current values display
        values_frame = ttk.Frame(scalar_frame)
        values_frame.pack(fill=tk.X, pady=5)

        self.scalar_value_labels = {}
        for i, name in enumerate(self.scalar_outputs):
            row = i // 4
            col = (i % 4) * 2

            label = ttk.Label(values_frame, text=f"{name}:", font=('Arial', 9))
            label.grid(row=row, column=col, padx=5, pady=2, sticky=tk.E)

            value_label = ttk.Label(values_frame, text="0.00", font=('Arial', 9, 'bold'))
            value_label.grid(row=row, column=col + 1, padx=5, pady=2, sticky=tk.W)

            self.scalar_value_labels[name] = value_label

    def create_control_panel(self, parent):
        """Create the control panel with input sliders and text boxes."""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        header = ttk.Label(header_frame, text="Input Parameters", font=('Arial', 16, 'bold'))
        header.pack()

        # Model status
        self.model_status_label = ttk.Label(header_frame,
                                            text="[WARNING] No model loaded - Showing placeholder values",
                                            font=('Arial', 9), foreground='orange')
        self.model_status_label.pack(pady=5)

        # Scrollable frame for inputs
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 10))

        # Input controls
        controls_frame = ttk.LabelFrame(scrollable_frame, text="Adjust Parameters", padding=10)
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_controls = {}

        for i, (key, info) in enumerate(self.inputs.items()):
            # Label
            label = ttk.Label(controls_frame, text=key, font=('Arial', 9))
            label.grid(row=i*2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

            # Slider
            slider = ttk.Scale(controls_frame, from_=info['min'], to=info['max'],
                             orient=tk.HORIZONTAL, length=200)
            slider.set(info['default'])
            slider.configure(command=lambda v, k=key: self.on_slider_changed(k, v))
            slider.grid(row=i*2+1, column=0, sticky=tk.EW, padx=(0, 5))

            # Text box
            text_var = tk.StringVar(value=f"{info['default']:.6f}")
            text_box = ttk.Entry(controls_frame, textvariable=text_var, width=12)
            text_box.bind('<Return>', lambda e, k=key: self.on_text_changed(k))
            text_box.bind('<FocusOut>', lambda e, k=key: self.on_text_changed(k))
            text_box.grid(row=i*2+1, column=1, padx=(5, 0))

            # Range label
            range_label = ttk.Label(controls_frame,
                                   text=f"[{info['min']:.3g}, {info['max']:.3g}]",
                                   font=('Arial', 8), foreground='gray')
            range_label.grid(row=i*2+1, column=2, padx=(5, 0))

            controls_frame.columnconfigure(0, weight=1)

            self.input_controls[key] = {
                'slider': slider,
                'text_var': text_var,
                'text_box': text_box,
                'info': info
            }

        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        predict_btn = ttk.Button(button_frame, text="Predict",
                                command=self.predict,
                                style='Accent.TButton')
        predict_btn.pack(fill=tk.X, pady=2)

        reset_btn = ttk.Button(button_frame, text="Reset to Defaults",
                              command=self.reset_to_defaults)
        reset_btn.pack(fill=tk.X, pady=2)

        clear_btn = ttk.Button(button_frame, text="Clear History",
                              command=self.clear_history)
        clear_btn.pack(fill=tk.X, pady=2)

        # Temperature unit toggle
        unit_frame = ttk.Frame(button_frame)
        unit_frame.pack(fill=tk.X, pady=5)

        ttk.Label(unit_frame, text="Temperature Unit:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 5))

        self.temp_unit_var = tk.StringVar(value="K")
        kelvin_rb = ttk.Radiobutton(unit_frame, text="K", variable=self.temp_unit_var,
                                    value="K", command=self.toggle_temperature_unit)
        kelvin_rb.pack(side=tk.LEFT, padx=2)

        celsius_rb = ttk.Radiobutton(unit_frame, text="°C", variable=self.temp_unit_var,
                                     value="C", command=self.toggle_temperature_unit)
        celsius_rb.pack(side=tk.LEFT, padx=2)

        # Colorbar range controls
        colorbar_frame = ttk.LabelFrame(button_frame, text="Colorbar Range (K)", padding=5)
        colorbar_frame.pack(fill=tk.X, pady=5)

        # Min value
        min_frame = ttk.Frame(colorbar_frame)
        min_frame.pack(fill=tk.X, pady=2)
        ttk.Label(min_frame, text="Min:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.colorbar_min_var = tk.StringVar(value=f"{self.colorbar_min:.1f}")
        min_entry = ttk.Entry(min_frame, textvariable=self.colorbar_min_var, width=10)
        min_entry.pack(side=tk.LEFT, padx=2)
        min_entry.bind('<Return>', lambda e: self.update_colorbar_range())

        # Max value
        max_frame = ttk.Frame(colorbar_frame)
        max_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_frame, text="Max:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.colorbar_max_var = tk.StringVar(value=f"{self.colorbar_max:.1f}")
        max_entry = ttk.Entry(max_frame, textvariable=self.colorbar_max_var, width=10)
        max_entry.pack(side=tk.LEFT, padx=2)
        max_entry.bind('<Return>', lambda e: self.update_colorbar_range())

        # Apply button
        apply_btn = ttk.Button(colorbar_frame, text="Apply Range",
                              command=self.update_colorbar_range)
        apply_btn.pack(fill=tk.X, pady=2)

        # Prediction info
        self.prediction_info = ttk.Label(button_frame, text="Prediction #0",
                                        font=('Arial', 9), foreground='gray')
        self.prediction_info.pack(pady=5)

    def on_slider_changed(self, key, slider_value):
        """Handle slider value change."""
        actual_value = float(slider_value)

        # Update text box
        self.input_controls[key]['text_var'].set(f"{actual_value:.6f}")

        # Don't auto-predict - user will click Predict button

    def on_text_changed(self, key):
        """Handle text box value change."""
        try:
            value = float(self.input_controls[key]['text_var'].get())
            info = self.input_controls[key]['info']

            # Clamp to range
            value = max(info['min'], min(info['max'], value))

            # Update text box with clamped value
            self.input_controls[key]['text_var'].set(f"{value:.6f}")

            # Update slider
            self.input_controls[key]['slider'].set(value)

            # Don't auto-predict - user will click Predict button
        except ValueError:
            pass

    def get_current_inputs(self):
        """Get current input values as a numpy array."""
        values = []
        # IMPORTANT: Training data used alphabetically sorted parameter names
        # NOT the order from model_inputs in the JSON
        for key in sorted(self.input_order):
            values.append(float(self.input_controls[key]['text_var'].get()))
        return np.array(values).reshape(1, -1)

    def kelvin_to_celsius(self, temp_k):
        """Convert temperature from Kelvin to Celsius."""
        if isinstance(temp_k, np.ndarray):
            return temp_k - 273.15
        else:
            return temp_k - 273.15

    def get_temp_value(self, temp_k):
        """Get temperature value in the current unit."""
        if self.use_kelvin:
            return temp_k
        else:
            return self.kelvin_to_celsius(temp_k)

    def get_temp_unit_label(self):
        """Get current temperature unit label."""
        return "K" if self.use_kelvin else "°C"

    def toggle_temperature_unit(self):
        """Toggle between Kelvin and Celsius."""
        self.use_kelvin = (self.temp_unit_var.get() == "K")
        # Redraw plots with new units
        self.update_scalar_plot()  # Redraw scalar plot
        # Trigger field plot update if we have predictions
        if hasattr(self, '_last_field_predictions'):
            self.update_field_plots(self._last_field_predictions)

    def update_colorbar_range(self):
        """Update the colorbar range and redraw field plots."""
        try:
            new_min = float(self.colorbar_min_var.get())
            new_max = float(self.colorbar_max_var.get())

            if new_min >= new_max:
                messagebox.showerror("Invalid Range", "Min value must be less than Max value")
                return

            self.colorbar_min = new_min
            self.colorbar_max = new_max

            # Redraw field plots with new range if we have predictions
            if hasattr(self, '_last_field_predictions'):
                self.update_field_plots(self._last_field_predictions)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values")

    def predict(self):
        """Make prediction with current inputs and update visualizations."""
        # Get current inputs
        inputs = self.get_current_inputs()

        # Debug: Print input values and order
        print(f"\n=== DEBUG: Prediction #{self.prediction_counter + 1} ===")
        print(f"Input order: {self.input_order}")
        print(f"Input values: {inputs[0]}")

        # Check if models are loaded
        if not self.models['scalars'] and not self.models['fields']:
            # Generate placeholder predictions
            scalar_predictions = {name: 300.0 for name in self.scalar_outputs}  # Room temperature
            field_predictions = {}
            for name in self.field_outputs:
                if name in self.field_coordinates:
                    n_points = len(self.field_coordinates[name]['coordinates'])
                    field_predictions[name] = np.full(n_points, 300.0)
                else:
                    field_predictions[name] = np.full(1000, 300.0)
        else:
            # Use actual trained models for prediction
            scalar_predictions = self.predict_scalars(inputs)
            field_predictions = self.predict_fields(inputs)

        # Store field predictions for unit toggle
        self._last_field_predictions = field_predictions

        # Update visualizations
        self.update_scalar_plot(scalar_predictions)
        self.update_field_plots(field_predictions)

        # Update counter
        self.prediction_counter += 1
        self.prediction_info.config(text=f"Prediction #{self.prediction_counter}")

    def predict_scalars(self, inputs):
        """Predict scalar outputs using loaded models."""
        predictions = {}

        for name in self.scalar_outputs:
            # Find matching model (e.g., "chip1_tavg" -> "chip1_tavg_temperature")
            model_name = None
            for key in self.models['scalars'].keys():
                if name in key or key in name:
                    model_name = key
                    break

            if model_name and model_name in self.models['scalars']:
                model_data = self.models['scalars'][model_name]
                try:
                    # Debug: Show scaling info for first prediction
                    if self.prediction_counter == 0 and name == 'chip1_tavg':
                        print(f"\n=== DEBUG: Model {model_name} ===")
                        print(f"param_scaler_mean shape: {model_data['param_scaler_mean'].shape}")
                        print(f"param_scaler_mean: {model_data['param_scaler_mean']}")
                        print(f"param_scaler_scale: {model_data['param_scaler_scale']}")
                        print(f"inputs shape: {inputs.shape}")
                        print(f"inputs: {inputs[0]}")

                    # Scale inputs
                    inputs_scaled = (inputs - model_data['param_scaler_mean']) / model_data['param_scaler_scale']

                    if self.prediction_counter == 0 and name == 'chip1_tavg':
                        print(f"inputs_scaled: {inputs_scaled[0]}")

                    # Predict (output is scaled)
                    output_scaled = model_data['nn_model'].predict(inputs_scaled, verbose=0)

                    if self.prediction_counter == 0 and name == 'chip1_tavg':
                        print(f"output_scaled: {output_scaled[0]}")

                    # Inverse scale output to get physical units
                    if model_data['output_scaler_mean'] is not None and model_data['output_scaler_scale'] is not None:
                        # Handle both scalar and array scalers
                        mean = model_data['output_scaler_mean']
                        scale = model_data['output_scaler_scale']
                        if isinstance(mean, np.ndarray) and len(mean) > 0:
                            mean = mean[0]
                        if isinstance(scale, np.ndarray) and len(scale) > 0:
                            scale = scale[0]
                        output = output_scaled * scale + mean
                        predictions[name] = float(output[0, 0] if output.ndim > 1 else output[0])

                        if self.prediction_counter == 0 and name == 'chip1_tavg':
                            print(f"output_scaler_mean: {mean}, output_scaler_scale: {scale}")
                            print(f"Final output: {predictions[name]}")
                    else:
                        predictions[name] = float(output_scaled[0, 0])
                except Exception as e:
                    print(f"Error predicting {name}: {e}")
                    predictions[name] = 300.0  # Default room temperature
            else:
                predictions[name] = 300.0  # Default room temperature

        return predictions

    def predict_fields(self, inputs):
        """Predict field outputs using loaded models."""
        predictions = {}

        for name in self.field_outputs:
            # Find matching model (e.g., "yz-mid" -> "yz-mid_temperature")
            model_name = None
            for key in self.models['fields'].keys():
                if name in key or key in name:
                    model_name = key
                    break

            if model_name and model_name in self.models['fields']:
                model_data = self.models['fields'][model_name]
                try:
                    # Scale inputs
                    inputs_scaled = (inputs - model_data['param_scaler_mean']) / model_data['param_scaler_scale']

                    # Predict POD modes
                    modes_scaled = model_data['nn_model'].predict(inputs_scaled, verbose=0)

                    # Inverse scale modes
                    modes = modes_scaled * model_data['mode_scaler_scale'] + model_data['mode_scaler_mean']

                    # Reconstruct field from POD modes
                    field_flat = model_data['pca_mean'] + np.dot(modes, model_data['pca_components'])

                    # Return as 1D array - plotting function will handle coordinates
                    predictions[name] = field_flat[0]

                except Exception as e:
                    print(f"Error predicting {name}: {e}")
                    # Return dummy data as fallback
                    if name in self.field_coordinates:
                        n_points = len(self.field_coordinates[name]['coordinates'])
                        predictions[name] = np.zeros(n_points)
                    else:
                        predictions[name] = np.zeros(1000)
            else:
                # Return dummy data if model not found
                if name in self.field_coordinates:
                    n_points = len(self.field_coordinates[name]['coordinates'])
                    predictions[name] = np.zeros(n_points)
                else:
                    predictions[name] = np.zeros(1000)

        return predictions

    def update_scalar_plot(self, predictions=None):
        """Update the scalar outputs real-time plot."""
        # If predictions provided, store in history
        if predictions is not None:
            self.counter_history.append(self.prediction_counter)
            for name, value in predictions.items():
                self.scalar_history[name].append(value)

                # Update text label
                if 'pdrop' in name.lower():
                    self.scalar_value_labels[name].config(text=f"{value:.2f} Pa")
                else:
                    display_value = self.get_temp_value(value)
                    unit_label = self.get_temp_unit_label()
                    self.scalar_value_labels[name].config(text=f"{display_value:.2f} {unit_label}")

            # Keep only last 50 predictions
            max_history = 50
            if len(self.counter_history) > max_history:
                self.counter_history = self.counter_history[-max_history:]
                for name in self.scalar_outputs:
                    self.scalar_history[name] = self.scalar_history[name][-max_history:]

        # Clear axes
        self.ax_temp.clear()
        self.ax_pressure.clear()

        # Plot temperature outputs on left axis (only if visible)
        temp_outputs = [n for n in self.scalar_outputs if 'pdrop' not in n.lower() and self.scalar_visibility[n].get()]
        for name in temp_outputs:
            # Convert temperature values for display
            display_values = [self.get_temp_value(v) for v in self.scalar_history[name]]
            self.ax_temp.plot(self.counter_history, display_values,
                            marker='o', label=name, linewidth=2, markersize=4)

        # Plot pressure outputs on right axis (only if visible)
        pressure_outputs = [n for n in self.scalar_outputs if 'pdrop' in n.lower() and self.scalar_visibility[n].get()]
        for name in pressure_outputs:
            self.ax_pressure.plot(self.counter_history, self.scalar_history[name],
                                marker='s', linestyle='--', label=name, linewidth=2, markersize=4)

        # Styling
        unit_label = self.get_temp_unit_label()
        self.ax_temp.set_xlabel('Prediction Index', fontsize=10, fontweight='bold')
        self.ax_temp.set_ylabel(f'Temperature ({unit_label})', fontsize=10, fontweight='bold', color='blue')

        # Position pressure label on the right with padding to avoid collision with tick labels
        self.ax_pressure.set_ylabel('Pressure Drop (Pa)', fontsize=10, fontweight='bold', color='red')
        self.ax_pressure.yaxis.set_label_coords(1.12, 0.5)  # Move label further right

        self.ax_temp.tick_params(axis='y', labelcolor='blue')
        self.ax_pressure.tick_params(axis='y', labelcolor='red')

        # Move legends to bottom to avoid data overlap
        if temp_outputs:
            self.ax_temp.legend(loc='lower left', fontsize=7, ncol=2)
        if pressure_outputs:
            self.ax_pressure.legend(loc='lower right', fontsize=7, ncol=2)

        self.ax_temp.grid(True, alpha=0.3)
        self.scalar_fig.tight_layout()
        self.scalar_canvas.draw()

    def update_scalar_plot_visibility(self):
        """Update scalar plot when visibility checkboxes change."""
        self.update_scalar_plot()  # Redraw without new predictions

    def update_field_plots(self, predictions):
        """Update the 2D scatter plots for field outputs using actual coordinates."""
        for field_name, data in predictions.items():
            plot_info = self.field_plots[field_name]
            ax = plot_info['ax']
            fig = plot_info['figure']

            # Remove old colorbar if it exists (before clearing axis)
            if plot_info['colorbar'] is not None:
                try:
                    plot_info['colorbar'].remove()
                except (KeyError, AttributeError, ValueError):
                    pass
                plot_info['colorbar'] = None

            # Clear axis
            ax.clear()

            # Check if we have coordinates for this field
            if field_name not in self.field_coordinates:
                # Fallback to simple plot if no coordinates
                ax.text(0.5, 0.5, f'No coordinates loaded for {field_name}',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{field_name} - No Data', fontsize=12, fontweight='bold')
                fig.tight_layout()
                plot_info['canvas'].draw()
                continue

            coord_info = self.field_coordinates[field_name]
            coordinates = coord_info['coordinates']
            varying_dims = coord_info['varying_dims']
            axis_names = coord_info['axis_names']

            # Get axis labels (swap for 90 degree rotation)
            xlabel = f'{axis_names[varying_dims[1]]} (m)'
            ylabel = f'{axis_names[varying_dims[0]]} (m)'

            # Flatten data if needed (handle different shapes from prediction)
            if len(data.shape) > 1:
                data_flat = data.flatten()
            else:
                data_flat = data

            # Ensure data matches coordinate size
            if len(data_flat) != len(coordinates):
                # Truncate or pad data to match coordinates
                if len(data_flat) > len(coordinates):
                    data_flat = data_flat[:len(coordinates)]
                else:
                    # Pad with zeros
                    padded = np.zeros(len(coordinates))
                    padded[:len(data_flat)] = data_flat
                    data_flat = padded

            # Convert temperature to current unit for display (but keep range in Kelvin)
            data_display = self.get_temp_value(data_flat)

            # Convert colorbar range to current unit
            vmin_display = self.get_temp_value(self.colorbar_min)
            vmax_display = self.get_temp_value(self.colorbar_max)

            # Plot 2D scatter with actual coordinates (rotated 90 degrees by swapping axes)
            scatter = ax.scatter(
                coordinates[:, varying_dims[1]],  # Swapped: use second dimension for X
                coordinates[:, varying_dims[0]],  # Swapped: use first dimension for Y
                c=data_display,
                cmap='jet',
                s=5,
                alpha=0.8,
                vmin=vmin_display,
                vmax=vmax_display
            )

            # Add new colorbar
            unit_label = self.get_temp_unit_label()
            cbar = fig.colorbar(scatter, ax=ax, label=f'Temperature ({unit_label})')
            plot_info['colorbar'] = cbar

            # Styling
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{field_name} - Temperature Distribution',
                        fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            plot_info['canvas'].draw()

    def reset_to_defaults(self):
        """Reset all inputs to default values."""
        for key, controls in self.input_controls.items():
            default = controls['info']['default']
            controls['text_var'].set(f"{default:.6f}")
            controls['slider'].set(default)

        self.predict()

    def clear_history(self):
        """Clear the scalar output history."""
        self.prediction_counter = 0
        self.counter_history = []
        for name in self.scalar_outputs:
            self.scalar_history[name] = []

        # Clear plot
        self.ax_temp.clear()
        self.ax_pressure.clear()
        self.scalar_canvas.draw()

        self.prediction_info.config(text="Prediction #0")

    def load_models(self):
        """Load all trained models from disk."""
        try:
            models_dir = self.case_dir / self.selected_model_folder

            if not models_dir.exists():
                print(f"[WARNING] Models directory not found: {models_dir}")
                self.model_status_label.config(text="[WARNING] No models found",
                                             foreground='orange')
                return

            # Import tensorflow/keras for loading .h5 models
            try:
                from tensorflow import keras
            except ImportError:
                print("[WARNING] TensorFlow not installed - cannot load models")
                self.model_status_label.config(text="[WARNING] TensorFlow not installed",
                                             foreground='orange')
                return

            # Load all model files
            npz_files = list(models_dir.glob("*.npz"))
            h5_files = list(models_dir.glob("*.h5"))

            if not npz_files or not h5_files:
                print(f"[WARNING] Incomplete model files in {models_dir}")
                self.model_status_label.config(text="[WARNING] Incomplete model files",
                                             foreground='orange')
                return

            print(f"\nLoading models from: {models_dir}")
            loaded_count = 0

            # Load each model
            for npz_file in npz_files:
                model_name = npz_file.stem  # e.g., "bottom_temperature"
                h5_file = npz_file.with_suffix('.h5')

                if not h5_file.exists():
                    print(f"  [WARNING] Skipping {model_name}: missing .h5 file")
                    continue

                try:
                    # Load metadata from .npz
                    data = np.load(npz_file, allow_pickle=True)

                    # Load neural network from .h5
                    nn_model = keras.models.load_model(h5_file, compile=False)

                    # Recompile to avoid issues
                    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                    # Store model data
                    model_data = {
                        'nn_model': nn_model,
                        'pca_components': data.get('pca_components'),
                        'pca_mean': data.get('pca_mean'),
                        'param_scaler_mean': data['param_scaler_mean'],
                        'param_scaler_scale': data['param_scaler_scale'],
                        'mode_scaler_mean': data.get('mode_scaler_mean'),
                        'mode_scaler_scale': data.get('mode_scaler_scale'),
                        'output_scaler_mean': data.get('output_scaler_mean'),
                        'output_scaler_scale': data.get('output_scaler_scale'),
                        'n_modes': int(data.get('n_modes', 0)) if 'n_modes' in data else None
                    }

                    # Categorize as scalar or field based on presence of PCA components
                    if model_data['pca_components'] is not None:
                        # Field model (has PCA)
                        self.models['fields'][model_name] = model_data
                    else:
                        # Scalar model (no PCA)
                        self.models['scalars'][model_name] = model_data

                    loaded_count += 1
                    print(f"  [OK] Loaded: {model_name}")

                except Exception as e:
                    print(f"  [ERROR] Error loading {model_name}: {str(e)}")

            # Update status
            if loaded_count > 0:
                print(f"\n[SUCCESS] Successfully loaded {loaded_count} models")
                print(f"  - Scalar models: {len(self.models['scalars'])}")
                print(f"  - Field models: {len(self.models['fields'])}")

                # Load coordinate data for field plots
                self.load_field_coordinates()

                self.model_status_label.config(
                    text=f"[SUCCESS] Loaded {loaded_count} models",
                    foreground='green')
            else:
                print("[WARNING] No models were loaded")
                self.model_status_label.config(text="[WARNING] Failed to load models",
                                             foreground='orange')

        except Exception as e:
            print(f"[ERROR] Error loading models: {str(e)}")
            self.model_status_label.config(text="[ERROR] Error loading models",
                                         foreground='red')

    def load_field_coordinates(self):
        """Load coordinate data for field visualizations."""
        try:
            dataset_dir = self.case_dir / "dataset"
            output_files = sorted(dataset_dir.glob("sim_*.npz"))

            if not output_files:
                print("[WARNING] No dataset files found for coordinates")
                return

            # Load coordinates from a middle file (most likely to be in training set)
            middle_file = output_files[len(output_files) // 2]
            data = np.load(middle_file, allow_pickle=True)

            for field_name in self.field_outputs:
                coord_key = f"{field_name}|coordinates"
                if coord_key in data.files:
                    coordinates = data[coord_key]

                    # Detect which 2 dimensions vary (for 2D surface plots)
                    variances = [np.var(coordinates[:, i]) for i in range(3)]
                    varying_dims = sorted(range(3), key=lambda i: variances[i], reverse=True)[:2]
                    varying_dims.sort()

                    self.field_coordinates[field_name] = {
                        'coordinates': coordinates,
                        'varying_dims': varying_dims,
                        'axis_names': ['X', 'Y', 'Z']
                    }
                    print(f"  [OK] Loaded coordinates for {field_name}: {coordinates.shape[0]} points")

        except Exception as e:
            print(f"[WARNING] Error loading coordinates: {str(e)}")


def main():
    """Run the digital twin interface."""
    root = tk.Tk()
    app = DigitalTwinInterface(root)
    root.mainloop()


if __name__ == '__main__':
    main()
