# Current Status - December 2025

## Summary

The Workflow Surrogate system has been enhanced with a **multi-model management system** that allows training and comparing multiple model configurations side-by-side. This was implemented to investigate and resolve a critical training data pairing issue.

---

## Recent Implementation: Multi-Model System ✓

### What Was Done:

1. **Model Storage Reorganization**
   - Moved existing models from `cases/{case}/models/` to `cases/{case}/model1/`
   - Each model set now stored in its own named folder

2. **Training Interface Updates** ([multi_model_trainer.py](modules/multi_model_trainer.py))
   - Added model name input prompt at training start
   - Validates model names (alphanumeric, underscores, hyphens only)
   - Creates named folder for each model set
   - All model files (.h5, .npz, metadata) stored in that folder

3. **Visualization Interface Updates** ([multi_model_visualizer.py](modules/multi_model_visualizer.py))
   - Added model selection menu at startup
   - Lists all available model folders with metadata:
     - Number of trained models
     - Training timestamp
   - User selects which model to visualize/validate

4. **Digital Twin Interface Updates** ([DT_sim.py](C:\Users\danny\OneDrive - University of Arkansas\.Machine Learning\Project\CP\DT_sim.py))
   - Added GUI dialog for model selection at startup
   - Shows model info (number of models, training date)
   - Auto-selects if only one model folder exists
   - Gracefully closes if no model selected

### Benefits:

- **Side-by-side comparison:** Train models with different configurations without overwriting
- **Version control:** Keep multiple model versions organized
- **Easy testing:** Compare model accuracy for different parameter orderings
- **Clean organization:** Each model set isolated in its own folder

---

## Outstanding Issue: Training Data Pairing

### The Problem:

During investigation of inverted physics predictions (higher heat flux → lower temperature), we discovered a potential **parameter ordering mismatch** between:

1. **Simulation execution** (`simulation_runner.py`): Uses `doe_config.items()` (insertion order)
2. **Model training** (`multi_model_trainer.py`): Uses `sorted(doe_config.keys())` (alphabetical order)

This mismatch could cause inputs to be paired with wrong outputs during training.

### What We Found:

1. **JSON is currently alphabetical:** `chip1, chip2, inlet1, inlet2` (verified in `model_setup.json`)
2. **DOE creation uses sorted():** The `doe_setup.py` function that created the JSON uses `sorted()` on both BC names and parameters (line 490)
3. **Key insight:** If the JSON was created via `doe_setup.py`, it should have been alphabetical from the start
4. **Implication:** Since `simulation_runner.py` uses `.items()` (which preserves insertion order), and the JSON is alphabetical, the pairing should be correct

### Investigation Status:

- ✓ Verified JSON is currently alphabetical
- ✓ Confirmed DOE creation code uses `sorted()`
- ✓ Created diagnostic scripts: `diagnose_pairing.py`, `determine_original_doe_order.py`
- ✓ Implemented multi-model system for testing different orderings
- ⚠️ **Cannot definitively prove original JSON order** (no backup/git history before fix script ran)

### Current Theory:

The most likely scenario is that the **JSON was alphabetical from the start** (created by `doe_setup.py`), which means:
- Current pairing is correct
- Just need to retrain models with existing data
- Physics should be correct after retraining

### Alternative Scenario:

If the JSON was manually created or modified before simulations ran, it might have had insertion order: `inlet2, inlet1, chip2, chip1`. This would require:
- Creating parameter permutation: `[4, 5, 0, 1, 2, 3]`
- Modifying trainer to apply permutation before zipping
- Retraining with permuted order

---

## Next Steps

### Option 1: Retrain with Current Order (Recommended)

**Assumption:** JSON was alphabetical when simulations ran

**Steps:**
1. Use training interface to create new model (name it "alphabetical" or "baseline")
2. Train all models with existing data
3. Test predictions in DT_sim.py:
   - Verify higher heat flux → higher temperature
   - Verify higher inlet temp → higher outlet temp
   - Verify higher flow rate → higher pressure drop
4. Check R² scores are high (>0.9)

**If this works:** Problem solved! The original order was alphabetical.

### Option 2: Test Permuted Order (If Option 1 Fails)

**Assumption:** JSON had different order when simulations ran

**Steps:**
1. Modify `multi_model_trainer.py` to add permutation:
   ```python
   # Before line 110 where parameters are zipped:
   PERMUTATION = [4, 5, 0, 1, 2, 3]  # inlet2, inlet1, chip2, chip1 → chip1, chip2, inlet1, inlet2
   param_values = [param_values[i] for i in PERMUTATION]
   param_names = [param_names[i] for i in PERMUTATION]
   ```
2. Train with permuted order (name it "permuted" or "test_permutation")
3. Test predictions to see if physics is correct
4. Compare R² scores between alphabetical vs permuted models

### Option 3: Systematic Permutation Testing

**If unsure which is correct:**

Create comprehensive diagnostic script that:
1. Tests all possible 2-parameter orderings (BC level permutations)
2. For each permutation:
   - Build parameter combinations
   - Calculate physical correlations:
     - heat_flux vs chip_temp (should be positive)
     - inlet_temp vs outlet_temp (should be positive)
     - flow_rate vs pressure_drop (should be positive)
3. Score each permutation by correlation strength
4. Identify best permutation

**Note:** Full permutation testing (720 permutations for 6 parameters) is too slow. Focus on BC-level permutations (24 total).

---

## DOE Structure (Important Context)

The dataset consists of **4500 simulations** split into two parts:

1. **Simulations 1-2500:** Full factorial design
   - Order doesn't matter (uses `itertools.product()`)
   - All combinations of discrete parameter values

2. **Simulations 2501-4500:** Latin Hypercube Sampling (LHS)
   - Order is CRITICAL (uses `zip()`)
   - Pre-shuffled parameter arrays
   - This is where ordering mismatch would cause problems

**Implication:** Focus testing on LHS portion (sims 2501-4500) to detect ordering issues.

---

## Files Modified

### Workflow Surrogate (Main Repository)
- [modules/multi_model_trainer.py](modules/multi_model_trainer.py): Added model name input and folder creation
- [modules/multi_model_visualizer.py](modules/multi_model_visualizer.py): Added model selection menu

### CP Project (Additional Directory)
- [DT_sim.py](C:\Users\danny\OneDrive - University of Arkansas\.Machine Learning\Project\CP\DT_sim.py): Added model selection dialog

### Diagnostic Scripts Created
- `determine_original_doe_order.py`: Permutation testing (too slow, abandoned)
- `diagnose_pairing.py`: Manual pairing verification

---

## Key Findings from Investigation

### From `doe_setup.py` (Line 490):
```python
for bc_name, params in sorted(doe_parameters.items()):  # ALPHABETICAL!
    for param_name, values in sorted(params.items()):   # ALPHABETICAL!
```

This means the DOE configuration was created alphabetically from the start.

### From `simulation_runner.py` (Line 118):
```python
for bc_name, doe_params in doe_config.items():  # INSERTION ORDER
```

This uses insertion order, but if JSON was created alphabetically, insertion order = alphabetical order.

**Conclusion:** The pairing should be correct if JSON was created via `doe_setup.py`.

---

## Model File Organization

### Old Structure:
```
cases/operation_conditions_1/
└── models/
    ├── 2D_temperature_1.h5
    ├── 2D_temperature_1.npz
    ├── 2D_temperature_1_metadata.json
    └── training_summary.json
```

### New Structure:
```
cases/operation_conditions_1/
├── model1/                    # Existing models (moved here)
│   ├── 2D_temperature_1.h5
│   ├── 2D_temperature_1.npz
│   ├── 2D_temperature_1_metadata.json
│   └── training_summary.json
├── alphabetical/              # New training with current order
│   └── (trained model files)
└── permuted/                  # New training with permuted order (if needed)
    └── (trained model files)
```

---

## Testing Checklist

### After Training New Models:

- [ ] Check R² scores (target: >0.9 for all models)
- [ ] Test physical relationships in DT_sim.py:
  - [ ] Higher chip heat flux → Higher chip temperature
  - [ ] Lower chip heat flux → Lower chip temperature
  - [ ] Higher inlet temperature → Higher outlet temperature
  - [ ] Higher flow rate → Higher pressure drop
  - [ ] Higher flow rate → Lower chip temperature
- [ ] Compare predictions between different model folders
- [ ] Verify no negative R² scores on custom predictions

### Red Flags:

- ❌ R² near 0.0 or negative
- ❌ Inverted relationships (higher input → lower output when should increase)
- ❌ Huge error in Fluent validation (>50% error)
- ❌ Training R² = 1.0 (overfitting or no variance)

---

## Additional Notes

### Simulation Runtime:
- **Cannot rerun simulations:** ~1 week for 4500 cases
- **Must work with existing data:** 4500 simulations already completed
- **Dataset is valuable:** Preserve all simulation outputs

### Data Quality Issues (Minor):
1. **Pressure drop units:** Mislabeled as "K" instead of "Pa" in some outputs
   - Not a critical issue for training
   - Models learn the values regardless of unit labels
2. **chip2_tmax placeholder:** Some sims have -1e20 value
   - Can be filtered during training if needed
   - Doesn't affect other outputs

---

## Contact & Resources

- **Plan Document:** [bright-kindling-sunset.md](C:\Users\danny\.claude\plans\bright-kindling-sunset.md)
- **Main README:** [README.md](README.md)
- **DOE Setup Code:** [modules/doe_setup.py](modules/doe_setup.py) (Line 490)
- **Simulation Runner:** [modules/simulation_runner.py](modules/simulation_runner.py) (Line 118)
- **Trainer Code:** [modules/multi_model_trainer.py](modules/multi_model_trainer.py)

---

**Last Updated:** December 2, 2025
**Status:** Multi-model system implemented ✓ | Ready for retraining and testing
