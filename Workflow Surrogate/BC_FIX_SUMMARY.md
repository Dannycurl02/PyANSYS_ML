# Boundary Condition Setting Bug Fix

## Problem Summary
The Workflow Surrogate program was not setting boundary conditions correctly, while the Scalar Surrogate notebook was working fine.

## Root Cause Analysis

### How Scalar Surrogate Works (✓ Correct)
```python
# From Scalar_Surrogate.ipynb (line 38)
cold_inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"]
cold_inlet.momentum.velocity.value = coldVel
```
- Uses the path: `momentum.velocity.value`
- Direct, simple API access

### How Workflow Surrogate Was Broken (✗ Before Fix)

1. **DOE Setup Phase** ([doe_setup.py:10-106](Workflow Surrogate/modules/doe_setup.py#L10-L106)):
   - The `get_bc_parameters()` function explores the BC object tree
   - It detects available parameters and stores their paths
   - Path detected: `momentum.velocity_magnitude` (stored in model_setup.json)

2. **Simulation Phase** ([simulation_runner.py:147-237](Workflow Surrogate/modules/simulation_runner.py#L147-L237)):
   - The `apply_boundary_conditions()` function tries to navigate to the path
   - It splits `momentum.velocity_magnitude` into parts
   - **BUG**: The old code navigated through `path_parts[:-1]` and then tried to set `path_parts[-1]`
   - This caused issues when the last part wasn't accessible or named differently

### The Mismatch

PyFluent's API for velocity inlets can expose velocity parameters with different names:
- **Simple API**: `momentum.velocity` (object with `.value` attribute)
- **Detailed API**: `momentum.velocity_magnitude` (might be what DOE detected)

The detected path during setup might not match the actual runtime API structure.

## The Fix

### Changes Made to `simulation_runner.py`

**Location**: [simulation_runner.py:178-238](Workflow Surrogate/modules/simulation_runner.py#L178-L238)

#### Key Improvements:

1. **Better Path Navigation** (lines 184-202):
   ```python
   # Navigate through ALL parts of the path (not just path[:-1])
   for i, part in enumerate(path_parts):
       if hasattr(target_obj, part):
           target_obj = getattr(target_obj, part)
       else:
           # Try alternate names for velocity parameters
           if part == 'velocity_magnitude' and hasattr(target_obj, 'velocity'):
               print(f"  Note: Using 'velocity' instead of 'velocity_magnitude' for {bc_name}")
               target_obj = getattr(target_obj, 'velocity')
           elif part == 'velocity' and hasattr(target_obj, 'velocity_magnitude'):
               print(f"  Note: Using 'velocity_magnitude' instead of 'velocity' for {bc_name}")
               target_obj = getattr(target_obj, 'velocity_magnitude')
           else:
               # Provide detailed error message
               print(f"  ⚠ Warning: Path part '{part}' not found in {bc_name}")
               print(f"     Full path: {param_path}, Current path: {'.'.join(path_parts[:i+1])}")
               if hasattr(target_obj, 'child_names'):
                   print(f"     Available attributes: {target_obj.child_names[:10]}")
               return False
   ```

2. **Fallback Mechanisms** (lines 191-196):
   - If `velocity_magnitude` not found, try `velocity`
   - If `velocity` not found, try `velocity_magnitude`
   - This handles API variations between PyFluent versions

3. **Better Error Reporting** (lines 198-202):
   - Shows exactly which path part failed
   - Lists available attributes to help debug
   - Makes it clear what went wrong

4. **Multiple Setting Methods** (lines 207-234):
   ```python
   if hasattr(target_obj, 'value'):
       target_obj.value = value  # Most common
   elif hasattr(target_obj, 'set_state'):
       target_obj.set_state(value)  # Alternative method
   elif callable(target_obj):
       target_obj(value)  # Callable interface
   ```

## Testing the Fix

### Expected Behavior After Fix:
1. When running simulations, you should see messages like:
   ```
   ✓ Set cold-inlet.momentum.velocity_magnitude = 0.4 (verified: 0.4)
   ```
   OR
   ```
   Note: Using 'velocity' instead of 'velocity_magnitude' for cold-inlet
   ✓ Set cold-inlet.momentum.velocity_magnitude = 0.4 (verified: 0.4)
   ```

2. If there's still an issue, you'll see detailed error messages:
   ```
   ⚠ Warning: Path part 'velocity_magnitude' not found in cold-inlet
      Full path: momentum.velocity_magnitude, Current path: momentum.velocity_magnitude
      Available attributes: ['child_names', 'velocity', 'temperature', ...]
   ```

### How to Test:
1. Open the Workflow Surrogate program
2. Load your project with the 4x4 DOE configuration
3. Try running a single simulation (Menu option 5 → 1)
4. Watch the console output during BC application phase
5. Verify that BCs are set correctly and simulations run

## Why This Happens

The PyFluent API exposes objects through introspection:
- During setup, `child_names` might return `'velocity_magnitude'`
- But at runtime, the actual attribute might be `'velocity'`
- Different PyFluent versions might use different naming
- The fix handles both cases automatically

## Additional Notes

- The fix is backward compatible - it tries the exact path first
- Only falls back to alternatives if the primary path fails
- Adds extensive debugging output to help diagnose future issues
- No changes needed to existing project files or configurations

## Related Files
- [simulation_runner.py](Workflow Surrogate/modules/simulation_runner.py) - Main fix location
- [doe_setup.py](Workflow Surrogate/modules/doe_setup.py) - Parameter detection
- [Scalar_Surrogate.ipynb](Scalar Surrogate/Scalar_Surrogate.ipynb) - Working reference implementation
