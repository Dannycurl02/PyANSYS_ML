# Side-by-Side Comparison: Scalar vs Workflow Surrogate

## BC Setting Code Comparison

### Scalar Surrogate (Working) ✓
```python
# Direct, simple approach
cold_inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"]
cold_inlet.momentum.velocity.value = coldVel

hot_inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["hot-inlet"]
hot_inlet.momentum.velocity.value = hotVel
```

### Workflow Surrogate (Now Fixed) ✓
```python
# Dynamic path-based approach with fallback
bc_obj = boundary_conditions.velocity_inlet["cold-inlet"]
param_path = "momentum.velocity_magnitude"  # From DOE config

# Navigate: momentum → velocity_magnitude (or velocity if not found)
path_parts = param_path.split('.')  # ['momentum', 'velocity_magnitude']
target_obj = bc_obj

for part in path_parts:
    if hasattr(target_obj, part):
        target_obj = getattr(target_obj, part)
    elif part == 'velocity_magnitude' and hasattr(target_obj, 'velocity'):
        # FALLBACK: Use 'velocity' if 'velocity_magnitude' not found
        target_obj = getattr(target_obj, 'velocity')
    else:
        # Error handling

# Set the value
target_obj.value = coldVel
```

## The Key Difference

| Aspect | Scalar Surrogate | Workflow Surrogate |
|--------|------------------|-------------------|
| **Approach** | Hardcoded paths | Dynamic paths from DOE config |
| **Path Used** | `momentum.velocity.value` | `momentum.velocity_magnitude.value` |
| **Flexibility** | Fixed (only velocity magnitude) | Configurable (any BC parameter) |
| **API Variation Handling** | Not needed (path is hardcoded) | **Now includes fallback logic** |

## What Was Wrong

The Workflow Surrogate's DOE setup would detect `momentum.velocity_magnitude` as the parameter path, but when trying to apply it during simulation:

**Before Fix (❌ Broken):**
```python
# Would fail if 'velocity_magnitude' doesn't exist
path_parts = ['momentum', 'velocity_magnitude']
target_obj = bc_obj
for part in path_parts[:-1]:  # Only navigate to 'momentum'
    target_obj = getattr(target_obj, part)

# Try to set path_parts[-1] as attribute
final_param = 'velocity_magnitude'
param_obj = getattr(target_obj, final_param)  # FAILS if doesn't exist
param_obj.value = value
```

**After Fix (✓ Working):**
```python
# Tries 'velocity_magnitude', falls back to 'velocity'
path_parts = ['momentum', 'velocity_magnitude']
target_obj = bc_obj
for part in path_parts:  # Navigate through ALL parts
    if hasattr(target_obj, part):
        target_obj = getattr(target_obj, part)
    elif part == 'velocity_magnitude' and hasattr(target_obj, 'velocity'):
        target_obj = getattr(target_obj, 'velocity')  # FALLBACK ✓

# Now target_obj IS the parameter object
target_obj.value = value  # Works!
```

## Why It Matters

PyFluent's API structure can vary:
- Different versions might expose parameters differently
- Object introspection (`child_names`) might return different names than actual attributes
- `velocity` vs `velocity_magnitude` is a common variation

The fix makes Workflow Surrogate robust to these API variations while maintaining its flexibility advantage over the hardcoded Scalar Surrogate approach.
