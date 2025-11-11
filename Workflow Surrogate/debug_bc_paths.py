"""
Debug script to compare BC parameter paths between Scalar Surrogate and Workflow Surrogate
"""

print("="*70)
print("BOUNDARY CONDITION PATH COMPARISON")
print("="*70)

print("\n1. SCALAR SURROGATE (WORKING):")
print("   Path used: cold_inlet.momentum.velocity.value")
print("   Code: cold_inlet.momentum.velocity.value = coldVel")

print("\n2. WORKFLOW SURROGATE (NOT WORKING):")
print("   Path detected: momentum.velocity_magnitude")
print("   Stored in model_setup.json as: 'momentum.velocity_magnitude'")
print("   Applied as: bc_obj.momentum.velocity_magnitude.value = value")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)

print("\nPotential Issues:")
print("  1. Path mismatch: 'velocity' vs 'velocity_magnitude'")
print("  2. The Scalar code uses 'velocity.value' directly")
print("  3. The Workflow code tries 'velocity_magnitude.value'")

print("\nFluent API has multiple ways to set velocity:")
print("  - Simple: bc.momentum.velocity.value (magnitude)")
print("  - Detailed: bc.momentum.velocity_magnitude.value")
print("  - Components: bc.momentum.velocity.x.value, y, z")

print("\n" + "="*70)
print("RECOMMENDED FIX:")
print("="*70)
print("\nThe issue is likely that 'velocity_magnitude' is being detected")
print("during DOE setup, but at runtime it doesn't exist or behaves differently.")
print("\nTwo possible solutions:")
print("  A. Force DOE setup to use 'momentum.velocity' instead of 'momentum.velocity_magnitude'")
print("  B. Add special handling in apply_boundary_conditions for velocity fields")

print("\n" + "="*70)
print("TO TEST:")
print("="*70)
print("\n1. Launch Fluent with the elbow case")
print("2. Get a velocity_inlet BC object")
print("3. Check what attributes it has:")
print("   - dir(cold_inlet.momentum)")
print("   - hasattr(cold_inlet.momentum, 'velocity')")
print("   - hasattr(cold_inlet.momentum, 'velocity_magnitude')")
print("4. Try setting both ways and see which works")
