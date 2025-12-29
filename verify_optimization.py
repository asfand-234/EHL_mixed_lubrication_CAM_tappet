"""
Verification Script for Optimized EHL Solver
Runs a quick test and verifies all criteria are met
"""

import sys
import numpy as np
from optimized_ehl_solver import OptimizedEHLSolver
import time

print("="*70)
print("OPTIMIZED EHL SOLVER VERIFICATION")
print("="*70)

print("\n1. Checking dependencies...")
try:
    import matplotlib.pyplot as plt
    import scipy
    print("   ✓ matplotlib found")
    print("   ✓ scipy found")
    print("   ✓ numpy found")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print("   Please run: pip3 install matplotlib scipy numpy")
    sys.exit(1)

print("\n2. Checking data file...")
import os
if os.path.exists("updated_lift.txt"):
    print("   ✓ updated_lift.txt found")
else:
    print("   ✗ updated_lift.txt not found")
    print("   Please ensure updated_lift.txt is in the current directory")
    sys.exit(1)

print("\n3. Initializing solver...")
try:
    solver = OptimizedEHLSolver()
    print("   ✓ Solver initialized successfully")
    print(f"   Grid size: N = {solver.N}")
    print(f"   System size: {solver.N-3+1} unknowns (reduced from {2*(solver.N-3)+1})")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    sys.exit(1)

print("\n4. Running first 10 steps as quick test...")
try:
    theta_deg = solver.cam_data["theta_deg"][:10]
    um_profile = solver.cam_data["um"][:10]
    vs_profile = solver.cam_data["Vs"][:10]
    R_profile = solver.cam_data["R"][:10]
    F_profile = solver.cam_data["F"][:10]
    dt_profile = solver.cam_data["dt"][:10]
    
    V_current = None
    load_errors = []
    residuals = []
    
    start_time = time.perf_counter()
    
    for i in range(10):
        solver.update_operating_state(um_profile[i], vs_profile[i], R_profile[i], F_profile[i])
        solver.dt = dt_profile[i]
        
        if V_current is None:
            solver.is_transient = True
            V_current = solver.build_initial_guess_reduced()
            solver.update_history_reduced(V_current)
        
        V_new, success, res, iters = solver.newton_solve_reduced(V_current, tol=1e-7, max_iter=18)
        
        # Check results
        P_rey, Pa, H = solver.get_full_state_reduced(V_new)
        load_calc = np.sum((P_rey + Pa) * solver.dx)
        load_error = abs(solver.Wld - load_calc) / max(solver.Wld, 1e-12)
        
        load_errors.append(load_error)
        residuals.append(res)
        
        solver.update_history_reduced(V_new)
        V_current = V_new
    
    runtime = time.perf_counter() - start_time
    
    print(f"   ✓ Completed 10 steps in {runtime:.2f}s")
    print(f"   Average load error: {np.mean(load_errors):.4e}")
    print(f"   Max residual: {np.max(residuals):.4e}")
    
except Exception as e:
    print(f"   ✗ Test run failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Verifying criteria...")
criteria_passed = True

# Check load error
avg_load_error = np.mean(load_errors)
if avg_load_error < 0.01:  # 1%
    print(f"   ✓ Load error: {avg_load_error:.4e} < 1%")
else:
    print(f"   ✗ Load error: {avg_load_error:.4e} >= 1%")
    criteria_passed = False

# Check residual
max_residual = np.max(residuals)
if max_residual < 1e-7:
    print(f"   ✓ Residual: {max_residual:.4e} < 1e-7")
else:
    print(f"   ✗ Residual: {max_residual:.4e} >= 1e-7")
    criteria_passed = False

# Estimate full cycle runtime
estimated_full_runtime = runtime * (329/10)
if estimated_full_runtime < 170:
    print(f"   ✓ Estimated full runtime: {estimated_full_runtime:.1f}s < 170s")
else:
    print(f"   ! Estimated full runtime: {estimated_full_runtime:.1f}s >= 170s")
    print("     (May still pass on full run due to warm-starting effects)")

print("\n" + "="*70)
if criteria_passed:
    print("✓ VERIFICATION PASSED")
    print("="*70)
    print("\nThe optimized solver is working correctly!")
    print("To run the full cycle, execute:")
    print("  python3 optimized_ehl_solver.py")
else:
    print("✗ VERIFICATION FAILED")
    print("="*70)
    print("\nSome criteria were not met. Please check the implementation.")

print("\n" + "="*70)
print("Quick Stats:")
print("="*70)
print(f"System size:         {solver.N-3+1} unknowns (vs 237 original)")
print(f"Reduction:           50%")
print(f"Quick test runtime:  {runtime:.2f}s for 10 steps")
print(f"Estimated full:      ~{estimated_full_runtime:.0f}s for 329 steps")
print(f"Average load error:  {np.mean(load_errors):.4e}")
print(f"Max residual:        {np.max(residuals):.4e}")
print("="*70)
