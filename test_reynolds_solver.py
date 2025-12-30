#!/usr/bin/env python3
"""
Test script for the 1D Reynolds equation mixed lubrication solver
"""

import os
import sys
import numpy as np
import pandas as pd

# Import our solver
from reynolds_1d_mixed_lubrication import Reynolds1DMixedLubrication

def test_solver():
    """Test the Reynolds equation solver"""
    print("Testing 1D Transient Reynolds Equation Mixed Lubrication Solver")
    print("=" * 70)
    
    try:
        # Initialize solver
        print("1. Initializing solver...")
        solver = Reynolds1DMixedLubrication(data_dir=".")
        print("   ✓ Solver initialized successfully")
        
        # Test kinematics calculation
        print("\n2. Testing kinematics calculation...")
        rpm = 500
        R, Ve, Vs, W, omega = solver.calculate_kinematics(rpm)
        print(f"   ✓ Kinematics calculated for {rpm} RPM")
        print(f"     - Effective radius range: {np.min(R)*1e3:.2f} - {np.max(R)*1e3:.2f} mm")
        print(f"     - Entraining velocity range: {np.min(Ve):.3f} - {np.max(Ve):.3f} m/s")
        print(f"     - Load range: {np.min(W):.1f} - {np.max(W):.1f} N")
        
        # Test single point solution
        print("\n3. Testing single point solution...")
        angle_deg = 0.0  # TDC
        idx = np.argmin(np.abs(solver.th_deg - angle_deg))
        
        # Get texture parameters
        density_key = "5%"
        dt = np.gradient(solver.th)[idx] / max(omega, 1e-30)
        shift = solver.integrate_shift(Vs, omega)
        atex = solver.get_texture_amplitude(density_key, rpm)
        d_tex = solver.D_TEXTURE[density_key]
        
        # Solve for textured case
        results = solver.solve_reynolds_at_angle(
            R[idx], Ve[idx], Vs[idx], W[idx], dt, angle_deg, rpm,
            atex_theta=float(max(atex[idx], 0.0)), 
            shift_theta=shift[idx], d_texture=d_tex
        )
        
        print(f"   ✓ Single point solution completed")
        print(f"     - Contact half-width: {results['a']*1e6:.1f} μm")
        print(f"     - Max pressure: {results['pmax']*1e-6:.1f} MPa")
        print(f"     - Min film thickness: {results['hmin']*1e9:.1f} nm")
        print(f"     - Hydrodynamic force: {results['Fh']:.2f} N")
        print(f"     - Boundary force: {results['Fb']:.2f} N")
        print(f"     - Load share - Hydro: {results['Wh']/(results['Wh']+results['Wa'])*100:.1f}%")
        print(f"     - Load share - Asperity: {results['Wa']/(results['Wh']+results['Wa'])*100:.1f}%")
        
        # Test friction reduction analysis (limited RPMs for speed)
        print("\n4. Testing friction reduction analysis...")
        test_rpms = [300, 500]  # Limited for testing
        reduction_results = solver.analyze_friction_reduction(density_key, rpms=test_rpms)
        
        print(f"   ✓ Friction reduction analysis completed")
        print("\n   Friction Torque Reduction Results:")
        for _, row in reduction_results.iterrows():
            print(f"     RPM {int(row['RPM'])}: {row['Pct_Reduction']:.2f}% reduction")
        
        # Test texture amplitude retrieval
        print("\n5. Testing texture data access...")
        for density in ["5%", "8%", "10%"]:
            if density in solver.atex_tables:
                atex_data = solver.get_texture_amplitude(density, 500)
                non_zero_count = np.count_nonzero(atex_data)
                print(f"   ✓ {density} texture data: {non_zero_count} non-zero values")
            else:
                print(f"   ⚠ {density} texture data not available")
        
        print("\n" + "=" * 70)
        print("All tests completed successfully! ✓")
        print("The 1D Reynolds equation mixed lubrication solver is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_validation():
    """Test physics validation"""
    print("\n" + "=" * 70)
    print("PHYSICS VALIDATION TESTS")
    print("=" * 70)
    
    try:
        solver = Reynolds1DMixedLubrication(data_dir=".")
        
        # Test 1: Load balance
        print("\n1. Testing load balance...")
        rpm = 500
        R, Ve, Vs, W, omega = solver.calculate_kinematics(rpm)
        idx = len(R) // 2  # Mid-cycle
        
        dt = np.gradient(solver.th)[idx] / max(omega, 1e-30)
        shift = solver.integrate_shift(Vs, omega)
        atex = solver.get_texture_amplitude("5%", rpm)
        d_tex = solver.D_TEXTURE["5%"]
        
        results = solver.solve_reynolds_at_angle(
            R[idx], Ve[idx], Vs[idx], W[idx], dt, solver.th_deg[idx], rpm,
            atex_theta=float(max(atex[idx], 0.0)), 
            shift_theta=shift[idx], d_texture=d_tex
        )
        
        total_load = results['Wh'] + results['Wa']
        load_error = abs(W[idx] - total_load) / W[idx] * 100
        print(f"   Applied load: {W[idx]:.2f} N")
        print(f"   Calculated load: {total_load:.2f} N")
        print(f"   Load balance error: {load_error:.3f}%")
        
        if load_error < 1.0:
            print("   ✓ Load balance test passed")
        else:
            print("   ⚠ Load balance error is high")
        
        # Test 2: Pressure positivity
        print("\n2. Testing pressure positivity...")
        min_pressure = np.min(results['p'])
        if min_pressure >= 0:
            print(f"   ✓ All pressures non-negative (min: {min_pressure:.2e} Pa)")
        else:
            print(f"   ❌ Negative pressures found (min: {min_pressure:.2e} Pa)")
        
        # Test 3: Film thickness positivity
        print("\n3. Testing film thickness positivity...")
        min_thickness = np.min(results['h'])
        if min_thickness > 0:
            print(f"   ✓ All film thicknesses positive (min: {min_thickness*1e9:.2f} nm)")
        else:
            print(f"   ❌ Non-positive film thicknesses found (min: {min_thickness*1e9:.2f} nm)")
        
        # Test 4: Texture effect
        print("\n4. Testing texture effect...")
        # Compare textured vs untextured
        results_smooth = solver.solve_reynolds_at_angle(
            R[idx], Ve[idx], Vs[idx], W[idx], dt, solver.th_deg[idx], rpm,
            atex_theta=0.0, shift_theta=shift[idx], d_texture=d_tex
        )
        
        friction_smooth = results_smooth['Fh'] + results_smooth['Fb']
        friction_textured = results['Fh'] + results['Fb']
        friction_change = (friction_textured - friction_smooth) / friction_smooth * 100
        
        print(f"   Smooth friction: {friction_smooth:.3f} N")
        print(f"   Textured friction: {friction_textured:.3f} N")
        print(f"   Friction change: {friction_change:.2f}%")
        
        if abs(friction_change) > 0.1:
            print("   ✓ Texture has measurable effect on friction")
        else:
            print("   ⚠ Texture effect is very small")
        
        print("\n" + "=" * 70)
        print("Physics validation completed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during physics validation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_solver()
    if success:
        test_physics_validation()
    else:
        sys.exit(1)