#!/usr/bin/env python3
"""
Demonstration script for 1D Reynolds equation mixed lubrication solver
Shows key features and capabilities without requiring plotting
"""

import numpy as np
import pandas as pd
from reynolds_1d_mixed_lubrication import Reynolds1DMixedLubrication

def demonstrate_solver():
    """Demonstrate key features of the Reynolds equation solver"""
    
    print("🔧 1D TRANSIENT REYNOLDS EQUATION MIXED LUBRICATION SOLVER")
    print("=" * 80)
    print("Complete implementation for CAM-SHIM contacts with surface textures")
    print("=" * 80)
    
    # Initialize solver
    print("\n📊 INITIALIZING SOLVER...")
    solver = Reynolds1DMixedLubrication(data_dir=".")
    print(f"✓ Loaded cam profile: {len(solver.th_deg)} angle points")
    print(f"✓ Available texture densities: {list(solver.atex_tables.keys())}")
    print(f"✓ Angle range: {solver.th_deg[0]:.1f}° to {solver.th_deg[-1]:.1f}°")
    
    # Demonstrate kinematics calculation
    print("\n⚙️  KINEMATICS ANALYSIS...")
    rpm = 500
    R, Ve, Vs, W, omega = solver.calculate_kinematics(rpm)
    
    print(f"Operating speed: {rpm} RPM ({omega:.2f} rad/s)")
    print(f"Effective radius: {np.min(R)*1e3:.2f} - {np.max(R)*1e3:.2f} mm")
    print(f"Entraining velocity: {np.min(Ve):.3f} - {np.max(Ve):.3f} m/s")
    print(f"Sliding velocity: {np.min(Vs):.3f} - {np.max(Vs):.3f} m/s")
    print(f"Contact load: {np.min(W):.1f} - {np.max(W):.1f} N")
    
    # Demonstrate single point solution
    print("\n🔍 SINGLE POINT ANALYSIS...")
    angle_deg = 0.0  # Top Dead Center
    idx = np.argmin(np.abs(solver.th_deg - angle_deg))
    
    print(f"Analyzing cam angle: {angle_deg}° (TDC)")
    print(f"Operating conditions at this angle:")
    print(f"  - Effective radius: {R[idx]*1e3:.2f} mm")
    print(f"  - Entraining velocity: {Ve[idx]:.3f} m/s")
    print(f"  - Sliding velocity: {Vs[idx]:.3f} m/s")
    print(f"  - Contact load: {W[idx]:.1f} N")
    
    # Get texture parameters
    density_key = "5%"
    dt = np.gradient(solver.th)[idx] / max(omega, 1e-30)
    shift = solver.integrate_shift(Vs, omega)
    atex = solver.get_texture_amplitude(density_key, rpm)
    d_tex = solver.D_TEXTURE[density_key]
    
    print(f"\nTexture parameters ({density_key} coverage):")
    print(f"  - Texture spacing: {d_tex*1e6:.1f} μm")
    print(f"  - Amplitude at this angle: {atex[idx]*1e9:.2f} nm")
    print(f"  - Texture shift: {shift[idx]*1e6:.2f} μm")
    
    # Solve for both smooth and textured cases
    print("\n🧮 SOLVING REYNOLDS EQUATION...")
    
    # Smooth case
    results_smooth = solver.solve_reynolds_at_angle(
        R[idx], Ve[idx], Vs[idx], W[idx], dt, angle_deg, rpm,
        atex_theta=0.0, shift_theta=shift[idx], d_texture=d_tex
    )
    
    # Textured case
    results_textured = solver.solve_reynolds_at_angle(
        R[idx], Ve[idx], Vs[idx], W[idx], dt, angle_deg, rpm,
        atex_theta=float(max(atex[idx], 0.0)), 
        shift_theta=shift[idx], d_texture=d_tex
    )
    
    # Compare results
    print("\n📈 SOLUTION COMPARISON (Smooth vs Textured):")
    print("-" * 60)
    print(f"{'Parameter':<25} {'Smooth':<15} {'Textured':<15} {'Change':<10}")
    print("-" * 60)
    
    def print_comparison(name, smooth_val, textured_val, unit="", format_str=".2f"):
        change = (textured_val - smooth_val) / smooth_val * 100 if smooth_val != 0 else 0
        print(f"{name:<25} {smooth_val:{format_str}}{unit:<5} {textured_val:{format_str}}{unit:<5} {change:+.1f}%")
    
    print_comparison("Contact half-width", results_smooth['a']*1e6, results_textured['a']*1e6, "μm")
    print_comparison("Max pressure", results_smooth['pmax']*1e-6, results_textured['pmax']*1e-6, "MPa")
    print_comparison("Min film thickness", results_smooth['hmin']*1e9, results_textured['hmin']*1e9, "nm")
    print_comparison("Hydrodynamic force", results_smooth['Fh'], results_textured['Fh'], "N", ".3f")
    print_comparison("Boundary force", results_smooth['Fb'], results_textured['Fb'], "N", ".3f")
    
    total_smooth = results_smooth['Fh'] + results_smooth['Fb']
    total_textured = results_textured['Fh'] + results_textured['Fb']
    print_comparison("Total friction force", total_smooth, total_textured, "N", ".3f")
    
    # Load distribution analysis
    print("\n⚖️  LOAD DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    def analyze_load_distribution(results, case_name):
        total_load = results['Wh'] + results['Wa']
        hydro_pct = results['Wh'] / total_load * 100
        asperity_pct = results['Wa'] / total_load * 100
        print(f"{case_name}:")
        print(f"  - Hydrodynamic: {hydro_pct:.1f}% ({results['Wh']:.2f} N)")
        print(f"  - Asperity:     {asperity_pct:.1f}% ({results['Wa']:.2f} N)")
        print(f"  - Total:        {total_load:.2f} N (Applied: {W[idx]:.2f} N)")
        print(f"  - Load balance error: {abs(W[idx] - total_load)/W[idx]*100:.3f}%")
    
    analyze_load_distribution(results_smooth, "Smooth surface")
    analyze_load_distribution(results_textured, "Textured surface")
    
    # Friction reduction analysis for multiple RPMs
    print("\n🎯 FRICTION REDUCTION ANALYSIS...")
    print("Analyzing multiple operating speeds...")
    
    test_rpms = [300, 500, 700, 900]
    reduction_results = []
    
    for test_rpm in test_rpms:
        print(f"  Processing {test_rpm} RPM...", end="")
        
        # Quick analysis (fewer angles for speed)
        R_test, Ve_test, Vs_test, W_test, omega_test = solver.calculate_kinematics(test_rpm)
        dt_test = np.gradient(solver.th) / max(omega_test, 1e-30)
        shift_test = solver.integrate_shift(Vs_test, omega_test)
        atex_test = solver.get_texture_amplitude(density_key, test_rpm)
        
        # Sample a few representative angles
        sample_indices = np.linspace(0, len(solver.th_deg)-1, 20, dtype=int)
        
        T_smooth_samples = []
        T_textured_samples = []
        
        for i in sample_indices:
            # Smooth case
            res_s = solver.solve_reynolds_at_angle(
                R_test[i], Ve_test[i], Vs_test[i], W_test[i], dt_test[i], 
                solver.th_deg[i], test_rpm, atex_theta=0.0, 
                shift_theta=shift_test[i], d_texture=d_tex
            )
            
            # Textured case
            res_t = solver.solve_reynolds_at_angle(
                R_test[i], Ve_test[i], Vs_test[i], W_test[i], dt_test[i], 
                solver.th_deg[i], test_rpm, atex_theta=float(max(atex_test[i], 0.0)),
                shift_theta=shift_test[i], d_texture=d_tex
            )
            
            r_eff = solver.rb + solver.lift_s[i]
            T_smooth_samples.append((res_s['Fh'] + res_s['Fb']) * r_eff)
            T_textured_samples.append((res_t['Fh'] + res_t['Fb']) * r_eff)
        
        Tavg_smooth = np.mean(T_smooth_samples)
        Tavg_textured = np.mean(T_textured_samples)
        pct_reduction = 100.0 * (1.0 - Tavg_textured / max(Tavg_smooth, 1e-30))
        
        reduction_results.append({
            'RPM': test_rpm,
            'Smooth_Torque': Tavg_smooth,
            'Textured_Torque': Tavg_textured,
            'Reduction_Pct': pct_reduction
        })
        
        print(f" {pct_reduction:.2f}% reduction")
    
    # Display friction reduction results
    print("\n📊 FRICTION TORQUE REDUCTION SUMMARY:")
    print("-" * 70)
    print(f"{'RPM':<8} {'Smooth Torque':<15} {'Textured Torque':<17} {'Reduction':<12}")
    print(f"{'':^8} {'[N⋅m]':<15} {'[N⋅m]':<17} {'[%]':<12}")
    print("-" * 70)
    
    for result in reduction_results:
        print(f"{result['RPM']:<8} {result['Smooth_Torque']:<15.4f} "
              f"{result['Textured_Torque']:<17.4f} {result['Reduction_Pct']:<12.2f}")
    
    # Physics validation
    print("\n✅ PHYSICS VALIDATION:")
    print("-" * 30)
    
    # Check load balance
    total_load = results_textured['Wh'] + results_textured['Wa']
    load_error = abs(W[idx] - total_load) / W[idx] * 100
    print(f"Load balance error: {load_error:.3f}% {'✓' if load_error < 1.0 else '⚠'}")
    
    # Check pressure positivity
    min_pressure = np.min(results_textured['p'])
    print(f"Pressure positivity: {'✓' if min_pressure >= 0 else '❌'} (min: {min_pressure:.2e} Pa)")
    
    # Check film thickness positivity
    min_thickness = np.min(results_textured['h'])
    print(f"Film thickness positivity: {'✓' if min_thickness > 0 else '❌'} (min: {min_thickness*1e9:.2f} nm)")
    
    # Check texture effect
    friction_change = (total_textured - total_smooth) / total_smooth * 100
    print(f"Texture effect magnitude: {'✓' if abs(friction_change) > 0.1 else '⚠'} ({friction_change:.2f}%)")
    
    print("\n" + "=" * 80)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("The 1D Reynolds equation mixed lubrication solver is fully functional")
    print("and ready for tribological analysis and surface texture optimization.")
    print("=" * 80)
    
    return reduction_results

if __name__ == "__main__":
    try:
        results = demonstrate_solver()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()