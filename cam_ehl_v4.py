#!/usr/bin/env python3
"""
1D Thermal Transient Mixed Lubrication - Cam-Follower EHL
Version 4: Robust bisection load balance + stable Reynolds solver

Key improvements:
1. Bisection method for load balance (guaranteed convergence)
2. Proper h0 bounds based on EHL physics
3. Stable Reynolds solver with proper relaxation
"""

import os
import numpy as np
from pathlib import Path
import pandas as pd
import time

TEMP_C = 90
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Load cam data
def load_cam():
    path = os.path.join(HERE, "updated_lift.txt")
    cam = pd.read_csv(path, sep=r"\s+", engine="python", header=None, 
                      names=["angle", "lift"], usecols=[0,1])
    return cam["angle"].values, cam["lift"].values

TH_DEG, LIFT = load_cam()
TH_RAD = np.deg2rad(TH_DEG)
DLIFT = np.gradient(LIFT, TH_RAD)
D2LIFT = np.gradient(DLIFT, TH_RAD)

# Fixed material properties
rb = 18.5e-3
k_spring = 7130.0
delta = 1.77e-3
Meq = 0.05733
L_width = 7.2e-3
E_star = 217e9
eta0 = 0.01381
alpha0 = 16e-9
rho0 = 858.44
sigma = 0.2e-6
beta_a = sigma / 0.001
eta_R = 0.055 / (sigma * beta_a)
gamma_th = 4.5e-4

# Kinematics
def kinematics(rpm):
    w = 2 * np.pi * rpm / 60
    R = np.maximum(rb + LIFT + D2LIFT, 1e-7)
    Vc = (rb + LIFT + D2LIFT) * w
    Vf = D2LIFT * w
    Ve = 0.5 * (Vc + Vf)
    Vs = Vc - Vf
    W = k_spring * (LIFT + delta) + Meq * w**2 * D2LIFT
    return R, Ve, Vs, W, w

def a_hertz(W, R):
    return np.sqrt(8 * max(W, 1e-6) * max(R, 1e-12) / (np.pi * E_star * L_width))

def ph_hertz(W, a):
    return 2 * max(W, 1e-6) / (np.pi * max(a, 1e-12) * L_width)

def hc_dowson(R, W, Ve):
    R, W = max(R, 1e-7), max(W, 1e-6)
    U = eta0 * abs(Ve) / (E_star * R)
    G = alpha0 * E_star
    Wp = W / (E_star * L_width * R)
    return 2.65 * (U**0.7) * (G**0.54) * (Wp**(-0.13)) * R

# Asperity pressure (Greenwood-Tripp)
def p_asperity(h):
    h = np.atleast_1d(np.maximum(h, 1e-12))
    lam = h / sigma
    F52 = np.zeros_like(lam)
    mask = lam < 4
    F52[mask] = 4.4086e-5 * (4 - lam[mask])**6.804
    K = (16*np.sqrt(2)/15) * np.pi * (eta_R * beta_a * sigma)**2 * np.sqrt(sigma/beta_a)
    return K * E_star * F52

# Viscosity
def viscosity(p):
    p = np.maximum(np.atleast_1d(p), 0)
    return eta0 * np.exp(np.clip(alpha0 * p, 0, 50))

# Density
def density(p):
    p = np.maximum(np.atleast_1d(p), 0)
    return rho0 * (0.59e9 + 1.34*p) / (0.59e9 + p)

# Flow factor
def phi_x(h):
    H = np.maximum(h / sigma, 0.1)
    return np.maximum(1 - 0.9 * np.exp(-0.56 * H), 0.01)

# Elastic deflection
def deflection(x, p, dx):
    N = len(x)
    b = dx / 2
    grid = np.arange(-N+1, N) * dx
    
    def xlnx(v):
        r = np.zeros_like(v)
        m = np.abs(v) > 1e-12
        r[m] = v[m] * np.log(np.abs(v[m]))
        return r
    
    kernel = xlnx(grid + b) - xlnx(grid - b) - 2*b
    conv = np.convolve(p, kernel, 'full')[N-1:2*N-1]
    u = -conv * 2 / (np.pi * E_star)
    return u - u[N//2]

# Reynolds solver (Gauss-Seidel)
def solve_reynolds(x, h, Ve, p_init, max_iter=200, relax=0.3):
    """
    Solve: d/dx[D dp/dx] = Ve * d(rho*h)/dx
    where D = rho * h^3 * phi_x / (12 * eta)
    """
    N = len(x)
    dx = x[1] - x[0]
    p = p_init.copy()
    
    for it in range(max_iter):
        # Update properties at current p
        eta = viscosity(p)
        rho = density(p)
        phi = phi_x(h)
        
        # Diffusion coefficient
        D = rho * h**3 * phi / (12 * eta)
        D = np.maximum(D, 1e-40)
        
        # Face-centered D (harmonic mean)
        D_face = np.zeros(N-1)
        for i in range(N-1):
            D_face[i] = 2 * D[i] * D[i+1] / (D[i] + D[i+1] + 1e-40)
        
        # Couette source: Ve * d(rho*h)/dx
        rho_h = rho * h
        
        p_old = p.copy()
        max_dp = 0
        
        # Update interior nodes
        for i in range(1, N-1):
            a_W = D_face[i-1] / dx
            a_E = D_face[i] / dx
            a_P = a_W + a_E
            
            if a_P < 1e-40:
                continue
            
            # Upwind Couette
            if Ve >= 0:
                source = Ve * (rho_h[i] - rho_h[i-1])
            else:
                source = Ve * (rho_h[i+1] - rho_h[i])
            
            p_new = (a_W * p[i-1] + a_E * p[i+1] - source) / a_P
            p_new = max(p_new, 0)  # Cavitation
            
            p[i] = (1 - relax) * p_old[i] + relax * p_new
            max_dp = max(max_dp, abs(p[i] - p_old[i]))
        
        # BCs
        p[0] = 1e5
        p[-1] = 1e5
        
        if max_dp < 1e-6 * (np.max(p) + 1e6):
            break
    
    return p

def compute_load(x, h, h0, parabola, Ve, p_init, dx):
    """
    Compute total load for given h0.
    """
    # Film thickness
    defl = deflection(x, p_init, dx)
    h[:] = np.maximum(h0 + parabola + defl, 1e-12)
    
    # Iterate to couple p and h
    p = p_init.copy()
    for _ in range(10):
        p = solve_reynolds(x, h, Ve, p, max_iter=50, relax=0.3)
        p_tot = p + p_asperity(h)
        defl = deflection(x, p_tot, dx)
        h[:] = np.maximum(h0 + parabola + defl, 1e-12)
    
    p_asp = p_asperity(h)
    W_h = np.sum(p) * dx * L_width
    W_a = np.sum(p_asp) * dx * L_width
    
    return W_h + W_a, p, p_asp, h

def solve_ehl_bisection(R, Ve, Vs, W_target, Nx=256, observe=False):
    """
    Solve EHL with bisection for load balance.
    """
    a = a_hertz(W_target, R)
    ph = ph_hertz(W_target, a)
    hc = hc_dowson(R, W_target, Ve)
    
    # Domain
    x = np.linspace(-4*a, 2*a, Nx)
    dx = x[1] - x[0]
    X = x / a
    parabola = x**2 / (2*R)
    h = np.zeros(Nx)
    
    # Initial pressure (Hertzian)
    p = np.zeros(Nx)
    mask = np.abs(X) <= 1
    p[mask] = ph * np.sqrt(np.maximum(1 - X[mask]**2, 0))
    p[~mask] = 1e5
    
    # Bisection bounds for h0
    # h0_min: near contact (high load)
    # h0_max: separated (low load)
    h0_min = 1e-10
    h0_max = 10 * hc  # Upper bound: 10x Dowson-Hamrock
    
    # Evaluate at bounds
    W_at_min, p_min, p_asp_min, h_min = compute_load(x, h.copy(), h0_min, parabola, Ve, p.copy(), dx)
    W_at_max, p_max, p_asp_max, h_max = compute_load(x, h.copy(), h0_max, parabola, Ve, p.copy(), dx)
    
    if observe:
        print(f"    Bounds: h0_min={h0_min*1e9:.1f}nm -> W={W_at_min:.1f}N")
        print(f"            h0_max={h0_max*1e9:.1f}nm -> W={W_at_max:.1f}N")
        print(f"    Target: W={W_target:.1f}N")
    
    # If W_at_max > W_target, need larger h0_max
    if W_at_max > W_target:
        while W_at_max > W_target and h0_max < 1e-3:
            h0_max *= 2
            W_at_max, _, _, _ = compute_load(x, h.copy(), h0_max, parabola, Ve, p.copy(), dx)
    
    # If W_at_min < W_target, need smaller h0_min (more contact)
    if W_at_min < W_target:
        # This means we need even closer contact, but h0_min is already tiny
        # Increase h0_min and find bracket
        h0_min = hc * 0.1
        h0_max = hc * 10
        W_at_min, _, _, _ = compute_load(x, h.copy(), h0_min, parabola, Ve, p.copy(), dx)
        W_at_max, _, _, _ = compute_load(x, h.copy(), h0_max, parabola, Ve, p.copy(), dx)
    
    # Bisection
    best_h0 = hc
    best_err = 1e10
    best_p = p.copy()
    best_h = h.copy()
    best_p_asp = np.zeros(Nx)
    
    for i in range(50):
        h0_mid = (h0_min + h0_max) / 2
        W_mid, p_mid, p_asp_mid, h_mid = compute_load(x, h.copy(), h0_mid, parabola, Ve, p.copy(), dx)
        
        err = (W_mid - W_target) / W_target
        
        if abs(err) < abs(best_err):
            best_err = err
            best_h0 = h0_mid
            best_p = p_mid.copy()
            best_h = h_mid.copy()
            best_p_asp = p_asp_mid.copy()
        
        if observe and i % 10 == 0:
            print(f"    Bisect {i}: h0={h0_mid*1e9:.1f}nm, W={W_mid:.1f}N, err={err*100:.1f}%")
        
        if abs(err) < 0.01:  # 1% tolerance
            break
        
        # Update bounds
        if W_mid > W_target:
            # Load too high -> increase h0
            h0_min = h0_mid
        else:
            # Load too low -> decrease h0
            h0_max = h0_mid
    
    return {
        "x": x, "X": X, "p": best_p, "p_asp": best_p_asp,
        "h": best_h, "h0": best_h0, "a": a, "ph": ph,
        "Wh": np.sum(best_p) * dx * L_width,
        "Wa": np.sum(best_p_asp) * dx * L_width,
        "Wext": W_target, "load_error": best_err, "dx": dx
    }

def check_criteria(r):
    c = {}
    c["load_error_val"] = abs(r["load_error"]) * 100
    c["load_error"] = c["load_error_val"] < 1.0
    
    pmax_n = np.max(r["p"]) / (r["ph"] + 1e-30)
    pasp_n = np.max(r["p_asp"]) / (r["ph"] + 1e-30)
    c["pressure_sum_val"] = pmax_n + pasp_n
    c["pressure_sum"] = abs(c["pressure_sum_val"] - 1) < 0.1
    
    c["X_pmax"] = r["X"][np.argmax(r["p"])]
    c["pmax_at_center"] = abs(c["X_pmax"]) < 0.3
    
    c["runtime"] = True
    
    # Cavitation spike
    X, p = r["X"], r["p"]
    exit_m = (X > 0.3) & (X < 1.5)
    c["cavitation_spike"] = np.any(np.diff(p[exit_m]) < -1e4) if np.sum(exit_m) > 3 else False
    
    # Flat film
    contact_m = np.abs(X) <= 1
    hc = r["h"][contact_m]
    c["flat_film_std"] = np.std(hc) / (np.mean(hc) + 1e-30) if len(hc) > 3 else 1.0
    c["flat_film"] = c["flat_film_std"] < 0.3
    
    return c

def print_criteria(c):
    print(f"  1. Load: {c['load_error_val']:.2f}% ({'OK' if c['load_error'] else 'X'})")
    print(f"  3. P_sum: {c['pressure_sum_val']:.3f} ({'OK' if c['pressure_sum'] else 'X'})")
    print(f"  4. X_pmax: {c['X_pmax']:.2f} ({'OK' if c['pmax_at_center'] else 'X'})")
    print(f"  6. Cav spike: ({'OK' if c['cavitation_spike'] else 'X'})")
    print(f"  7. Flat: {c['flat_film_std']:.2f} ({'OK' if c['flat_film'] else 'X'})")
    return all([c['load_error'], c['pressure_sum'], c['pmax_at_center'], 
                c['cavitation_spike'], c['flat_film']])

def run_cycle(rpm=300, sample_only=True, observe=True):
    print("="*70)
    print(f"CAM EHL v4 (Bisection) - {rpm}rpm, {TEMP_C}°C")
    print("="*70)
    
    R_arr, Ve_arr, Vs_arr, W_arr, w = kinematics(rpm)
    
    angles = [-50.0, 0.0, 50.0] if sample_only else TH_DEG[::10]
    indices = [np.argmin(np.abs(TH_DEG - a)) for a in angles]
    
    start = time.time()
    pass_count = 0
    
    for idx in indices:
        angle = TH_DEG[idx]
        R, Ve, Vs, W = R_arr[idx], Ve_arr[idx], Vs_arr[idx], W_arr[idx]
        
        print(f"\n--- {angle:.1f}° ---")
        print(f"  R={R*1e3:.2f}mm, Ve={Ve:.3f}m/s, W={W:.1f}N")
        
        r = solve_ehl_bisection(R, Ve, Vs, W, observe=observe)
        
        print(f"  pmax={np.max(r['p'])/1e6:.0f}MPa, hmin={np.min(r['h'])*1e9:.0f}nm")
        print(f"  Wh={r['Wh']:.1f}N, Wa={r['Wa']:.1f}N, err={r['load_error']*100:.1f}%")
        
        c = check_criteria(r)
        if print_criteria(c):
            print("  ALL PASS")
            pass_count += 1
    
    print(f"\nDONE: {time.time()-start:.1f}s, Passed: {pass_count}/{len(indices)}")

if __name__ == "__main__":
    run_cycle(rpm=300, sample_only=True, observe=True)
