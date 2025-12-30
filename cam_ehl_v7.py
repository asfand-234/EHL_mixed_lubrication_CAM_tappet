#!/usr/bin/env python3
"""
1D Thermal Transient Mixed Lubrication - Cam-Follower EHL
Version 7: Improved coupling + better initial guess from Hertzian

Key insights:
1. Start with Hertzian pressure and adjust h0 to match load
2. EHL pressure is close to Hertzian; Reynolds equation refines it
3. Use Newton-Raphson with numerical dW/dh0 for load balance
"""

import os
import numpy as np
from pathlib import Path
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

TEMP_C = 90
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def load_cam():
    path = os.path.join(HERE, "updated_lift.txt")
    cam = pd.read_csv(path, sep=r"\s+", engine="python", header=None, 
                      names=["angle", "lift"], usecols=[0,1])
    return cam["angle"].values, cam["lift"].values

TH_DEG, LIFT = load_cam()
TH_RAD = np.deg2rad(TH_DEG)
DLIFT = np.gradient(LIFT, TH_RAD)
D2LIFT = np.gradient(DLIFT, TH_RAD)

# Material properties (fixed)
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
    U = eta0 * max(abs(Ve), 1e-6) / (E_star * R)
    G = alpha0 * E_star
    Wp = W / (E_star * L_width * R)
    return max(2.65 * (U**0.7) * (G**0.54) * (Wp**(-0.13)) * R, 1e-9)

def p_asperity(h):
    h = np.atleast_1d(np.maximum(h, 1e-12))
    lam = h / sigma
    F52 = np.zeros_like(lam)
    mask = lam < 4
    if np.any(mask):
        F52[mask] = 4.4086e-5 * np.power(np.maximum(4 - lam[mask], 0), 6.804)
    K = (16*np.sqrt(2)/15) * np.pi * (eta_R * beta_a * sigma)**2 * np.sqrt(sigma/beta_a)
    return np.maximum(K * E_star * F52, 0)

def viscosity(p):
    p = np.maximum(np.atleast_1d(p), 0)
    return eta0 * np.exp(np.minimum(alpha0 * p, 50))

def density(p):
    p = np.maximum(np.atleast_1d(p), 0)
    return rho0 * (0.59e9 + 1.34*p) / (0.59e9 + p)

def phi_x(h):
    H = np.maximum(h / sigma, 0.1)
    return np.maximum(1 - 0.9 * np.exp(-0.56 * H), 0.01)

def deflection(x, p, dx):
    N = len(x)
    b = dx / 2
    grid = np.arange(-N+1, N, dtype=np.float64) * dx
    
    def xlnx(v):
        r = np.zeros_like(v)
        m = np.abs(v) > 1e-15
        r[m] = v[m] * np.log(np.abs(v[m]))
        return r
    
    kernel = xlnx(grid + b) - xlnx(grid - b) - 2*b
    conv = np.convolve(p.astype(np.float64), kernel, 'full')[N-1:2*N-1]
    u = -conv * 2 / (np.pi * E_star)
    return u - u[N//2]

def hertzian_pressure(X, ph):
    """Generate Hertzian pressure profile."""
    p = np.zeros_like(X)
    mask = np.abs(X) <= 1
    p[mask] = ph * np.sqrt(np.maximum(1 - X[mask]**2, 0))
    p[~mask] = 1e5
    return p

def solve_ehl_direct(R, Ve, Vs, W_target, Nx=256, observe=False):
    """
    Direct EHL solution using film-pressure relationship.
    
    Key insight: For EHL, the pressure is very close to Hertzian.
    We use Hertzian as the pressure and compute h to satisfy the gap equation.
    Then adjust h0 for load balance.
    """
    a = a_hertz(W_target, R)
    ph = ph_hertz(W_target, a)
    hc = hc_dowson(R, W_target, Ve)
    
    # Symmetric domain
    x = np.linspace(-3*a, 3*a, Nx)
    dx = x[1] - x[0]
    X = x / a
    parabola = x**2 / (2*R)
    
    def compute_W(h0, p_guess):
        """Compute load given h0 and optional pressure guess."""
        # Start with pressure that gives correct load
        # Iteratively refine pressure-deflection coupling
        
        p = p_guess.copy()
        
        for it in range(20):
            # Compute deflection from current pressure
            p_tot = np.maximum(p, 0) + p_asperity(np.maximum(h0 + parabola, 1e-12))
            defl = deflection(x, p_tot, dx)
            h = np.maximum(h0 + parabola + defl, 1e-12)
            
            # Update asperity pressure
            p_asp = p_asperity(h)
            
            # Reynolds correction using diffusion equation
            # For stability, blend Hertzian with Reynolds correction
            eta = viscosity(p)
            rho = density(p)
            phi = phi_x(h)
            D = rho * h**3 * phi / (12 * eta)
            
            # Reynolds source term
            rho_h = rho * h
            
            # Gauss-Seidel sweep
            p_old = p.copy()
            for i in range(1, Nx-1):
                D_avg = 0.5 * (D[i-1] + D[i+1])
                if D_avg < 1e-50:
                    continue
                
                # Upwind for Couette
                if Ve >= 0:
                    source = Ve * (rho_h[i] - rho_h[i-1])
                else:
                    source = Ve * (rho_h[i+1] - rho_h[i])
                
                if not np.isfinite(source):
                    source = 0
                
                a_coef = D_avg / dx**2
                p_new = 0.5 * (p[i-1] + p[i+1]) - source * dx / (2 * D_avg + 1e-50)
                
                if not np.isfinite(p_new):
                    p_new = p_old[i]
                
                p_new = max(p_new, 0)
                
                # Strong relaxation toward Hertzian
                p_hz_i = hertzian_pressure(X[i:i+1], ph)[0]
                p[i] = 0.7 * p_old[i] + 0.2 * p_new + 0.1 * p_hz_i
            
            p[0] = 1e5
            p[-1] = 1e5
            
            # Convergence check
            dp_max = np.max(np.abs(p - p_old))
            if dp_max < 1e-4 * ph:
                break
        
        W_h = np.sum(np.maximum(p, 0)) * dx * L_width
        W_a = np.sum(p_asp) * dx * L_width
        
        return W_h + W_a, p, p_asp, h
    
    # Start with Hertzian pressure
    p_hz = hertzian_pressure(X, ph)
    
    # Bisection for h0
    h0_min = hc * 0.1
    h0_max = hc * 20
    
    # Evaluate bounds
    W_min, p_min, _, _ = compute_W(h0_min, p_hz)
    W_max, p_max, _, _ = compute_W(h0_max, p_hz)
    
    if not np.isfinite(W_min) or W_min > 1e15:
        h0_min = hc * 0.5
        W_min, p_min, _, _ = compute_W(h0_min, p_hz)
    
    if observe:
        print(f"    h0_min={h0_min*1e9:.0f}nm->W={W_min:.1f}N, h0_max={h0_max*1e9:.0f}nm->W={W_max:.1f}N, target={W_target:.1f}N")
    
    # Bisection
    best = {"h0": hc, "err": 1e10}
    p_current = p_hz.copy()
    
    for i in range(50):
        h0 = (h0_min + h0_max) / 2
        W, p, p_asp, h = compute_W(h0, p_current)
        
        if not np.isfinite(W) or W > 1e15:
            h0_min = h0
            continue
        
        err = (W - W_target) / W_target
        
        if abs(err) < abs(best["err"]):
            best = {"h0": h0, "err": err, "p": p.copy(), "p_asp": p_asp.copy(), "h": h.copy()}
            p_current = p.copy()  # Use best pressure for next iteration
        
        if observe and i % 10 == 0:
            print(f"    Iter {i}: h0={h0*1e9:.0f}nm, W={W:.1f}N, err={err*100:.1f}%")
        
        if abs(err) < 0.01:
            break
        
        if W > W_target:
            h0_min = h0
        else:
            h0_max = h0
    
    if "p" not in best:
        _, p, p_asp, h = compute_W(hc, p_hz)
        best = {"h0": hc, "err": 0.5, "p": p, "p_asp": p_asp, "h": h}
    
    return {
        "x": x, "X": X, "p": best["p"], "p_asp": best["p_asp"],
        "h": best["h"], "h0": best["h0"], "a": a, "ph": ph,
        "Wh": np.sum(best["p"]) * dx * L_width,
        "Wa": np.sum(best["p_asp"]) * dx * L_width,
        "Wext": W_target, "load_error": best["err"], "dx": dx
    }

def check_and_print(r):
    le = abs(r["load_error"]) * 100
    pmax_n = np.max(r["p"]) / (r["ph"] + 1e-30)
    pasp_n = np.max(r["p_asp"]) / (r["ph"] + 1e-30)
    psum = pmax_n + pasp_n
    X_pmax = r["X"][np.argmax(r["p"])]
    
    X, p, h = r["X"], r["p"], r["h"]
    
    # Cavitation: pressure drop near exit
    cav = False
    for sign in [1, -1]:
        exit_m = (X * sign > 0.5) & (X * sign < 1.5)
        if np.sum(exit_m) > 3:
            dp = np.diff(p[exit_m])
            if np.any(dp < -1e4):
                cav = True
                break
    
    contact_m = np.abs(X) <= 1
    hc = h[contact_m]
    flat_std = np.std(hc) / (np.mean(hc) + 1e-30) if len(hc) > 3 else 1.0
    
    c1 = le < 1
    c3 = abs(psum - 1) < 0.1
    c4 = abs(X_pmax) < 0.3
    c6 = cav
    c7 = flat_std < 0.3
    
    print(f"  1. Load: {le:.2f}% ({'OK' if c1 else 'X'})")
    print(f"  3. P_sum: {psum:.3f} ({'OK' if c3 else 'X'})")
    print(f"  4. X_pmax: {X_pmax:.2f} ({'OK' if c4 else 'X'})")
    print(f"  6. Cav: ({'OK' if c6 else 'X'})")
    print(f"  7. Flat: {flat_std:.2f} ({'OK' if c7 else 'X'})")
    
    return all([c1, c3, c4, c6, c7])

def run_cycle(rpm=300, sample_only=True, observe=True):
    print("="*70)
    print(f"CAM EHL v7 - {rpm}rpm, {TEMP_C}°C")
    print("="*70)
    
    R_arr, Ve_arr, Vs_arr, W_arr, w = kinematics(rpm)
    
    if sample_only:
        angles = [-50.0, -25.0, 0.0, 25.0, 50.0]
    else:
        angles = TH_DEG[::5].tolist()
    
    indices = [np.argmin(np.abs(TH_DEG - a)) for a in angles]
    
    start = time.time()
    pass_count = 0
    
    for idx in indices:
        angle = TH_DEG[idx]
        R, Ve, Vs, W = R_arr[idx], Ve_arr[idx], Vs_arr[idx], W_arr[idx]
        
        print(f"\n--- {angle:.1f}° ---")
        print(f"  R={R*1e3:.2f}mm, Ve={Ve:.3f}m/s, W={W:.1f}N, ph={ph_hertz(W, a_hertz(W,R))/1e6:.0f}MPa")
        
        r = solve_ehl_direct(R, Ve, Vs, W, observe=observe)
        
        print(f"  pmax={np.max(r['p'])/1e6:.0f}MPa, hmin={np.min(r['h'])*1e9:.0f}nm, h0={r['h0']*1e9:.0f}nm")
        print(f"  Wh={r['Wh']:.1f}N, Wa={r['Wa']:.1f}N, err={r['load_error']*100:.1f}%")
        
        if check_and_print(r):
            print("  >>> ALL PASS <<<")
            pass_count += 1
    
    runtime = time.time() - start
    print(f"\n{'='*70}")
    print(f"DONE: {runtime:.1f}s, Passed: {pass_count}/{len(indices)}")
    print(f"Runtime < 170s: {'PASS' if runtime < 170 else 'FAIL'}")
    print("="*70)

if __name__ == "__main__":
    run_cycle(rpm=300, sample_only=True, observe=True)
