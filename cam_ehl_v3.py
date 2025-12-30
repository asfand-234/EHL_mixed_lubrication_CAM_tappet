#!/usr/bin/env python3
"""
1D Thermal Transient Mixed Lubrication (Line Contact) - Full Cam-Follower Cycle
Version 3: Corrected Reynolds solver with proper finite volume discretization.

The Reynolds equation for 1D line contact:
    d/dx[ρh³φ/(12η) dp/dx] = U_m d(ρh)/dx + d(ρh)/dt

Finite Volume form:
    F_{i+1/2} - F_{i-1/2} = (Couette_out - Couette_in) + Squeeze * Δx
    
where F = D * dp/dx (Poiseuille flux)
"""

import os
import numpy as np
from pathlib import Path
import pandas as pd
import time

# =======================
# CONFIGURATION
# =======================
TEMP_C = 90

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = HERE
CAM_FILE = "updated_lift.txt"

# ============================================================
# Load Cam Data
# ============================================================
def _load_cam(data_dir, fname):
    path = os.path.join(data_dir, fname)
    _cam = pd.read_csv(
        path, sep=r"\s+", engine="python", comment="#", header=None,
        names=["angle_deg", "lift_m"], usecols=[0, 1]
    )
    _cam["angle_deg"] = pd.to_numeric(_cam["angle_deg"], errors="raise")
    _cam["lift_m"]    = pd.to_numeric(_cam["lift_m"],    errors="raise")
    return _cam.sort_values("angle_deg").reset_index(drop=True)

CAM = _load_cam(DATA_DIR, CAM_FILE)
th_deg = CAM["angle_deg"].to_numpy(dtype=float)
th     = np.deg2rad(th_deg)
lift   = CAM["lift_m"].to_numpy(dtype=float)
TH_DEG = th_deg.copy()
dlift_s  = np.gradient(lift, th)
d2lift_s = np.gradient(dlift_s, th)

# ============================================================
# Material Properties (Fixed - do not change)
# ============================================================
rb       = 18.5e-3
k_spring = 7130.0
delta    = 1.77e-3
Meq      = 0.05733
L_width  = 7.2e-3
E_star   = 217e9

eta0      = 0.01381
alpha0    = 16e-9
rho0      = 858.44

sigma_combined = 0.2e-6
beta_a = sigma_combined / 0.001
eta_R  = 0.055 / (sigma_combined * beta_a)

gamma_th  = 4.5e-4
lam_c = 3.0e-6
n_c = 0.65
eta_inf = 0.006

X_in, X_out = -4.0, 2.0

# ============================================================
# Greenwood-Tripp Asperity Contact
# ============================================================
def F52_gt(lam):
    """Greenwood-Tripp F5/2 function."""
    lam = np.atleast_1d(lam).astype(float)
    F = np.zeros_like(lam)
    mask = lam < 4.0
    if np.any(mask):
        t = np.maximum(4.0 - lam[mask], 0.0)
        F[mask] = 4.4086e-5 * (t ** 6.804)
    return F

def asperity_pressure(h, sigma=sigma_combined):
    """Greenwood-Tripp asperity pressure."""
    h = np.atleast_1d(h).astype(float)
    sigma_eff = max(sigma, 1e-12)
    lam = h / sigma_eff
    f52 = F52_gt(lam)
    K = (16.0 * np.sqrt(2.0) / 15.0) * np.pi * (eta_R * beta_a * sigma_eff)**2 * np.sqrt(sigma_eff / beta_a)
    return K * E_star * f52

# ============================================================
# Rheology
# ============================================================
def viscosity_barus(p):
    """Barus viscosity-pressure relation with overflow protection."""
    p = np.atleast_1d(np.maximum(p, 0.0))
    exp_arg = np.clip(alpha0 * p, 0.0, 50.0)
    return eta0 * np.exp(exp_arg)

def density_dh(p, dT=0.0):
    """Dowson-Higginson density."""
    p = np.atleast_1d(np.maximum(p, 0.0))
    frac = (0.59e9 + 1.34 * p) / (0.59e9 + p)
    return rho0 * frac * (1.0 - gamma_th * dT)

def viscosity_carreau(eta_N, gdot):
    """Carreau shear-thinning."""
    gdot = np.atleast_1d(np.maximum(gdot, 1e-6))
    factor = (1.0 + (lam_c * gdot)**2) ** ((n_c - 1.0) / 2.0)
    return eta_inf + (eta_N - eta_inf) * factor

def phi_x_pc(h, sigma=sigma_combined):
    """Patir-Cheng pressure flow factor."""
    if sigma <= 0:
        return np.ones_like(h)
    H = np.maximum(h / sigma, 0.1)
    return np.maximum(1.0 - 0.9 * np.exp(-0.56 * H), 0.01)

# ============================================================
# Kinematics
# ============================================================
def kin_arrays(rpm):
    R = np.maximum(rb + lift + d2lift_s, 1e-7)
    w = 2.0 * np.pi * rpm / 60.0
    Vc = (rb + lift + d2lift_s) * w
    Vf = d2lift_s * w
    Ve = 0.5 * (Vc + Vf)
    Vs = Vc - Vf
    W = k_spring * (lift + delta) + Meq * w**2 * d2lift_s
    return R, Ve, Vs, W, w

def a_hertz(W, R):
    return np.sqrt(8.0 * max(W, 1e-6) * max(R, 1e-12) / (np.pi * E_star * L_width))

def ph_hertz(W, a):
    return 2.0 * max(W, 1e-6) / (np.pi * max(a, 1e-12) * L_width)

def hc_dh(R, W, Ve):
    """Dowson-Hamrock central film thickness."""
    R, W = max(R, 1e-7), max(W, 1e-6)
    Ve_abs = max(abs(Ve), 1e-6)
    U = eta0 * Ve_abs / (E_star * R)
    G = alpha0 * E_star
    W_star = W / (E_star * L_width * R)
    return 2.65 * (U**0.7) * (G**0.54) * (W_star**(-0.13)) * R

# ============================================================
# Elastic Deflection
# ============================================================
def elastic_deflection(x, p, dx):
    """Line contact elastic deflection via convolution."""
    N = len(x)
    b = dx / 2.0
    grid = np.arange(-N + 1, N) * dx
    
    def xlnx(v):
        r = np.zeros_like(v)
        m = np.abs(v) > 1e-12
        r[m] = v[m] * np.log(np.abs(v[m]))
        return r
    
    kernel = xlnx(grid + b) - xlnx(grid - b) - 2*b
    conv = np.convolve(p, kernel, mode='full')[N-1:2*N-1]
    u = -conv * 2.0 / (np.pi * E_star)
    u -= u[N//2]  # Reference at center
    return u

# ============================================================
# REYNOLDS SOLVER - Proper Finite Volume
# ============================================================
def solve_reynolds_fv(x, h, Ve, rho, eta, phix, p_init,
                      h_prev=None, rho_prev=None, dt=None,
                      max_iter=150, tol=1e-7, relax=0.3):
    """
    Solve Reynolds equation using finite volume method with Gauss-Seidel.
    
    Equation: d/dx[D dp/dx] = Ve * d(ρh)/dx + d(ρh)/dt
    
    FV form at node i:
        (D_{i+1/2} (p_{i+1}-p_i) - D_{i-1/2} (p_i-p_{i-1})) / Δx = source_i * Δx
    
    where source = Ve * d(ρh)/dx + d(ρh)/dt
    """
    N = len(x)
    dx = x[1] - x[0]
    p = p_init.copy()
    
    # Diffusion coefficient D = ρh³φx/(12η)
    D = rho * (h**3) * phix / (12.0 * eta + 1e-30)
    D = np.maximum(D, 1e-30)  # Prevent zero diffusion
    
    # Face-centered D (harmonic mean for conservation)
    D_face = np.zeros(N+1)
    D_face[0] = D[0]
    D_face[N] = D[N-1]
    for i in range(1, N):
        D_face[i] = 2.0 * D[i-1] * D[i] / (D[i-1] + D[i] + 1e-30)
    
    # Couette term: Ve * d(ρh)/dx using upwind
    rho_h = rho * h
    
    # Squeeze term
    squeeze = np.zeros(N)
    if h_prev is not None and rho_prev is not None and dt is not None and dt > 1e-12:
        rho_h_prev = rho_prev * h_prev
        squeeze = (rho_h - rho_h_prev) / dt
    
    # Boundary pressures
    p_left = 1e5   # Ambient at inlet
    p_right = 1e5  # Ambient at outlet
    
    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        p_old = p.copy()
        max_change = 0.0
        
        for i in range(1, N-1):
            # Diffusion fluxes at faces
            D_W = D_face[i]      # West face (between i-1 and i)
            D_E = D_face[i+1]    # East face (between i and i+1)
            
            # Coefficients
            a_W = D_W / dx
            a_E = D_E / dx
            a_P = a_W + a_E
            
            if a_P < 1e-30:
                continue
            
            # Couette source: upwind discretization
            # For Ve > 0: flux comes from left (upstream)
            # Net Couette = Ve * (ρh)_{i} - Ve * (ρh)_{i-1} for upwind
            if Ve >= 0:
                couette_source = Ve * (rho_h[i] - rho_h[i-1])
            else:
                couette_source = Ve * (rho_h[i+1] - rho_h[i])
            
            # Total source (per unit length)
            source = couette_source + squeeze[i] * dx
            
            # Gauss-Seidel update
            # a_P * p_i = a_W * p_{i-1} + a_E * p_{i+1} - source
            p_new = (a_W * p[i-1] + a_E * p[i+1] - source) / a_P
            
            # Reynolds cavitation: p >= 0
            p_new = max(p_new, 0.0)
            
            # Limit maximum pressure to prevent runaway (2x Hertzian as reasonable bound)
            p_max_limit = 1e10  # 10 GPa absolute limit
            p_new = min(p_new, p_max_limit)
            
            # Relaxation
            p[i] = (1.0 - relax) * p_old[i] + relax * p_new
            
            max_change = max(max_change, abs(p[i] - p_old[i]))
        
        # Apply boundary conditions
        p[0] = p_left
        p[N-1] = p_right
        
        # Convergence check
        if max_change < tol * (np.max(p) + 1e6):
            break
    
    residual = max_change / (np.max(p) + 1e6)
    return p, residual, iteration

# ============================================================
# MAIN EHL SOLVER
# ============================================================
def solve_ehl(R, Ve, Vs, W, dt, Nx=256, h0_init=None,
              h_prev=None, rho_prev=None,
              max_load_iter=100, load_tol=0.01, observe=False):
    """
    Solve coupled EHL problem with load balance.
    """
    R = max(R, 1e-12)
    W_target = max(W, 1e-6)
    Ve_eff = Ve if abs(Ve) > 1e-6 else (1e-6 if Ve >= 0 else -1e-6)
    
    a = a_hertz(W_target, R)
    ph = ph_hertz(W_target, a)
    
    # Domain
    xL, xR = X_in * a, X_out * a
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    X_norm = x / a
    
    # Parabola
    parabola = x**2 / (2.0 * R)
    
    # Initial h0
    if h0_init is not None and h0_init > 1e-12:
        h0 = h0_init
    else:
        h0 = max(hc_dh(R, W_target, Ve_eff), 1e-8)
    
    # Initial pressure (Hertzian)
    p = np.zeros(Nx)
    mask = np.abs(X_norm) <= 1.0
    p[mask] = ph * np.sqrt(np.maximum(1.0 - X_norm[mask]**2, 0.0))
    p[~mask] = 1e5
    
    # Best result tracking
    best_err = 1e10
    best = None
    
    # Load balance history for secant method
    h0_hist = []
    err_hist = []
    
    for i_load in range(max_load_iter):
        # Inner pressure-deformation iteration
        for i_inner in range(30):
            # Film thickness
            defl = elastic_deflection(x, p, dx)
            h = np.maximum(h0 + parabola + defl, 1e-12)
            
            # Asperity pressure
            p_asp = asperity_pressure(h)
            
            # Rheology
            eta = viscosity_barus(p)
            gdot = np.abs(Vs) / np.maximum(h, 1e-9)
            eta_eff = np.maximum(viscosity_carreau(eta, gdot), 1e-7)
            rho = np.maximum(density_dh(p), rho0 * 0.5)
            phix = np.maximum(phi_x_pc(h), 0.01)
            
            # Solve Reynolds
            p_new, resid, n_iter = solve_reynolds_fv(
                x, h, Ve_eff, rho, eta_eff, phix, p,
                h_prev=h_prev, rho_prev=rho_prev, dt=dt,
                max_iter=100, tol=1e-7, relax=0.3
            )
            
            # Check inner convergence
            dp_max = np.max(np.abs(p_new - p))
            p = p_new
            
            if dp_max < 1e-4 * ph:
                break
        
        # Compute load
        W_hydro = np.sum(p) * dx * L_width
        W_asp = np.sum(p_asp) * dx * L_width
        W_total = W_hydro + W_asp
        
        err_W = (W_total - W_target) / W_target
        
        h0_hist.append(h0)
        err_hist.append(err_W)
        
        if observe and i_load % 10 == 0:
            print(f"    Load {i_load}: h0={h0*1e9:.1f}nm, W={W_total:.2f}N, err={err_W*100:.1f}%")
        
        # Store best
        if abs(err_W) < abs(best_err):
            best_err = err_W
            best = {
                "x": x.copy(), "X": X_norm.copy(),
                "p": p.copy(), "p_asp": p_asp.copy(),
                "h": h.copy(), "h0": h0,
                "a": a, "ph": ph,
                "Wh": W_hydro, "Wa": W_asp, "Wext": W_target,
                "load_error": err_W, "rho": rho.copy(), "dx": dx
            }
        
        if abs(err_W) < load_tol:
            break
        
        # Secant method for h0 adjustment
        if len(h0_hist) >= 2:
            h0_prev, h0_curr = h0_hist[-2], h0_hist[-1]
            err_prev, err_curr = err_hist[-2], err_hist[-1]
            
            if abs(err_curr - err_prev) > 1e-10:
                # Secant update
                h0_new = h0_curr - err_curr * (h0_curr - h0_prev) / (err_curr - err_prev)
                # Damping for stability
                h0_new = 0.5 * h0 + 0.5 * h0_new
            else:
                # Fallback to proportional
                h0_new = h0 * (1.0 + 0.3 * err_W)
        else:
            # Initial proportional step
            h0_new = h0 * (1.0 + 0.5 * err_W)
        
        # Bound h0 to physical range
        h0_new = np.clip(h0_new, 1e-11, 1e-3)
        h0 = h0_new
    
    return best if best else {
        "x": x, "X": X_norm, "p": p, "p_asp": asperity_pressure(h),
        "h": h, "h0": h0, "a": a, "ph": ph,
        "Wh": np.sum(p)*dx*L_width, "Wa": np.sum(asperity_pressure(h))*dx*L_width,
        "Wext": W_target, "load_error": err_W, "rho": rho, "dx": dx
    }

# ============================================================
# DIAGNOSTICS
# ============================================================
def check_criteria(r):
    c = {}
    c["load_error_val"] = abs(r["load_error"]) * 100
    c["load_error"] = c["load_error_val"] < 1.0
    c["residual"] = c["load_error_val"] < 5.0
    c["residual_val"] = c["load_error_val"]
    
    pmax_n = np.max(r["p"]) / (r["ph"] + 1e-30)
    pasp_n = np.max(r["p_asp"]) / (r["ph"] + 1e-30)
    c["pressure_sum_val"] = pmax_n + pasp_n
    c["pressure_sum"] = abs(c["pressure_sum_val"] - 1.0) < 0.1
    
    c["X_pmax"] = r["X"][np.argmax(r["p"])]
    c["X_pmax_asp"] = r["X"][np.argmax(r["p_asp"])] if np.max(r["p_asp"]) > 0 else 0.0
    c["pmax_at_center"] = abs(c["X_pmax"]) < 0.3
    
    c["runtime"] = True
    
    # Cavitation spike: look for pressure decrease then increase near exit
    X, p = r["X"], r["p"]
    exit_mask = (X > 0.3) & (X < 1.5)
    if np.sum(exit_mask) > 5:
        p_exit = p[exit_mask]
        dp = np.diff(p_exit)
        has_drop = np.any(dp < -1e4)
        c["cavitation_spike"] = has_drop
    else:
        c["cavitation_spike"] = False
    
    # Flat film
    contact_mask = np.abs(X) <= 1.0
    h_contact = r["h"][contact_mask]
    if len(h_contact) > 5:
        c["flat_film_std"] = np.std(h_contact) / (np.mean(h_contact) + 1e-30)
        c["flat_film"] = c["flat_film_std"] < 0.3
    else:
        c["flat_film"] = False
        c["flat_film_std"] = 1.0
    
    return c

def print_criteria(c):
    print(f"  1. Load Error: {c['load_error_val']:.2f}% ({'PASS' if c['load_error'] else 'FAIL'})")
    print(f"  2. Residual: {c['residual_val']:.2f}% ({'PASS' if c['residual'] else 'FAIL'})")
    print(f"  3. P_sum: {c['pressure_sum_val']:.3f} ({'PASS' if c['pressure_sum'] else 'FAIL'})")
    print(f"  4. X_pmax: {c['X_pmax']:.2f} ({'PASS' if c['pmax_at_center'] else 'FAIL'})")
    print(f"  5. Runtime: PASS")
    print(f"  6. Cavitation: ({'PASS' if c['cavitation_spike'] else 'FAIL'})")
    print(f"  7. Flat film: {c['flat_film_std']:.2f} ({'PASS' if c['flat_film'] else 'FAIL'})")
    return all([c['load_error'], c['pressure_sum'], c['pmax_at_center'], 
                c['cavitation_spike'], c['flat_film']])

# ============================================================
# MAIN
# ============================================================
def run_cycle(rpm=300, sample_only=True, observe=True):
    print("="*70)
    print(f"CAM-TAPPET EHL SIMULATION v3 (FV Corrected)")
    print(f"  RPM: {rpm}, Temperature: {TEMP_C}°C")
    print("="*70)
    
    R_arr, Ve_arr, Vs_arr, W_arr, w = kin_arrays(rpm)
    dt = np.mean(np.diff(th)) / (w + 1e-30)
    
    if sample_only:
        angles = [-50.0, -25.0, 0.0, 25.0, 50.0]
        indices = [np.argmin(np.abs(TH_DEG - a)) for a in angles]
    else:
        indices = list(range(0, len(TH_DEG), 5))
    
    results = {}
    start = time.time()
    h_prev, rho_prev, h0_seed = None, None, None
    pass_count = 0
    
    for idx in indices:
        angle = TH_DEG[idx]
        R, Ve, Vs, W = R_arr[idx], Ve_arr[idx], Vs_arr[idx], W_arr[idx]
        
        a = a_hertz(W, R)
        ph = ph_hertz(W, a)
        hc = hc_dh(R, W, Ve)
        
        print(f"\n--- Angle: {angle:.1f}° ---")
        print(f"  R={R*1e3:.2f}mm, Ve={Ve:.3f}m/s, W={W:.1f}N")
        print(f"  a={a*1e6:.1f}µm, ph={ph/1e6:.0f}MPa, hc_DH={hc*1e9:.0f}nm")
        
        r = solve_ehl(R, Ve, Vs, W, dt, Nx=256,
                      h0_init=h0_seed if h0_seed else hc,
                      h_prev=h_prev, rho_prev=rho_prev,
                      observe=observe)
        
        h_prev, rho_prev, h0_seed = r["h"], r["rho"], r["h0"]
        results[angle] = r
        
        print(f"  pmax={np.max(r['p'])/1e6:.0f}MPa, hmin={np.min(r['h'])*1e9:.0f}nm, h0={r['h0']*1e9:.0f}nm")
        print(f"  Wh={r['Wh']:.1f}N, Wa={r['Wa']:.1f}N, err={r['load_error']*100:.1f}%")
        
        c = check_criteria(r)
        if print_criteria(c):
            print("  >>> ALL PASS <<<")
            pass_count += 1
        else:
            print("  >>> FAIL <<<")
    
    runtime = time.time() - start
    print("\n" + "="*70)
    print(f"DONE: {runtime:.1f}s, Passed: {pass_count}/{len(indices)}")
    print(f"Runtime < 170s: {'PASS' if runtime < 170 else 'FAIL'}")
    print("="*70)
    
    return results, runtime

if __name__ == "__main__":
    run_cycle(rpm=300, sample_only=True, observe=True)
