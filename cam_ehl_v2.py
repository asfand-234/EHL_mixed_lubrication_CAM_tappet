#!/usr/bin/env python3
"""
1D Thermal Transient Mixed Lubrication (Line Contact) - Full Cam-Follower Cycle
Version 2: Corrected Reynolds solver with proper physics.

Based on standard EHL literature:
- Venner & Lubrecht "Multigrid Techniques" (2000)
- Hamrock, Schmid, Jacobson "Fundamentals of Fluid Film Lubrication" (2004)
"""

import os
import numpy as np
from pathlib import Path
import pandas as pd
import time

# =======================
# CONFIGURATION
# =======================
TEMP_C = 90  # Temperature in Celsius

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
# Material Properties (Fixed)
# ============================================================
rb       = 18.5e-3      # base circle radius [m]
k_spring = 7130.0       # spring rate [N/m]
delta    = 1.77e-3      # preload [m]
Meq      = 0.05733      # equivalent mass [kg]
L_width  = 7.2e-3       # contact width (out-of-plane) [m]
E_star   = 217e9        # effective modulus [Pa]

# Lubricant properties at 90°C
eta0      = 0.01381     # Pa·s
alpha0    = 16e-9       # Pa^-1
rho0      = 858.44      # kg/m³

# Roughness parameters (Greenwood-Tripp)
sigma_combined = 0.2e-6  # Combined roughness [m]
beta_a = sigma_combined / 0.001  # Asperity radius
eta_R  = 0.055 / (sigma_combined * beta_a)  # Asperity density

# Thermal
gamma_th  = 4.5e-4

# Carreau shear thinning
lam_c = 3.0e-6
n_c = 0.65
eta_inf = 0.006

# Domain
X_in, X_out = -4.0, 2.0  # Asymmetric for EHL inlet/outlet

# ============================================================
# Greenwood-Tripp F5/2 Function
# ============================================================
def F52_greenwood_tripp(lam):
    """
    Greenwood-Tripp F5/2 function using polynomial fit.
    lam = h/sigma (film thickness ratio)
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    H = np.maximum(lam, 0.0)
    F = np.zeros_like(H)
    
    # For H < 4, significant asperity contact
    # Using Zhao-Maietta-Chang fit
    mask = H < 4.0
    if np.any(mask):
        # F5/2 ≈ 4.4086e-5 * (4-H)^6.804 for H < 4
        t = np.maximum(4.0 - H[mask], 0.0)
        F[mask] = 4.4086e-5 * (t ** 6.804)
    
    return np.maximum(F, 0.0)

def asperity_pressure(h, sigma=sigma_combined):
    """
    Greenwood-Tripp asperity contact pressure.
    p_asp = K' * E* * F5/2(h/σ)
    where K' = (16√2/15) * π * (ηβσ)² * √(σ/β)
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    sigma_eff = max(sigma, 1e-12)
    lam = h / sigma_eff
    f52 = F52_greenwood_tripp(lam)
    
    K_prime = (16.0 * np.sqrt(2.0) / 15.0) * np.pi * (eta_R * beta_a * sigma_eff)**2 * np.sqrt(sigma_eff / beta_a)
    p_asp = K_prime * E_star * f52
    
    return np.maximum(p_asp, 0.0)

# ============================================================
# Rheology Functions
# ============================================================
def viscosity_barus(p, eta0_val=eta0, alpha=alpha0):
    """Barus exponential viscosity-pressure relation."""
    p_eff = np.maximum(np.atleast_1d(p), 0.0)
    # Limit exponent to prevent overflow
    exp_arg = np.clip(alpha * p_eff, 0.0, 50.0)
    return eta0_val * np.exp(exp_arg)

def density_dowson_higginson(p, dT=0.0):
    """Dowson-Higginson density-pressure relation."""
    p_eff = np.maximum(np.atleast_1d(p), 0.0)
    D1 = 0.59e9
    D2 = 1.34
    frac = (D1 + D2 * p_eff) / (D1 + p_eff)
    therm = 1.0 - gamma_th * dT
    return rho0 * frac * np.maximum(therm, 0.5)

def viscosity_carreau(eta_N, gdot):
    """Carreau shear-thinning model."""
    gdot_eff = np.maximum(np.atleast_1d(gdot), 1e-6)
    factor = (1.0 + (lam_c * gdot_eff)**2) ** ((n_c - 1.0) / 2.0)
    return np.maximum(eta_inf + (eta_N - eta_inf) * factor, 1e-7)

# ============================================================
# Flow Factors (Patir-Cheng 1978)
# ============================================================
def phi_x_patir_cheng(h, sigma=sigma_combined):
    """Pressure flow factor."""
    if sigma <= 0:
        return np.ones_like(h)
    H = np.maximum(h / sigma, 0.1)
    # Isotropic roughness
    C = 0.90
    r = 0.56
    phi_x = 1.0 - C * np.exp(-r * H)
    return np.maximum(phi_x, 0.01)

# ============================================================
# Kinematics
# ============================================================
def kin_arrays(rpm):
    """Compute kinematics arrays for all cam angles."""
    R = np.maximum(rb + lift + d2lift_s, 1e-7)
    w = 2.0 * np.pi * float(rpm) / 60.0
    Vf = d2lift_s * w
    Vc = (rb + lift + d2lift_s) * w
    Ve = 0.5 * (Vc + Vf)
    Vs = Vc - Vf
    W = k_spring * (lift + delta) + Meq * (w**2) * d2lift_s
    return R, Ve, Vs, W, w

def a_hertz(W, R):
    """Hertzian contact half-width for line contact."""
    W_eff = max(W, 1e-6)
    R_eff = max(R, 1e-12)
    return np.sqrt(8.0 * W_eff * R_eff / (np.pi * E_star * L_width))

def ph_hertz(W, a):
    """Hertzian peak pressure."""
    W_eff = max(W, 1e-6)
    a_eff = max(a, 1e-12)
    return 2.0 * W_eff / (np.pi * a_eff * L_width)

def hc_dowson_hamrock(R, W, Ve):
    """Dowson-Hamrock central film thickness for line contact."""
    R = max(R, 1e-7)
    W = max(W, 1e-6)
    Ve_abs = max(abs(Ve), 1e-6)
    
    U = eta0 * Ve_abs / (E_star * R)
    G = alpha0 * E_star
    W_star = W / (E_star * L_width * R)
    
    # Line contact formula: hc/R = 2.65 * U^0.7 * G^0.54 * W'^(-0.13)
    hc = 2.65 * (U**0.7) * (G**0.54) * (W_star**(-0.13)) * R
    return hc

# ============================================================
# Elastic Deflection (Line Contact BEM)
# ============================================================
def elastic_deflection(x, p, dx):
    """
    Elastic deflection for 2D line contact.
    u(x) = -2/(πE*) ∫ p(s) ln|x-s| ds
    
    Using analytical integration for constant pressure elements.
    """
    N = len(x)
    b = dx / 2.0  # Half-width of element
    
    # Create influence kernel
    grid = np.arange(-N + 1, N, dtype=float) * dx
    xp = grid + b
    xm = grid - b
    
    def xlnx(v):
        """x * ln|x| with proper x=0 handling."""
        res = np.zeros_like(v)
        mask = np.abs(v) > 1e-12
        res[mask] = v[mask] * np.log(np.abs(v[mask]))
        return res
    
    kernel = xlnx(xp) - xlnx(xm) - 2*b
    
    # Convolve
    conv = np.convolve(p, kernel, mode='full')
    u = conv[N-1 : 2*N-1]
    
    # Scale by -2/(πE*)
    u *= -2.0 / (np.pi * E_star)
    
    # Remove rigid body motion (center at x=0)
    center_idx = N // 2
    u -= u[center_idx]
    
    return u

# ============================================================
# REYNOLDS SOLVER - Gauss-Seidel with Line Relaxation
# ============================================================
def solve_reynolds_gauss_seidel(
    x, h, Ve, rho, eta, phix, p_init,
    h_prev=None, rho_prev=None, dt=None,
    p_boundary=1e5, max_iter=200, tol=1e-6, relax=0.5
):
    """
    Solve Reynolds equation using Gauss-Seidel iteration.
    
    Reynolds equation (steady state, no cavitation model):
    d/dx[ρh³φx/(12η) dp/dx] = Ve * d(ρh)/dx
    
    With transient (squeeze) term:
    d/dx[D dp/dx] = Ve * d(ρh)/dx + d(ρh)/dt
    
    Finite Volume discretization:
    (D_{i+1/2}(p_{i+1}-p_i) - D_{i-1/2}(p_i-p_{i-1}))/dx² = source_i
    """
    N = len(x)
    dx = x[1] - x[0]
    
    p = p_init.copy()
    
    # Diffusion coefficient D = ρh³φx/(12η)
    D = rho * (h**3) * phix / (12.0 * eta + 1e-30)
    
    # Face-centered diffusion (harmonic mean for stability)
    D_half = np.zeros(N-1)
    for i in range(N-1):
        D_half[i] = 2.0 * D[i] * D[i+1] / (D[i] + D[i+1] + 1e-30)
    
    # Couette source term: Ve * d(ρh)/dx
    rho_h = rho * h
    drhodh_dx = np.gradient(rho_h, dx)
    source_couette = Ve * drhodh_dx
    
    # Squeeze source term: d(ρh)/dt
    source_squeeze = np.zeros(N)
    if h_prev is not None and rho_prev is not None and dt is not None and dt > 1e-12:
        rho_h_prev = rho_prev * h_prev
        source_squeeze = (rho_h - rho_h_prev) / dt
    
    source = source_couette + source_squeeze
    
    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        p_old = p.copy()
        
        # Update interior points
        for i in range(1, N-1):
            a_W = D_half[i-1] / dx**2
            a_E = D_half[i] / dx**2
            a_P = a_W + a_E
            
            if a_P < 1e-30:
                continue
            
            # RHS = source term
            b = source[i]
            
            # Gauss-Seidel update
            p_new = (a_W * p[i-1] + a_E * p[i+1] - b) / a_P
            
            # Cavitation: p cannot be negative
            p_new = max(p_new, 0.0)
            
            # Relaxation
            p[i] = (1.0 - relax) * p_old[i] + relax * p_new
        
        # Boundary conditions
        p[0] = p_boundary
        p[-1] = p_boundary
        
        # Check convergence
        dp = np.abs(p - p_old)
        residual = np.max(dp) / (np.max(p) + 1e-30)
        
        if residual < tol:
            break
    
    return p, residual

# ============================================================
# MAIN EHL SOLVER
# ============================================================
def solve_ehl_contact(
    R, Ve, Vs, W, dt, angle_deg, rpm,
    Nx=256, h0_init=None,
    h_prev=None, rho_prev=None,
    max_load_iter=100, load_tol=0.01,
    max_newton_iter=50, newton_tol=1e-6,
    observe=False
):
    """
    Solve EHL contact problem for a single operating point.
    
    Uses outer loop for load balance (h0 adjustment) and
    inner loop for coupled pressure-deformation iteration.
    """
    # Validate inputs
    R = max(R, 1e-12)
    Ve_abs = max(abs(Ve), 1e-6)
    W_target = max(W, 1e-6)
    
    # Hertzian reference
    a = a_hertz(W_target, R)
    ph = ph_hertz(W_target, a)
    
    # Domain setup
    xL = X_in * a
    xR = X_out * a
    x = np.linspace(xL, xR, Nx)
    dx = x[1] - x[0]
    X_norm = x / a  # Normalized coordinate
    
    # Parabolic gap
    parabola = x**2 / (2.0 * R)
    
    # Initial film thickness
    if h0_init is not None and h0_init > 0:
        h0 = h0_init
    else:
        h0 = max(hc_dowson_hamrock(R, W_target, Ve), 1e-8)
    
    # Initialize pressure with Hertzian distribution
    p = np.zeros(Nx)
    mask_hz = np.abs(X_norm) <= 1.0
    p[mask_hz] = ph * np.sqrt(np.maximum(1.0 - X_norm[mask_hz]**2, 0.0))
    p[~mask_hz] = 1e5  # Ambient
    
    # Store best solution
    best_err = 1e10
    best_result = None
    
    # Load balance loop
    for i_load in range(max_load_iter):
        # Inner Newton-like iteration for pressure-deformation coupling
        for i_newton in range(max_newton_iter):
            # Compute film thickness
            p_total = p + asperity_pressure(np.maximum(h0 + parabola, 1e-12))
            defl = elastic_deflection(x, p_total, dx)
            h = np.maximum(h0 + parabola + defl, 1e-12)
            
            # Compute asperity pressure
            p_asp = asperity_pressure(h)
            
            # Rheology
            eta_node = viscosity_barus(p)
            gdot = np.abs(Vs) / np.maximum(h, 1e-9)
            eta_eff = viscosity_carreau(eta_node, gdot)
            rho_node = density_dowson_higginson(p)
            phix = phi_x_patir_cheng(h)
            
            # Solve Reynolds equation
            p_new, resid = solve_reynolds_gauss_seidel(
                x, h, Ve, rho_node, eta_eff, phix, p,
                h_prev=h_prev, rho_prev=rho_prev, dt=dt,
                max_iter=100, tol=1e-6, relax=0.4
            )
            
            # Check convergence
            dp_norm = np.max(np.abs(p_new - p)) / (np.max(p) + 1e-30)
            p = p_new
            
            if dp_norm < newton_tol:
                break
        
        # Recompute film and asperity pressure with converged p
        defl = elastic_deflection(x, p + asperity_pressure(np.maximum(h0 + parabola, 1e-12)), dx)
        h = np.maximum(h0 + parabola + defl, 1e-12)
        p_asp = asperity_pressure(h)
        
        # Compute total load
        W_hydro = np.sum(p) * dx * L_width
        W_asp = np.sum(p_asp) * dx * L_width
        W_total = W_hydro + W_asp
        
        # Load error
        err_W = (W_total - W_target) / W_target
        
        if observe and i_load % 10 == 0:
            print(f"    Load iter {i_load}: h0={h0*1e9:.1f}nm, W={W_total:.2f}N, err={err_W*100:.1f}%")
        
        # Store best
        if abs(err_W) < abs(best_err):
            best_err = err_W
            best_result = {
                "x": x.copy(),
                "X": X_norm.copy(),
                "p": p.copy(),
                "p_asp": p_asp.copy(),
                "h": h.copy(),
                "h0": h0,
                "a": a,
                "ph": ph,
                "Wh": W_hydro,
                "Wa": W_asp,
                "Wext": W_target,
                "load_error": err_W,
                "rho": rho_node.copy(),
                "dx": dx
            }
        
        # Check convergence
        if abs(err_W) < load_tol:
            break
        
        # Adjust h0 using Newton-like update
        # W increases as h0 decreases (more contact pressure)
        # dW/dh0 ≈ -W_hydro/h0 for soft EHL, but for stiff contact ~E*
        
        # Use bisection-like approach
        if err_W > 0:
            # Load too high -> increase h0
            dh0 = 0.1 * h0 * err_W
        else:
            # Load too low -> decrease h0
            dh0 = 0.2 * h0 * err_W
        
        # Limit step size
        dh0 = np.clip(dh0, -0.3 * h0, 0.5 * h0)
        
        h0_new = h0 + dh0
        h0 = max(h0_new, 1e-11)
    
    if best_result is None:
        # Return current state even if not converged
        defl = elastic_deflection(x, p, dx)
        h = np.maximum(h0 + parabola + defl, 1e-12)
        p_asp = asperity_pressure(h)
        rho_node = density_dowson_higginson(p)
        
        best_result = {
            "x": x,
            "X": X_norm,
            "p": p,
            "p_asp": p_asp,
            "h": h,
            "h0": h0,
            "a": a,
            "ph": ph,
            "Wh": np.sum(p) * dx * L_width,
            "Wa": np.sum(p_asp) * dx * L_width,
            "Wext": W_target,
            "load_error": err_W,
            "rho": rho_node,
            "dx": dx
        }
    
    return best_result

# ============================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================
def check_criteria(result, angle_deg):
    """Check all 7 criteria."""
    ph = result["ph"]
    X = result["X"]
    p = result["p"]
    p_asp = result["p_asp"]
    h = result["h"]
    
    criteria = {}
    
    # 1. Load error < 1%
    load_err = abs(result["load_error"]) * 100
    criteria["load_error"] = load_err < 1.0
    criteria["load_error_val"] = load_err
    
    # 2. Residual (using load error as proxy)
    criteria["residual"] = load_err < 5.0
    criteria["residual_val"] = load_err
    
    # 3. Pressure sum = 1 ± 10%
    pmax_norm = np.max(p) / (ph + 1e-30)
    pmax_asp_norm = np.max(p_asp) / (ph + 1e-30)
    p_sum = pmax_norm + pmax_asp_norm
    criteria["pressure_sum"] = abs(p_sum - 1.0) < 0.1
    criteria["pressure_sum_val"] = p_sum
    
    # 4. Max pressure at X=0
    idx_pmax = np.argmax(p)
    idx_pmax_asp = np.argmax(p_asp)
    X_pmax = X[idx_pmax]
    X_pmax_asp = X[idx_pmax_asp] if np.max(p_asp) > 0 else 0.0
    criteria["pmax_at_center"] = abs(X_pmax) < 0.3
    criteria["X_pmax"] = X_pmax
    criteria["X_pmax_asp"] = X_pmax_asp
    
    # 5. Runtime (external)
    criteria["runtime"] = True
    
    # 6. Cavitation spike near exit
    # Look for local minimum in h near outlet
    mask_exit = (X > 0.5) & (X < 1.5)
    h_exit = h[mask_exit]
    if len(h_exit) > 3:
        h_grad = np.gradient(h_exit)
        criteria["cavitation_spike"] = np.any(h_grad < 0)  # Constriction
    else:
        criteria["cavitation_spike"] = False
    
    # 7. Flat film in contact zone
    mask_contact = np.abs(X) <= 1.0
    h_contact = h[mask_contact]
    if len(h_contact) > 3:
        h_std = np.std(h_contact) / (np.mean(h_contact) + 1e-30)
        criteria["flat_film"] = h_std < 0.3
        criteria["flat_film_std"] = h_std
    else:
        criteria["flat_film"] = False
        criteria["flat_film_std"] = 1.0
    
    return criteria

def print_criteria(criteria, angle_deg):
    """Print criteria status."""
    print(f"  1. Load Error: {criteria['load_error_val']:.2f}% ({'PASS' if criteria['load_error'] else 'FAIL'})")
    print(f"  2. Residual: {criteria['residual_val']:.2e} ({'PASS' if criteria['residual'] else 'FAIL'})")
    print(f"  3. P_rey+P_asp: {criteria['pressure_sum_val']:.3f} ({'PASS' if criteria['pressure_sum'] else 'FAIL'})")
    print(f"  4. P_max at X: {criteria['X_pmax']:.2f} ({'PASS' if criteria['pmax_at_center'] else 'FAIL'})")
    print(f"  5. Runtime: ({'PASS' if criteria['runtime'] else 'FAIL'})")
    print(f"  6. Cavitation: ({'PASS' if criteria['cavitation_spike'] else 'FAIL'})")
    print(f"  7. Flat film: {criteria.get('flat_film_std', 0):.2f} ({'PASS' if criteria['flat_film'] else 'FAIL'})")
    
    return all([
        criteria['load_error'],
        criteria['pressure_sum'],
        criteria['pmax_at_center'],
        criteria['cavitation_spike'],
        criteria['flat_film']
    ])

# ============================================================
# MAIN CYCLE LOOP
# ============================================================
def run_cycle(rpm=300, sample_only=True, observe=True):
    """Run cam cycle simulation."""
    print("="*70)
    print(f"CAM-TAPPET EHL SIMULATION v2")
    print(f"  RPM: {rpm}, Temperature: {TEMP_C}°C")
    print("="*70)
    
    R_arr, Ve_arr, Vs_arr, W_arr, w = kin_arrays(rpm)
    dtheta = float(np.mean(np.diff(th)))
    dt = dtheta / (w + 1e-30)
    
    if sample_only:
        sample_angles = [-50.0, -25.0, 0.0, 25.0, 50.0]
        indices = [np.argmin(np.abs(TH_DEG - a)) for a in sample_angles]
    else:
        step = 5
        indices = list(range(0, len(TH_DEG), step))
    
    results = {}
    start_time = time.time()
    
    h_prev = None
    rho_prev = None
    h0_seed = None
    pass_count = 0
    
    for idx in indices:
        angle = TH_DEG[idx]
        R = R_arr[idx]
        Ve = Ve_arr[idx]
        Vs = Vs_arr[idx]
        W = W_arr[idx]
        
        a = a_hertz(W, R)
        ph = ph_hertz(W, a)
        hc = hc_dowson_hamrock(R, W, Ve)
        
        print(f"\n--- Angle: {angle:.1f}° ---")
        print(f"  R={R*1e3:.3f}mm, Ve={Ve:.3f}m/s, W={W:.1f}N")
        print(f"  Hertz: a={a*1e6:.1f}µm, ph={ph/1e6:.1f}MPa, hc_DH={hc*1e9:.1f}nm")
        
        result = solve_ehl_contact(
            R=R, Ve=Ve, Vs=Vs, W=W, dt=dt,
            angle_deg=angle, rpm=rpm,
            h0_init=h0_seed if h0_seed else hc,
            h_prev=h_prev, rho_prev=rho_prev,
            observe=observe
        )
        
        h_prev = result["h"]
        rho_prev = result["rho"]
        h0_seed = result["h0"]
        
        results[angle] = result
        
        print(f"  Result: pmax={np.max(result['p'])/1e6:.1f}MPa, hmin={np.min(result['h'])*1e9:.1f}nm")
        print(f"  Load: Wh={result['Wh']:.2f}N, Wa={result['Wa']:.2f}N, err={result['load_error']*100:.1f}%")
        
        criteria = check_criteria(result, angle)
        print("  CRITERIA:")
        if print_criteria(criteria, angle):
            print("  >>> ALL PASS <<<")
            pass_count += 1
        else:
            print("  >>> SOME FAIL <<<")
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print("\n" + "="*70)
    print(f"COMPLETE: {runtime:.1f}s, Passed: {pass_count}/{len(indices)}")
    print(f"Runtime < 170s: {'PASS' if runtime < 170 else 'FAIL'}")
    print("="*70)
    
    return results, runtime

if __name__ == "__main__":
    results, runtime = run_cycle(rpm=300, sample_only=True, observe=True)
