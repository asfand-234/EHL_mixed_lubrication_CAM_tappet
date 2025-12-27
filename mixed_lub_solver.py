#!/usr/bin/env python3
"""
1D transient Reynolds equation solver for mixed lubrication (cam–shim line contact).

Features
- Hydrodynamic pressure from transient mass-conserving Reynolds formulation (content transport)
- Pressure-dependent density and viscosity (Dowson–Higginson, Houpert)
- Shear-thinning (Carreau) and boundary shear limit (Eyring-like)
- Greenwood–Tripp asperity contact load sharing
- Elastic deflection (line-contact log kernel)
- Optional 1D periodic Gaussian textures with theta-dependent amplitude from data files

Inputs
- Cam kinematics: CamAngle_vs_Lift_smooth.txt (angle [deg], lift [m])
- Optional a_texture tables: a_texture_data_5pct.txt, a_texture_data_8pct.txt, a_texture_data_10pct.txt

CLI examples
- Smooth (no textures):
  python mixed_lub_solver.py --data-dir . --rpms 300 500
- With 5% textures:
  python mixed_lub_solver.py --data-dir . --density 5 --rpms 300 500

Outputs
- Prints average friction torque for smooth vs textured (when a texture density is selected)
- Saves optional CSV with per-angle quantities if --write-csv is provided

Note
- This script mirrors the physics and numerics used in the provided research script
  but packaged as a standalone, readable module with a simple CLI.
"""
from __future__ import annotations

import argparse
import os
from math import pi
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Load cam data
# ============================================================

def load_cam_table(data_dir: str, fname: str) -> pd.DataFrame:
    path = os.path.join(data_dir, fname)
    cam = pd.read_csv(
        path, sep=r"\s+", engine="python", comment="#", header=None,
        names=["angle_deg", "lift_m"], usecols=[0, 1]
    )
    cam = cam.sort_values("angle_deg").reset_index(drop=True)
    cam["angle_deg"] = pd.to_numeric(cam["angle_deg"], errors="raise")
    cam["lift_m"] = pd.to_numeric(cam["lift_m"], errors="raise")
    return cam


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = int(max(3, k)) | 1  # odd
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


# ============================================================
# Materials / geometry / fluid
# ============================================================
rb = 18.5e-3  # base circle radius [m]
k_spring = 7130.0  # spring rate [N/m]
delta = 1.77e-3  # preload [m]
Meq = 0.05733  # equivalent mass [kg]
L = 7.2e-3  # out-of-plane length [m]

E_cam = 209e9
E_tap = 216e9
nu = 0.30
E_star = 1.0 / ((1.0 - nu**2) / E_cam + (1.0 - nu**2) / E_tap)

# Fluid / rheology
eta0 = 0.01381  # Pa·s
alpha_p = 15e-9  # 1/Pa
mu_b = 0.12  # boundary friction coefficient
rho0 = 858.44  # kg/m^3

# Greenwood–Tripp constants
sigma_combined = 0.265e-6
beta_a = 2.65e-4
eta_R = (0.05 / (sigma_combined * beta_a))

# Precompute GT kernel lookup F_{3/2}(Lambda)
_gt_w = np.linspace(0.0, 8.0, 400)
_gt_w_pow = _gt_w**1.5
_gt_norm = np.sqrt(2.0 * np.pi)
_lam_grid = np.linspace(0.0, 6.0, 360)
_kern = _gt_w_pow[None, :] * np.exp(-0.5 * (_lam_grid[:, None] + _gt_w) ** 2)
_F32_lookup = np.trapz(_kern, _gt_w, axis=1) / _gt_norm
_F32_lookup[-1] = 0.0


def eyring_gamma_limit() -> float:
    log = np.log10
    eta1, eta2, eta3 = 129.0, 13.5, 15.5
    T1, T2 = 40.0, 100.0
    rho0_local = 858.44
    ASTM = (log((log(eta1 + 0.7)) / (log(eta2 + 0.7)))) / (T2 / T1)
    g = (
        -5.0662
        + 8.8630 * (log(eta3)) ** (-0.07662)
        + 0.0312 * (ASTM**3.3611) * (log(eta3)) ** (-0.6271)
        - 0.1189 * (log(eta3)) ** (-5.4743) * (rho0_local) ** (-23.5841)
    ) / 100.0
    return max(g, 0.0)


gamma_lim = eyring_gamma_limit()


def eta_houpert(p: np.ndarray) -> np.ndarray:
    return np.maximum(eta0 * np.exp(np.clip(alpha_p * np.maximum(p, 0.0), 0.0, 23.0)), 1e-7)


eta_inf = 0.05 * eta0
lam_c = 1.5e-6
n_c = 0.65


def eta_carreau(etaN: np.ndarray, h: np.ndarray, Vs: np.ndarray | float) -> np.ndarray:
    abs_Vs = np.abs(Vs)
    gdot = np.where(h > 1e-12, abs_Vs / h, 0.0)
    return np.maximum(eta_inf + (etaN - eta_inf) * (1.0 + (lam_c * gdot) ** 2.0) ** ((n_c - 1.0) / 2.0), 1e-7)


def rho_dowson_higginson(p: np.ndarray) -> np.ndarray:
    p_eff = np.maximum(p, 0.0)
    return rho0 * (1.0 + 0.6e-9 * p_eff) / (1.0 + 1.7e-9 * p_eff)


# Greenwood–Tripp asperity pressure

def asperity_pressure_greenwood_tripp(h: np.ndarray) -> np.ndarray:
    lam = np.maximum(np.asarray(h, float) / (sigma_combined + 1e-18), 0.0)
    lam_clipped = np.clip(lam, _lam_grid[0], _lam_grid[-1])
    F32 = np.interp(lam_clipped, _lam_grid, _F32_lookup)
    pref = (4.0 / 3.0) * E_star * np.sqrt(beta_a) * eta_R * (sigma_combined**1.5)
    return (pref * F32).reshape(lam.shape)


# Hamrock–Dowson central film estimate (seed)

def central_film_thickness(R: float, W: float, Ve: float) -> float:
    R = max(float(R), 1e-6)
    W = max(float(W), 1.0)
    U = (eta0 * abs(Ve)) / (E_star * R + 1e-30)
    G = alpha_p * E_star
    W_star = W / (E_star * L * R + 1e-30)
    hc = 2.69 * (U**0.67) * (G**0.53) * (W_star**-0.067) * R
    return float(np.clip(hc, 40e-9, 600e-9))


# Kinematics arrays per RPM

def kinematics_for_rpm(th: np.ndarray, lift_s: np.ndarray, dlift_s: np.ndarray, d2lift_s: np.ndarray, rpm: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    w = 2.0 * pi * rpm / 60.0
    Vc = w * (rb + lift_s + d2lift_s)
    Vf = w * dlift_s
    R = np.abs(Vc / (w + 1e-30))
    Ve = 0.5 * (Vc + Vf)
    Vs = np.abs(Vc - Vf)
    W = k_spring * (lift_s + delta) + Meq * (w**2) * d2lift_s
    return R, Ve, Vs, W, w


# Hertz line-contact relations

def a_hertz(W: float, R: float) -> float:
    return np.sqrt(np.maximum(2.0 * np.maximum(W, 0.0) * np.maximum(R, 1e-12), 0.0) / (np.pi * E_star * L + 1e-30))


def ph_hertz(W: float, a: float) -> float:
    return 2.0 * np.maximum(W, 0.0) / (np.pi * np.maximum(a, 1e-12) * L + 1e-30)


# Elastic deflection (O(N^2) log kernel)

def elastic_deflection(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    N = len(x)
    dx = x[1] - x[0]
    eps = 0.5 * dx
    u = np.zeros_like(x)
    for i in range(N):
        u[i] = np.sum(p * np.log(np.sqrt((x[i] - x) ** 2 + eps * eps))) * dx
    u *= (2.0 / (np.pi * E_star))
    u -= np.mean(u)
    return u


# Rusanov advection for content transport

def rusanov_div_bc(u: float, q: np.ndarray, dx_nd: float, q_in_left: float, q_in_right: float) -> np.ndarray:
    N = len(q)
    qL = np.empty(N + 1)
    qR = np.empty(N + 1)
    qL[1:] = q
    qR[:-1] = q
    qL[0] = q_in_left
    qR[0] = q[0]
    qL[-1] = q[-1]
    qR[-1] = q_in_right
    F = 0.5 * (u * (qL + qR)) - 0.5 * abs(u) * (qR - qL)
    return (F[1:] - F[:-1]) / (dx_nd + 1e-30)


# ============================================================
# TEXTURE MODEL (theta-dependent amplitude from files)
# ============================================================
# Global texture shape params
w_texture = 35e-6  # [m]
g_val = 1e-9  # [m]
x_start = 0.0
X_in, X_out = -4.0, 3.0  # window scaling by a(theta)

# texture spacing per area density
D_TEXTURE: Dict[str, float] = {"5%": 700e-6, "8%": 437.5e-6, "10%": 350e-6}

ATEX_FILES: Dict[str, str] = {
    "5%": "a_texture_data_5pct.txt",
    "8%": "a_texture_data_8pct.txt",
    "10%": "a_texture_data_10pct.txt",
}


def load_atex_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for k, fn in ATEX_FILES.items():
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        tbl = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
        tbl = tbl.sort_values("angle_deg").reset_index(drop=True)
        tables[k] = tbl
    return tables


def atex_series_for_rpm(tables: Dict[str, pd.DataFrame], density_key: str, rpm: int, th_deg: np.ndarray) -> np.ndarray:
    df = tables[density_key]
    col = f"RPM{int(rpm)}"
    if col not in df.columns:
        raise ValueError(f"{density_key}: column {col} not found in {list(df.columns)}")
    if not np.allclose(df["angle_deg"].to_numpy(), th_deg, atol=1e-9):
        return np.interp(th_deg, df["angle_deg"].to_numpy(), df[col].to_numpy())
    return df[col].to_numpy(dtype=float)


def integrate_shift(Vs: np.ndarray, w: float, th: np.ndarray) -> np.ndarray:
    dtheta = np.gradient(th)
    integrand = Vs / (w + 1e-30)
    shift = np.zeros_like(integrand)
    shift[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dtheta[1:])
    return shift


def htex_profile(x: np.ndarray, a_theta: float, atex_theta: float, shift_theta: float, d_texture: float) -> np.ndarray:
    if atex_theta <= 0.0 or a_theta <= 0.0:
        return np.zeros_like(x)
    u = ((x - x_start - shift_theta + d_texture / 2.0) % d_texture) - d_texture / 2.0
    expo = (np.log(g_val / atex_theta) * (u**2)) / (w_texture + 1e-30)
    h = atex_theta * np.exp(expo)
    return np.where((x >= -a_theta) & (x <= a_theta), h, 0.0)


# ============================================================
# CORE SOLVER (per cam angle) — mixed lub with texture
# ============================================================

def solve_theta(
    R: float,
    Ve: float,
    Vs: float,
    W: float,
    dt: float,
    angle_deg: float,
    rpm: int,
    atex_theta: float,
    shift_theta: float,
    d_texture: float,
    Nx: int = 171,
    iters: int = 52,
    substep_cap: int = 6,
    relax_p: float = 0.85,
    relax_h: float = 0.55,
    M_core: int = 451,
) -> Dict[str, float | np.ndarray]:
    R = float(max(R, 1e-12))
    W = float(max(W, 1.0))
    a = max(a_hertz(W, R), 2e-6)
    ph = max(ph_hertz(W, a), 1e3)

    # Extended geometry window; pressure lives only in |x| <= a
    xL, xR = X_in * a, X_out * a
    x = np.linspace(xL, xR, Nx)

    # Normalized core for pressure on [-a,a]
    s = np.linspace(-1.0, 1.0, int(M_core))
    xs = a * s
    dS = s[1] - s[0]

    # Base film seed (Hamrock–Dowson)
    h0 = central_film_thickness(R, W, Ve)

    # Texture contribution
    htex = htex_profile(x, a, atex_theta, shift_theta, d_texture)

    # Initial film
    h = np.maximum(h0 + x**2 / (2 * R) + htex, 5e-9)

    # Content Phi = (rho / rho0) * (h * R / ph)
    p_zero = np.zeros_like(x)
    Phi = (rho_dowson_higginson(p_zero) / rho0) * ((h * R) / ph)
    phi_in = 0.5
    G = phi_in * Phi.copy()

    # Seed a Hertzian-shaped core pressure
    P_core = np.sqrt(np.maximum(1.0 - s**2, 0.0))

    # Transport substepping (CFL)
    dX = (x[1] - x[0]) / max(a, 1e-12)
    cfl = abs(Ve) * dt / ((x[1] - x[0]) + 1e-30)
    substeps = int(min(max(2, np.ceil(cfl / 0.35)), substep_cap))
    dts = dt / max(substeps, 1)
    dT = dts * max(abs(Ve), 0.05) / max(a, 1e-12)

    def embed_p(P_core_vec: np.ndarray) -> np.ndarray:
        p_full = np.zeros_like(x)
        inside = (x >= -a) & (x <= a)
        if inside.any():
            P_vals = np.interp(x[inside], xs, np.maximum(P_core_vec, 0.0))
            p_full[inside] = P_vals * ph
        return p_full

    # Gentle smoothing weights
    K0, K1 = 0.55, 0.225

    for _sub in range(substeps):
        # Transport source (from content update)
        p_tr = embed_p(P_core)
        rho_nd = rho_dowson_higginson(p_tr) / rho0
        H_nd = (h * R) / ph
        Phi = rho_nd * H_nd
        G_in_L = phi_in * Phi[0]
        G_in_R = phi_in * Phi[-1]
        div_phi = rusanov_div_bc(1.0, G, dX, G_in_L, G_in_R)
        S_nd = (Phi - G) / max(dT, 1e-12) + div_phi
        S_core_nd = np.interp(xs, x, S_nd)

        for _it in range(iters):
            # Diffusion ND on core (pressure >= 0 ensured)
            p_embed = embed_p(P_core)
            rho_nd = rho_dowson_higginson(p_embed) / rho0
            H_nd = (h * R) / ph
            eta_nd = eta_houpert(p_embed) / eta0

            rho_core = np.interp(xs, x, rho_nd)
            H_core = np.interp(xs, x, H_nd)
            eta_core = np.interp(xs, x, np.maximum(eta_nd, 1e-7))
            D_core = np.maximum(rho_core * H_core**3 / eta_core, 1e-12)

            # Tridiagonal assembly (Dirichlet p=0 at ±a via P_core[0]=P_core[-1]=0)
            M = len(xs)
            A = np.zeros(M)
            B = np.zeros(M)
            C = np.zeros(M)
            RHS = np.zeros(M)
            invdS2 = 1.0 / (dS * dS + 1e-30)
            B[0] = 1.0
            RHS[0] = 0.0
            for j in range(1, M - 1):
                Dw = 0.5 * (D_core[j] + D_core[j - 1])
                De = 0.5 * (D_core[j] + D_core[j + 1])
                A[j] = -Dw * invdS2
                C[j] = -De * invdS2
                B[j] = -(A[j] + C[j]) + 1e-12
                RHS[j] = S_core_nd[j]
            B[M - 1] = 1.0
            RHS[M - 1] = 0.0

            # Thomas solve
            for j in range(1, M):
                wfac = A[j] / (B[j - 1] + 1e-30)
                B[j] -= wfac * C[j - 1]
                RHS[j] -= wfac * RHS[j - 1]
            P_new = np.zeros(M)
            P_new[-1] = RHS[-1] / (B[-1] + 1e-30)
            for j in range(M - 2, -1, -1):
                P_new[j] = (RHS[j] - C[j] * P_new[j + 1]) / (B[j] + 1e-30)

            # positivity and hydro load prescale
            P_new = np.maximum(P_new, 0.0)
            Wh_trial = np.trapz(P_new * ph, xs) * L
            s_load = 1.0 if Wh_trial <= 1e-20 else np.clip(W / Wh_trial, 1e-3, 1e3)

            # relax + smooth
            P_core = (1 - relax_p) * P_core + relax_p * np.maximum(P_new * s_load, 0.0)
            Ptmp = P_core.copy()
            for j in range(1, len(P_core) - 1):
                P_core[j] = K1 * Ptmp[j - 1] + K0 * Ptmp[j] + K1 * Ptmp[j + 1]
            P_core[0] = 0.0
            P_core[-1] = 0.0

            # Mixed closure (Wh + Wa = W), update film with elastic deflection & TEXTURE
            p_embed = embed_p(P_core)
            defl = elastic_deflection(x, p_embed)

            # re-evaluate texture (keeps shift & amplitude fixed within this angle)
            htex = htex_profile(x, a, atex_theta, shift_theta, d_texture)

            h_nom = np.maximum(h0 + x**2 / (2 * R) + defl + htex, 5e-9)
            h = np.maximum(relax_h * h + (1.0 - relax_h) * h_nom, 5e-9)

            p_asp = asperity_pressure_greenwood_tripp(h)
            Wa = np.trapz(p_asp, x) * L
            Wh = np.trapz(p_embed, x) * L
            Wmix = Wh + Wa
            s_mix = (W / Wmix) if Wmix > 1e-20 else 1.0
            s_mix = max(s_mix, 0.0)
            P_core *= s_mix

            # smooth again for stability
            Ptmp = P_core.copy()
            for j in range(1, len(P_core) - 1):
                P_core[j] = K1 * Ptmp[j - 1] + K0 * Ptmp[j] + K1 * Ptmp[j + 1]
            P_core[0] = 0.0
            P_core[-1] = 0.0

        # conservative update of content field
        p_tr = embed_p(P_core)
        rho_nd = rho_dowson_higginson(p_tr) / rho0
        Phi = rho_nd * ((h * R) / ph)
        G_in_L = 0.5 * Phi[0]
        G_in_R = 0.5 * Phi[-1]
        div_phi = rusanov_div_bc(1.0, G, dX, G_in_L, G_in_R)
        G = np.clip(G + (dT) * (-div_phi), 0.0, Phi)

    # Final mixed enforcement
    p = embed_p(P_core)
    p_asp_final = asperity_pressure_greenwood_tripp(h)
    Wa_final = np.trapz(p_asp_final, x) * L
    Wh_now = np.trapz(p, x) * L
    if Wh_now + Wa_final > 1e-20:
        s_final = W / (Wh_now + Wa_final)
        s_final = max(s_final, 0.0)
        p *= s_final
        defl_final = elastic_deflection(x, p)
        htex = htex_profile(x, a, atex_theta, shift_theta, d_texture)
        h = np.maximum(h0 + x**2 / (2 * R) + defl_final + htex, 5e-9)

    # Friction forces
    eta_eff = eta_carreau(eta_houpert(p), h, abs(Ve))
    tau_h = np.where(h > 1e-12, eta_eff * abs(Ve) / h, 0.0)
    Fh = np.trapz(tau_h, x) * L

    tau_lim = gamma_lim * np.maximum(p, 0.0)
    Fb = L * np.trapz(tau_lim + mu_b * p_asp_final, x)

    return {
        "x": x,
        "p": p,
        "h": h,
        "Fh": float(Fh),
        "Fb": float(Fb),
        "Wa": float(Wa_final),
        "a": float(a),
        "pmax": float(np.max(p)),
    }


# ============================================================
# CAM CYCLE DRIVER
# ============================================================

def run_cycle(
    cam: pd.DataFrame,
    density_key: str | None,
    rpms: List[int],
    data_dir: str,
    write_csv: bool = False,
    csv_prefix: str | None = None,
) -> pd.DataFrame:
    th_deg = cam["angle_deg"].to_numpy(dtype=float)
    th = np.deg2rad(th_deg)
    lift = cam["lift_m"].to_numpy(dtype=float)

    def movavg3(x: np.ndarray) -> np.ndarray:
        return moving_average(moving_average(moving_average(x, 9), 21), 41)

    lift_s = movavg3(lift)
    dlift_s = np.gradient(lift_s, th)
    d2lift_s = np.gradient(dlift_s, th)

    atex_tables = load_atex_tables(data_dir)

    rows = []
    for rpm in rpms:
        R, Ve, Vs, W, w = kinematics_for_rpm(th, lift_s, dlift_s, d2lift_s, rpm)
        dt = np.gradient(th) / max(w, 1e-30)
        shift = integrate_shift(Vs, w, th)

        # texture spacing if selected
        if density_key is None:
            d_tex = D_TEXTURE["5%"]  # unused
            atex = np.zeros_like(th_deg)
        else:
            d_tex = D_TEXTURE[density_key]
            atex = atex_series_for_rpm(atex_tables, density_key, rpm, th_deg)
            atex = np.nan_to_num(atex, nan=0.0)

        T_smooth, T_tex = [], []
        for i, ang in enumerate(th_deg):
            # Untextured reference
            res_s = solve_theta(
                R[i], Ve[i], Vs[i], W[i], dt[i], ang, rpm, atex_theta=0.0, shift_theta=shift[i], d_texture=d_tex
            )

            r_eff = rb + lift_s[i]
            T_smooth.append((res_s["Fh"] + res_s["Fb"]) * r_eff)

            if density_key is not None:
                res_t = solve_theta(
                    R[i], Ve[i], Vs[i], W[i], dt[i], ang, rpm,
                    atex_theta=float(max(atex[i], 0.0)), shift_theta=shift[i], d_texture=d_tex,
                )
                T_tex.append((res_t["Fh"] + res_t["Fb"]) * r_eff)

        Tavg_s = float(np.mean(T_smooth))
        result = {"RPM": rpm, "Tavg_smooth": Tavg_s}
        if density_key is not None and len(T_tex) == len(T_smooth):
            Tavg_t = float(np.mean(T_tex))
            pct_reduction = 100.0 * (1.0 - Tavg_t / max(Tavg_s, 1e-30))
            result.update({"Tavg_textured": Tavg_t, "%Reduction": pct_reduction})
        rows.append(result)

        if write_csv and csv_prefix is not None:
            # save per-angle torques for inspection
            out = pd.DataFrame({"angle_deg": th_deg, "T_smooth": T_smooth})
            if density_key is not None and len(T_tex) == len(T_smooth):
                out["T_tex"] = T_tex
            out_path = f"{csv_prefix}_rpm{rpm}.csv"
            out.to_csv(out_path, index=False)

    return pd.DataFrame(rows).sort_values("RPM").reset_index(drop=True)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="1D transient Reynolds solver for mixed lubrication (cam–shim)")
    parser.add_argument("--data-dir", default=".", help="Directory with cam and a_texture data files")
    parser.add_argument("--cam-file", default="CamAngle_vs_Lift_smooth.txt", help="Cam angle vs lift file name")
    parser.add_argument("--density", choices=["smooth", "5", "8", "10"], default="smooth", help="Texture area density (%). 'smooth' disables textures")
    parser.add_argument("--rpms", nargs="+", type=int, default=[300], help="RPM values to simulate")
    parser.add_argument("--write-csv", action="store_true", help="Write per-angle torque CSVs")
    args = parser.parse_args()

    data_dir = args.data_dir
    cam = load_cam_table(data_dir, args.cam_file)

    density_key = None if args.density == "smooth" else f"{args.density}%"

    df = run_cycle(
        cam=cam,
        density_key=density_key,
        rpms=args.rpms,
        data_dir=data_dir,
        write_csv=args.write_csv,
        csv_prefix="cycle_output" if args.write_csv else None,
    )

    if density_key is None:
        print("\nAverage friction torque (smooth)")
        print(df.to_string(index=False, formatters={"Tavg_smooth": lambda v: f"{v:10.4e}"}))
    else:
        print(f"\nAverage friction torque and % reduction vs smooth ({density_key})")
        print(
            df.to_string(
                index=False,
                formatters={
                    "Tavg_smooth": lambda v: f"{v:10.4e}",
                    "Tavg_textured": lambda v: f"{v:10.4e}",
                    "%Reduction": lambda v: f"{v:6.2f}",
                },
            )
        )


if __name__ == "__main__":
    main()
