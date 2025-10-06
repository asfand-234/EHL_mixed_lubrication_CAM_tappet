#!/usr/bin/env python3
"""
Complete 1D Transient Reynolds Equation Solution for Mixed Lubrication Theory
CAM-SHIM (Bucket Tappet) with Line Textures

This code implements a comprehensive 1D mixed lubrication solver that includes:
- Transient Reynolds equation with mass-conserving cavitation
- Mixed lubrication with hydrodynamic and asperity contact
- Texture modeling with theta-dependent amplitude
- Elastic deformation effects
- Non-Newtonian fluid behavior (Eyring, Carreau models)
- Friction torque calculation and reduction analysis

Author: AI Assistant
Date: 2025-10-06
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for better visualization
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

class Reynolds1DMixedLubrication:
    """
    Complete 1D Transient Reynolds Equation Solver for Mixed Lubrication
    
    This class implements a comprehensive mixed lubrication solver for cam-tappet
    contacts with surface textures, including:
    - Mass-conserving Reynolds equation with cavitation
    - Greenwood-Tripp asperity contact model
    - Elastic deformation using influence functions
    - Non-Newtonian fluid behavior
    - Texture modeling with variable amplitude
    """
    
    def __init__(self, data_dir: str = ".", cam_file: str = "CamAngle_vs_Lift_smooth.txt"):
        """
        Initialize the mixed lubrication solver
        
        Args:
            data_dir: Directory containing input data files
            cam_file: Cam profile data file name
        """
        self.data_dir = data_dir
        self.cam_file = cam_file
        
        # Material and geometric properties
        self._init_material_properties()
        
        # Fluid properties
        self._init_fluid_properties()
        
        # Texture properties
        self._init_texture_properties()
        
        # Numerical parameters
        self._init_numerical_parameters()
        
        # Load cam data
        self._load_cam_data()
        
        # Initialize Greenwood-Tripp lookup table
        self._init_greenwood_tripp()
        
        # Load texture data
        self._load_texture_data()
        
    def _init_material_properties(self):
        """Initialize material and geometric properties"""
        # Geometry
        self.rb = 18.5e-3          # Base circle radius [m]
        self.L = 7.2e-3            # Out-of-plane length [m]
        
        # Spring properties
        self.k_spring = 7130.0     # Spring rate [N/m]
        self.delta = 1.77e-3       # Preload [m]
        self.Meq = 0.05733         # Equivalent mass [kg]
        
        # Material properties
        E_cam = 209e9              # Cam elastic modulus [Pa]
        E_tap = 216e9              # Tappet elastic modulus [Pa]
        nu = 0.30                  # Poisson's ratio
        self.E_star = 1.0 / ((1-nu**2)/E_cam + (1-nu**2)/E_tap)  # Reduced modulus
        
    def _init_fluid_properties(self):
        """Initialize fluid and rheological properties"""
        # Base fluid properties
        self.eta0 = 0.01381        # Base viscosity [Pa·s]
        self.alpha_p = 15e-9       # Pressure-viscosity coefficient [1/Pa]
        self.rho0 = 858.44         # Base density [kg/m³]
        
        # Boundary friction
        self.mu_b = 0.12           # Boundary friction coefficient
        
        # Eyring stress limit
        self.gamma_lim = self._calculate_eyring_limit()
        
        # Carreau model parameters
        self.eta_inf = 0.05 * self.eta0  # Infinite shear viscosity
        self.lam_c = 1.5e-6        # Carreau time constant [s]
        self.n_c = 0.65            # Carreau power law index
        
    def _init_texture_properties(self):
        """Initialize texture model parameters"""
        self.w_texture = 35e-6     # Texture width [m]
        self.g_val = 1e-9          # Texture parameter [m]
        self.x_start = 0.0         # Texture start position
        self.X_in, self.X_out = -4.0, 3.0  # Domain scaling factors
        
        # Texture density for different area coverage
        self.D_TEXTURE = {
            "5%": 700e-6,
            "8%": 437.5e-6,
            "10%": 350e-6
        }
        
        # Texture data files
        self.ATEX_FILES = {
            "5%": "a_texture_data_5pct.txt",
            "8%": "a_texture_data_8pct.txt",
            "10%": "a_texture_data_10pct.txt",
        }
        
    def _init_numerical_parameters(self):
        """Initialize numerical solver parameters"""
        self.Nx = 171              # Grid points in x-direction
        self.M_core = 451          # Core pressure grid points
        self.iters = 52            # Maximum iterations per time step
        self.substep_cap = 6       # Maximum transport substeps
        self.relax_p = 0.85        # Pressure relaxation factor
        self.relax_h = 0.55        # Film thickness relaxation factor
        
    def _calculate_eyring_limit(self) -> float:
        """Calculate Eyring shear stress limit"""
        log = np.log10
        eta1, eta2, eta3 = 129.0, 13.5, 15.5
        T1, T2 = 40.0, 100.0
        rho0_local = 858.44
        
        ASTM = (log((log(eta1+0.7))/(log(eta2+0.7)))) / (T2/T1)
        g = (-5.0662 + 8.8630*(log(eta3))**(-0.07662) + 
             0.0312*(ASTM**3.3611)*(log(eta3))**(-0.6271) - 
             0.1189*(log(eta3))**(-5.4743)*(rho0_local)**(-23.5841)) / 100.0
        return max(g, 0.0)
        
    def _load_cam_data(self):
        """Load and smooth cam profile data"""
        path = os.path.join(self.data_dir, self.cam_file)
        cam_data = pd.read_csv(
            path, sep=r"\s+", engine="python", comment="#", header=None,
            names=["angle_deg", "lift_m"], usecols=[0, 1]
        )
        cam_data["angle_deg"] = pd.to_numeric(cam_data["angle_deg"], errors="raise")
        cam_data["lift_m"] = pd.to_numeric(cam_data["lift_m"], errors="raise")
        cam_data = cam_data.sort_values("angle_deg").reset_index(drop=True)
        
        self.th_deg = cam_data["angle_deg"].to_numpy(dtype=float)
        self.th = np.deg2rad(self.th_deg)
        lift_raw = cam_data["lift_m"].to_numpy(dtype=float)
        
        # Apply smoothing
        self.lift_s = self._moving_average(
            self._moving_average(
                self._moving_average(lift_raw, 9), 21), 41)
        
        # Calculate derivatives
        self.dlift_s = np.gradient(self.lift_s, self.th)
        self.d2lift_s = np.gradient(self.dlift_s, self.th)
        
    def _moving_average(self, x: np.ndarray, k: int) -> np.ndarray:
        """Apply moving average filter"""
        k = int(max(3, k)) | 1  # Ensure odd number >= 3
        w = np.ones(k) / k
        return np.convolve(x, w, mode="same")
        
    def _init_greenwood_tripp(self):
        """Initialize Greenwood-Tripp asperity contact lookup table"""
        # Surface roughness parameters
        self.sigma_combined = 0.265e-6  # Combined RMS roughness [m]
        self.beta_a = 2.65e-4           # Asperity density [1/m²]
        self.eta_R = 0.05 / (self.sigma_combined * self.beta_a)  # Radius parameter
        
        # Create lookup table for F₅/₂ function
        gt_w = np.linspace(0.0, 8.0, 400)
        gt_w_pow = gt_w**1.5
        gt_norm = np.sqrt(2.0 * np.pi)
        self._lam_grid = np.linspace(0.0, 6.0, 360)
        
        kern = gt_w_pow[None, :] * np.exp(-0.5 * (self._lam_grid[:, None] + gt_w) ** 2)
        self._F32_lookup = np.trapz(kern, gt_w, axis=1) / gt_norm
        self._F32_lookup[-1] = 0.0
        
    def _load_texture_data(self):
        """Load texture amplitude data for different area densities"""
        self.atex_tables = {}
        for density_key, filename in self.ATEX_FILES.items():
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                table = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
                table = table.sort_values("angle_deg").reset_index(drop=True)
                self.atex_tables[density_key] = table
            else:
                print(f"Warning: Texture file {filename} not found")
                
    def calculate_kinematics(self, rpm: float) -> Tuple[np.ndarray, ...]:
        """
        Calculate kinematic parameters for given RPM
        
        Args:
            rpm: Rotational speed [rev/min]
            
        Returns:
            Tuple of (R, Ve, Vs, W, omega) arrays
        """
        omega = 2 * pi * rpm / 60.0
        
        # Velocities
        Vc = omega * (self.rb + self.lift_s + self.d2lift_s)  # Cam velocity
        Vf = omega * self.dlift_s                             # Follower velocity
        
        # Effective radius and velocities
        R = np.abs(Vc / (omega + 1e-30))
        Ve = 0.5 * (Vc + Vf)  # Entraining velocity
        Vs = np.abs(Vc - Vf)  # Sliding velocity
        
        # Load
        W = self.k_spring * (self.lift_s + self.delta) + self.Meq * (omega**2) * self.d2lift_s
        
        return R, Ve, Vs, W, omega
        
    def hertz_contact(self, W: float, R: float) -> Tuple[float, float]:
        """
        Calculate Hertzian contact parameters
        
        Args:
            W: Normal load [N]
            R: Effective radius [m]
            
        Returns:
            Tuple of (half-width, max pressure)
        """
        a = np.sqrt(max(2*max(W,0.0)*max(R,1e-12), 0.0) / (np.pi*self.E_star*self.L + 1e-30))
        ph = 2.0*max(W,0.0) / (np.pi*max(a,1e-12)*self.L + 1e-30)
        return a, ph
        
    def central_film_thickness(self, R: float, W: float, Ve: float) -> float:
        """
        Hamrock-Dowson central film thickness formula
        
        Args:
            R: Effective radius [m]
            W: Normal load [N]
            Ve: Entraining velocity [m/s]
            
        Returns:
            Central film thickness [m]
        """
        R = max(float(R), 1e-6)
        W = max(float(W), 1.0)
        
        U = (self.eta0 * abs(Ve)) / (self.E_star * R + 1e-30)
        G = self.alpha_p * self.E_star
        W_star = W / (self.E_star * self.L * R + 1e-30)
        
        hc = 2.69 * (U**0.67) * (G**0.53) * (W_star**-0.067) * R
        return float(np.clip(hc, 40e-9, 600e-9))
        
    def viscosity_houpert(self, p: np.ndarray) -> np.ndarray:
        """Houpert pressure-dependent viscosity model"""
        return np.maximum(
            self.eta0 * np.exp(np.clip(self.alpha_p * np.maximum(p, 0.0), 0, 23.0)),
            1e-7
        )
        
    def viscosity_carreau(self, eta_N: np.ndarray, h: np.ndarray, Vs: float) -> np.ndarray:
        """Carreau shear-thinning viscosity model"""
        gdot = np.where(h > 1e-12, abs(Vs) / h, 0.0)
        return np.maximum(
            self.eta_inf + (eta_N - self.eta_inf) * (1 + (self.lam_c * gdot)**2)**((self.n_c - 1) / 2),
            1e-7
        )
        
    def density_dowson_higginson(self, p: np.ndarray) -> np.ndarray:
        """Dowson-Higginson pressure-dependent density model"""
        p_eff = np.maximum(p, 0.0)
        return self.rho0 * (1.0 + 0.6e-9 * p_eff) / (1.0 + 1.7e-9 * p_eff)
        
    def asperity_pressure_greenwood_tripp(self, h: np.ndarray) -> np.ndarray:
        """
        Greenwood-Tripp asperity contact pressure
        
        Args:
            h: Film thickness array [m]
            
        Returns:
            Asperity pressure array [Pa]
        """
        lam = np.maximum(np.asarray(h, float) / (self.sigma_combined + 1e-18), 0.0)
        lam_clipped = np.clip(lam, self._lam_grid[0], self._lam_grid[-1])
        F32 = np.interp(lam_clipped, self._lam_grid, self._F32_lookup)
        
        pref = (4.0/3.0) * self.E_star * np.sqrt(self.beta_a) * self.eta_R * (self.sigma_combined**1.5)
        return (pref * F32).reshape(lam.shape)
        
    def elastic_deflection(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Calculate elastic deflection using influence functions
        
        Args:
            x: Spatial coordinate array [m]
            p: Pressure distribution [Pa]
            
        Returns:
            Deflection array [m]
        """
        x = np.asarray(x, float)
        p = np.asarray(p, float)
        N = len(x)
        dx = x[1] - x[0]
        eps = 0.5 * dx
        
        u = np.zeros_like(x)
        for i in range(N):
            u[i] = np.sum(p * np.log(np.sqrt((x[i] - x)**2 + eps*eps))) * dx
            
        u *= (2.0 / (np.pi * self.E_star))
        u -= np.mean(u)  # Remove rigid body motion
        return u
        
    def get_texture_amplitude(self, density_key: str, rpm: float) -> np.ndarray:
        """
        Get texture amplitude series for given density and RPM
        
        Args:
            density_key: Texture density key ("5%", "8%", "10%")
            rpm: Rotational speed [rev/min]
            
        Returns:
            Texture amplitude array [m]
        """
        if density_key not in self.atex_tables:
            return np.zeros_like(self.th_deg)
            
        df = self.atex_tables[density_key]
        col = f"RPM{int(rpm)}"
        
        if col not in df.columns:
            print(f"Warning: Column {col} not found for {density_key}")
            return np.zeros_like(self.th_deg)
            
        # Check if angle grids match
        if not np.allclose(df["angle_deg"].to_numpy(), self.th_deg, atol=1e-9):
            # Interpolate if grids differ
            return np.interp(self.th_deg, df["angle_deg"].to_numpy(), df[col].to_numpy())
            
        return df[col].to_numpy(dtype=float)
        
    def integrate_shift(self, Vs: np.ndarray, omega: float) -> np.ndarray:
        """
        Calculate texture shift due to sliding
        
        Args:
            Vs: Sliding velocity array [m/s]
            omega: Angular velocity [rad/s]
            
        Returns:
            Shift array [m]
        """
        dtheta = np.gradient(self.th)
        integrand = Vs / (omega + 1e-30)
        
        shift = np.zeros_like(integrand)
        shift[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dtheta[1:])
        return shift
        
    def texture_profile(self, x: np.ndarray, a_theta: float, atex_theta: float, 
                       shift_theta: float, d_texture: float) -> np.ndarray:
        """
        Calculate texture profile at given cam angle
        
        Args:
            x: Spatial coordinate array [m]
            a_theta: Contact half-width [m]
            atex_theta: Texture amplitude [m]
            shift_theta: Texture shift [m]
            d_texture: Texture spacing [m]
            
        Returns:
            Texture height array [m]
        """
        if atex_theta <= 0.0 or a_theta <= 0.0:
            return np.zeros_like(x)
            
        # Periodic texture with shift
        u = ((x - self.x_start - shift_theta + d_texture/2.0) % d_texture) - d_texture/2.0
        expo = (np.log(self.g_val / atex_theta) * (u**2)) / (self.w_texture + 1e-30)
        h_tex = atex_theta * np.exp(expo)
        
        # Mask to contact region
        return np.where((x >= -a_theta) & (x <= a_theta), h_tex, 0.0)
        
    def rusanov_advection(self, u: np.ndarray, q: np.ndarray, dx: float,
                         q_in_left: float, q_in_right: float) -> np.ndarray:
        """
        Rusanov flux for advection equation with boundary conditions
        
        Args:
            u: Velocity array
            q: Transported quantity array
            dx: Grid spacing
            q_in_left: Left boundary value
            q_in_right: Right boundary value
            
        Returns:
            Divergence array
        """
        N = len(q)
        qL = np.empty(N + 1)
        qR = np.empty(N + 1)
        
        qL[1:] = q
        qR[:-1] = q
        qL[0] = q_in_left
        qR[0] = q[0]
        qL[-1] = q[-1]
        qR[-1] = q_in_right
        
        F = 0.5 * (u * (qL + qR)) - 0.5 * np.abs(u) * (qR - qL)
        return (F[1:] - F[:-1]) / (dx + 1e-30)
        
    def solve_reynolds_at_angle(self, R: float, Ve: float, Vs: float, W: float, 
                               dt: float, angle_deg: float, rpm: float,
                               atex_theta: float, shift_theta: float, 
                               d_texture: float, **kwargs) -> Dict[str, Any]:
        """
        Solve Reynolds equation at a single cam angle
        
        Args:
            R: Effective radius [m]
            Ve: Entraining velocity [m/s]
            Vs: Sliding velocity [m/s]
            W: Normal load [N]
            dt: Time step [s]
            angle_deg: Cam angle [degrees]
            rpm: Rotational speed [rev/min]
            atex_theta: Texture amplitude [m]
            shift_theta: Texture shift [m]
            d_texture: Texture spacing [m]
            **kwargs: Additional solver parameters
            
        Returns:
            Dictionary containing solution results
        """
        # Extract solver parameters
        Nx = kwargs.get('Nx', self.Nx)
        iters = kwargs.get('iters', self.iters)
        substep_cap = kwargs.get('substep_cap', self.substep_cap)
        relax_p = kwargs.get('relax_p', self.relax_p)
        relax_h = kwargs.get('relax_h', self.relax_h)
        M_core = kwargs.get('M_core', self.M_core)
        
        # Ensure minimum values
        R = float(max(R, 1e-12))
        W = float(max(W, 1.0))
        
        # Hertzian contact parameters
        a, ph = self.hertz_contact(W, R)
        a = max(a, 2e-6)
        ph = max(ph, 1e3)
        
        # Create spatial grid
        xL, xR = self.X_in * a, self.X_out * a
        x = np.linspace(xL, xR, Nx)
        dx = x[1] - x[0]
        
        # Core pressure grid (normalized coordinates)
        s = np.linspace(-1.0, 1.0, M_core)
        xs = a * s
        dS = s[1] - s[0]
        
        # Initial film thickness estimate
        h0 = self.central_film_thickness(R, W, Ve)
        
        # Add texture contribution
        htex = self.texture_profile(x, a, atex_theta, shift_theta, d_texture)
        
        # Initial film thickness distribution
        h = np.maximum(h0 + x**2/(2*R) + htex, 5e-9)
        
        # Initialize content field (mass conservation)
        p_zero = np.zeros_like(x)
        Phi = (self.density_dowson_higginson(p_zero) / self.rho0) * ((h * R) / ph)
        phi_in = 0.5
        G = phi_in * Phi
        
        # Initialize core pressure with Hertzian shape
        P_core = np.sqrt(np.maximum(1.0 - s**2, 0.0))
        
        # Transport parameters
        u_nd = 1.0
        dX = dx / max(a, 1e-12)
        cfl = abs(Ve) * dt / (dx + 1e-30)
        substeps = int(min(max(2, np.ceil(cfl / 0.35)), substep_cap))
        dts = dt / max(substeps, 1)
        dT = dts * max(abs(Ve), 0.05) / max(a, 1e-12)
        
        def embed_pressure(P_core_vec: np.ndarray) -> np.ndarray:
            """Embed core pressure into full domain"""
            p_full = np.zeros_like(x)
            inside = (x >= -a) & (x <= a)
            if inside.any():
                P_vals = np.interp(x[inside], xs, np.maximum(P_core_vec, 0.0))
                p_full[inside] = P_vals * ph
            return p_full
            
        # Smoothing parameters
        K0, K1 = 0.55, 0.225
        
        # Main time-stepping loop
        for substep in range(substeps):
            # Transport step
            p_tr = embed_pressure(P_core)
            rho_nd = self.density_dowson_higginson(p_tr) / self.rho0
            H_nd = (h * R) / ph
            Phi = rho_nd * H_nd
            
            G_in_L = phi_in * Phi[0]
            G_in_R = phi_in * Phi[-1]
            div_phi = self.rusanov_advection(u_nd, G, dX, G_in_L, G_in_R)
            S_nd = (Phi - G) / max(dT, 1e-12) + div_phi
            S_core_nd = np.interp(xs, x, S_nd)
            
            # Pressure iteration loop
            for iteration in range(iters):
                # Calculate diffusion coefficients
                p_embed = embed_pressure(P_core)
                rho_nd = self.density_dowson_higginson(p_embed) / self.rho0
                H_nd = (h * R) / ph
                eta_nd = self.viscosity_houpert(p_embed) / self.eta0
                
                rho_core = np.interp(xs, x, rho_nd)
                H_core = np.interp(xs, x, H_nd)
                eta_core = np.interp(xs, x, np.maximum(eta_nd, 1e-7))
                D_core = np.maximum(rho_core * H_core**3 / eta_core, 1e-12)
                
                # Assemble tridiagonal system
                M = len(xs)
                A = np.zeros(M)
                B = np.zeros(M)
                C = np.zeros(M)
                RHS = np.zeros(M)
                invdS2 = 1.0 / (dS * dS + 1e-30)
                
                # Boundary conditions (P = 0 at edges)
                B[0] = 1.0
                RHS[0] = 0.0
                
                # Interior points
                for j in range(1, M-1):
                    Dw = 0.5 * (D_core[j] + D_core[j-1])
                    De = 0.5 * (D_core[j] + D_core[j+1])
                    A[j] = -Dw * invdS2
                    C[j] = -De * invdS2
                    B[j] = -(A[j] + C[j]) + 1e-12
                    RHS[j] = S_core_nd[j]
                    
                B[M-1] = 1.0
                RHS[M-1] = 0.0
                
                # Thomas algorithm
                for j in range(1, M):
                    wfac = A[j] / (B[j-1] + 1e-30)
                    B[j] -= wfac * C[j-1]
                    RHS[j] -= wfac * RHS[j-1]
                    
                P_new = np.zeros(M)
                P_new[-1] = RHS[-1] / (B[-1] + 1e-30)
                for j in range(M-2, -1, -1):
                    P_new[j] = (RHS[j] - C[j] * P_new[j+1]) / (B[j] + 1e-30)
                    
                # Ensure positivity and load balance
                P_new = np.maximum(P_new, 0.0)
                Wh_trial = np.trapz(P_new * ph, xs) * self.L
                s_load = 1.0 if Wh_trial <= 1e-20 else np.clip(W / Wh_trial, 1e-3, 1e3)
                
                # Relaxation and smoothing
                P_core = (1 - relax_p) * P_core + relax_p * np.maximum(P_new * s_load, 0.0)
                Ptmp = P_core.copy()
                for j in range(1, len(P_core)-1):
                    P_core[j] = K1 * Ptmp[j-1] + K0 * Ptmp[j] + K1 * Ptmp[j+1]
                P_core[0] = 0.0
                P_core[-1] = 0.0
                
                # Mixed lubrication closure
                p_embed = embed_pressure(P_core)
                defl = self.elastic_deflection(x, p_embed)
                
                # Update texture profile
                htex = self.texture_profile(x, a, atex_theta, shift_theta, d_texture)
                
                # Update film thickness
                h_nom = np.maximum(h0 + x**2/(2*R) + defl + htex, 5e-9)
                h = np.maximum(relax_h * h + (1.0 - relax_h) * h_nom, 5e-9)
                
                # Asperity contact
                p_asp = self.asperity_pressure_greenwood_tripp(h)
                Wa = np.trapz(p_asp, x) * self.L
                Wh = np.trapz(p_embed, x) * self.L
                Wmix = Wh + Wa
                
                s_mix = (W / Wmix) if Wmix > 1e-20 else 1.0
                s_mix = max(s_mix, 0.0)
                P_core *= s_mix
                
                # Final smoothing
                Ptmp = P_core.copy()
                for j in range(1, len(P_core)-1):
                    P_core[j] = K1 * Ptmp[j-1] + K0 * Ptmp[j] + K1 * Ptmp[j+1]
                P_core[0] = 0.0
                P_core[-1] = 0.0
                
            # Update content field
            p_tr = embed_pressure(P_core)
            rho_nd = self.density_dowson_higginson(p_tr) / self.rho0
            Phi = rho_nd * ((h * R) / ph)
            G_in_L = 0.5 * Phi[0]
            G_in_R = 0.5 * Phi[-1]
            div_phi = self.rusanov_advection(1.0, G, dx/max(a,1e-12), G_in_L, G_in_R)
            G = np.clip(G + dT * (-div_phi), 0.0, Phi)
            
        # Final solution
        p = embed_pressure(P_core)
        p_asp_final = self.asperity_pressure_greenwood_tripp(h)
        Wa_final = np.trapz(p_asp_final, x) * self.L
        Wh_now = np.trapz(p, x) * self.L
        
        # Final load balance
        if Wh_now + Wa_final > 1e-20:
            s_final = W / (Wh_now + Wa_final)
            s_final = max(s_final, 0.0)
            p *= s_final
            defl_final = self.elastic_deflection(x, p)
            htex = self.texture_profile(x, a, atex_theta, shift_theta, d_texture)
            h = np.maximum(h0 + x**2/(2*R) + defl_final + htex, 5e-9)
            
        # Calculate friction forces
        eta_eff = self.viscosity_carreau(self.viscosity_houpert(p), h, abs(Ve))
        tau_h = np.where(h > 1e-12, eta_eff * abs(Ve) / h, 0.0)
        Fh = np.trapz(tau_h, x) * self.L
        
        tau_lim = self.gamma_lim * np.maximum(p, 0.0)
        Fb = self.L * np.trapz(tau_lim + self.mu_b * p_asp_final, x)
        
        return {
            "x": x,
            "p": p,
            "h": h,
            "p_asp": p_asp_final,
            "Fh": float(Fh),
            "Fb": float(Fb),
            "Wa": float(Wa_final),
            "Wh": float(Wh_now),
            "a": float(a),
            "pmax": float(np.max(p)),
            "hmin": float(np.min(h)),
            "htex": htex
        }
        
    def analyze_friction_reduction(self, density_key: str = "5%", 
                                 rpms: Optional[list] = None) -> pd.DataFrame:
        """
        Analyze friction torque reduction for textured vs untextured surfaces
        
        Args:
            density_key: Texture density key
            rpms: List of RPMs to analyze (if None, uses available RPMs)
            
        Returns:
            DataFrame with friction reduction results
        """
        if density_key not in self.atex_tables:
            raise ValueError(f"Texture data for {density_key} not available")
            
        # Get available RPM columns
        df = self.atex_tables[density_key]
        rpm_cols = [c for c in df.columns if c.startswith("RPM")]
        available_rpms = sorted(int(c.replace("RPM", "")) for c in rpm_cols)
        
        if rpms is None:
            rpms = available_rpms
        else:
            rpms = [rpm for rpm in rpms if rpm in available_rpms]
            
        if not rpms:
            raise ValueError("No valid RPMs found")
            
        results = []
        d_tex = self.D_TEXTURE[density_key]
        
        for rpm in rpms:
            print(f"Analyzing RPM {rpm}...")
            
            # Calculate kinematics
            R, Ve, Vs, W, omega = self.calculate_kinematics(rpm)
            dt = np.gradient(self.th) / max(omega, 1e-30)
            shift = self.integrate_shift(Vs, omega)
            atex = np.nan_to_num(self.get_texture_amplitude(density_key, rpm), nan=0.0)
            
            # Accumulate torque over cam cycle
            T_smooth, T_textured = [], []
            
            for i, ang in enumerate(self.th_deg):
                # Untextured case
                res_smooth = self.solve_reynolds_at_angle(
                    R[i], Ve[i], Vs[i], W[i], dt[i], ang, rpm,
                    atex_theta=0.0, shift_theta=shift[i], d_texture=d_tex
                )
                
                # Textured case
                res_textured = self.solve_reynolds_at_angle(
                    R[i], Ve[i], Vs[i], W[i], dt[i], ang, rpm,
                    atex_theta=float(max(atex[i], 0.0)), 
                    shift_theta=shift[i], d_texture=d_tex
                )
                
                # Calculate torque (force × effective radius)
                r_eff = self.rb + self.lift_s[i]
                T_smooth.append((res_smooth["Fh"] + res_smooth["Fb"]) * r_eff)
                T_textured.append((res_textured["Fh"] + res_textured["Fb"]) * r_eff)
                
            # Calculate average torques and reduction
            Tavg_smooth = float(np.mean(T_smooth))
            Tavg_textured = float(np.mean(T_textured))
            pct_reduction = 100.0 * (1.0 - Tavg_textured / max(Tavg_smooth, 1e-30))
            
            results.append({
                "RPM": rpm,
                "Tavg_smooth": Tavg_smooth,
                "Tavg_textured": Tavg_textured,
                "Pct_Reduction": pct_reduction
            })
            
        return pd.DataFrame(results)
        
    def plot_results(self, results: Dict[str, Any], title: str = "Mixed Lubrication Results"):
        """
        Plot solution results
        
        Args:
            results: Solution dictionary from solve_reynolds_at_angle
            title: Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        x = results["x"] * 1e6  # Convert to μm
        
        # Pressure distribution
        ax1.plot(x, results["p"] * 1e-6, 'b-', linewidth=2, label='Hydrodynamic')
        ax1.plot(x, results["p_asp"] * 1e-6, 'r--', linewidth=2, label='Asperity')
        ax1.set_xlabel('Position [μm]')
        ax1.set_ylabel('Pressure [MPa]')
        ax1.set_title('Pressure Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Film thickness
        ax2.plot(x, results["h"] * 1e9, 'g-', linewidth=2, label='Total')
        if "htex" in results:
            ax2.plot(x, results["htex"] * 1e9, 'k:', linewidth=1, label='Texture')
        ax2.set_xlabel('Position [μm]')
        ax2.set_ylabel('Film Thickness [nm]')
        ax2.set_title('Film Thickness Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Load distribution
        total_load = results["Wh"] + results["Wa"]
        load_ratio_h = results["Wh"] / max(total_load, 1e-30) * 100
        load_ratio_a = results["Wa"] / max(total_load, 1e-30) * 100
        
        ax3.bar(['Hydrodynamic', 'Asperity'], [load_ratio_h, load_ratio_a], 
                color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Load Share [%]')
        ax3.set_title('Load Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""Contact Half-width: {results["a"]*1e6:.1f} μm
Max Pressure: {results["pmax"]*1e-6:.1f} MPa
Min Film Thickness: {results["hmin"]*1e9:.1f} nm
Hydrodynamic Force: {results["Fh"]:.2f} N
Boundary Force: {results["Fb"]:.2f} N
Total Force: {results["Fh"]+results["Fb"]:.2f} N"""
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Solution Summary')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=14)
        plt.show()
        
    def run_example(self):
        """Run example analysis"""
        print("1D Transient Reynolds Equation Mixed Lubrication Solver")
        print("=" * 60)
        
        # Example: Analyze single operating point
        rpm = 500
        angle_deg = 0.0  # TDC
        density_key = "5%"
        
        print(f"\nAnalyzing single point: RPM={rpm}, Angle={angle_deg}°, Texture={density_key}")
        
        # Calculate kinematics
        R, Ve, Vs, W, omega = self.calculate_kinematics(rpm)
        
        # Find index closest to desired angle
        idx = np.argmin(np.abs(self.th_deg - angle_deg))
        
        # Get texture parameters
        dt = np.gradient(self.th)[idx] / max(omega, 1e-30)
        shift = self.integrate_shift(Vs, omega)
        atex = self.get_texture_amplitude(density_key, rpm)
        d_tex = self.D_TEXTURE[density_key]
        
        # Solve for textured case
        results = self.solve_reynolds_at_angle(
            R[idx], Ve[idx], Vs[idx], W[idx], dt, angle_deg, rpm,
            atex_theta=float(max(atex[idx], 0.0)), 
            shift_theta=shift[idx], d_texture=d_tex
        )
        
        # Plot results
        self.plot_results(results, f"Mixed Lubrication at {rpm} RPM, {angle_deg}°")
        
        # Friction reduction analysis
        print(f"\nPerforming friction reduction analysis for {density_key} texture...")
        reduction_results = self.analyze_friction_reduction(density_key, rpms=[300, 500, 700, 900])
        
        print("\nFriction Torque Reduction Results:")
        print(reduction_results.to_string(index=False, 
              formatters={"Pct_Reduction": lambda v: f"{v:6.2f}%"}))
        
        return results, reduction_results


def main():
    """Main function to run the mixed lubrication solver"""
    try:
        # Initialize solver
        solver = Reynolds1DMixedLubrication(data_dir=".")
        
        # Run example analysis
        results, reduction_results = solver.run_example()
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()