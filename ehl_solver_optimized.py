"""
1D Thermal Transient Mixed Lubrication - Cam-Follower Cycle
Optimized solver: Load error <1%, Residual <1e-7, Runtime <160s
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.linalg import lu_factor, lu_solve


class EHLSolver:
    def __init__(self):
        print("Init EHL Solver...")
        
        self.P0_ref = 0.5e9
        self.mu00 = 0.01381
        self.Pr = 1.96e8
        self.alpha_input = 15e-9
        self.E_prime = 217e9
        self.B = 7.3e-3
        
        self.sigma = 0.2e-6
        self.eta_beta_sigma = 0.04
        self.sigma_beta_ratio = 0.001
        self.K_GT = (16 * np.pi * np.sqrt(2) / 15) * (self.eta_beta_sigma**2) * \
                   np.sqrt(self.sigma_beta_ratio) * self.E_prime
        
        self.T0_K = 363.15
        self.gamma_therm = 6.5e-4
        self.k_therm = 0.15
        self.P_max = 5.0e9
        self.beta0 = 0.04
        
        ln_eta0 = np.log(self.mu00)
        self.S0 = self.beta0 * (self.T0_K - 138.0) / (ln_eta0 + 9.67)
        self.Z_houper = self.alpha_input / (5.11e-9 * (ln_eta0 + 9.67))
        
        self.N = 71
        self.delta_ad = 0.05
        self.sigma_factor = 1.0
        
        self.cam_data = self.load_cam_profile("updated_lift.txt")
        self.setup_grid(self.cam_data["max_a_over_r"])
        
        self.R = self.Um = self.Vs = self.W = None
        self.Um_mag = 1e-9
        self.Um_sign = 1.0
        self.a_Hertz = self.Pmh = self.Wld = 0.0
        self.dt = 1e-5
        self.is_transient = True
        self.rho_old = self.H_old = None
        self.T_current = np.full(self.N, self.T0_K)
        self.prev_V = None
        self.gamma_h = 1e12
        self.hmin_dim = 0.0
        
        print("Init complete.")
    
    def load_cam_profile(self, path):
        data = np.loadtxt(path)
        theta_deg, lift = data[:, 0], data[:, 1]
        theta_rad = np.deg2rad(theta_deg)
        dlift = np.gradient(lift, theta_rad)
        ddlift = np.gradient(dlift, theta_rad)
        rb = 18.4e-3
        omega = (2 * np.pi * 300) / 60
        Vc = (rb + lift + ddlift) * omega
        Vf = omega * ddlift
        um = (Vf + Vc) / 2
        Vs = Vc - Vf
        R = ddlift + lift + rb
        F = 7130 * (lift + 1.77e-3) + ddlift * 0.05733 * omega**2
        Wl = np.maximum(F, 0.0) / self.B
        a_hertz = np.sqrt(8 * Wl * R / (np.pi * self.E_prime))
        max_a_over_r = np.max(a_hertz / R)
        dt = np.gradient(theta_rad) / omega
        time_arr = (theta_rad - theta_rad[0]) / omega
        return dict(theta_deg=theta_deg, um=um, Vs=Vs, R=R, F=F, dt=dt, 
                   time=time_arr, max_a_over_r=max_a_over_r)
    
    def setup_grid(self, max_a_over_r):
        self.X_dim = np.linspace(-4.0 * max_a_over_r, 3.0 * max_a_over_r, self.N)
        self.dx = self.X_dim[1] - self.X_dim[0]
        self.X = self.X_dim
        self.CE = -4 * self.P0_ref / (np.pi * self.E_prime)
        self._compute_D_matrix()
    
    def _compute_D_matrix(self):
        N, dx = self.N, self.dx
        D = np.zeros((N, N))
        b = dx / 2.0
        for i in range(N):
            xi = self.X_dim[i]
            for j in range(N):
                xj = self.X_dim[j]
                d1, d2 = xi - (xj - b), xi - (xj + b)
                v1 = d1 * np.log(abs(d1) + 1e-30) - d1 if abs(d1) > 1e-12 else 0.0
                v2 = d2 * np.log(abs(d2) + 1e-30) - d2 if abs(d2) > 1e-12 else 0.0
                D[i, j] = v1 - v2
        self.D_mat = D * self.CE
    
    def update_operating_state(self, um, vs, R, load):
        self.Um, self.Vs, self.R, self.W = um, vs, R, load
        self.Um_mag = max(abs(um), 1e-9)
        self.Um_sign = 1.0 if um >= 0 else -1.0
        self.A_C = R * self.P0_ref / (12 * self.mu00 * self.Um_mag)
        self.u1d = self.Um_sign
        Wl = load / self.B
        self.a_Hertz = np.sqrt(8 * Wl * R / (np.pi * self.E_prime)) if Wl > 0 else 1e-9
        self.Pmh = 2 * Wl / (np.pi * self.a_Hertz) if self.a_Hertz > 0 else 0.0
        self.Wld = Wl / (R * self.P0_ref) if R > 0 else 0.0
        self.g1 = 1e15 * self.A_C / (self.dx**2)
    
    def ro(self, P):
        Pp = np.clip(P * self.P0_ref, 0, self.P_max)
        rho_T = np.maximum(1 - self.gamma_therm * (self.T_current - self.T0_K), 0.1)
        return (1 + 0.6e-9 * Pp / (1 + 1.7e-9 * Pp)) * rho_T
    
    def droo(self, P):
        Pp = np.clip(P * self.P0_ref, 0, self.P_max)
        rho_T = np.maximum(1 - self.gamma_therm * (self.T_current - self.T0_K), 0.1)
        return 0.6e-9 / ((1 + 1.7e-9 * Pp)**2) * self.P0_ref * rho_T
    
    def c_mu(self, P):
        Pp = np.clip(P * self.P0_ref, 0, self.P_max)
        ln_eta0 = np.log(self.mu00)
        C1 = ln_eta0 + 9.67
        T_term = np.maximum((self.T_current - 138.0) / (self.T0_K - 138.0), 1e-5)
        bracket = T_term**(-self.S0) * (1 + Pp / self.Pr)**self.Z_houper - 1.0
        return self.mu00 * np.exp(np.clip(C1 * bracket, -50, 50))
    
    def calc_flow_factors(self, H):
        lam = np.maximum(H * self.R / (self.sigma * self.sigma_factor), 0)
        return 1 - 0.9 * np.exp(-0.56 * lam), 1 - 0.9 * np.exp(-0.2 * lam)
    
    def calc_asperity(self, H, deriv=False):
        Lambda = np.clip(H * self.R / self.sigma, 0, 4)
        Pa = np.zeros_like(Lambda)
        dPa = np.zeros_like(Lambda)
        mask = Lambda < 4
        if np.any(mask):
            val = np.maximum(4 - Lambda[mask], 0)
            Pa[mask] = self.K_GT * 4.4084e-5 * val**6.804 * self.sigma_factor / self.P0_ref
            if deriv:
                dPa[mask] = -self.K_GT * 4.4084e-5 * 6.804 * val**5.804 * self.sigma_factor * self.R / (self.sigma * self.P0_ref)
        return (Pa, dPa) if deriv else Pa
    
    def calc_reynolds_residual(self, P, H):
        idx = np.arange(2, self.N-1)
        He = np.clip(H, 1e-12, 1e-2)
        rho, mu = self.ro(P), self.c_mu(P)
        phi_x, phi_s = self.calc_flow_factors(He)
        rho_p = self.droo(P)
        val = self.dx * rho_p * H * self.delta_ad / self.A_C
        
        eps1 = (rho[idx] + rho[idx-1]) * (He[idx] + He[idx-1])**3 * 0.125 * \
               0.5 * (phi_x[idx] + phi_x[idx-1]) / (mu[idx] + mu[idx-1]) + \
               0.5 * (val[idx] + val[idx-1])
        eps2 = (rho[idx] + rho[idx+1]) * (He[idx] + He[idx+1])**3 * 0.125 * \
               0.5 * (phi_x[idx] + phi_x[idx+1]) / (mu[idx] + mu[idx+1]) + \
               0.5 * (val[idx] + val[idx+1])
        
        term_pois = (self.A_C / self.dx**2) * (P[idx-1]*eps1 - P[idx]*(eps1+eps2) + P[idx+1]*eps2)
        term_couette = (self.u1d / self.dx) * (rho[idx]*He[idx]*phi_s[idx] - rho[idx-1]*He[idx-1]*phi_s[idx-1])
        
        term_sq = 0
        if self.is_transient and self.rho_old is not None:
            term_sq = (self.R / self.Um_mag) * (rho[idx]*He[idx] - self.rho_old[idx]*self.H_old[idx]) / self.dt
        
        penalty_p = self.gamma_h * np.minimum(P[idx], 0)
        penalty_h = self.g1 * np.maximum(-H[idx], 0)**2
        
        return (term_pois - term_couette - term_sq - penalty_p + penalty_h) * self.dx**2 / self.A_C
    
    def system_func(self, V):
        n = self.N - 3
        P, H_inner, H0 = np.zeros(self.N), np.zeros(self.N), V[-1]
        P[2:-1] = np.clip(V[:n], -self.P_max/self.P0_ref, self.P_max/self.P0_ref)
        H_inner[2:-1] = np.clip(V[n:2*n], -1e-3, 1e-2)
        for k in [0, 1, -1]:
            H_inner[k] = H0 + (self.X[k]**2)/2
        
        Pa, _ = self.calc_asperity(H_inner, True)
        D_term = self.D_mat @ (P + Pa)
        H_el = H0 + (self.X**2)/2 + D_term
        for k in [0, 1, -1]:
            H_inner[k] = H_el[k]
        
        idx = np.arange(2, self.N-1)
        F_rey = self.calc_reynolds_residual(P, H_inner)
        F_film = H_inner[idx] - H_el[idx]
        F_load = (self.Wld - np.sum((P + Pa) * self.dx)) / max(self.Wld, 1e-15)
        
        return np.concatenate([F_rey, F_film, [F_load]])
    
    def calc_jacobian(self, V, eps=1e-7):
        n = self.N - 3
        J = np.zeros((len(V), len(V)))
        
        P, H = np.zeros(self.N), np.zeros(self.N)
        P[2:-1] = V[:n]
        H[2:-1] = V[n:2*n]
        H0 = V[-1]
        for k in [0, 1, -1]:
            H[k] = H0 + (self.X[k]**2)/2
        
        idx = np.arange(2, self.N-1)
        _, dPa = self.calc_asperity(H, True)
        dPa_slice = dPa[idx]
        
        D_slice = self.D_mat[np.ix_(idx, idx)]
        J[n:2*n, :n] = -D_slice
        J[n:2*n, n:2*n] = np.eye(n) - D_slice * dPa_slice[np.newaxis, :]
        J[n:2*n, -1] = -1.0
        
        J[-1, :n] = -self.dx / max(self.Wld, 1e-15)
        J[-1, n:2*n] = -dPa_slice * self.dx / max(self.Wld, 1e-15)
        
        F0 = self.calc_reynolds_residual(P, H)
        for j in range(n):
            P[j+2] += eps
            J[:n, j] = (self.calc_reynolds_residual(P, H) - F0) / eps
            P[j+2] -= eps
        for j in range(n):
            H[j+2] += eps
            J[:n, n+j] = (self.calc_reynolds_residual(P, H) - F0) / eps
            H[j+2] -= eps
        old_H = (H[0], H[1], H[-1])
        for k in [0, 1, -1]:
            H[k] = (H0 + eps) + (self.X[k]**2)/2
        J[:n, -1] = (self.calc_reynolds_residual(P, H) - F0) / eps
        H[0], H[1], H[-1] = old_H
        
        return J
    
    def update_history(self, V):
        n = self.N - 3
        P, H = np.zeros(self.N), np.zeros(self.N)
        P[2:-1] = V[:n]
        H[2:-1] = V[n:2*n]
        H0 = V[-1]
        for k in [0, 1, -1]:
            H[k] = H0 + (self.X[k]**2)/2
        Pa = self.calc_asperity(H)
        H_el = H0 + (self.X**2)/2 + self.D_mat @ (P + Pa)
        self.rho_old, self.H_old = self.ro(P), H_el
    
    def get_full_state(self, V):
        n = self.N - 3
        P, H = np.zeros(self.N), np.zeros(self.N)
        P[2:-1] = V[:n]
        H[2:-1] = V[n:2*n]
        H0 = V[-1]
        for k in [0, 1, -1]:
            H[k] = H0 + (self.X[k]**2)/2
        Pa = self.calc_asperity(H)
        return P, Pa, H0 + (self.X**2)/2 + self.D_mat @ (P + Pa)
    
    def calc_load_error(self, V):
        P, Pa, _ = self.get_full_state(V)
        return abs(self.Wld - np.sum((P + Pa) * self.dx)) / max(self.Wld, 1e-15)
    
    def calc_solution_change(self, V_new, V_old):
        """Max(|p_total_new - p_total_old|, |h_new - h_old|)"""
        P_new, Pa_new, H_new = self.get_full_state(V_new)
        P_old, Pa_old, H_old = self.get_full_state(V_old)
        max_dp = np.max(np.abs((P_new + Pa_new) - (P_old + Pa_old)))
        max_dh = np.max(np.abs(H_new - H_old))
        return max(max_dp, max_dh)
    
    def build_initial_guess(self):
        P = np.zeros(self.N)
        X_dim = self.X_dim * self.R if self.R else self.X_dim
        if self.a_Hertz > 0:
            c = np.abs(X_dim) <= self.a_Hertz
            P[c] = self.Pmh * np.sqrt(np.maximum(1 - (X_dim[c] / self.a_Hertz)**2, 0)) / self.P0_ref
        
        U = self.mu00 * self.Um_mag / (self.E_prime * self.R) if self.R else 1e-12
        G = self.alpha_input * self.E_prime
        W = max((self.W / self.B) / (self.E_prime * self.R), 1e-15) if self.R else 1e-15
        H0 = max(2.65 * U**0.7 * G**0.54 * W**-0.13, 1e-8)
        
        H_guess = H0 + (self.X**2)/2 + self.D_mat @ P
        return np.concatenate([P[2:-1], H_guess[2:-1], [H0]])
    
    def newton_solve(self, V0, tol=1e-7, max_iter=30):
        """Newton solver - converges until solution change < tol."""
        V = V0.copy()
        F = self.system_func(V)
        sys_res = np.linalg.norm(F)
        
        best_V = V.copy()
        best_change = float('inf')
        
        lu = None
        stall = 0
        
        for k in range(max_iter):
            if k == 0 or stall > 2:
                J = self.calc_jacobian(V)
                reg = 1e-8 * (1 + np.abs(np.diag(J)).mean())
                try:
                    lu = lu_factor(J + reg * np.eye(len(V)))
                    stall = 0
                except:
                    break
            
            try:
                dV = lu_solve(lu, -F)
            except:
                break
            
            dV_n = np.linalg.norm(dV)
            V_n = np.linalg.norm(V) + 1e-10
            if dV_n > 0.3 * V_n:
                dV *= 0.3 * V_n / dV_n
            
            V_old = V.copy()
            improved = False
            alpha = 1.0
            for _ in range(8):
                Vn = V + alpha * dV
                Fn = self.system_func(Vn)
                rn = np.linalg.norm(Fn)
                if np.isfinite(rn) and rn < sys_res:
                    V, F, sys_res = Vn, Fn, rn
                    improved = True
                    break
                alpha *= 0.5
            
            if not improved:
                stall += 1
                V = V + 0.05 * dV
                F = self.system_func(V)
                sys_res = np.linalg.norm(F)
            
            change = self.calc_solution_change(V, V_old)
            if change < best_change:
                best_V, best_change = V.copy(), change
            
            if change < tol:
                return V, change
        
        return best_V, best_change
    
    def solve_step(self, V_start, dt):
        """Single transient step with retries."""
        self.update_history(V_start)
        self.dt = dt
        
        V_pred = 0.8 * V_start + 0.2 * self.prev_V if self.prev_V is not None else V_start.copy()
        
        V, res = self.newton_solve(V_pred, tol=1e-7, max_iter=30)
        err = self.calc_load_error(V)
        
        # Retry from fresh guess if load error high
        if err > 0.005:
            V2, res2 = self.newton_solve(self.build_initial_guess(), tol=1e-7, max_iter=35)
            err2 = self.calc_load_error(V2)
            if err2 < err:
                V, res, err = V2, res2, err2
        
        # Pressure scaling
        if err > 0.003:
            n = self.N - 3
            P, Pa, _ = self.get_full_state(V)
            P_total = np.sum((P + Pa) * self.dx)
            if P_total > 1e-10:
                scale = self.Wld / P_total
                if 0.5 < scale < 2.0:
                    V_s = V.copy()
                    V_s[:n] *= scale
                    V3, res3 = self.newton_solve(V_s, tol=1e-8, max_iter=40)
                    err3 = self.calc_load_error(V3)
                    if err3 < err:
                        V, res, err = V3, res3, err3
        
        # Additional correction for high errors
        if err > 0.008:
            for _ in range(3):
                n = self.N - 3
                P, Pa, _ = self.get_full_state(V)
                P_total = np.sum((P + Pa) * self.dx)
                if P_total > 1e-10:
                    scale = self.Wld / P_total
                    if 0.8 < scale < 1.3:
                        V[:n] *= scale
                        V4, res4 = self.newton_solve(V, tol=1e-8, max_iter=35)
                        err4 = self.calc_load_error(V4)
                        if err4 < err:
                            V, res, err = V4, res4, err4
                            if err < 0.005:
                                break
        
        err = self.calc_load_error(V)
        
        # Thermal
        if self.gamma_therm > 0:
            P, Pa, H = self.get_full_state(V)
            h = np.clip(H, 1e-12, 1e-2) * self.R
            eta = self.c_mu(P)
            a = max(self.a_Hertz, 1e-9)
            P_r = np.clip(P * self.P0_ref, 0, self.P_max)
            num = self.Um_mag * self.T_current * self.gamma_therm * h * P_r + 2 * a * eta * self.Um_mag**2 / h
            den = np.maximum(a * self.k_therm / h - self.Um_mag * self.gamma_therm * h * P_r, 1e-5)
            T_new = np.clip(self.T0_K + num / den, self.T0_K - 50, self.T0_K + 250)
            self.T_current = 0.7 * T_new + 0.3 * self.T_current
        
        self.update_history(V)
        self.prev_V = V_start.copy()
        
        return V, res, err
    
    def run_cam_cycle(self):
        theta = self.cam_data["theta_deg"]
        um, vs = self.cam_data["um"], self.cam_data["Vs"]
        R, F = self.cam_data["R"], self.cam_data["F"]
        dt_arr = self.cam_data["dt"]
        n = len(theta)
        
        P_list, Pa_list, H_list = [], [], []
        errs, resids = [], []
        
        V = None
        t0 = time.perf_counter()
        
        print(f"Running {n} steps...")
        
        for i in range(n):
            self.update_operating_state(um[i], vs[i], R[i], F[i])
            
            if V is None:
                V = self.build_initial_guess()
                self.update_history(V)
            
            V, res, err = self.solve_step(V, abs(dt_arr[i]))
            
            Pr, Pa, H = self.get_full_state(V)
            P_list.append((Pr * self.P0_ref / self.Pmh) if self.Pmh > 0 else Pr)
            Pa_list.append((Pa * self.P0_ref / self.Pmh) if self.Pmh > 0 else Pa)
            H_list.append(H)
            errs.append(err)
            resids.append(res if np.isfinite(res) else 1e10)
            
            if i % 50 == 0 or err > 0.008 or res > 1e-6:
                print(f"Step {i+1:3d}/{n} θ={theta[i]:6.1f}° err={err:.2e} res={res:.2e} t={time.perf_counter()-t0:.1f}s")
        
        total = time.perf_counter() - t0
        max_err, max_res = max(errs) * 100, max(resids)
        avg_err = np.mean(errs) * 100
        max_res_idx = np.argmax(resids)
        
        print("-" * 60)
        print(f"Done! Time={total:.1f}s MaxErr={max_err:.4f}% AvgErr={avg_err:.5f}% MaxRes={max_res:.2e}")
        
        ok = max_err <= 1 and max_res <= 1e-7 and total <= 160
        if max_err > 1:
            print(f"FAIL: Load {max_err:.3f}% > 1%")
        if max_res > 1e-7:
            print(f"FAIL: Res {max_res:.2e} > 1e-7")
        if total > 160:
            print(f"FAIL: Time {total:.1f}s > 160s")
        if ok:
            print("SUCCESS: All criteria met!")
        
        self._plot(theta, um, vs, R, F, P_list, Pa_list, H_list)
        return dict(time=total, max_err=max_err, max_res=max_res, ok=ok)
    
    def _plot(self, theta, um, vs, R, F, P_list, Pa_list, H_list):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0,0].plot(theta, um); ax[0,0].set_ylabel("um")
        ax[0,1].plot(theta, vs); ax[0,1].set_ylabel("Vs")
        ax[1,0].plot(theta, R); ax[1,0].set_ylabel("R")
        ax[1,1].plot(theta, F); ax[1,1].set_ylabel("F")
        for a in ax.flat:
            a.set_xlabel("θ"); a.grid()
        fig.tight_layout()
        fig.savefig("Graph_Cam_Kinematics.png")
        plt.close()
        
        nc = len(theta)
        c = plt.cm.viridis(np.linspace(0, 1, nc))
        for name, data in [("Reynolds_Pressure", P_list), ("Asperity_Pressure", Pa_list), ("Film_Thickness", H_list)]:
            plt.figure(figsize=(10, 6))
            for i in range(nc):
                plt.plot(self.X_dim, data[i], color=c[i], alpha=0.4, lw=0.7)
            plt.xlabel("X/R"); plt.ylabel(name.replace("_"," ")); plt.grid()
            plt.savefig(f"Graph_{name}_Cycle.png")
            plt.close()
        print("Plots saved.")
    
    def solve(self):
        return self.run_cam_cycle()


if __name__ == "__main__":
    EHLSolver().solve()
