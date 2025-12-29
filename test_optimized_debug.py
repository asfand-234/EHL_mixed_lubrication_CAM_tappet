"""Debug test for optimized solver - run first 5 steps with diagnostics"""

import sys
sys.path.insert(0, '/workspace')

from optimized_ehl_solver import OptimizedEHLSolver
import numpy as np

solver = OptimizedEHLSolver()

# Run first 5 steps with debug
theta_deg = solver.cam_data["theta_deg"][:5]
um_profile = solver.cam_data["um"][:5]
vs_profile = solver.cam_data["Vs"][:5]
R_profile = solver.cam_data["R"][:5]
F_profile = solver.cam_data["F"][:5]
dt_profile = solver.cam_data["dt"][:5]

V_current = None

for i in range(5):
    print(f"\n{'='*60}")
    print(f"Step {i+1}/5 - Angle={theta_deg[i]:.2f} deg")
    print(f"{'='*60}")
    
    solver.update_operating_state(um_profile[i], vs_profile[i], R_profile[i], F_profile[i])
    solver.dt = dt_profile[i]
    
    print(f"Operating state: Um={solver.Um:.4e}, R={solver.R:.4e}, W={solver.W:.2f} N")
    print(f"                 a_Hertz={solver.a_Hertz:.4e}, Pmh={solver.Pmh:.4e}")
    print(f"                 Wld={solver.Wld:.4e}")
    
    if V_current is None:
        solver.is_transient = True
        V_current = solver.build_initial_guess_reduced()
        solver.update_history_reduced(V_current)
        print(f"Initial guess: P_inner range=[{V_current[:-1].min():.4e}, {V_current[:-1].max():.4e}]")
        print(f"              H0={V_current[-1]:.4e}")
    
    # Check residual of initial guess
    F_init = solver.system_func_reduced(V_current)
    res_init = np.linalg.norm(F_init) / np.sqrt(len(F_init))
    print(f"Initial residual: {res_init:.4e}")
    print(f"Residual components: Rey={F_init[:-1].max():.4e}, Load={abs(F_init[-1]):.4e}")
    
    # Try Newton solve with debug
    print("\nNewton solve:")
    V_new, success, res, iters = solver.newton_solve_reduced(V_current, tol=1e-7, max_iter=15, debug=True)
    
    print(f"\nResult: success={success}, res={res:.4e}, iters={iters}")
    
    # Check load error
    P_rey, Pa, H = solver.get_full_state_reduced(V_new)
    load_calc = np.sum((P_rey + Pa) * solver.dx)
    load_error = abs(solver.Wld - load_calc) / max(solver.Wld, 1e-12)
    print(f"Load: target={solver.Wld:.4e}, calculated={load_calc:.4e}, error={load_error:.4e}")
    print(f"Pressure: P_rey max={P_rey.max():.4e}, Pa max={Pa.max():.4e}")
    print(f"Film: H min={H.min():.4e}, H max={H.max():.4e}")
    
    solver.update_history_reduced(V_new)
    V_current = V_new

print("\n" + "="*60)
print("Debug test complete")
