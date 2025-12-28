You are given my Python EHL solver in text file,  (test6666.txt) for 1D transient thermal mixed lubrication line contact of a cam–follower. The cam cycle is -84° to 80° with 329 angle points; I currently solve a nonlinear system at every cam step using a dense Newton method with a dense Jacobian that is partly finite-differenced (calc_jacobian loops over all P and H unknowns) and solved by np.linalg.solve. Runtime is >12 minutes per cycle.

Task: refactor/upgrade the numerical scheme to drastically reduce runtime WITHOUT compromising accuracy.

Constraints:
- Keep the same governing physics (Reynolds + elastic film thickness with integral D_mat + Greenwood–Tripp asperity pressure + load balance + thermal viscosity/density coupling).
- Maintain solution accuracy vs current code (define regression checks: max relative change in P_rey, Pa, H and load error, with tolerances).
- Keep results at the original 329 angles (you may internally use adaptive stepping + interpolation, but final outputs must match those 329 angles).

Required improvements (prioritize):
1) Eliminate redundant unknowns: currently V = [P_inner, H_inner, H0] but film residual enforces H = H_elastic. Reformulate to solve only for [P_inner, H0] (or [P_inner, H0, optional cavitation var]) and compute H_elastic directly from H0 and P_tot each nonlinear iteration. This should cut system size ~in half and remove the need for FD on H_inner.
2) Replace dense FD Jacobian with an efficient approach:
   - Build an analytic or semi-analytic SPARSE Jacobian for the reduced system using the local stencil structure of the discretized Reynolds residual (banded) plus the elastic/asperity coupling terms.
   - Use scipy.sparse (CSR) + a sparse direct solve (splu/spsolve) or Newton–Krylov with preconditioning.
   - Implement Jacobian reuse policy “once per time step” or “minimal” (update only when convergence stalls), similar to COMSOL.
3) Add a predictor–corrector for cam stepping: extrapolate [P, H0, T] from previous 1–2 steps to warm-start Newton and cut iterations.
4) Implement adaptive cam-step control inspired by implicit BDF(1/2): if Newton converges fast, increase step; if it struggles/fails, reduce step. Interpolate cam kinematics/load/Um/Vs/R onto intermediate angles. Output still required on the 329 original angles.
5) Reduce nested thermal iteration cost: treat temperature update as a relaxed fixed-point (Aitken or under-relaxation) and avoid multiple full Newton solves per thermal iteration; only re-solve if ΔT exceeds a threshold.

TESTING CRITERIA:
every time you must run the script and monitor live the load error, residual and runtime.
Criteria:
Load error < 1% (even 0.3% is acceptable)
Residual <1e-7
Full cycle runtime < 170 seconds.
If at any cam step or any point the above criteria is not achieved then find the rot causes and keep fixing and keep running and keep monitoring the code.
