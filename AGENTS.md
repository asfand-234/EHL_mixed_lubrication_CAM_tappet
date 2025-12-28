Read and deeply analyze python code in text file "test6666.txt". It is 1D thermal, Transient mixed Lubrication Line contact in cam and tappet mechanism

Your task is to optimize the numerical scheme and time dependent approach by replacing with below mention numerical loop to reduce overll cycle runtime while keep accuracy same.

TESTING CRITERIA:
1) replace current approach with below numerical loop.
2) Run every time entire code and monitor live load error, residual and runtime.
3) Criteria:
Load error < 1% (even 0.3% is acceptable)
Residual < 1e-7 
Total cycle runtime <160 seconds.
4) If any step fail in any above criteria then keep optimizing and keep fixing root cause and keep running the code and analyzing by yourself.

Numerical loop:

START
  |
  v
Initialize u(t0), choose method (BDF / gen-α), tolerances, dt0, order q
  |
  v
WHILE t < t_end:
    |
    v
  (1) Predict u_guess at t+dt using history (BDF predictor / extrapolation)
    |
    v
  (2) Nonlinear solve for this time step:
        set k = 0
        decide Jacobian update policy (often: build J only at k=0)
        REPEAT:
            - Assemble residual R(u^(k)) including implicit time terms
            - If k==0 (or policy says update): assemble/update Jacobian J
            - Solve linear system:  J * Δu = -R   (sparse direct or Krylov+PC)
            - Damping/line search: u^(k+1) = u^(k) + λ * Δu
            - Check nonlinear convergence
            - k = k + 1
        UNTIL converged OR max iters
    |
    v
  (3) If NOT converged:
        dt = dt * shrink_factor
        possibly reduce order q
        retry step
    |
    v
  (4) If converged:
        estimate local time error (BDF LTE controller)
        if error acceptable:
            accept step: t = t + dt, store history, maybe increase dt and/or q
        else:
            reject step: dt = dt * shrink_factor, retry
  |
  v
END WHILE
  |
  v
OUTPUT u(t) over [t0, t_end]
