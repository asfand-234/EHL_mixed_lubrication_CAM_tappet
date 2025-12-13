# Codex Agent Instructions — Reynolds Solver Stabilization (CAM–Tappet, 1D Line Contact)

## Non-negotiable behavior
You MUST work iteratively.
keep Performing improve-cycles until all criteria pass.

Each improve-cycle MUST include:
1) Run the verification runner (all requested angles)
2) Print a PASS/FAIL table per criterion per angle
3) Choose the top 1–2 root-cause failures (see Priority Rules)
4) Implement ONE targeted code change
5) Re-run verification and report delta in metrics

Do NOT stop after printing results once.
Do NOT say "time is short" unless you have completed at least 8 cycles.

## Allowed scope of changes
- You MAY modify only the Reynolds/cavitation/load-balance parts of `solve_theta` and helper routines it calls.
- You MUST NOT change cam kinematics formulas or input arrays.
- You MUST NOT add non-physical clamps/floors besides:
  - p >= 0
  - 0 <= theta <= 1
  - h >= h_floor (numerical only, <= 1e-12 to 1e-10 m range)
- If you add any stabilization, it must be a standard numerical stabilization (e.g., upwind/Rusanov, active-set cavitation, under-relaxation), and you must justify it by which criterion it fixes.

## Target runs (execute every cycle)
Run:
- test6666.txt --rpm 300 --angles -8, -2, 0, 2, 6, 31, -45

## Priority rules (what to fix first)
Always fix in this order:
P1) Complementarity / cavitation violations (theta outside [0,1], p<0, or p*(1-theta) large)
P2) Discrete Reynolds residual too large
P3) Flux continuity failure at cavitation boundary
P4) Load balance closure failure
P5) Shape metrics (multiple peaks, early cutoff, leakage outside |x|>a)
P6) Runtime optimization

Never optimize runtime until P1–P4 pass.

## Acceptance criteria (must pass for all angles at rpm=300)
A) Load balance:
- rel_load_err = abs((Wh+Wa)-Wext)/max(Wext,1e-30) <= 5e-4

B) Complementarity:
- min(p) >= -1e-12*ph
- min(theta) >= -1e-6 and max(theta) <= 1+1e-6
- eps_C = max( (p/ph)*(1-theta) ) <= 1e-6

C) Discrete Reynolds residual:
- eps_R_L2 <= 1e-3
- eps_R_Linf <= 5e-3*max(|RHS|) + 1e-30

D) Cavitation boundary flux continuity:
- Find boundary after peak where p<thr and theta<1
- Relative flux jump across boundary <= 1%

E) Shape stability (for untextured runs or when texture amplitude=0):
- Exactly 1 local maximum in p for |x|<=a
- downstream cutoff x_cut in [0.85a,1.10a]
- leakage: max_{|x|>a} p/pmax <= 1e-6

F) Grid convergence (rpm=300 angles -8,0,6 between Nx=271 and 401):
- pmax difference <= 3%
- hmin difference <= 3%

G) Initial-condition independence:
- Run each angle with p_init=0 and p_init=Hertz; pmax/hmin differ <= 2%

## Mandatory printed output (every cycle)
For each angle print:
- angle, rpm, R, Ve, Vs, Wext
- a, ph, h0, pmax, hmin, Wh, Wa, rel_load_err
- eps_C, eps_R_L2, eps_R_Linf
- cav boundary x_cut and flux jump
- number of peaks in p (|x|<=a), leakage ratio

Also print runtime per solve_theta.

## Diagnosis map (use this to decide the next code change)
If B fails (eps_C):
- Likely cavitation update or theta-mobility inconsistency.
- Implement active-set complementarity: enforce p=0 in cavitated nodes; theta=1 where p>0.
- Remove any theta-dependent mobility floors; if operator becomes singular, use active-set instead of floors.

If C fails (residual):
- Discretization inconsistency.
- Ensure RHS uses the same conservative flux form as LHS (Poiseuille) and advection.
- Do NOT compute dp/dt from nonlinear iteration deltas.

If D fails (flux jump at boundary):
- Cavitation boundary treatment mismatch.
- Use conservative upwinding for advective flux and consistent face mobility for Poiseuille flux.
- Ensure boundary conditions at x=±a are applied consistently.

If A fails (load closure):
- Outer loop is unstable.
- Replace secant with bracketed Illinois regula falsi and warm-start using previous fields.

If E fails (multiple peaks/early cutoff):
- Check embedding mask and boundary conditions at ±a.
- Ensure core grid includes x=±a in interpolation; avoid strict comparisons.

## Runtime optimization allowed only after P1–P4 pass
Then:
- Precompute deflection kernel if it is physically identical to current elastic_deflection
- Reduce iters dynamically based on residual drop
- Use adaptive relaxation (increase relax_p as residual decreases)
