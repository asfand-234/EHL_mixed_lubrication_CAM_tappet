# — Reynolds Solver Load balance Stabilization (CAM–Tappet, 1D Line Contact)

## Non-negotiable behavior
You must keep Performing until the below criteria pass.
## CODE
Read the python code in text file "test6666.txt" and run the script for only  --rpm 300 -- cam angles -8, -2, 0, 2, 6, 31, -45

Each Run you MUST include:
1) Run the verification runner (all requested angles)
2) Print a PASS/FAIL table for criteria per angle
3) Implement ONE targeted code change after doing deeply research from standard relevant literature from web sources.
4) Re-run verification and report delta in metrics.
5) If criteria does not pass then again optimize the physics and so on.

Do NOT stop after printing results once.
Do NOT say "time is short" unless you have completed targets.

## Allowed scope of changes
- You MAY modify only the Reynolds/cavitation/load-balance parts of solve_theta and helper routines it calls.
- You MUST NOT change cam kinematics formulas or input arrays.
- You MUST NOT add non-physical clamps/floors besides:
  - p >= 0
  - 0 <= theta <= 1
  - h >= h_floor (numerical only, <= 1e-12 to 1e-10 m range)
- If you add any stabilization, it must be a standard numerical stabilization (e.g., upwind/Rusanov, active-set cavitation, under-relaxation), and you must justify it by which criterion it fixes.


## Acceptance criteria 
*(must pass for all angles at rpm=300)*

** Load balance:**
- rel_load_err = abs((Wh+Wa)-Wext)/max(Wext,1e-30) <= 0.01


## Mandatory printed output (every cycle)
For each angle print:
- angle, rpm, R, Ve, Vs, Wext
- a, ph, h0, pmax, hmin, Wh, Wa, rel_load_err
Also print runtime per solve_theta.

## Deliverable:
once the criteria pass for all angles then provide only the updated code for me.
