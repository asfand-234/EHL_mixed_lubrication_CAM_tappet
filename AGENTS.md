# Codex Agent: Reynolds/EHL Verification Protocol (1D Line Contact CAM–Tappet)

## Objective
Make the Reynolds + cavitation + load-balance solution numerically verified and stable.
Do not stop early. Continue iterating until all acceptance criteria pass.

## Files
- Primary script: test6666.txt
- Create a verification runner: verify_reynolds.py (new)

## Required Run Commands
1) python verify_reynolds.py --rpm 300 --angles -8 -2 0 2 6 31 -45
Run every time the solver changes.

## Angles
Use the nearest available TH_DEG entry to each requested angle.

## Acceptance Criteria (must all pass)
For each angle:

A) Load balance closure:
- abs((Wh+Wa)-Wext)/max(Wext,1e-30) <= 5e-4

B) Complementarity (Elrod–Adams):
- min(p) >= -1e-12*ph
- min(theta) >= -1e-6 and max(theta) <= 1+1e-6
- max( (p/ph)*(1-theta) ) <= 1e-6

C) Discrete Reynolds residual on the solved pressure grid:
- eps_R_L2 = ||L(P)-RHS||2/(||RHS||2+1e-30) <= 1e-3
- eps_R_Linf = max|L(P)-RHS| <= 5e-3*max(|RHS|)+1e-30

D) Cavitation front flux continuity:
- Identify downstream cavitation boundary index ic (after p peak where p<thr and theta<1)
- Relative flux jump across ic <= 1%

E) Profile shape stability:
- p has exactly one local maximum in |x|<=a (untextured runs)
- downstream cutoff x_cut where p falls to ~0 after peak satisfies: 0.85a <= x_cut <= 1.10a
- max_{|x|>a} p / pmax <= 1e-6

F) Grid convergence (run at Nx=271,401,601 for rpm=300 angles -8,0,6):
- pmax difference between 401 and 601 <= 2%
- hmin difference between 401 and 601 <= 2%
- load closure still within tolerance

G) Initial-condition independence:
- Re-run each angle with p_init=0 and p_init=Hertz; final pmax and hmin differ <= 2%

## Output Requirements (print every run)
For each (rpm,angle):
- R, Ve, Vs, Wext
- a, ph, pmax, hmin, h0, Wh, Wa, load_err_rel
- eps_C (complementarity), eps_R_L2, eps_R_Linf
- cavitation boundary x_cut and flux jump
- peak count, leakage ratio max(|x|>a) p/pmax
- runtime per solve

Also print PASS/FAIL per criterion.

## Implementation Notes (do not ignore)
- Any artificial mobility floor, hard clamping of physical variables (beyond p>=0, 0<=theta<=1, h>=h_floor), or nonphysical scaling must be justified or removed.
- If residuals pass but shapes are wrong, treat it as a boundary/cavitation-set identification issue.
- If shapes pass but residual is large, treat as discretization inconsistency.

## Completion
Do not claim completion until ALL criteria A–G pass for both rpm sets.
