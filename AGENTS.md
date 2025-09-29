# AGENTS.md

> **Read me fully, then start immediately.**
> This repository allows **only a single-file workflow** and a **pure-Python (NumPy-only)** implementation. No extra folders, no subprocesses, no external packages beyond NumPy. Do **not** refuse due to “environment limits.” If something is unavailable, **choose an equivalent pure-Python method** and proceed. **Iterate internally** and, after each attempt, **print the 4×3 table of predicted % averaged friction torque**. When all targets pass, **print only the final Google Colab script** (and nothing else).

## Problem
We model **1‑D mixed lubrication** for a **cam–bucket tappet** with a **textured shim**. Inputs available:
- `MAIN_SCRIPT.txt` — legacy text of current model (for reference only).
- `CamAngle_vs_Lift_smooth.txt` — cam angle (deg) vs lift (m).

**Target table** (columns are scenarios **S5, S8, S10**):
```
RPM    5_percnt   8_percnt   10_percnt
300    3.40%      7.95%      3.40%
500    6.12%      8.92%      10.71%
700    4.21%      14.11%     9.40%
900    18.33%     11.91%     6.87%
```

## Non‑Negotiables
1) **No calibration/fitting/non‑physical scaling.** Physics/numerics improvements only.  
2) **Mass‑conserving cavitation** (Elrod–Adams/JFO) with saturation \(\theta\in[0,1]\).  
3) **Stable, convergent, deterministic** Reynolds solver under textures.  
4) **Single-file, pure-Python (NumPy-only)** solution that runs **here**. No extra project scaffolding.  
5) **Output policy**: During iteration, **print only** the evolving 4×3 table. Upon success, **print only** the final Colab script source. No other text, comments, or diagnostics in chat.

## Physics (single-file implementation)
- Film thickness: \(h(x,t)=h_{geom}(x,t)+h_{tex}(x)\), with \(h_{geom}\) from kinematics and \(h_{tex}\) periodic grooves (depth, duty, pitch, volume-neutral, C¹ ramps).
- Reynolds (1‑D, mass-conserving): implicit finite difference/volume; switch to Elrod–Adams with flux continuity at rupture/reformation; enforce non-negative pressures.
- Viscosity: Barus `μ=μ0 exp(αp)` or Roelands (choose by regime); optional Eyring cap `τ≤τ0` if shear rates justify.
- Mixed regime: Patir–Cheng flow factors (1‑D approximation) and Greenwood–Tripp asperity contact **from literature** (not tuned to targets).
- Friction/torque: `τ_h=μU/h + 0.5 h ∂x p` + asperity shear; integrate over x; convert to torque with cam radius; average over full cycle; compute % reduction vs untextured.
- Operating points: RPM ∈ {300,500,700,900}; scenarios ∈ {S5,S8,S10}.

## Acceptance Gate (all must pass)
- Per cell: `abs(pred - target)/max(target,1e-6) ≤ 0.15` for all 12 cells.  
- (Recommended) MAPE ≤ 0.10.  
- Numerics: residual L∞ for pressure and θ < 1e‑8 each cam step; no negative film thickness; non‑negative dissipation; mesh refinement (internal) medium vs fine torque diff < 1%.

## Environment-Safe Auto‑Runner (MANDATORY)
**Start now and loop until the Gate passes.** Everything happens inside a **single Python program in memory** using NumPy. No file creation except the final Colab script text printed at success.

1. **Bootstrap (in memory)**
   - Load `CamAngle_vs_Lift_smooth.txt` via NumPy. If missing, ask the user to upload it, then continue.
   - Read `MAIN_SCRIPT.txt` only to understand prior assumptions; do **not** reuse any non-physical parts.

2. **Baseline attempt**
   - Implement solver & model in one script with sensible SI defaults (μ0≈0.013 Pa·s, α≈1–3×10⁻⁸ Pa⁻¹, τ0≈2–5 MPa if needed; cam radius & clearance reasonable and documented in comments).
   - Compute all 12 cases; **print only** the 4×3 table.

3. **Iterate (physics only) if not passing**
   - Improve cavitation numerics (θ update, under-relaxation, flux limiter).
   - Switch Barus↔Roelands or enable Eyring based on computed shear rate.
   - Correct texture phase/entrainment, enforce volume neutrality, and C¹ ramps.
   - Refine mesh (≥400 nodes/period or adaptive clustering around texture edges); use implicit TDMA or stabilized tridiagonal solves (NumPy-only).
   - Re-run and **print only** the updated table.
   - Repeat until all cells meet ≤15% error.

4. **Success ⇒ emit one Colab script**
   - Produce a **single, executable Google Colab script** named `cam_textured_mixed_lubrication_colab.py` whose code:
     - Loads `CamAngle_vs_Lift_smooth.txt` (user will upload in Colab).
     - Contains the finalized pure-Python solver.
     - Runs RPM×{S5,S8,S10} and **prints only** the final 4×3 table.
   - **Now print only the script source** in the chat (no headers/extra text).

## Anti‑Refusal Clause
- If an operation seems “not permitted,” **choose an equivalent pure-Python approach**: no subprocesses, no file I/O beyond reading the two inputs, no plotting. **Do not stop.** Continue iterations until the Gate passes.

## Minimal Iteration Print Format (strict)
At each iteration (including baseline), print exactly four lines:
```
300 {'S5': X1, 'S8': X2, 'S10': X3}
500 {'S5': X4, 'S8': X5, 'S10': X6}
700 {'S5': X7, 'S8': X8, 'S10': X9}
900 {'S5': X10, 'S8': X11, 'S10': X12}
```
No extra text or decoration.

## Final Reminder
- **No curve fitting.** Fix physics and numerics only.
- **Single-file, NumPy-only.**
- **During iteration: print only tables.**  
- **At success: print only the Colab script.**
