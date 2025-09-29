# AGENTS.md

> **Codex (in ChatGPT), read carefully and start immediately.**
> **Do not scaffold a project tree. Do not create many files/folders.** Work in-memory / scratch as needed. Run full simulations **in the background on your side**, and **periodically print the predicted % averaged friction torque table** after each improvement cycle so I can see progress. **Only when the targets are met** (per Acceptance Gate) you will output **one complete Google Colab executable script** (and nothing else at that time).

## Problem (authoritative)
We have a **1‑D mixed lubrication** model for a **cam–bucket tappet** with a **textured shim**. Inputs present in the repo:
- `MAIN_SCRIPT.txt` — current raw Python text that solves Reynolds and computes **% reduction of averaged friction torque** vs. untextured.
- `CamAngle_vs_Lift_smooth.txt` — cam angle (deg) vs lift (m).

**Target table** (treat columns as three distinct scenarios **S5, S8, S10**):
```
RPM    5_percnt   8_percnt   10_percnt
300    3.40%      7.95%      3.40%
500    6.12%      8.92%      10.71%
700    4.21%      14.11%     9.40%
900    18.33%     11.91%     6.87%
```

## Hard Rules
1) **No calibration / fitting / non‑physical scaling.** Only physics‑based, literature‑defensible changes.  
2) **Mass‑conserving cavitation** (Elrod–Adams or JFO). No negative pressures.  
3) **Stable, convergent Reynolds solver** under textures; document residuals and step acceptance internally.  
4) **Deterministic**: same inputs ⇒ same outputs.  
5) **Minimal outputs**: During iteration you **print only** the evolving 4×3 prediction table; when targets pass, you **print only** the **final Colab script** source code. No extra commentary/plots/files.

## Physics to (re)implement/upgrade (single‑file approach)
- **Film thickness**: `h(x,t) = h_geom(x,t) + h_tex(x)`; `h_geom` from cam kinematics; `h_tex` supports duty, depth, pitch, and phase with **C¹** ramps; enforce **volume neutrality** over a period.
- **Reynolds (1‑D, mass‑conserving)**: implicit finite‑difference/finite‑volume, with saturation variable θ ∈ [0,1] and flux continuity at rupture/reformation boundaries.
- **Viscosity**: Barus `μ = μ0 exp(α p)` or Roelands (choose based on regime).
- **Mixed regime**: Patir–Cheng flow factors (1‑D approximation) and Greenwood–Tripp asperity contact for shear/normal support (no tuning to targets).
- **Friction & torque**: τ_hyd = μU/h + 0.5 h ∂x p; add asperity shear; integrate over x, map to cam torque, **average over a full cam cycle**; compute % reduction vs untextured.
- **Operating points**: RPM ∈ {300, 500, 700, 900}; scenarios ∈ {S5, S8, S10}.

## Acceptance Gate (ALL must pass)
- Per‑cell error ≤ **15%**: `abs(pred - target)/max(target,1e-6) ≤ 0.15` for all 12 cells.  
- (Optional but recommended) MAPE across 12 cells ≤ **10%**.  
- Numerical: pressure & θ residual L∞ < 1e‑8 at each cam step; no negative film thickness; energy dissipation non‑negative; mesh refinement (medium vs fine) torque diff < 1% (checked internally by you).

## Auto‑Run Workflow (single‑file, minimal artifacts)
**Start now and loop until the Acceptance Gate passes.**

1. **Bootstrap in memory**
   - Load `CamAngle_vs_Lift_smooth.txt`.
   - Parse `MAIN_SCRIPT.txt` only to understand current assumptions; do **not** reuse non‑physical parts.
   - Define a **single Python program in memory** (one file) containing: kinematics, texture generator, Patir–Cheng/Greenwood–Tripp helpers, Reynolds+EA solver, friction/torque integration, and a runner.

2. **Baseline run**
   - Compute the 4×3 table (RPM × {S5,S8,S10}) with reasonable defaults: SI units; μ0≈0.013 Pa·s; α≈(1–3)×10⁻⁸ Pa⁻¹; τ₀≈2–5 MPa if needed; cam radius and clearances documented in code.
   - **Print the table** (e.g., `300 {'S5': xx, 'S8': yy, 'S10': zz}`, etc.).

3. **Improve physics only (iterate)**
   - If any cell fails ≤15%: adjust **physics/modeling** (never numeric fitting):
     - Switch Barus ↔ Roelands; enable/disable Eyring cap based on computed shear rate.
     - Refine mass‑conserving cavitation (θ update, flux limiter, under‑relaxation).
     - Improve texture phase/entrainment mapping; enforce volume neutrality per period.
     - Refine Patir–Cheng factors using literature ranges for composite roughness (no tuning to target).
     - Increase spatial resolution/adaptive clustering near steep ∂x p or texture edges; switch to line‑relaxed/TDMA or multigrid if needed.
   - Re‑run full cycle; **print the updated 4×3 table**.
   - Repeat until Acceptance Gate passes.

4. **When Gate passes ⇒ emit final deliverable only**
   - Generate **one** complete **Google Colab** executable script named `cam_textured_mixed_lubrication_colab.py` that **runs end‑to‑end** and **prints only** the final 4×3 table.
   - **Now print only the script source code** in the chat (no headings, no extra text).

## What to print (strict)
- **During iteration**: Only the 4×3 table per iteration (no other logs).  
- **At success**: Only the **full Colab script** text.  
- **Never** print partial code or other artifacts before success.

## Guardrails & defaults (internal, may adjust with justification in code comments)
- Units: SI. RPM→ω=2π·RPM/60; U=R·ω.  
- Mesh: ≥400 nodes/texture‑period or adaptive; implicit scheme; under‑relaxation 0.1–0.6.  
- Boundary conditions: Ambient inlet with meniscus; JFO outlet.  
- Data handling: Full cam cycle averaging; consistent scenario mapping S5/S8/S10 to (duty, depth, pitch).  
- Determinism: fixed numerical tolerances; no randomness.

## Final reminder
- **Do not** use curve‑fitting or secret multipliers. Achieve the targets by fixing **physics and numerics** only.  
- Keep the repository clean: **do not** create extra folders/files. Compute and iterate internally; only the final Colab script is emitted at the end.
