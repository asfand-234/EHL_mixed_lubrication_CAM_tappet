
# AGENT.md — CODEX Plan for Physics-First Optimization of CAM–Tappet Texturing Script

**Repository context**
- `MAIN_SCRIPT.txt`: raw Python script that computes **percentage reduction of averaged friction torque vs. untextured** using a 1D mixed-lubrication model for a cam–tappet (bucket + textured shim) with deterministic textures/grooves.
- `CamAngle_vs_Lift_smooth.txt`: cam angle (deg) ↔ lift (m) dataset.
- **Target results** to match (≥85% agreement per entry):

  | RPM | 5_percnt | 8_percnt | 10_percnt |
  |---:|---:|---:|---:|
  | 300 | 3.4% | 7.95% | 3.4% |
  | 500 | 6.12% | 8.92% | 10.71% |
  | 700 | 4.21% | 14.11% | 9.4% |
  | 900 | 18.33% | 11.91% | 6.87% |

> **Interpretation of columns**: “5/8/10_percnt” are taken to denote a **texture duty cycle or area fraction (%)** used by the current script. If the existing code defines them differently (e.g., dimple depth % of nominal film, groove width %, etc.), **auto-detect from variable names and comments** in `MAIN_SCRIPT.txt` and map the computed cases to these columns accordingly (do not change the target table).

---

## Non‑negotiable constraints (read carefully)

1. **No calibration/fitting/non-physical scaling.** You may **only** vary quantities by:
   - Adding **missing physics** 
2. **Convergence priority**: the Reynolds solver must be **unconditionally stable** under strong texturing. Use monotone discretization, under‑relaxation, and residual-based adaptive stepping. No non-physical pressure clipping.
3. **Acceptance**: Only when **all 12 target entries** are within **±15%** (≥85% match) may you output the **single final Google Colab script**. Until then, iterate internally and **print current predicted table** for developer testing.
4. **Reproducibility**: deterministic runs, fixed seeds, unit-checked I/O. All constants with **units**.

---

## Physics model — Baseline and upgrades

### Kinematics & loadin
### Hydrodynamics (1D line contact along entrainment
### Roughness/texture averaging**: **Patir–Cheng** flow factors `φx, φy, φs` from local deterministic micro-geometry or tabulated factors if the script already uses them.
### Asperity contact (mixed lubrication)

### Texture & grooves (deterministic or average-flow)
- Read existing texture parameters (depth, width, wavelength, skew, orientation, duty cycle). Permit **physically realistic** adjustments onl
- Use **inlet-biased** textures on shim to enhance replenishment at opening flank; avoid outlet texturing that increases cavitation length (per literature).

### Friction & torque

## Numerics & convergence playbook

- **Solver**: line-by-line TDMA (tridiagonal) or projected SOR on the EA complementarity system; optionally multigrid V-cycle for pressure.
- **Under-relaxation**: adaptive ω based on residual norm trend; start ω_p = 0.6 (pressure), ω_θ = 0.5 (saturation).
- **Mesh**: at least 200–400 nodes across the Hertzian length; **refine near textures** and cavitation fronts; CFL-like time step for transient marching in angle.
- **Stopping**: L∞ residual of Reynolds flux balance < 1e−6 × (inlet mass flux), and cavitation complementarity violation < 1e−8; max 500 inner iterations per angle step.
- **Safeguards**: positivity of h, θ; cap |∂p/∂x| growth; backtracking on divergence; switch to average-flow mode if deterministic sub-grid oscillations persist.

---

## Execution plan for CODEX

### Stage 0 — Repository probe
1. Open `MAIN_SCRIPT.txt`. Identify:
   - Governing equations, rheology, cavitation handling, texture parameterization, boundary conditions.
   - The meaning of “5/8/10_percnt”. If needed, deduce and map programmatically.
2. Confirm data units for `CamAngle_vs_Lift_smooth.txt`. If missing, assume: angle in degrees; lift in meters (or mm → convert). Validate monotonicity per lobe.

### Stage 1 — Test harness (kept internal until acceptance)
- Implement `evaluate_cases(rpms=[300,500,700,900], percents=[5,8,10])`:
  - For each RPM & percent:
    - Run **untextured** baseline to compute averaged torque `T0`.
    - Run **textured** case to compute `T_tex` with current physics stack.
    - Record `%Δ = 100*(T0 - T_tex)/T0`.
  - Print a table and the **% match** vs target: `match = 100 * (1 - |pred - target|/target)` elementwise.

**If any entry < 85% match** → trigger Stage 2.

### Stage 2 — Physics-first improvements (no fitting)
Apply in this order until acceptance:

1. **Switch to Elrod–Adams JFO** if not already. Verify mass conservation by integrating flux across domain and cycle.
2. **Introduce starvation & inlet saturation** linked to cam lift phase; texture on shim acts as reservoir → compute `θ_inlet(φ)`.
3. **Texture placement & duty fraction**: bias to inlet side for early pressurization; avoid outlet textures. Keep only **physically allowed** ranges.
4. **Average-flow factors**: if deterministic mesh shows oscillations, switch to Patir–Cheng factors matched to pattern aspect ratio.
5. **Add any other necessary physics to meet the target**

Re-run `evaluate_cases` after each addition. Stop once all 12 entries ≥85% match.

### Stage 3 — Final deliverable (single Colab script)
When acceptance reached, **generate a single, self-contained Google Colab script** that:
- Loads both repo files from paths or via upload cell.
- Implements the **accepted physics stack** with clean, modular functions.
- Runs the 12 cases and **prints only** the predicted `% averaged friction torque reduction` table (no extra plots or logs).
- Includes a short **header comment** citing the physics used with dates (from the list above), constants with units, and any assumptions.

> **Important**: Do not show intermediate attempts or partial outputs to the user. Only present the final Colab script once the ≥85% criterion is satisfied.

---

## Acceptance checklist (automated)

- [ ] **Mass conservation** residual over full cycle < 0.5% of total entrained mass.
- [ ] **EA complementarity** max violation < 1e−8.
- [ ] **Pressure non-negativity** (`p ≥ p_cav` everywhere).
- [ ] **Mesh independence**: halving Δx changes any of the 12 entries by < 1%.
- [ ] **Thermal stability**: max ΔT per angle step < 5 K or under-relaxed.
- [ ] **Target match**: all 12 entries ≥ 85% match.

---

### End of AGENT.md
