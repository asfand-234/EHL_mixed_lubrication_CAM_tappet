
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
   - Adding **missing physics** (documented below).
   - Choosing **physically justified parameter values/ranges** from literature (oil rheology, cavitation formulation, roughness/flow factors, realistic texture geometry ranges). Cite sources in comments.
2. **Mass conservation & cavitation** must be **JFO-consistent** (Elrod–Adams formulation) with complementarity between pressure `p` and saturation `θ`. Use mass conservation at rupture and reformation fronts.
3. **Convergence priority**: the Reynolds solver must be **unconditionally stable** under strong texturing. Use monotone discretization, under‑relaxation, and residual-based adaptive stepping. No non-physical pressure clipping.
4. **Acceptance**: Only when **all 12 target entries** are within **±15%** (≥85% match) may you output the **single final Google Colab script**. Until then, iterate internally and **print current predicted table** for developer testing.
5. **Reproducibility**: deterministic runs, fixed seeds, unit-checked I/O. All constants with **units**.

---

## Physics model — Baseline and upgrades

### Kinematics & loading
- Parse `CamAngle_vs_Lift_smooth.txt` → cubic spline for lift `h(φ)` with `φ` cam angle [deg]. Compute velocity `ḣ` and acceleration `ẍ` vs. **rotational speed** Ω (RPM). Obtain **normal load** from valve spring model or use script’s existing load law; if absent, derive an **effective normal force** from Hertzian line contact with cam radius & follower geometry as in typical cam/tappet contacts (document assumptions).

### Hydrodynamics (1D line contact along entrainment)
- Solve **Reynolds equation** with **JFO mass-conserving cavitation** (Elrod–Adams variables `p, θ`), including texture-induced film variation `h(x,φ)` and time dependence via sliding speed `U(φ)`.
  - Discretization: conservative finite differences; staggered mass flux `J = - (h^3/12η) ∂p/∂x + (U h/2) φ_x` (Patir–Cheng flow factors when roughness/texture is averaged).
  - Cavitation: complementarity `p ≥ p_cav; 0 ≤ θ ≤ 1; (p - p_cav)(1-θ) = 0`. Track rupture/reattachment fronts.
  - **Starvation**: entrainment-limited inlet boundary (`θ_inlet ≤ 1`) with supply film `h_supp` / oil feed height. Estimate `θ_inlet` from previous cycle carry‑over.
  - **Roughness/texture averaging**: **Patir–Cheng** flow factors `φx, φy, φs` from local deterministic micro-geometry or tabulated factors if the script already uses them.
    - Sources: Patir & Cheng 1978/1979; Tripp 1982. [Citations]
- **Rheology**:
  - Pressure–viscosity: **Roelands/Houpert** (literature parameters) with realistic α (check oil type). [Citations]
  - Shear-thinning: **Carreau** at moderate shear + **Eyring** cap at high shear stress; take the lower apparent viscosity to avoid unphysical traction. [Citations]
  - Thermal correction: **simple energy balance** for flash temperature rise; adjust viscosity with local temperature (explicit coupling, under-relaxed).

### Asperity contact (mixed lubrication)
- Use **Greenwood–Tripp/Greenwood–Williamson** style contact for load share and boundary shear in regions where `h(·)` comparable to composite roughness `σ`. [Citations]
- Boundary friction term: τ_b = **m·p_a** with m ≈ friction coefficient of boundary film (literature 0.05–0.15) OR Eyring-limited shear stress for boundary layer; choose physically defensible values only.

### Texture & grooves (deterministic or average-flow)
- Read existing texture parameters (depth, width, wavelength, skew, orientation, duty cycle). Permit **physically realistic** adjustments only:
  - Depth: 1–15 μm; Duty: 2–20%; Pitch scaled to local Hertzian contact width; **orientation** aligned to entrainment to assist inlet pressurization and cavitation control.
- Two pathways:
  1. **Deterministic micro-geometry**: resolve `h(x)` at texture scale with refined mesh and EA cavitation.
  2. **Average-flow reduction**: convert texture to **flow factors** dependent on local film/roughness ratio and pattern aspect ratio (Patir–Cheng-style), validated against deterministic tests.
- Use **inlet-biased** textures on shim to enhance replenishment at opening flank; avoid outlet texturing that increases cavitation length (per literature).

### Friction & torque
- Film shear: τ_h = η_eff * (U_rel / h) with wall‑slip off.
- Contact shear: τ_a from asperity contacts (see above).
- Friction force F_f = ∫ (τ_h + τ_a) dA (1D: multiply by cam width `b`), then **torque** = F_f·effective radius. Compute **cycle-averaged** torque and **percent reduction vs. untextured** at each RPM and each “% case.”

---

## Numerics & convergence playbook

- **Solver**: line-by-line TDMA (tridiagonal) or projected SOR on the EA complementarity system; optionally multigrid V-cycle for pressure.
- **Under-relaxation**: adaptive ω based on residual norm trend; start ω_p = 0.6 (pressure), ω_θ = 0.5 (saturation), ω_T = 0.3 (temperature).
- **Mesh**: at least 200–400 nodes across the Hertzian length; **refine near textures** and cavitation fronts; CFL-like time step for transient marching in angle.
- **Stopping**: L∞ residual of Reynolds flux balance < 1e−6 × (inlet mass flux), and cavitation complementarity violation < 1e−8; max 500 inner iterations per angle step.
- **Safeguards**: positivity of h, θ; cap |∂p/∂x| growth; backtracking on divergence; switch to average-flow mode if deterministic sub-grid oscillations persist.

---

## Literatures to ground parameters (inline comments should cite these)

- **Mass-conserving cavitation (JFO; Elrod–Adams)**: Elrod (1981); Adams (1983); Woods (NASA, 1988); modern numerical overviews (2024). citeturn0search7turn0search14
- **Texture hydrodynamics review**: Gropper et al., 2016. citeturn0search6
- **Average-flow model & flow factors**: Patir & Cheng, 1978/1979; Tripp, 1982; Prat (overview). citeturn0search1turn0search8turn0search16turn0search23
- **Asperity contact**: Greenwood–Williamson (1966); Greenwood–Tripp (review). citeturn0search24turn0search17
- **Pressure–viscosity & film correlations**: Roelands (1966); Bair (1993, 2022); recent assessments (2025). citeturn0search25turn0search3turn0search18
- **Shear thinning / limiting shear**: Carreau (1972); Eyring (classical) and modern validations (2004–2025). citeturn0search11turn0search19turn0search4
- **Cam–tappet EHL**: Zhou & Huang (2002); recent mixed-EHL works. citeturn0search20turn0search5
- **Textured cavitation optimization**: Mao et al., 2016; Nitzschke et al., 2016. citeturn0search21turn0search22

(Use these as **bounds and modeling choices**, not as curve fits.)

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
2. **Add shear-thinning (Carreau) + Eyring cap** for traction at high shear; compute local Weissenberg `Wi = λ γ̇` and stress `τ`, choose smaller η_eff.
3. **Introduce starvation & inlet saturation** linked to cam lift phase; texture on shim acts as reservoir → compute `θ_inlet(φ)`.
4. **Texture placement & duty fraction**: bias to inlet side for early pressurization; avoid outlet textures. Keep only **physically allowed** ranges.
5. **Asperity contact**: Greenwood–Tripp for mixed regime; boundary shear with realistic m or limiting τ_0 (~2–10 MPa depending on oil/additives) — cite assumption.
6. **Thermal correction**: update viscosity with flash temperature from friction work balance; under-relax.
7. **Average-flow factors**: if deterministic mesh shows oscillations, switch to Patir–Cheng factors matched to pattern aspect ratio.

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

## Pseudocode sketch (for Codex implementation)

```python
# Load data
lift_tbl = load_cam_lift("CamAngle_vs_Lift_smooth.txt")
params = parse_script_defaults("MAIN_SCRIPT.txt")

# Physics options
phys = {
  "cavitation": "Elrod-Adams",
  "viscosity": {"p-v": "Roelands-Houpert", "shear": ["Carreau","Eyring_cap"]},
  "contact": "Greenwood-Tripp",
  "starvation": True,
  "thermal": "explicit_underrelaxed",
  "texture_model": "deterministic_or_avgflow",
}

def run_case(RPM, duty_pct):
    # build kinematics for RPM
    # set texture geometry consistent with duty_pct and existing code’s definition
    # solve mixed lubrication with EA + rheology + contact + thermal
    # compute cycle-averaged torque for textured and untextured
    return percent_reduction

table_pred = evaluate_cases([300,500,700,900], [5,8,10])
match = compare_to_target(table_pred, TARGET_TABLE)
if (match >= 85%).all():
    emit_final_colab_script(phys, params)
else:
    apply_next_physics_improvement()
```

---

## Notes to self (implementation hints)

- Prefer **complementarity formulations** (e.g., primal-dual) for EA to avoid manual switching; reference modern 2024 solvers. citeturn0search14
- Validate viscosity law vs expected α range (e.g., 10–25 GPa⁻¹ for common engine oils at 40–100 °C) using Bair’s discussions on pressure–viscosity definitions (2022). citeturn0search3
- Keep texture depth small enough to avoid **spurious cavitation elongation**; see review guidance (Gropper 2016). citeturn0search6
- For cam–tappet specifics, cross-check trends with Zhou & Huang (2002) for sanity (friction vs speed & roughness). citeturn0search20

---

### End of AGENT.md
