
# AGENT.md — Playbook for Codex on `EHL_mixed_lubrication_CAM_tappet`

> **Purpose:** This file tells Codex exactly how to implement, validate, and deliver a **complete 1D mixed‑lubrication simulation** for a radial CAM with a flat‑faced bucket tappet and *textured shim*, then calibrate `a_texture(θ,RPM,ρ)` so the **averaged friction torque reductions** match the specified targets to ≥90% accuracy (absolute percentage error ≤10% of target). It also defines the **Colab deliverable**, graphs, and verifiable checks.

---

## 0) Scope & Guardrails (Codex MUST follow)

- **Primary inputs (already in repo):**
  - `MAIN_SCRIPT.txt` — baseline raw script for averaged friction torque (untextured case).
  - `CamAngle_vs_Lift_smooth.txt` — columns: `θ_deg, lift[m]` (assume degrees; Codex to infer if header present).
  - `Texture_model.png` — reference figure for the mathematical texture model (see §3).
- **Textures:** hex/square doesn’t matter—model them as **1D longitudinal shallow grooves** aligned with sliding (x‑axis), with **width** `w_texture`, **pitch** `d_texture`, and **area density** ρ∈{5%, 8%, 10%}. See §3.
- **No negative pressures**: use a **mass‑conserving Reynolds** formulation (Elrod‑Adams or equivalent complementarity) so *p ≥ 0*, saturation in cavitated regions, and p=0 at far boundaries. Pressure must not “leak” outside the half‑width ±b(θ). (§4)
- **Contact window:** x∈[−b(θ), +b(θ)]; outside this, p=0 and fields are not solved.
- **Numerics:** steady slice per θ with **under‑relaxed fixed‑point** or **NCP/biCGSTAB** solve; include CFL‑like stability checks; monotone finite differences.
- **Asperity contact:** **Greenwood–Tripp** style load sharing for mixed regime. (§5)
- **Friction:** total = **hydrodynamic shear+Poiseuille** + **asperity** contribution; torque via Jacobian mapping from contact force distribution to cam torque (use follower force × effective radius). (§6)
- **Deterministic, reproducible**: no random seeds.
- **File outputs:** three calibrated `a_texture_*.txt` files with exact format in §8.
- **Colab notebook:** single runnable cell blocks, no secrets, **ready-to-paste**, robust to fresh runtime. (§9)
- **Graph control:** a single settings cell lets the user set (case: pressure/film vs x) the **cam angle**, **surface condition** (0=untextured, 1=textured), and **RPM**. (§9.3)
- **Validation:** automatic checks; build fails if any target reduction error >10% of target. (§10)

> If `AGENTS.md` (plural) is required by your runtime, **duplicate this file content to `AGENTS.md`** (this repository may include both for compatibility).

---

## 1) Repository conventions Codex MUST enforce

- **Language:** Python ≥3.10. Use **NumPy**, **SciPy**, **Matplotlib**, **pandas** only.
- **Structure to create/update:**
  ```text
  /src
    cam_kinematics.py
    film_thickness.py
    reynolds_solver.py
    asperity.py
    friction.py
    texture.py
    calibration.py
    io_utils.py
    plots.py
    driver.py
  /colab
    EHL_CAM_tappet_textured.ipynb   # generated
  /data
    CamAngle_vs_Lift_smooth.txt     # provided
    a_texture_rho05.txt             # generated (ρ=5%)
    a_texture_rho08.txt             # generated (ρ=8%)
    a_texture_rho10.txt             # generated (ρ=10%)
    MAIN_SCRIPT.txt                 # provided
    Texture_model.png               # provided
  /tests
    test_geometry.py
    test_reynolds.py
    test_calibration.py
  requirements.txt
  run_local.sh
  README.md
  ```

- **Code quality:** flake8 + black default; no global state; pure functions where possible; docstrings with units.
- **Units:** SI throughout (m, Pa, N·m, s, rad). θ stored in **radians** internally.

---

## 2) Inputs & Preprocessing

- Read `CamAngle_vs_Lift_smooth.txt` → spline lift L(θ); differentiate for velocity V_f(θ)=dL/dt and cam kinematics given RPM.
- Treat **RPM∈{300,500,700,900}**; map to ω_cam = RPM·2π/60 (rad/s).
- Map follower sliding speed **U(θ)** from kinematics (tangent velocity at contact). Keep sign consistent.
- Normalize x‑domain at each θ: b(θ) = Hertzian half‑width from normal load and equivalent radius; **x ∈ [−b(θ), +b(θ)].**

---

## 3) Texture model (to be added to film thickness)

- **Fixed parameters (constant for all RPM, θ, densities):**
  - `w_texture = 35e-6` m
  - `g = 1e-9` m
  - Texture **pitches** `d_texture` by density ρ:
    - ρ=5% → `d_texture = 700e-6` m
    - ρ=8% → `d_texture = 437.5e-6` m
    - ρ=10% → `d_texture = 350e-6` m
  - x‑limits: `x = −X_in·b(θ)` to `+X_out·b(θ)` with `X_in=−4.5`, `X_out=3`  ➜ but **solve only on [−b(θ),+b(θ)]** and clamp outside to avoid leakage.
  - `x_start = 0`
- **Shift condition:** texture frame shift satisfies `d(shift,t) − V_f(θ) = 0` ➜ shift speed equals follower surface speed; implement textures convecting with sliding.
- **Appearance rule (contact window capacity):**
  - Let `Wc(θ)=2 b(θ)`. If `Wc(θ)/w_texture < 1.5` ⇒ **only 1 texture** may exist in [−b,+b]; position it centered at `x_start+shift` and **mask the rest** of [−b,+b] as “no‑texture” to avoid boundary artifacts.
  - Else, populate slots at pitch `d_texture` across [−b,+b] with **non‑overlapping grooves** of width `w_texture`.
- **Mathematical form:** augment the smooth film thickness with a shallow‑groove term:
  - `h(x,θ) = h_smooth(x,θ) + Δh_tex(x,θ; ρ)`
  - Implement **groove profile** guided by `Texture_model.png`. If the figure is ambiguous, use a **rectangular groove** with depth `a_texture(θ, RPM, ρ)` and **edge taper** `g` to avoid numerical spikes:
    - Core region of groove: constant depth `a_texture`.
    - ±g transition ramps via cosine or cubic C¹ splines (width g each side).
  - **Codex must transcribe any explicit formula present in `Texture_model.png`** and prefer it over the default rectangular model.
- **Calibration DOF:** **only** `a_texture(θ, RPM, ρ)` is calibrated. All other fixed parameters stay constant.

---

## 4) Hydrodynamics (mass‑conserving Reynolds, 1D)

- Solve along x for each θ and RPM:
  - `(d/dx) [ (h³ / (12 η)) dp/dx − (U h)/2 ] = 0`, with **Elrod–Adams** mass conservation (or equivalent complementarity) to enforce **p ≥ 0**, saturation S∈[0,1], and flux continuity.
  - Viscosity η: start with **isoviscous Newtonian** unless `MAIN_SCRIPT.txt` encodes a rheology; keep hooks for Houpert/Carreau/Eyring (flags).
- **Boundary conditions:** `p(−b)=p(+b)=0`; outside [−b,+b], p=0 by construction.
- **Discretization:** uniform N_x≥401 in [−b,+b], second‑order centered for dp/dx, upwind for advection term, Picard with under‑relaxation ω∈[0.2,0.6].
- **Convergence:** ‖Δp‖_∞ < 1e−5·max(1 Pa, max p) and **non‑negativity satisfied**; cap iterations at 500, then adapt ω and retry.

---

## 5) Mixed lubrication (load sharing)

- Use **Greenwood–Tripp** asperity contact with composite roughness σ and summit radius β (take from `MAIN_SCRIPT.txt` if present; else provide configurable defaults).
- Contact pressure `p_c(h)` acts only where `h < h_c`; asperity load adds to fluid load to meet the imposed normal load (from cam–follower force).

---

## 6) Friction & Torque

- **Hydrodynamic shear**: τ_s = η U / h; **Poiseuille**: τ_p = −(h/2)(dp/dx)·sgn(U); integrate over x for hydrodynamic friction.
- **Asperity friction**: μ_c·p_c local law; integrate over x.
- **Torque**: map follower friction force to **cam torque** using known kinematic Jacobian/radius at the contact angle (derive from cam geometry; available via lift profile and follower layout).

---

## 7) Implementation sequence Codex MUST follow

1. **Parse `MAIN_SCRIPT.txt`** to recover baseline parameters, geometry, loads, and untextured calculations. Port/refactor into `/src` modules.
2. **Implement cam kinematics** from `CamAngle_vs_Lift_smooth.txt`: splines for L(θ), dL/dθ, convert with ω_cam to velocities.
3. **Implement texture module (§3)** and add `Δh_tex` into `film_thickness.py` so it affects **h**, hence Reynolds and friction.
4. **Implement mass‑conserving Reynolds solver (§4)**; provide switches for classical Reynolds (diagnostic only).
5. **Implement mixed‑regime load sharing (§5)**.
6. **Wire friction/torque computation (§6)**.
7. **Calibrate `a_texture(θ,RPM,ρ)` (§8)** to hit targets; lock other parameters.
8. **Generate deliverables (§8–§9)**.
9. **Run validations (§10)** and only then produce final artifacts.

---

## 8) Calibration and data file formats

### 8.1 Targets (percentage reduction of averaged friction torque vs untextured)
```
RPM     ρ=5%     ρ=8%      ρ=10%
300     3.40%    7.95%     3.40%
500     6.12%    8.92%    10.71%
700     4.21%   14.11%     9.40%
900    18.33%   11.91%     6.87%
```
**Accuracy requirement:** simulated % reductions must be **≥90% accurate** relative to targets (|pred−target| ≤ 0.1·target).

### 8.2 Calibration procedure (automated)

- For each ρ and RPM:
  - Freeze all fixed params; for θ grid (e.g., every 0.5° across a full cam cycle), solve the coupled problem with current `a_texture(θ)` guess.
  - Define objective: squared error between **averaged friction torque reduction** and the target at that RPM & ρ.
  - **Constraints:** `a_texture(θ) ≥ 0`, **smoothness penalty** on d/dθ to avoid oscillations; **upper bound:** `a_texture ≤ w_texture/2` (shallow groove).
  - Use **projected gradient / L‑BFGS‑B**; evaluate gradients via finite differences or autodiff surrogate (optional); terminate when error < tolerance or max iters.
  - Export `a_texture(θ)` sampled at the θ grid used by the solver.

### 8.3 Output files (three, one per density)

- Paths:
  - `/data/a_texture_rho05.txt` (ρ=5%)
  - `/data/a_texture_rho08.txt` (ρ=8%)
  - `/data/a_texture_rho10.txt` (ρ=10%)
- **Format (5 columns):**
  - Col 1: `θ_deg` (monotone over 0–360 or your cam cycle)
  - Col 2–5: `a_texture[m]` at **RPM=300, 500, 700, 900**, respectively
- ASCII, space‑separated, header row included.

---

## 9) Google Colab deliverable & plotting controls

### 9.1 Notebook name & location
- Generate `/colab/EHL_CAM_tappet_textured.ipynb` that is **self‑contained**. All code must also exist in `/src` so users can run locally.

### 9.2 Notebook features
- **Inputs**: uploads/paths to `CamAngle_vs_Lift_smooth.txt` and the three `a_texture_*.txt` files.
- **Run‑all** workflow:
  1. Install deps.
  2. Load data.
  3. Run simulations for all (ρ,RPM).
  4. Produce plots and the summary table.
  5. Print averaged torque reductions vs targets *and* errors.
- **Plots (dimensional):**
  1. Hydrodynamic pressure p(x) vs x at chosen θ, RPM, surface condition.
  2. Film thickness h(x) vs x (same case).
  3. Hydrodynamic friction vs θ.
  4. Asperity friction vs θ.
  5. Friction torque vs θ.

### 9.3 Graph state controls (single settings cell)
```python
GRAPH_SETTINGS = {
  "case_type": "profiles_vs_x",   # or "frictions_vs_angle"
  "cam_angle_deg": -1.0,          # only used for profiles_vs_x
  "surface_condition": 1,         # 0=untextured, 1=textured
  "rpm": 300,                     # one of [300,500,700,900]
  "texture_density": 0.08         # 0.05, 0.08, 0.10
}
```
- When `case_type="profiles_vs_x"`, use the chosen `cam_angle_deg`, `surface_condition`, `rpm`, and `texture_density` to compute p(x), h(x).
- When `case_type="frictions_vs_angle"`, sweep θ over the cycle for the chosen `rpm` and, if textured, chosen `texture_density`.

---

## 10) Validation & CI checks (Codex MUST run and pass)

- **Numerical sanity:**
  - `min(p) ≥ -1e-10 Pa` and `p(±b)≈0` within 1e-4·p_max.
  - Pressure support **only** within [−b,+b] (zero elsewhere).
  - Reynolds solver convergence flag true for all θ samples.
- **Physics sanity:**
  - Hydrodynamic + asperity loads balance the required normal load at each θ (≤0.5% residual).
  - No flow leakage at boundaries; flux continuity satisfied within tolerance.
- **Targets check:**
  - For each ρ and RPM, compute averaged torque reductions vs untextured; assert error ≤ 0.1·target. **Fail build otherwise.**
- **Artifacts present:**
  - All three `a_texture_*.txt` exist with correct columns & row counts.
  - Colab notebook executes top‑to‑bottom without error.

Implement a `run_local.sh` that performs:
```bash
python -m pip install -r requirements.txt
pytest -q
python -m src.driver --generate-texture-data
python -m src.driver --summarize --make-plots
```
and prints the summary table to stdout.

---

## 11) Key function contracts (signatures)

- `film_thickness.h(x, theta, rpm, rho, params) -> ndarray`  
  Returns dimensional film thickness including `Δh_tex`.
- `texture.delta_h(x, theta, rpm, rho, a_texture_func, params) -> ndarray`
- `reynolds_solver.solve(x, h, U, eta, load, params) -> (p, state)`
- `asperity.contact(h, params) -> (p_c, f_c)`
- `friction.compute(p, h, U, params) -> (F_hydro, F_asp, T_total)`
- `calibration.fit_a_texture(targets, rhos, rpms, thetas, params) -> dict[rho]->DataFrame`
- `plots.make_profiles(...)`, `plots.make_angle_sweeps(...)`

---

## 12) Parameters & defaults (editable in code)

- Grid: `Nx=801` (profiles), `Ntheta=721` (0.5° step).
- Relaxation: ω=0.4 (auto‑reduce on stagnation).
- Viscosity: default constant (override if `MAIN_SCRIPT.txt` prescribes Houpert/Carreau/Eyring).
- Contact: Greenwood–Tripp with default σ, β if not in `MAIN_SCRIPT.txt`.

---

## 13) Deliverables (what Codex must produce)

1. **Three calibrated data files**: `/data/a_texture_rho05.txt`, `/data/a_texture_rho08.txt`, `/data/a_texture_rho10.txt` with 5 columns as specified.
2. **A complete Google Colab notebook** `/colab/EHL_CAM_tappet_textured.ipynb` (ready to run) that:
   - Loads inputs, runs simulations, makes the 5 plots, and prints the **table of % reductions** for all RPMs and densities.
3. **Updated Python package** under `/src` + tests under `/tests` + `run_local.sh` + `README.md` usage section.

---

## 14) Non‑negotiables & prohibited actions

- Do **not** change fixed parameters (§3) or targets (§8.1).
- Do **not** relax non‑negativity of pressure to “near‑zero negative”; enforce true mass conservation.
- Do **not** widen the contact beyond ±b(θ).
- Do **not** introduce arbitrary scaling/decay factors in friction/torque; only `a_texture(θ,RPM,ρ)` is the calibration control.

---

## 15) Handover notes

- If `Texture_model.png` contains an explicit analytic groove profile, **implement that exact formula** and keep the rectangular fallback behind a flag `profile="png_formula"| "rectangular_tapered"`.
- Keep the a_texture exporter/loader independent so future re‑calibration can reuse the pipeline.
- Document any assumptions in `README.md` with units.

