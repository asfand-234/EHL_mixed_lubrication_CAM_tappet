
# AGENT.md — Minimal, precise playbook for Codex on `EHL_mixed_lubrication_CAM_tappet`

> **Purpose:** Direct Codex to implement a **complete 1D mixed-lubrication simulation** for a radial CAM with a flat-faced bucket tappet and a **textured shim**, where the **only calibrated quantity** is the **texture amplitude/depth `a_texture(θ, RPM, ρ)`** strictly following the geometric model in **`Texture_model.png`**. Codex must deliver three calibrated data files and **one** self-contained Google Colab notebook that reads those files and reproduces the target averaged friction-torque reductions to **≥90% accuracy**, with no additional scaling, fitting, or correction inside the notebook.

---

## 0) Scope & Guardrails (Codex MUST comply)

- **Inputs already in repo:**
  - `MAIN_SCRIPT.txt` — baseline raw script (untextured averaged friction torque).
  - `CamAngle_vs_Lift_smooth.txt` — cam angle vs lift (smooth).
  - `Texture_model.png` — **authoritative geometry** for the texture; **implement this exact model** (no substitute shapes, no rectangular fallback). The **only degree of freedom to calibrate is the amplitude/depth** `a_texture`.
- **Textures (fixed parameters for all RPM, θ, densities):**
  - Area densities ρ ∈ {5%, 8%, 10%}
  - `w_texture = 35e-6` m
  - `g = 1e-9` m (edge smoothing/transition consistent with the provided model)
  - Texture pitch by density:
    - ρ=5%  → `d_texture = 700e-6` m
    - ρ=8%  → `d_texture = 437.5e-6` m
    - ρ=10% → `d_texture = 350e-6` m
  - x-domain linkage: `x ∈ [-X_in·b(θ), +X_out·b(θ)]`, with `X_in = -4.5`, `X_out = 3`, but **the hydrodynamic solution domain is strictly [−b(θ), +b(θ)]**.
  - `x_start = 0`
- **Shift constraint:** `d(shift, t) − V_f(θ) = 0` (texture frame convects with follower velocity).
- **Appearance rule (capacity):** Let `Wc(θ)=2 b(θ)`. If `Wc(θ)/w_texture < 1.5` at a given θ, **only one texture** is allowed within [−b,+b]; mask the rest to avoid boundary artifacts.
- **Hydrodynamics:** **Mass-conserving Reynolds** (Elrod–Adams or equivalent complementarity), `p ≥ 0`, `p(±b)=0`, no leakage outside [−b,+b].
- **Mixed regime:** Greenwood–Tripp type load sharing; total load = fluid + asperity.
- **Friction & torque:** hydrodynamic shear + Poiseuille + asperity; torque via appropriate cam–follower mapping.
- **No extra data files** beyond the three `a_texture` files and **one** Colab notebook. Codex may create temporary data during calibration but **must not commit or deliver** any kinematics/pressure/exported datasets.

---

## 1) Minimal deliverables (and only these)

1) **Three calibrated data files** (space-separated text, with header):
   - `a_texture_rho05.txt`
   - `a_texture_rho08.txt`
   - `a_texture_rho10.txt`
   **Format (5 columns):**
   - Col 1: `theta_deg` (monotone grid covering the cam cycle)
   - Col 2–5: `a_texture[m]` for **RPM=300, 500, 700, 900** respectively

   **Strict requirement:** Every value in every column of each file is calibrated so that when imported by the notebook **without any internal scaling/fitting/decay**, the simulation reproduces the target averaged friction-torque reductions to **≥90% accuracy** for each (ρ, RPM).

2) **One single Google Colab notebook** (self-contained, detailed, ready-to-paste, and executable in a clean environment) that:
   - Imports `CamAngle_vs_Lift_smooth.txt` and the three `a_texture_*.txt` files.
   - Implements the full simulation (hydrodynamics + mixed lubrication + friction + torque).
   - Produces the required plots and the summary table.
   - Allows manual graph-state control from a single settings cell (see §3.3).
   - Contains extensive comments and clear layout.

> Do **not** create additional modules/packages or persistent outputs. All Python code can live inside the Colab in clearly labeled cells.

---

## 2) Targets for calibration (must be met to ≥90% accuracy)

```
RPM     ρ=5%     ρ=8%      ρ=10%
300     3.40%    7.95%     3.40%
500     6.12%    8.92%    10.71%
700     4.21%   14.11%     9.40%
900    18.33%   11.91%     6.87%
```

- The accuracy condition is: `abs(predicted - target) ≤ 0.1 * target` for each line item.
- **Only `a_texture(θ, RPM, ρ)` is calibrated.** All other fixed parameters remain fixed.

---

## 3) Notebook specifications Codex must implement

### 3.1 Numerical grid and angles
- **Ntheta = 328** samples per cam cycle (uniform in degrees unless `MAIN_SCRIPT.txt` prescribes otherwise).
- **Nx = 401** points across the hydrodynamic domain **x ∈ [−b(θ), +b(θ)]**.

### 3.2 Physics & numerics
- **Film thickness:** `h(x,θ) = h_smooth(x,θ) + Δh_tex(x,θ; ρ)`, where **`Δh_tex` is derived *directly and only* from the geometry in `Texture_model.png`**. The **amplitude/depth is `a_texture(θ, RPM, ρ)`** (to be read from the data files).
- **Reynolds (mass-conserving):** 1D, isoviscous Newtonian unless `MAIN_SCRIPT.txt` indicates otherwise; `p ≥ 0`, `p(±b)=0`; stabilized Picard/biCGSTAB with under-relaxation.
- **Mixed lubrication:** Greenwood–Tripp load sharing; ensure load balance within ≤0.5% residual per θ.
- **Friction & torque:** hydrodynamic (shear+Poiseuille) + asperity; compute friction torque vs θ, then average over cycle.

### 3.3 Graph-state controls (single settings cell)
```python
GRAPH_SETTINGS = {
  "case_type": "profiles_vs_x",  # or "frictions_vs_angle"
  "cam_angle_deg": -1.0,         # only for profiles_vs_x
  "surface_condition": 1,        # 0=untextured, 1=textured
  "rpm": 300,                    # one of [300,500,700,900]
  "texture_density": 0.08        # 0.05, 0.08, 0.10
}
```
- When `case_type="profiles_vs_x"`: compute and plot **p(x)** and **h(x)** at the specified cam angle, RPM, and surface condition/density.
- When `case_type="frictions_vs_angle"`: sweep θ for the chosen RPM and (if textured) density; plot hydrodynamic friction, asperity friction, and friction torque vs cam angle.

### 3.4 Plots & outputs (dimensional)
- (i) Hydrodynamic pressure p(x) vs x  
- (ii) Film thickness h(x) vs x  
- (iii) Hydrodynamic friction vs cam angle  
- (iv) Asperity friction vs cam angle  
- (v) Friction torque vs cam angle  
- (vi) **Printed table**: predicted % reduction of **averaged** friction torque for all RPMs and all three densities, side-by-side with targets and errors.

---

## 4) Implementation sequence Codex must follow

1. Read and interpret `MAIN_SCRIPT.txt` to extract baseline parameters/assumptions (untextured).
2. Implement CAM kinematics from `CamAngle_vs_Lift_smooth.txt` (internal to the notebook; do **not** export datasets).
3. Implement the **texture geometry strictly from `Texture_model.png`**. **Do not** substitute any alternate groove shapes. The **only** tunable parameter is the **amplitude/depth** `a_texture(θ, RPM, ρ)`.
4. Add `Δh_tex` to the film thickness; ensure linkage to x via contact geometry `b(θ)`.
5. Enforce the **appearance capacity rule**: if `2 b(θ)/w_texture < 1.5`, only one texture within [−b,+b].
6. Solve the **mass-conserving Reynolds** equation with non-negativity and p(±b)=0; ensure no leakage outside [−b,+b].
7. Mixed-lubrication load sharing and total torque as specified.
8. **Calibration**: fit `a_texture(θ, RPM, ρ)` so that cycle-averaged torque reductions hit targets (error ≤10% of target). Use smoothness constraints in θ and bounds `a_texture ≥ 0`, with a reasonable upper bound (e.g., a small fraction of `w_texture`) to keep the shallow-groove assumption. **Calibration happens offline within the notebook’s internal logic, but the only persistent outputs are the three final data files.**
9. Export the three `a_texture_*.txt` files with the exact 5-column format.
10. Demonstrate that **importing those files back** (with no scaling/fits) reproduces the results and passes the accuracy checks.

---

## 5) Stability & correctness checks (must pass)

- **Numerical:** convergence for all θ; `min(p) ≥ 0` (up to numerical epsilon) and `p(±b)=0` within tolerance; flux continuity satisfied.
- **Physics:** load balance within ≤0.5% per θ; pressure support confined to [−b,+b].
- **Targets:** each (ρ, RPM) meets the ≥90%-accuracy criterion when using the **imported** `a_texture` files with **no internal rescaling**.
- **No extra artifacts:** besides the **three** data files and the **one** notebook, nothing else is produced or saved.

---

## 6) File names and formats (final)

- `/a_texture_rho05.txt`, `/a_texture_rho08.txt`, `/a_texture_rho10.txt`
  - 5 columns: `theta_deg  atex_300[m]  atex_500[m]  atex_700[m]  atex_900[m]`
  - Header line required; ASCII, space-separated.
- `/EHL_CAM_tappet_textured.ipynb` (Google Colab)
  - Clear sections, heavy inline comments, robust error handling, and a top “Run all” note.

---

## 7) Non-negotiables

- Do **not** change the fixed parameters or targets listed above.
- Do **not** introduce any groove shape other than the geometry expressed in `Texture_model.png`.
- Do **not** produce or commit kinematics/dynamics/pressure datasets or any files besides the three `a_texture` files and the single notebook.
- Do **not** apply any scaling, fitting, or decay to `a_texture` inside the simulation when importing the calibrated files; values must be used as-is.

