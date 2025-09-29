# AGENTS.md

## Mission
You are **Codex** acting as a **senior research engineer** for this repository. Your job is to **fix the physics and numerics** of a 1‑D mixed‑lubrication model for a **cam–bucket tappet system with textured shim** so that the predicted **percentage reduction of *averaged friction torque* vs. the untextured case** matches provided target results **within ≥85% agreement (≤15% relative error) for every target point**.  
**Never use calibration, fitting, or non‑physical scaling.** All improvements must be **physics‑based** and **numerically verifiable**.

**Repository inputs**
- `MAIN_SCRIPT.txt`: current raw Python **text file** (not .py) that computes % reduction via a 1‑D mixed‑lubrication model solving the Reynolds equation for textured vs. untextured surfaces.
- `CamAngle_vs_Lift_smooth.txt`: cam kinematics (angle vs. lift).

**Target results** (to be matched by a physics‑only model):
```
RPM    5_percnt   8_percnt   10_percnt
300    3.40%      7.95%      3.40%
500    6.12%      8.92%      10.71%
700    4.21%      14.11%     9.40%
900    18.33%     11.91%     6.87%
```

> The three columns correspond to three texture scenarios in the current study (e.g., texture area fractions or groove fill ratios). Treat them as **three distinct operating scenarios** to be simulated at each RPM.

---

## Non‑Negotiable Rules
1. **No tuning to the targets.** Do **not** introduce arbitrary multipliers, empirical fudge factors, or data fits whose sole purpose is to match numbers.  
2. **Physics first.** You may refine or extend the physics, but only using **documented, defensible models** appropriate for 1‑D mixed lubrication of line/point contacts under cam–follower kinematics.
3. **Numerical rigor.** The Reynolds solver must be **stable, convergent, and verified** under textured film thickness with cavitation handling.
4. **Determinism.** Same inputs ⇒ same outputs. No randomness.
5. **Single final deliverable.** **Only when** all acceptance criteria pass, generate **one** self‑contained, error‑free **Google Colab** script (`.ipynb` or `.py` for Colab) that **prints only the final table of predicted % reductions** (nothing else). Until then, do not output any script to the user.

---

## Physics Requirements (Minimum Set)
Implement or upgrade to the following **without violating Rule #1**:

### Film Thickness & Texture Representation
- 1‑D domain along sliding direction. Use **composite film thickness**: `h(x,t) = h_geom(x,t) + h_tex(x)`.
- `h_geom(x,t)` from cam kinematics and follower compliance; compute normal approach velocity and nominal clearance from `CamAngle_vs_Lift_smooth.txt` (finite‑difference angle → lift → velocity, acceleration).
- `h_tex(x)` for textured shim/grooves:
  - Support **area‑fraction scenarios** (5%, 8%, 10%) or equivalent operating configurations already implied by the current script.
  - Parameterize texture **depth, length, pitch, and duty cycle**; allow **phase vs. entrainment** inversion to capture relative motion of pattern.
  - Enforce **volume neutrality** over one pattern period unless contradicted by geometry.
  - Smooth transitions with **C¹ ramps** to prevent numerical ringing.

### Reynolds Equation (Compressible, Isoviscous‑Incompressible base is acceptable if justified)
- Discretize 1‑D steady‑quasi‑static form at each time/cam‑angle step with **finite volume** or **finite difference (2nd order where possible)**:

  \( \partial_x (\Phi h^3 \partial_x p) = 6\mu U\partial_x h + 12\mu \partial_t h \), with appropriate switch to Elrod‑Adams mass‑conserving cavitation (see below).
- **Mass‑conserving cavitation**: **Elrod–Adams** or **JFO** implementation with saturation variable \(\theta\in[0,1]\); ensure flux continuity and no negative pressures.
- **Viscosity model**: start Newtonian with **Roelands** or **Barus** pressure–viscosity (choose one; document). Optionally, **Eyring shear‑thinning cap** if shear rates demand it (cap the shear stress at \(\tau_0\)).
- **Density**: constant (if isoviscous‑incompressible), or simple pressure‑dependent if using compressible model; be consistent.
- **Slip**: assume no‑slip unless justified by regime; if mixed, incorporate a **shear support** partition via asperity contact model.
- **Roughness/Mixed term**: Use **Greenwood–Tripp** or **Patir–Cheng** flow factors **\(\Phi\)** for 1‑D (at minimum, pressure and shear flow factors). If unavailable, start with **Patir–Cheng** approximations calibrated **from literature only** (not to target data).

### Friction & Torque
- Friction force per unit width: \( f = \int (\tau_{hyd} + \tau_{asperity})\,dx \).
  - \( \tau_{hyd} = \mu U/h + 0.5 h \partial_x p \) (Couette + Poiseuille shear contribution).
  - \( \tau_{asperity} \) from mixed model (e.g., Greenwood–Tripp) using composite roughness and real area of contact.
- Convert to **torque** using cam radius mapping. Compute **averaged friction torque** over one full cam cycle for **textured** and **untextured** cases; report **percentage reduction**.

### Boundary & Operating Conditions
- Inlet boundary: ambient pressure with **inlet meniscus** handling under cavitation.
- Outlet: JFO condition.
- Entrainment speed \(U\) from cam–follower kinematics; vary with RPM.
- Temperature assumed isothermal unless thermal coupling is introduced (optional). If added, solve **1‑D energy balance** with adiabatic side boundaries; still no tuning to targets.

---

## Numerical Requirements
- Spatial grid: **≥ 400 nodes** per texture period or **adaptive refinement** near steep gradients. For full cam cycle, either tile periodic texture or resolve multiple periods—document choice.
- Time/cam‑angle marching: choose **implicit** (recommended) or semi‑implicit scheme with **CFL ≤ 0.5** equivalent for stability if explicit components exist.
- **Solver**: **line‑relaxed Gauss–Seidel (TDMA/Thomas)** or **SOR** with **under‑relaxation 0.1–0.6**; optionally **multigrid V‑cycle** for acceleration.  
- **Convergence** at each step:
  - Pressure residual \( \ell_\infty < 1e-8 \) *and* relative change in film thickness < 1e-8.
  - Cavitation saturation residual \( \ell_\infty < 1e-8 \).
- **Verification**:
  - Grid refinement study (coarse ×1, medium ×2, fine ×4); show friction torque convergence to within **<1%** between medium and fine.
  - Energy check: non‑negativity of dissipated power; no negative film thickness; no negative density/viscosity.

---

## Engineering Workflow (MANDATORY)
1. **Understand & Plan (before coding)**
   - Restate task, list physics gaps, propose fixes, and enumerate risks.
   - Define **acceptance tests** (see next section). **Do not code** before you present this plan.
2. **Design**
   - List files to add/modify (with paths). Provide discrete algorithms: discretization, cavitation handling, solver strategy, boundary conditions, and friction post‑processing.
3. **Implement (small, atomic commits)**
   - Convert `MAIN_SCRIPT.txt` to a clean module (`src/model.py`) and CLI/runner (`src/run_case.py`).
   - Keep functions pure; isolate I/O and plotting (no plots in final deliverable).
4. **Validate**
   - Run the **Test Harness** (below). If any metric fails, **iterate physics**, not parameters.
5. **Self‑Review Checklist (tick in PR description)**
   - [ ] No non‑physical factors added.
   - [ ] Reynolds solver converges on all grids.
   - [ ] Cavitation is mass‑conserving (Elrod–Adams/JFO).
   - [ ] Friction decomposition documented.
   - [ ] All acceptance tests pass with margin.
6. **Final Delivery (only after all tests pass)**
   - Generate **one** Colab script that **prints only** the final prediction table (no other logs).

---

## Test Harness & Acceptance Criteria
Create `tests/test_targets.py` that:
1. Loads `CamAngle_vs_Lift_smooth.txt`.
2. Runs the model for RPM ∈ {300, 500, 700, 900} and for three scenarios {`S5`, `S8`, `S10`} corresponding to the three target columns.
3. Produces a table of predicted **% averaged friction torque reduction**.

**Target table (as floats, %):**
```python
TARGET = {
    300: {"S5": 3.40, "S8": 7.95, "S10": 3.40},
    500: {"S5": 6.12, "S8": 8.92, "S10": 10.71},
    700: {"S5": 4.21, "S8": 14.11, "S10": 9.40},
    900: {"S5": 18.33, "S8": 11.91, "S10": 6.87},
}
```
**Compute metrics:**
- **Per‑cell relative error**: `abs(pred - tgt) / max(tgt, 1e-6)`.
- **MAPE** across all 12 cells.
- **PASS criteria (all must hold):**
  - Every cell relative error ≤ **0.15** (≥85% agreement).
  - MAPE ≤ **0.10** (overall ≥90% agreement) — stronger aggregate guard.
  - Grid refinement check: medium vs. fine torque difference < **1%**.

If any check fails:
- **Print the current 12 predictions to console (for you, the developer)** and **automatically start an improvement cycle**:
  - Re‑evaluate physics assumptions (viscosity law, cavitation implementation, texture phase mapping, mixed contact model).
  - Adjust **modeling choices only** (e.g., enable Eyring cap if shear rates exceed threshold; update flow factors per literature), **never numeric fitting to targets**.
  - Re‑run tests until all pass.

> During iterative runs while developing, you may print diagnostics. **But do not include diagnostics in the final Colab script.**

---

## Project Structure (what you should create)
```
src/
  __init__.py
  geometry.py          # film thickness & texture generators
  kinematics.py        # cam speed & entrainment, from CamAngle_vs_Lift_smooth.txt
  flow_factors.py      # Patir–Cheng/Greenwood–Tripp helpers
  reynolds.py          # solver (Elrod–Adams/JFO), linear algebra
  friction.py          # hydrodynamic + asperity contributions, torque mapping
  scenarios.py         # define S5/S8/S10 texture configs
  runner.py            # CLI entry; runs all RPM×scenario and prints table
  grids.py             # mesh & refinement utilities
tests/
  test_targets.py      # acceptance tests (see above)
MAIN_SCRIPT.txt        # original reference (kept; not executed after refactor)
CamAngle_vs_Lift_smooth.txt
```

---

## Implementation Notes & Defaults (override if you justify)
- **Units:** SI (m, s, Pa, N, J). RPM → rad/s via `ω = 2π * RPM/60`; entrainment **U = R·ω** at contact (or appropriate mapping from cam profile).
- **Material/Fluid:** Start with base viscosity `η0` (e.g., 0.01–0.02 Pa·s) at reference pressure; use **Barus** law `η = η0 exp(α p)` with `α ≈ 1e-8–3e-8 Pa⁻¹` from literature. Document chosen values.  
- **Eyring cap:** `τ = min(η γ̇, τ0)`, with `τ0 ≈ 2–5 MPa` if enabled; justify via shear rates.  
- **Surface:** composite RMS roughness for Patir–Cheng; pick reasonable σ and skewness from literature; **do not** tweak to fit targets.
- **Numerics:** Prefer **finite volume**; center‑based fluxes with **upwind correction** for shear term; TDMA for tridiagonal systems; residual norms in double precision.  
- **Cavitation:** Maintain `θ∈[0,1]`; flux continuity at rupture/reformation boundaries; under‑relax `θ` updates.

---

## Commands (local dev)
```bash
# (Codex may emulate these in its environment)
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip pytest numpy scipy
pytest -q
python -m src.runner
```

---

## What to Output to the User (very important)
- While iterating: **Do not** output any intermediate scripts or numbers to the user.
- **Only when** all acceptance tests pass, generate **one** Google Colab script named `cam_textured_mixed_lubrication_colab.py` (or `.ipynb`) that:
  - Embeds or loads the two repository files (`MAIN_SCRIPT.txt`, `CamAngle_vs_Lift_smooth.txt`).
  - Runs the final validated solver across RPM × {S5, S8, S10}.
  - **Prints only** the final 4×3 table of predicted % reductions (no other logs, no plots).
  - Exits.

**If tests do not pass**: keep working internally; **do not** produce any user‑facing script. Iterate physics and numerics per this AGENTS.md until they do.

---

## Quality Gates
- Lint: PEP8/flake8 (internal).  
- Types: optional `pyright` for strict typing.  
- **Physics gate**: enforcement of Rules #1–#5 via unit tests and reviewer checklist.  
- **Numerics gate**: residual and refinement criteria must pass in CI run.

---

## Known Pitfalls
- Non‑mass‑conserving cavitation leads to over‑prediction of friction reduction in textures.
- Insufficient axial resolution per texture period causes spurious diffusion and target mismatch.
- Ignoring pressure–viscosity (α=0) can under‑predict load capacity and distort friction trends vs. RPM.
- Treating texture phase static in the follower frame may be wrong; ensure correct **entrainment‑based phase drift**.
- Failing to average torque over **full cam cycle** leads to erroneous comparison vs. targets.

---

## Documentation & Decision Log
For any physics change, add a short docstring with: **(a)** rationale, **(b)** equations, **(c)** source(s). If you substitute models (e.g., Roelands ↔ Barus), justify with shear/pressure regime estimates.

---

## Final Reminder
**No parameter fitting to targets.** Fix physics and numerics. Deliver the **single Colab script** only after the acceptance suite passes and predicted reductions meet the ≥85% per‑cell and ≤10% MAPE criteria.
