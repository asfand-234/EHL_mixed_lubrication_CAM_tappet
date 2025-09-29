# AGENT.md

**Purpose:** In this repository there is a plain‑text Python script named **`UPDATED_MAIN_SCRIPT`** (no extension). Your task (as a coding agent) is to reconstruct it into a **complete, executable** Python module that runs locally and in **Google Colab**, consumes the required input data, and **prints the percentage averaged friction torque reduction for all cases**. You must **not modify any input data files**.

---

## 0) Ground rules (MUST follow)

- **Do not modify any input data files.** Treat them as read‑only and verify via **SHA‑256** checksums before and after execution.
- **Remove all ellipses (`…`) and placeholder stubs** (e.g., `pass`, incomplete functions) found in `UPDATED_MAIN_SCRIPT`. Replace them with fully functional code.
- **Preserve original intent/semantics.** Use surrounding code/comments/equations in `UPDATED_MAIN_SCRIPT` to infer exact behavior.
- **No missing lines.** Deliver a single, ready‑to‑paste Python file with **all definitions completed** and **no placeholders**.
- **Determinism:** Results must be deterministic given the same inputs (seed any randomness, avoid time‑dependent logic).
- **Portability:** Works on Python ≥ 3.10, macOS/Linux/Windows, and **Colab** (no GUI backends; use only stdout/files).

---

## 1) Inputs and interface

### Required input files
1. **Three `a_texture` files** (text/CSV/TSV/whitespace). These contain the textured‑case data.
2. **One `CamAngle_vs_Lift_smooth`** file (text/CSV/TSV/whitespace).

> The file formats may vary. Implement robust loaders that autodetect delimiter and header:
> - Use `pandas.read_csv(..., sep=None, engine="python")` for delimiter sniffing.
> - Fallback encodings: try `utf-8`, then `latin-1`.
> - Normalize column names (lowercase, strip, replace spaces with `_`).

### Output
- **Printed report to stdout**: per‑case **percentage averaged friction torque reduction** and overall average.
- **CSV summary file** with at least columns:
  - `case_id`, `a_texture_file`, `cam_angle_file`, `n_samples`, `avg_torque_baseline`, `avg_torque_textured`, `pct_reduction`.

### CLI (to be implemented)
Create **`UPDATED_MAIN_SCRIPT_completed.py`** with:

```bash
python UPDATED_MAIN_SCRIPT_completed.py   --a-texture path/to/a_tex1.csv path/to/a_tex2.csv path/to/a_tex3.csv   --cam-angle path/to/CamAngle_vs_Lift_smooth.csv   --out path/to/results_summary.csv
```

- `--a-texture` accepts **exactly three** paths (order preserved).
- `--cam-angle` accepts **one** path.
- `--out` optional; default `./outputs/results_summary.csv` (create folder if missing).

---

## 2) Computation requirements

- Use the torque equations and averaging logic implied by the original **`UPDATED_MAIN_SCRIPT`**. When explicitly defined there, **follow those definitions** verbatim.
- If no explicit formula is present, use the standard definition (document this in the generated file header and CSV):

\[ \%\,reduction = 100 \times \frac{T_{baseline} - T_{textured}}{T_{baseline}} \]

- **Averaging:** Compute averages over the relevant dimension (samples/timesteps/operating points) as implied by the original script. If ambiguous, choose the minimal, conventional interpretation consistent with variable names and comments and state this in docstrings.

---

## 3) Implementation plan (what you must do)

1. **Locate** `UPDATED_MAIN_SCRIPT` (plain text file) in the repo.
2. **Reconstruct** a complete Python program:
   - Add all **imports**, type hints, and docstrings.
   - Replace **ellipses** and incomplete functions with full implementations.
   - Add `if __name__ == "__main__":` entry point using `argparse` to implement the CLI above.
3. **I/O helpers** (robust, reusable):
   - `load_frame(path: Path) -> pd.DataFrame`  (delimiter/encoding autodetect).
   - `normalize_columns(df: pd.DataFrame) -> pd.DataFrame` (lowercase, sanitize).
   - `validate_inputs(tex_dfs: list[pd.DataFrame], cam_df: pd.DataFrame) -> None` (ensure required columns are present; raise actionable errors).
4. **Core computation**:
   - Implement `compute_friction_torque(...)` exactly as intended by the original script (e.g., rheology or kinematics it uses). **Do not invent** new models if the original defines them.
   - Implement `percent_reduction(baseline: float, textured: float) -> float` with robust divide‑by‑zero/NaN guards.
5. **Aggregation/reporting**:
   - Produce per‑case and overall summaries.
   - Print a **compact table** to stdout and write a CSV (see columns above).
6. **Integrity checks**:
   - Before any processing, compute SHA‑256 for the **three `a_texture`** files and the **`CamAngle_vs_Lift_smooth`** file.
   - After processing, recompute and **assert they are unchanged**. If changed, abort with a non‑zero exit code and a clear error message.
7. **Colab compatibility**:
   - Provide a callable `main(argv: list[str] | None = None) -> int` so it can be invoked from a Colab cell.
   - Avoid OS‑specific features; use `pathlib` for paths.
8. **Style**:
   - PEP 8 compliant; Black‑format‑friendly.
   - Clear error messages (include path and missing column names).

---

## 4) Acceptance criteria (must pass)

- **Executability:** `python UPDATED_MAIN_SCRIPT_completed.py --help` works; full run completes on both local Python and **Google Colab**.
- **Determinism:** Same inputs → same outputs.
- **Data integrity:** Input checksums identical before/after.
- **Numerical check:** For a small in‑memory synthetic example (not written to disk), `percent_reduction` matches the formula within `1e-12`.
- **Outputs:** Stdout table + CSV contain per‑case rows and overall summary, with units and assumptions documented.
- **Completeness:** No ellipses/`pass`/TODOs remain; all lines present.

---

## 5) Dependencies

Target Python: **3.10–3.12**

Minimum packages (install only if missing):

```
pandas>=2.0
numpy>=1.24
tabulate>=0.9
pyarrow>=15 ; python_version >= "3.10"
```

---

## 6) Example Colab usage

```python
# In Google Colab
!pip -q install pandas numpy tabulate pyarrow
from UPDATED_MAIN_SCRIPT_completed import main
main([
  "--a-texture",
  "/content/a_tex1.csv", "/content/a_tex2.csv", "/content/a_tex3.csv",
  "--cam-angle",
  "/content/CamAngle_vs_Lift_smooth.csv",
  "--out",
  "/content/results_summary.csv",
])
```

---

## 7) Reference implementation shape (for guidance only)

> The delivered file must be **complete**; do not leave TODOs. This sketch shows expected structure only.

```python
#!/usr/bin/env python3
"""
UPDATED_MAIN_SCRIPT_completed.py
Reconstructed from `UPDATED_MAIN_SCRIPT` (text). Fully executable locally and in Colab.
Inputs: three a_texture files + one CamAngle_vs_Lift_smooth file.
Outputs: stdout report + CSV summary of percentage averaged friction torque reduction per case & overall.
"""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def percent_reduction(baseline: float, textured: float) -> float:
    if baseline == 0:
        raise ValueError("Baseline torque is zero; cannot compute percentage reduction.")
    return 100.0 * (baseline - textured) / baseline

# Agent: implement compute_friction_torque(...) exactly as intended by the original script.
# Agent: build per-case results DataFrame and write CSV + print table.

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a-texture", nargs=3, required=True, type=Path)
    p.add_argument("--cam-angle", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/results_summary.csv"))
    args = p.parse_args(argv)

    in_paths = list(args.a_texture) + [args.cam_angle]
    before = {p: sha256_of(p) for p in in_paths}

    tex_dfs = [normalize_columns(load_frame(p)) for p in args.a_texture]
    cam_df = normalize_columns(load_frame(args.cam_angle))

    # Agent: validate inputs; compute per-case torques and reductions; assemble results DataFrame.
    # results.to_csv(args.out, index=False)
    # print(tabulate(results, headers="keys", tablefmt="github", floatfmt=".6g"))

    after = {p: sha256_of(p) for p in in_paths}
    if before != after:
        raise RuntimeError("Input data files were modified during execution; aborting.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 8) Handoff

**Deliverables the agent must produce:**
- `UPDATED_MAIN_SCRIPT_completed.py` (single ready-to-paste file; no placeholders).
- `outputs/results_summary.csv` (unless `--out` overrides path).
- `RUNTIME_LOG.md` summarizing environment, exact command used, runtime, and the input file checksums.

**Commit message suggestion:**
> Reconstruct UPDATED_MAIN_SCRIPT into complete executable; add CLI, robust I/O, integrity checks; compute and report percentage averaged friction torque reduction per case; add CSV output.
