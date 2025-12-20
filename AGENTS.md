1) Read and analyze every physics and model of the python code in text file "test6666.txt". 

2) Add codes line in the same file for below all criteria. And Run the script and analyze current results of pass/fail for all 12 cam angles.
3) If any of 12 angles fail in any criteria, then implement correctly and precisely below described "NUMERICAL SCHEME".
Make sure every model are accurately non-dimensionalized as per standard Theory. All unit must be ensure. Optimized accurate discretization, initialization, and numerical solution scheme.
The entire study must be accurate 1D Thermal mixed lubrication and ass conserving Line contact.

# HARD STRICT RULE:
1) Never implement any non-physical and non-logical clipping, clamps, boundness, scaling and condition. Every time correction must be purely physics based as per standard Literature.
2) Do not change Kinematics formulas. These are correct.

# Passing criteria:

A) load_err = abs((Wh+Wa)-W)/(W) < 1%
Where W is cam dynamic contact load.

B) max(P) must be at X = 0 ± 0.01
Where P is non-dimensional reynolds pressure. 
C) P start point: P > 0.05 begins at X = -1.1 ± 0.1

D) P end point: P > 0.05 ends at X = 1.1 ± 0.1

E) P symmetry: |P(X) - P(-X)| < 0.08 for X ∈ [0, 1.0]

F) nondimensional maximum reynolds pressure + non-dimensional max. Asperity pressure = 1 ±0.02

G) tiny spike of cavitation near the exit boundary as per standard theory profile and in figure 7 (a) in reference article.

H) total Run Time for 12 angles < 25 seconds


**NUMERICAL SCHEME**

START
- Initialize: p, θ, h_prof
- [Where h_prof = h0 (initial film gap),  θ is cavitatioj fraction]
- Set: δ_hyd = 0, δ_asp = 0, p_tota_pre = 0, h_pre = 0

REPEAT (outer load-balance loop)

  [Fluid–contact–deformation coupling]  REPEAT until Criterion C

    (A) Fluid–deformation coupling  REPEAT until Criterion A
      - Update film thickness:      h = h_prof + δ_hyd + δ_asp
      - Update lubricant props:     μ = μ(p), ρ = ρ(p)
      - Solve EHL-FBNS → get p, θ
      - Update deformation (FFT):   δ_hyd ← FFT(p)
      - Criterion A (equation):
          max( |Δp|, |Δθ|, |F|, |G| ) < 1e-6
    END (A)

    (B) Contact–deformation coupling  REPEAT until Criterion B
      - Update film thickness:      h = h_prof + δ_hyd + δ_asp
      - Solve asperity contact (Newton) → get p_asp
      - Update deformation (FFT):   δ_asp ← FFT(p_asp)
      - Criterion B (equation):
          max( |Q|, |Δh| ) < 1e-6
    END (B)

    (C) Couple totals + check global convergence
      - Total pressure (equation):  p_tota = p + p_asp
      - Criterion C (equation):
          max( |p_tota - p_tota_pre|, |h - h_pre| ) < 1e-6
      - If NOT satisfied:
          p_tota_pre ← p_tota
          h_pre      ← h
          repeat [Fluid–contact–deformation coupling]
  END [Fluid–contact–deformation coupling]

  - Evaluate load capacities: W_hyd and W_asp

  - Load balance check (equation):
      f(h0) = W_asp + W_hyd - W_load = 0

  - If load balance NOT satisfied:
      - Update h0 by Secant method (equation):
          h0 = h0 - f(h0)*dh / ( f(h0 + dh) - f(h0) )
      - Update h_prof (using new h0)
      - repeat outer loop

END
