START
- Initialize: p, θ, h_prof
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
