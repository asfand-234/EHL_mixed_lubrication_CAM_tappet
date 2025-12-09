Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:
import numpy as np

# 12-component reference model (your current E_model)
_COMPONENTS = np.array([...])  # as given

def E_full(S, F):
    S = np.asarray(S, float)
    F = np.asarray(F, float)
    acc = np.zeros_like(np.broadcast_to(S, np.broadcast(S, F).shape))
    for S0, F0, wS, wF, A in _COMPONENTS:
        acc += A * np.exp(-(((S - S0)/wS)**2 + ((F - F0)/wF)**2))
    return np.clip(acc, 0.0, 1.2)  # or clip to [0,1] if you prefer

def E_approx(S, F, A, S0, F0, wS, wF, p):
    r2 = ((S - S0)/wS)**2 + ((F - F0)/wF)**2
    return A * np.exp(-(r2**p))
Fit the 6-parameter analytic model E_approx(S, F, A, S0, F0, wS, wF, p) to approximate
the existing 12-component model E_full(S, F).

1. Sample a grid of S and F values that cover the full range of _COMPONENTS coordinates,
   e.g. S in [min(S0_k) - 1, max(S0_k) + 1], F in [min(F0_k) - 1, max(F0_k) + 1].
2. For each grid point, compute E_full(S,F).
3. Minimize the mean squared error between E_approx(S,F,...) and E_full(S,F) over this grid.
4. Use the following initial guesses:
   A = 1.0
   S0 = 6.4
   F0 = 2.5
   wS = 3.0
   wF = 1.5
   p  = 1.7
5. Constrain:
   A in (0, 1.2], wS>0, wF>0, p>=1.

Return:
- The fitted parameter values,
- The mean squared error, max absolute error,
- And a final Python function E_approx_model(S, F) with constants embedded.

**Constraints:**
- First optimize the constants. If still error remain greater then add or modify the model with mathematica functions and constants. But **Do Not generate polynomial, Inverse distance weighting type,Radial Basis function or Neural base network. It must remain analytical**
- Report the fitted constants, R^2, max absolute error, and a Python function E_model(S,F) with these constants embedded.

TESTING CRITERIA:
1) You must find error between each value of E. And error must be less than 20% for each value of E. 
Must check and verify each value of predicted E before commiting, else keep optimizing and improving function
2)Do not generate radial basis functions (RBF). Try the baseline above model and keep optiming its constants, mathematical functioms etc.
3) accuracy must be above 80% for each value of E.
4) E must not be input variable in formula. As E is dependent variable.
5) Once target achieved then provide precisely the final model in commit with neccessary explanations.
