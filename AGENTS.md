Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:
Your task is to model the dimensionless correction factor E(S, F) :

def E_model(S_val, F_val):
    """Callable model with the fitted constants embedded."""
    S0, F0, a, b, c, p = params
    dS_val = np.asarray(S_val) / S0 - 1.0
    dF_val = np.asarray(F_val) / F0 - 1.0
    Q_val = a * dS_val ** 2 + b * dF_val ** 2 + c * dS_val * dF_val
    Q_val = np.maximum(Q_val, 1e-12)
    return np.exp(-(np.abs(Q_val) ** p))
S0=2.2540050844, F0=4.0902888336, a=0.0006811763, b=0.0069066459, c=0.0036143950, p=0.0527759716

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
