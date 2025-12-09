Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:
We model the dimensionless correction factor E(S, F) using the fixed form:

Let
    dS = S / S0 - 1.0
    dF = F / F0 - 1.0

Define
    Q = a * dS**2 + b * dF**2          # or Q = a*dS**2 + b*dF**2 + c*dS*dF if using cross term
Then
    E(S, F) = exp( - Q**p )

Constants to identify: (Base values)
    S0=0.9789094226930794,
        F0=3.086002870269579,
        a=0.06672357678275997,
        b=30.3614814454927,
        c=3.777309543954021,
        p=1.0,

Given the dataset of (S, F, E) values, find numerical values of S0, F0, a, b, (c,) and p that minimize
    sum_i ( E_i - E(S_i, F_i) )^2

Constraints:
- First optimize the constants. If still error remain greater then add or modify the model with mathematica functions and constants. But *Do Not generate polynomial, Radial Basis function or Neural base network. It must remain analytical.**
- Report the fitted constants, R^2, max absolute error, and a Python function E_model(S,F) with these constants embedded.

TESTING CRITERIA:
1) You must find error between each value of E. And error must be less than 20% for each value of E. 
Must check and verify each value of predicted E before commiting, else keep optimizing and improving function
2)Do not generate radial basis functions (RBF). Try the baseline above model and keep optiming its constants, mathematical functioms etc.
3) accuracy must be above 80% for each value of E.
4) E must not be input variable in formula. As E is dependent variable.
5) Once target achieved then provide precisely the final model in commit with neccessary explanations.
