Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:

The following 6th degree polynomial (about 28 terms) gives 100% fitted results for data of E. 

E = -99.411294107003
    + 250.065187868297*S - 194.154147029667*F
    - 104.423541300749*S^2 + 82.718669111793*S*F - 216.807172791364*F^2
    + 35.865907681079*S^3 - 132.363302439310*S^2*F + 282.586580057985*S*F^2 + 47.189677968289*F^3
    - 4.375555144577*S^4 + 11.195407460734*S^3*F + 14.478967940004*S^2*F^2 - 101.654509179886*S*F^3 - 24.250089893800*F^4
    + 0.547181394841*S^5 - 4.560284932299*S^4*F + 19.939816832655*S^3*F^2 - 50.044244694836*S^2*F^3 + 68.559374278790*S*F^4 - 11.228974166543*F^5
    - 0.017442502192*S^6 + 0.097982682392*S^5*F + 0.171318030992*S^4*F^2 - 2.928407540264*S^3*F^3 + 9.342694809872*S^2*F^4 - 12.280945920685*S*F^5 + 3.576755063620*F^6

Your Task is to reduce the terms of polynomial and its complexity to a maximum level by two ways:

1) Try to incorporate support of other mathematical functions like exp, log, powers of fractiona/integers, sin/cos etc. wherever helpful.
2) Reduce the accuracy of each cell of data from 100% to 80%. 

HARD STRICT RULES:

1)  Prioritize interpretability and simplicity but maintain perfect accuracy.
2) The accuracy must be upto 80% of each cell data.
3) Do not only calculate overall R^2, but also verify every time by putting S and F in equation and find error between predicted E and actual E. If error greater than re-optimize the equations. Fhe accuracy for each value of E should be above 80%
TESTING CRITERIA:
Once fitted equation achieved then put each value of S and F and calculate E and compare it with original each data of E if error is greater then optimize the polynomial equation to highest extend. Once accuracy achieved then generate only the optimized fitted equation for me in commit.
