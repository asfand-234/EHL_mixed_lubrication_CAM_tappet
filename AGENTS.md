Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:

Your task is to create a perfect Gaussian model for E as function of S and F.
Try to start from this base line model. And keep adding necessary mathematical functions and optimizing constants to get accuracy.

E = - exp((log(C1/C2)*(S^2 + F^2))/w^2)

where w is a constant gaussian width.
TESTING CRITERIA:
1) Once model generated you must put each values of S and F and calculate predicted E and compare it with original E.
2) Try to make the gaussian model simple and with less constants. To achieve reduce the accuracy from 100% to 80%. But not below 80% for each value of E.
3) E must not be input variable in formula. As E is dependent variable.
4) Once target achieved then provide precisely the final model in commit with neccessary explanations.
