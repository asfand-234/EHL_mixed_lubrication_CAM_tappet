Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S and F).
And third column is dependent variable (E).

TASK:
Your task is to create simple analytic formulas for a scalar functions through below implementaton criteria steps.

IMPLEMENTATION CRITERIA:
1) first split the E data columns into two columns y1 and y2 in such a way that the product of y1*y2 must be equal to E. 
2) The splitting of E must be so optimum so that the trend of y1 and y2 data get a regular/better trend shape when plotting with respect to S or F. Keep optimizing splitting until both columns of y1 and y2 get maximum regular nature or trend.
3) Then fit either S vs y1 and F vs y2  seperately or F vs y1 and S vs y2 (whatever suitable) and get two sperate equations with accuracy must be greater than 90%. For these equations, use only +, -, *, /, exp(), log(), powers with small integer exponents, sin/cose. 
4) if the accuracy in equation in step 3 not achieved then move back to step 2 and optimize the splitting of E again and repeat all steps again until you get optimum results.


HARD STRICT RULES:

1) DO NOT generate radial basis functions, splines, neural networks, or high-degree polynomials.
2) Prioritize interpretability and simplicity but maintain perfect accuracy.
3) Avoid making it very long or very complex.
4) The accuracy must be upto 90%

TESTING CRITERIA:
Once both fitted equations achieved then put each value of S and F and calculate y1 and y2 and find the error. If error is greater then keep optimizing. 
Once achieved then only provide the fitted two equations and data of y1 and y2 in commit.
