Read the text file "symbolic_data.txt". It is data having first three columns are  independent variables (S, L and F).
And third column is dependent variable (E).
***TO ACHIEVE THIS TASK YOU HAVE NO TIME LIMIT. TAKE YOUR TIME AS MUCH YOU NEED. KEEP OPTIMIZING AND KEEP ITERATING UNTIL YOU MEET ACCURACY***
TASK:
Your task is to create three simple analytic formulas for y1, y2 and y3 through below implementaton criteria steps.

IMPLEMENTATION CRITERIA:
1) First split the E data columnsbwith inital gusses into three columns y1, y2 and y3 in such a way that:
Product of y1*y2*y3 = E    OR
Adddition of y1+y2+y3 = E   OR
Subtraction of y1-y2-y3 = E   OR
Division of y1/y2/y3  = E   OR
combination of any above.
2) Then optimize the data of y1, y2 and y3 in each cell so that the trend of y1, y2 and y3 data get a regular trend shapes when plotting with respect to S or F. Keep optimizing splitting until all columns of y1, y2 and y3 get maximum regular nature or trend.
3) Then fit either S vs y1 and F vs y2  seperately or F vs y1 and S vs y2, L vs y3 (whatever suitable) and get three seperate equations with accuracy must be greater than 90%. For these equations, use only +, -, *, /, exp(), log(), powers with small integer exponents, sin/cose. 
4) if the accuracy in equation in step 3 not achieved then move back to step 2 and optimize the splitting of E again manually and repeat all steps again until you get optimum results. While making sure it obeys step 1. 


HARD STRICT RULES:

1) DO NOT generate radial basis functions, splines, neural networks, or high-degree polynomials.
2) Prioritize interpretability and simplicity but maintain perfect accuracy.
3) The accuracy must be upto 90%

TESTING CRITERIA:
Once fitted equations achieved then put each value of S, F and Land calculate y1, y2 and y3 and compare with your original y1,y2,y3 you kept in step 1 and find the error. If error is greater then keep optimizing. 
Once achieved then only provide the fitted three equations and splitted data of y1, y2 and y3 in commit.
