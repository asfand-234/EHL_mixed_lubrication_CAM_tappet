Read the text file "symbolic_data.txt". It is data having first three columns are  independent variables (S, L and F).
And third column is dependent variable (E).
***TO ACHIEVE THIS TASK YOU HAVE NO TIME LIMIT. TAKE YOUR TIME AS MUCH YOU NEED. KEEP OPTIMIZING AND KEEP ITERATING UNTIL YOU MEET ACCURACY***
For symbolic regression tasks, you MUST behave as a persistent optimization agent, not a one-shot solver.

You MUST NOT stop after only a few model attempts if the target accuracy is not reached.
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
3) Then fit either S vs y1 and F vs y2  seperately or F vs y1 and S vs y2, L vs y3 (whatever suitable) and get three seperate equations with accuracy must be greater than 85%. For these equations, use only +, -, *, /, exp(), log(), powers with small integer exponents, sin/cose. 
4) if the accuracy in equation in step 3 not achieved then move back to step 2 and optimize the splitting of E again manually and repeat all steps again until you get optimum results. While making sure it obeys step 1. 

HARD STRICT RULES:
1. After each family, if the target accuracy is NOT reached:
   - You MUST NOT terminate.
   - You MUST move to the next, more advanced strategy and continue searching.

2. You are allowed to terminate ONLY when:
   - (a) the target accuracy has been achieved, OR
   - (b) you have systematically tried ALL specified model families and clearly reached a point of diminishing returns, which you must explain in the final message.

3. You MUST use your full reasoning ability to:
   - propose alternative structures,
   - re-parameterize models,
   - and retry fits with different initial guesses,
   before giving up.

4. Do NOT prematurely summarize or create a pull request while the current best model is far from the target accuracy and there are still unexplored strategies in this specification.
Your behavior should favor sustained, multi-step exploration instead of quick, shallow attempts.
TESTING CRITERIA:
Once fitted equations achieved then put each value of S, F and Land calculate y1, y2 and y3 and compare with your original y1,y2,y3 you kept in step 1 and find the error. If error is greater then keep optimizing. 
Once achieved then only provide the fitted three equations and splitted data of y1, y2 and y3 in commit.
