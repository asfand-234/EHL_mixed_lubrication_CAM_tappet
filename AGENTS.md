Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S and F).
And third column is dependent variable (E).

TASK:
Your task is to optimze a simple analytic formula for E; 
E ≈ a + b * exp(-c * F) + d * exp(-e * S)
a, b, c, d, e = -0.57121192, 0.94495082, 0.11055355, 3.69613574, 0.53906929
using only a restricted set of mathematical operators.The current model gives large error. Optimize/improve it by incorporating below given mathematical symbols if needed to enhance accurate upto 90%.

HARD STRICT RULES:
Try different powers (in decimals) with either E or F or both and calculate error with respect to original value of E in data file for each row. If not improved try another symbol or operator.
I want a *short closed-form formula* for E(S, F) that is:
- built only from +, -, *, /, exp(), log(), and powers with small integer exponents.
- *no* radial basis functions, splines, neural networks, or high-degree polynomials.
- at most ~3 main terms in the expression.
- prioritize interpretability and simplicity over perfect accuracy.
4) Try to not make it very long or very complex.
5) The accuracy must be upto 90%

TESTING CRITERIA:
Once the symbolic expression created then run with every value of independent data and verify the accuracy must be 90%. If achieved then generate only the final mathematical expression to me.
