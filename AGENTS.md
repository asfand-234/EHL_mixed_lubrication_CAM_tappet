Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S and F).
And third column is dependent variable (E).

TASK:
Your task is to create a very simple analytic formula for a scalar function y=f(x1,x2) through symbolic regression
using only a restricted set of mathematical operators.

HARD STRICT RULES:
I want a *short closed-form formula* for E(S, F) that is:
- built only from +, -, *, /, exp(), log(), and powers with small integer exponents.
- *no* radial basis functions, splines, neural networks, or high-degree polynomials.
- at most ~4 main terms in the expression.
- prioritize interpretability and simplicity but maintain perfect accuracy.
4) Try to not make it very long or very complex.
5) The accuracy must be upto 86%

TESTING CRITERIA:
Once the symbolic expression created then run with every value of S and F from data and verify the accuracy of E with respect to original data upto 86%. If achieved then generate only the final mathematical expression to me.
