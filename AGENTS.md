Read the text file "symbolic_data.txt". It is data having first three columns are  independent variables (S, L and F).
And third column is dependent variable (E).

TASK:
Your task is to create a simple analytic formula for a scalar function y=f(x1,x2, x3) through symbolic regression
using only a restricted set of mathematical operators.

HARD STRICT RULES:
I want a short closed-form formula for E(S, L,F) that is:
- built only from +, -, *, /, exp(), log(), and powers with small integer exponents.
- NO radial basis functions, splines, neural networks, or high-degree polynomials.
- at most ~4 main terms in the expression.
- prioritize interpretability and simplicity but maintain perfect accuracy.
4) Try to not make it very long or very complex.
5) The accuracy must be upto 90%

TESTING CRITERIA:
Once the symbolic expression created then put  every value of S L, and F from data and verify the accuracy of E with respect to original data upto 90%. If achieved then generate only the final mathematical expression to me. If error is greater then keep optimizing the model and keep reducing error.
