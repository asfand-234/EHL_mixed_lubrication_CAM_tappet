Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S and F).
And third column is dependent variable (E).

TASK:
Your task is to optimze a very simple analytic formula for E = 1.56 - 0.24 * F - 0.18 * (S / F)
using only a restricted set of mathematical operators.The current model gives large error. Optimize/improve it by incorporating below given mathematical symbols if needed to enhance accurate upto 90%.

HARD STRICT RULES:
I want a *short closed-form formula* for E(S, F) that is:
- built only from +, -, *, /, exp(), log(), and powers with small integer exponents.
- *no* radial basis functions, splines, neural networks, or high-degree polynomials.
- at most ~3 main terms in the expression.
- prioritize interpretability and simplicity over perfect accuracy.
4) Try to not make it very long or very complex.
5) The accuracy must be greater than 90%

TESTING CRITERIA:
Once the symbolic expression created then run with every value of independent data and verify the accuracy must be 90%. If achieved then generate only the final mathematical expression to me.
