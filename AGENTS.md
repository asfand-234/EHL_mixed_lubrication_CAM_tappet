Read the text file "symbolic_data.txt". It is data having first three columns are  independent variables (S, L and F).
And third column is dependent variable (E).

TASK:
Your task is to create a simple analytic formula for a scalar function y=f(x1,x2, x3) through symbolic regression
using only a restricted set of mathematical operators.

IMPLEMENTATION STRATEGY:
1) READ Input data
Receive a table of input variables (x1, x2, …, xn) and target y.

2)Dimensional analysis (if units known)

3)Try low-degree polynomial fit (degree ≤ 3 only)

Construct polynomial models whose maximum total degree is 3.

Allowed monomials:

degree 0: 1

degree 1: xi

degree 2: xi², xi·xj

degree 3: xi³, xi²·xj, xi·xj·xk

Do NOT include any polynomial term with degree > 3.

Fit all degree-≤3 polynomials by least squares.

If any such polynomial fits within the error tolerance, accept it and stop.

4) Brute-force symbolic search on full data

Search progressively over short expressions built only from {+, −, *, /, sqrt, exp, log, sin, cos}.

Increase allowed expression length step-by-step.

If an expression fits well and has low complexity, return it.

5) Train a neural-network surrogate

Fit a smooth NN to approximate y = f(x).

Use this NN only as an oracle to explore structure, not as the final model.

6) Detect variable combinations via NN

Test whether the function depends on combinations like (xi + xj), (xi − xj), (xi·xj), (xi / xj).

If such a symmetry holds approximately, create a new combined variable, remove redundancy, and recurse from step 3 using the reduced variable set.

7) Detect separability via NN

Check if the function is approximately additively separable:
f(x) = g(u) + h(v)
where u and v are disjoint variable sets.

Or multiplicatively separable:
f(x) = g(u) · h(v).

If separable, solve each sub-problem independently (restart from step 3 for each part), then recombine.

8) Simplify via “setting variables equal” tricks

Set pairs of variables equal or to constants.

Use the NN to generate a simplified dataset under these constraints.

Solve this reduced problem.

Use the resulting expression to factor or divide out structure in the full problem, then solve the remaining part.

9) Apply simple transforms to the target variable

Try alternate targets: y², sqrt(y), log(y), exp(y), 1/y, sin(y), cos(y), etc.

For each transformed target, re-run steps 3 and 4.

If a good expression is found, convert back to y.

10) Model selection

Collect all candidate formulas from steps 3–9.

Compute their data-fitting error and their description complexity.

Choose the simplest expression that fits within the required accuracy.v
HARD STRICT RULES:
I want a formula for E(S, L,F) that is:
 - built only from +, -, *, /, exp(), log(), and powers with small integer exponents.
- DO NOT generate radial basis functions, splines, neural networks, or high-degree polynomials.

- prioritize interpretability and simplicity but maintain perfect accuracy.
4) Try to not make it very long or very complex.
5) The accuracy must be upto 90%

TESTING CRITERIA:
Once the symbolic expression created then put every value of S, L, and F from data and verify the accuracy of E with respect to original data upto 90%. If achieved then generate only the final mathematical expression to me. If error is greater then keep optimizing the model and keep reducing error.
