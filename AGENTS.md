Read deeply every paragraph, equation and figures in research article "mixed_lubrication.pdf" in my repository. It is related to EHL/mixed lubrication line contact including asperity model. 

Your task is to generate a complete error free ready to paste google colab code that solve complete non-dimensional reynolds equation in eq. 9 from pdf using simutanelously other equations including eqs. 5, 6,7, 11, 13, 17 etc. The code should plot same non-dimensional hydrodynamic pressure Ph vs X, non-dimensional asperity pressure Pa vs X, and non-dimensional film thickness H vs X separately three graphs for only:
U = 1e-11, W = 1e-4, G= 4500, V = 0.01
Sigma = 2e-5, lam = 1.01.
Suppose parameter values by yourself which are not given in pdf paper.
**DO NOT simplify anything. Do not make anything shorter. You have to read the entire pdf and its equations and relevant pubslished research article to achieve the target.***
 You can only have one right that if you want you can skip the last two terms of nondimensional asperity pressure Pa in equation 17 and only include elastic term of asperity. Other than that you are not allow to make any simplification.
***NUMERICAL PROCEDURE:***
The governing Eqs. (9), (11), (13), and (17) are discretized using the finite-difference method and solved simultaneously for pressure and film profile. The input dimensionless parameters are W, U, G, r, b, and V. For N nodes, the system consists of N equations and N unknowns. N-1 equations come from the Reynolds equation, Eq. (9), and one from the load balance, Eq. (11). The unknowns are H00, Kr, and the hydrodynamic pressure at nodes 2 to N-1 (the pressure is zero at the boundaries, i.e., nodes 1 and N).

***TESTING CRITERIA:***
once complete code generated then run the entire code and visualize all the graphs by yourself. Do not generate graphs for me. If these graphs profiles 100% match the pdf paper and meet the standard (and NO shrinkages, irregular spikes, pointed tops etc errors) and Converges smoothly then generate only complete ready to paste script for me. If not, then find the issues and fix them.
