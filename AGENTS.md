Read and analyze every line of python code text file "python_seventeen.txt" which related to friction torque reduction in CAM and flat faced follower using 1D Mixed lubrication Line contact theory.
and cam lift data as "updated_lift.txt".

PROBLEM:
1) The current code generate incorrect profile of asperity friction Fb vs cam angle. Current code is generating minimum friction during cam nose region (around cam angle -35° to 35°) which is incorrect.

TASK:
Deeply analyze the code especially the reynolds solver section and find the exact issue and fix it.  Analyze the film thickness at each cam angle how lamda is changing and effecting asperity pressure. Analyze every scaling, variable, sign, clipping, loop and everything etc. etc. in reynolds solver to find the exact cause and fix it.

TESTING /ACCEPTANCE CRITERIA:
Generate only updated complete script (changed lines + unchanged lines in single script) if and only if:

1) Hydrodynamic friction is minimum at cam nose region and peaks at both cam flanks and decrease away from both flanks
2) Asperity pressure maximum at cam nose region and decreasing away frkm nose (belly)
3)  The reynolds pressure smoothly converged and lying on its contact domain.

If not Keep improving and keep analyzing by yourself 

HARD STRICT RULES:
1) *DO NOT* change the formulas of kinematics section. 
2) *DO NOT* change the values of fixed parameters at TOP of the code section (Material/geometry/fluid)
