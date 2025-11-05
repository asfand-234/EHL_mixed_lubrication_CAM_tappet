Read and analyze every line of python code text file "python_seventeen.txt" which related to friction torque reduction in CAM and flat faced follower using 1D Mixed lubrication Line contact theory.
and cam lift data as "updated_lift.txt".
PROBLEM:
1) The current code generate reynolds pressure profile little bit condense, pressure is shifted to left side and not stable. And pressure data lying somewhere between x=-2 to x = 0. Which is incorrect. 
2) The film thickness vs x axis profile is not as per standard. There is no flat region at contact zone. 
TASK:
Deeply analyze the code especially the reynolds solver section and find the exact issue and fix it.  Analyze the film thickness at each cam angle and other input vriables of reynolds. Analyze every term of reynolds (RHS, D_core, D_full etc.). Analyze every scaling, variable, sign, clipping, loop and everything etc. etc. in reynolds solver to find the exact cause and fix it.
Deep studu from standard relevant literature and also from git.hub available sample codes.

TESTING /ACCEPTANCE CRITERIA:
Generate only updated complete script (changed lines + unchanged lines in single script) if and only if:

1) on non- dimensional X domain the contact zone is -1 to 1. So pressure must lie *ON* -1 to 1 not *within* it. And away from it must be zero as per standard Mixed lubrication/EHL line contact theory. Which means the pressure data must have zero values away from X =-1 and X= 1. (With a little mergen is accpetable upto X = - 1.2 to X = 1.2). And the pressure profile must be standard (belly shape or quite hertzian similar shape). In data it means the pressure values start increase from X = -1 towards center (X =0) and also pressure data increase from x= 1 towards center.
Check data/plot for differenr CAM angles randomly (untextured case). If it meets the target then Switch on surface texture state for like 10% texture area density and run for only cam angles -19° and -20° and check the pressure data if any pressure value does not leak outside the contact zone then it is acceptable. If not, then try to stablize it for onlu untextured scenario. 
NOTE: For untextuted scenario the pressure must start from Left boundary (X= -1.2) and ends ON right boundary (X = 1
2). But for textured case, the only acceptance criteria is the pressure must not leak out of the X domain it doesn't how it is behaving inside. 

2) As per standard theory, film thickness h profile remain almost constant within Contact zone and exponentially increasing when moving left contact boundary -a to  towards X_in. Due to elastic deflection and h0 values at each cam angle. And at center film thickness must be less than 0.2e6 m.  Deep analyze and find fhe exact cause and fix it.

When the criteria 1 and 2 is achieved and rjn the script and analyze the data for hydrodynamic friction vs cam angle and asperith friction vs cam angle so that,

3) Hydrodynamic friction must be minimum at cam nose region and peaks at both cam flanks and decrease away from both flanks
4) Asperity pressure must be maximum at cam nose region and decreasing away from nose (belly)
If any of above 4 criteria does not meet then keep improving and keep analyzing.

HARD STRICT RULES:
1) *DO NOT* change the formulas of kinematics section. 
2) *DO NOT* change the values of fixed parameters at TOP of the code section (Material/geometry/fluid)
