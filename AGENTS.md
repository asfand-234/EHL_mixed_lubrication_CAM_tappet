Read Python code in text file "ehl_python.txt" which is related to average friction torque reduction due to surface textures using 1D Mixed lubrication Line contact having compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

PROBLEM: The current code is generating very low (<1%) percentage of average friction torque reduction wuth respect to untextured for all rpms and all texture area densities.

TASK:
1) Run the entire code for 300 rpm and at 5% texture density and analyze the % averaged friction torque reduction.
2) Deeply analyze the entire code. Find the exact bugs/inconsistencies due to which the percentage of friction torque reduction is too low. It must be greater than 5% as per experimentally.

3) Fix the issues and re-run the code and verify it it is solved. If not, re optimize the code and re run and re print the % average friction torque reduction


Testing and acceptance criteria:
Run the entire code each time. Once % of reduction is equal or greater than 5% then generate only updated complete script resdy to paste for me.

HARD STRICT RULES:
1) Do not change "KINEMATICS"  equations in the code.
2) Do not change parameters values of texture model.
3) Do not change fixed geometry/material values.
4) Do not oversimplify any physics.
