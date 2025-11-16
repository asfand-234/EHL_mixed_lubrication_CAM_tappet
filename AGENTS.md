Read Python code in text file "ehl_python.txt" which is related to average friction torque reduction due to surface textures using 1D Mixed lubrication Line contact having compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

PROBLEM:
The current code is generating very unrealistic large % averaged friction torque reduction for RPMs 500, 700 and 900 for all texture area densities.

TASK
1) Run the entire code and print % averaged friction torque reduction at 500 RPM for texture area densities 5%, 8% and 10%. And analyze the results. 
2) If the % average friction torque reduction is unrealistic large figure then deep analyze and find the bugs/causes and fix them all. And again run the entire code and print the results.
3) If results becomes realistic reduction of friction to reasonabme percentage then print results for 700 and 900 and do the same procedure. If not, then keep analyze the physics and keep optimizing.
4) Once all cases improved then provide only ready to paste complete script to me. Do not provide results.
ACCEPTANCE CRITERIA:
1) % reduction should be positive not negative. Which indicates decrease in friction due to textures. 
2) % reduction must be reasonable i.e less then 15% 

HARD STRICT RULES:
1) Do not change "KINEMATICS" formulas
2) Do not change fixed texture parameters values.
3) Do not change fixed geometry/material values. Rather focus on physics and inconsistencies.
