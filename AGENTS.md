Read Python code in text file "ehl_python.txt" which is related to average friction torque reduction due to surface textures using 1D Mixed lubrication Line contact having compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

PROBLEM:
The current code setup generating wrong asperity friction vs cam angle profile.
The current asperity friction is lower around cam nose region (about -30° to 30° cam angles) as compare to cam flanks. It may be due to low asperity pressure at cam nose angles then cam flanks. Which is may be due to greater film thickness at nose than flanks. And which may be due to h0 or any other variable. 

TASK & IMPLEMENTATION STRATEGY:
1) Run the code and print data of only asperity friction vs cam angle at 300 rpm and untextured case. See aound cam nose region the friction is lower at flanks peaks. Which is incorrect.
2) Print maximum asperiry pressure values for different regions of cam and film thickness. And analyze and find the exact true causes. Fix them and re-run and print asperity friction vs cam angle.
3) If the asperity friction is maximum at nose region and decreasing away from it (belly shape) then accepted else keep analyze the entire code and keep optimizing.

TESTING AND ACCEPTANCE CRITERIA:
Keep running the code and printing results and finding the exact cause. Once got the target results then provide only complete updated final script ready to paste to me.

HARD STRICT RULES:
 1) Do not change "KINEMATICS" equations in the code.
2) Do not change fix geometry/material parameters values.
3) Do not oversimplify any physics.
4) Do not add any non-logical, non-physical clipping/constraint/conditions.
