*Read* text file "python_eight.txt" code which calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory. And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
Only 1 parameter in texture model is not fixed i.e. a_tex (amplitude). The code takes 3 files as input ""a_texture_data_5pct.txt", "a_texture_data_8pct.txt", and "a_texture_data_10pct.txt". 

**PROBLEM:**
The current results for 5% texture at only 900 RPM case is not as per target result.

**TARGET RESULTS:**
The target results are given as,
RPM      % Avg. friction Tq reduction         
900         18.33%      

**TASK:**
Your task is only tune/calibrate the data set of 900 RPM only in""a_texture_data_5pct.txt" file in non-zero cells only to meet at least 80% same target results for only 5%. Do not tune other file data just focus on 5% results for 900 RPM.

**IMPLEMENTATION STRATEGY:**
Start tune every random non-zeros cell values in RPM column and run the entire simulation and focus only on results of % reduction of averaged friction torque for 5% at 900 RPM. Keep tuning every value in entire column perform different treatments and understand the behaviour until you achieved at least 80% same target results.

**HARD STRICT RULES:**
i) DO NOT make changes in zero cells, only calibrate values of non-zeros cells. And *STRICTLY* the calibration only allowed greater then 0 and less than 7e-6.
ii) DO NOT put any calibration/fitting/non-physical scaling in script, rather must only tune/calibrate non-zero cells values in each column to meet the target.
iii) DO NOT make any single change or do not add any factor in any physics in entire code.
iv) Run the entire simulation every time and compare your results with 10% target results.

**DELIVERABLES:**
Only and only once the codex achieve target then generate the updated a_texture data file of 5% only. 
Do not print any other results to me other than final 5% updated a_texture file.
