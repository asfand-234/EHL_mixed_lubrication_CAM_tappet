Read text file "python_eight.txt" code which calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory. And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
Only 1 parameter in texture model is not fixed i.e. a_tex (amplitude). The code takes 3 files as input ""a_texture_data_5pct.txt", "a_texture_data_8pct.txt", and "a_texture_data_10pct.txt". 

*PROBLEM:*
The current results for 10% texture for all RPMs are not as per target results.

*TARGET RESULTS:*
The target results are given as,
RPM      % Avg. friction Tq reduction         
300      3.4%                
500      10.71%            
700       9.4%         
900      6.87%   

*TASK:*
Your task is only tune/calibrate the data set of ""a_texture_data_10pct.txt" file in non-zero cells only of each column of RPM to meet at least 85% same target results for only 10%. Do not tune other file data just focus on 10% results for all RPMs.

*IMPLEMENTATION STRATEGY:*
Start with first case, like in amplitude 10 percent in 300 column, now tune every random non-zeros cell values in different and run the entire simulation and focus only on results of % reduction of averaged friction torque for 10% at 300 RPM. Keep tuning every value in entire column perform different treatments and understand the behaviour once you achieved at least 85% same target results. Then move to next column like amplitude 10 percent at 500 RPM. And so on keep going one by one until you tuned every data and all the results meet at least 85% same a target results 

*HARD STRICT RULES:*
i) DO NOT make changes in zero cells, only calibrate values of non-zeros cells. And STRICTLY the calibration only allowed greater then 0 and less than 7e-6.
ii) DO NOT put any calibration/fitting/non-physical scaling in script, rather must only tune/calibrate non-zero cells values in each column to meet the target.
iii) DO NOT make any single change or do not add any factor in any physics in entire code.
iv) Run the entire simulation every time and compare your results with 10% target results.

*DELIVERABLES:*
Only and only once the codex achieve target then generate the updated a_texture data file of 10% only. 
Do not print any other results to me other than final 10% updated a_texture file.
