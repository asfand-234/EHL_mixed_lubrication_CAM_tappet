1) Read text file "MAIN_SCRIPT.txt" which calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory. And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
Only 1 parameter in texture model is not fixed i.e. a_tex (amplitude). The code takes 3 files as input ""amplitude_5_percent.txt", "amplitude_8_percent.txt", and "amplitude_10_percent.txt". 
But the current results are not as per target results.
The target results are given as,
 TARGET RESULTS:
RPM    5_percnt        
300     3.4%                
500    6.12%            
700    4.21%         
900    18.33%      
2) Your task is only tune the data set of ""amplitude_8_percent.txt" file in each cell of each column to meet at least 85% same target results for only 8%. Do not tune other file data just focus on 8% results for all RPM.
3) IMPLEMENTATION STRATEGY:
Start with first case, like in amplitude 8 percent in 300 column, now tune every random values in different cells  and run the entire simulation and focus only on results of % reduction of averaged friction torque for 8% at 300 RPM. Keep tuning  every value in entire column perform different treatments and understand the behaviour once you achieved at least 85% same target results. Then move to next column like amplitude 8 percent at 500 RPM. And so on keep going one by one until you tuned every data and all the results meet at least 85% same a target results 
4) Deliverables:
Only and only once the codex achieve target then generate the updated amplutude data file of 8% only. 
Do not print any other results to me other than final 8% ampliude file.
5) HARD STRICT RULES:
i) *DO NOT* put any calibration/fitting/non-physical scaling in script, rather it must only tuning each value in amplitude data set to meet the target.
ii) *DO NOT* make any single change or do not add any factor in any physics in entire code.
iii) Yo have to tune every single value in each column and row where necessary of ONLY 8% amplitude file and run the entire simulation every time and compare only with 8% target results.
