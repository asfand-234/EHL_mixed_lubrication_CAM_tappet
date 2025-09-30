1) Read text file "MAIN_SCRIPT.txt" which calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory.
And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
Only 1 parameter in texture model is not fixed i.e. a_tex (amplitude). The code takes 3 files as input ""amplitude_5_percent.txt", "amplitude_8_percent.txt", and "amplitude_10_percent.txt". 
But the current results are not as per target results.
The target results are given as,
 TARGET RESULTS:
RPM    5_percnt     8_percnt    10_percnt
300     3.4%           7.95%         3.4%
500    6.12%           8.92%         10.71%
700    4.21%           14.11%        9.4%
900    18.33%         11.91%        6.87%

2) Your task is only tune the data set of each amplitude file in each cell of each column to meet at least 85% same target results. 
3) IMPLEMENTATION STRATEGY:
Start with first case, like in amplitude 5 percent in 300 column, now tune every random values in different cells and run the entire simulation and focus only on results of % reduction of averaged friction torque for 5% at 300 RPM. Keep tuning values in entire column and understand the behaviour once you achieved at least 85% same targer results. Then move to next column like amplitude 5 percent at 500 RPM. And so on keep going one by one until you tuned every data and all the results meet at least 85% same a target results 

**Remember**: If your session time out after working long or any interuption occurs then must generate the last updated/tuned three files of amplitude without caring about wether all results meet the targer or not"
4) Deliverables:
Only and only once the codex achieve target then generate three updated amplutude data files. 
Do not print any other results to me other than final 3 ampliude files.
5) HARD STRICT RULES:
i) You must *DO NOT* put any calibration/fitting/non-physical scaling . *DO NOT* perform any surrogation or regression, rather it must only tune each value in amplitude data set to meet the target.
ii) Do not make any single change in any physics in entire code.
iii) You must have to tune every single value in each column of all three amplitude files and run the entire simulation every time.
