1) codex should read text file "MAIN_SCRIPT.txt" which calculates the percentage reduction of averaged friction torque with respect to untextured case. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory.
And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
But the current results are not as per target results.
The target results are given as,

 TARGET RESULTS:
RPM    5_percnt     8_percnt    10_percnt
300     3.4%         7.95%        3.4%
500    6.12%         8.92%        10.71%
700    4.21%         14.11%        9.4%
900    18.33%        11.91%        6.87%


2) Codex must optimize all physics present in script and/or add more real physics where necessary and optimize the texture model to meet at least 85% same target results. 

3) NEVER do any calibration/fitting/non-physical scaling. NEVER do any surrogation or regression, rather ONLY focus on all physics to get target results.
During testing, codex must print its predicted % averaged friction torque reduction and if the results are not at least 85% then codex start with improved strategies to meet the target.

4) Deliverables:
Only and only once the codex achieve target then generate a single error free complete executable Google Colab script for me which print predicted % averaged frictiom torque.
Do not print any other results to me other than final script.
