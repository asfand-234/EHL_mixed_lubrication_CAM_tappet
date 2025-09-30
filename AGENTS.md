1) Read text file "MAIN_SCRIPT.txt" which is python code that calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory. And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction. Only 1 texture parameter a_tex (amplitude) varies for each RPM in each texture area density. Itbhas data set in repository.

2) Your task as an expert COMSOL API agent, generate a complete COMSOL 6.1 API script of ''.java'', which must be error free executable in "Method" in COMSOL.
Read every section in the code make understanding about all physics and convert into comsol .java  syntax. 
For your help there is a sample in respository named "sample_comsol_API". Do not pick any value or data from it. It is just for your reference so that you must generate script as per compatible with COMSOL 6.1. 

3) Make sure, the implementation of 1D  Reynolds equation must be accurate, stable and must converge. 

4) Remember: COMSOL does not support any descrete array. So your script must be as per comsol available settings and syntax. Include a_tex data set and cam lift in interpolation tables in comsol.  

5) make sure to implement accaurate kinematics/dynamics and moxed lubrication relations and time steps as per code to get exact same results.
6) At the end run comsol java script and verify whether it is giving correct results are not. if not then fix it.
