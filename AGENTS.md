1) Read text file "MAIN_SCRIPT.txt" which python code that calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory.
And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction.
2) Your task is to convert entire code/script "MAIN_SCRIPT.txt" into MATLAB language/syntax and preserving all physics and results same. Make sure there must no any error and MALTAB must give the same results as the current code.

3) Also add a new section in script that must control the graphs plotting when i run code in MATLAB. 
It should,
I) plot reynolds pressure p vs x axis.
With following settings must be available in script so i can change manually like,
Cam angle = 1 [degree], 2, 3 .. (so that wether i want to plot profile for single cam angle or multiple cam angles in single graph)
RPM = 300, 500,......  (so that wether i want to plot profile for single RPM or multiple RPMs in single graph)
Surface State = 1, 0  [for textured = 1, for untextured = 0] (must be flexible so rhat wether i want to keep to plot only textured or both)

II) film thickness vs x axis. ( the same manual settings options must be available in script as mentioned for reynolds pressure).

III) Hydrodynamic friction Fh vs cam angles.. (same all settings must be available as mentioned above cases except "cam angle option).
IV) Asperity friction Fb vs cam angle.(same all settings must be available as mentioned above cases except "cam angle option).
V) Friction Torque T vs cam angle. (same all settings must be available as mentioned above cases except "cam angle option). 

4) All sections in MATLAB script must be clearly and precisly presented with necessary comments inside script. Make the plots more asthetic and professional

Deliverable: Once you sure your MATLAB code is error free then generate only complete script
