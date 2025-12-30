Read and deeply analyze every model, physics, layout etc in my python code in text file "test6666.txt". It is related to 1D thermal, transient Mixed Lubrication Line contact in cam & tappet mechanism.

HARD STRICT RULES:
To complete below tasks you must remember:
1) Do not implement any non-physical or artificial factor, scaling, clamp, clipping, constraints etc anywhere in the code. Every correction and optimization must be pure physics base as per standard EHL literature prove. 
2)Do not change kinematics formulas
3) Do not change material and roughness input parameter values 

TASKS:
a) Add print commands and Run the code for entire cam cycle i.e. 329 cam lift data points [at 300 rpm, 90°C and untextured case only] and live print below 8 criteria and keep monitoring.
b) If any cam angle does not pass any below 8 criteria than start optimizing the entire code.
   CRITERIA:
1) load error < 1%
2) Residual <1e-6
3) Max. Non-dimensional Reynolds + Max. Non-dimensional asperity pressure = 1 (with 10% error margin).
4) both max. Pressures lie at X = 0
5) Full cycle runtime < 170 seconds
6) Cavitation spike near exit (must as per flow direction theory)
7) flat region on film thickness profile on contact zone X [-1 1] as per theory.
8) Making sure the reynolds pressure profile for any cam angle is not a false/fake hertzian profile rather it must be real hydrodynamic pressure naturally calculated through coupled equations.

c) Deeply and more critical analyze every line of the code. Especially more focus on Reynolds equation discritization, initiation, scaling, units, coupling. Then perform some side diagnostic calculations if needed and fix all inconsistencies. Then more attentively analyze the current numerical solution scheme and convergence method. Optimize it to highest possible level if it doesn't work then change the scheme as per standard literature.
d) Optimize time stepping, squeez film, load balacing without implement any artificial scaling or factor or clamps.
For every cam angles load balance and all coupled equations must satisfy naturally. But keep in mind h0 value from Downson formula [But do not implement] and everytime figure out why load balance is not producing this value naturally. And so on.

f) Once all above 8 criteria acheived then generate single python code file ready to paste for google colab which import cam lift data file and print steps, load error, residual and three seperate plots of nondimensional reynolds pressure, asperity pressure and film thickness vs X for entire cam cycle.
