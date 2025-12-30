Read and deeply analyze every model, physics, layout etc in my python code in text file "test6666.txt". It is related to 1D thermal, transient Mixed Lubrication Line contact in cam & tappet mechanism.

PROBLEM:
The reynolds and asperity pressure profiles are not correctly shaped and not lying on its contact zone from current code. Especially for cam flank angles the reynolds pressure is shifted towards left.  And runtime of cycle and load error is very greater.

HARD STRICT RULES:
To complete below tasks you must remember:
1) Do not implement any non-physical or artificial factor, scaling, clamp, clipping, constraints etc anywhere in the code. Every correction and optimization must be pure physics base as per standard EHL literature prove. 
2)Do not change kinematics formulas
3) Do not change material and roughness input parameter values 

TASKS:
1) Add print commands and Run the code for entire cam cycle [at 300 rpm, 90°C and untextured case only] and live print below 7 criteria and keep monitoring.
2) If any cam angle does not pass any below 7 criteria than start optimizing the entire code. 
3) Deeply and more critical analyze every line of the code. Especially more focus on Reynolds equation discritization, initiation, scaling, units, coupling. Then perform some side diagnostic calculations if needed ad fix all inconsistencies. Fhen mofe attentively analyze the current numerical simulation scheme and convergence criteria. Optimize it to highest possible level if it doesn't work then change the scheme as per standard literature.
4) Optimize time stepping, squeez film, load balacing without implement any artificial scaling or factor or clamps.
For every cam angles load balance and all coupled equations must satisfy naturally. But keep in mind h0 value from Downson formula [But do not implement] and everytime figure out why load balance is not producing this value naturally. And so on. Do not implement any artificial forcefull constraints rather deeply do research from standard literature and find techniques.
5) Do not confuse yourself with any comment mentioned in code. These may be wrong. Focus on physics and deeply do research from standard web sources. 
6) When you feel hard to meet the all 7 criteria. Then pick 2 angles from nose region and 2 angles from flank peak and optimize the entire study so that the code successfully pass all criteria and all profiles are as per standard theory. Then put this updated physics for full cycle and monitor results until all cam angles pass all criteria. 

CRITERIA:
1) load error < 1%
2) Residual <1e-6
3) Max. Non-dimensional Reynolds + Max. Non-dimensional asperity pressure = 1 (with 10% error margin).
4) both max. Pressure lie at X = 0
5) Full cycle runtime < 170 seconds
6) Cavitation spike near exit 
7) flat region on film thickness profile along contact zone X [-1 1].
