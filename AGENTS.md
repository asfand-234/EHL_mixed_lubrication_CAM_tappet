Read python code in text file, "ehl_python.txt" which is related to 1D mixed lubrication line contact solution.

PROBLEM:
The current code is very simple and generate inccorect reynolds pressure and film thickness profiles.

TASKS:

Your Task is to:
1) Implement the complete and professional discretization and solution of 1D transient Reynolds equation using complete mixed lubrication line contact load balacing. 

2) Your Python code must be only for 1D Mixed lubrication (hydrodynamic + asperity loads) and unsteady.
It must solve complete standard rheological models of viscosity and density same as the current code is doing.
The reynolds equation must be compressible and mass conserving.
It must solve the same complete cavitation algorithm (Elord) and stabilization. 
It must contain complete standard elastic deformation alogrithm.
It must use proper hertzian contact theory (hertzian width and pressure).
For asperity pressure/load use standard Greenwood statistical model. Do not simplify anything.

3) It must plot on 3 graphs in 1D separately. 
reynolds (hydrodynamic) pressure vs x axis.
Film thickness vs x axis.
Asperity pressure vs x axis 

TESTING AND ACCEPTANCE CRITERIA:
Once the above all 3 points achieved then run the entire code and analyze the pressure and film thickness data. 

1) If the pressure profile is dome-like/hertzian-like and lying in contact zone (-a to a) and away from it must be zero.

2) and film thickness has flat/constant region along contact zone and data 100% meet the standard theory 
3) and script is smoothly converging then provide only "Ready to paste google colab complete script".
No need to provide results data.

HARD STRICT RULES:
1) DO NOT simplify anything. 
2) You must go through the relevant literature when needed to make the code perfect.
