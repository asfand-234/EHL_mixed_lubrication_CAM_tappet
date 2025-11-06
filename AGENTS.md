
Read python code in text file "python_slippy.txt" which is related to complete mixed lubrication/EHL solution of reynolds equation.
TASK:
Your task is to:
1) Run the entire script and find the errors and fix them all. 
2) Once the code converged successfully then analyze the entire code deeply line by line.
3) Make it perfect "1D mixed lubrication (hydrodynamic + asprity (roughness) loads ) Line contact ". 
It must solve complete standard rheological models of viscosity and density. 
The reynolds equation must be Transient and compressible and mass conserving.
It must be contain Elord cavitation and stabilization.
The asperity pressure must be calculated from Greenwood statistical model with following roughness values.
DATA:
Poison ratio for both = 0.3
Dynamic Viscosity at 90°C = 0.01381 Pa.s
Density at 90°C = 858.44 kg/m^3
Pressure viscosity coefficient at 90°C= 15e-9 Pa^-1
Boundary shear strength coefficient = 0.11
Combined RMS roughness = 0.2 micrometers
Contact length = 7.2e-3 m

Roughness parameters:
combined RMS = sigma = 0.2e-6 m
Average asperity tip radius = beta_a = 2e-4 average summit radius =  k = 2e-4 
Asperity distribution per unit area = eta_R = 1.25e9
4) Make sure the script must plot only non-dimensional reynolds pressure vs X axis, and Non-dimensional film thickness vs X axis seperately in two graphs. No other reference plot should be there.

5) once script organized and converged then analyze both the profiles and their data. Their profile must meet the standard mixed lubrication/EHL Line contact reynolds (dome like or hertzian shape on X = -1 to X = 1 and away from it pressure must be zero. And pressure peak should be P = 1). Make sure reynolds pressure must not be shrink/shifted towards one side rather it must be stable with true hydrodynamic behaviour.
   1) Do not create multiple .py files/script. Your final script must be a single complete ready to paste google colab code.
6) Do not simplify everything. Make it perfect.
7) You can also take help from other codes open available on www.git.hub.com for references.

 and film thickness profile must be as per standard literature (flat constant on contact zone from X = -1 to X = 1 and increasing exponentially towards inlet domin boundary).
8) If solution smoothly converge and both profiles are 100% correct then remove unnecessary code lines, comments and any extra lines which are not the part of solution from raw code.

9) Then finally generate complete ready to paste google colab script. Do not show me changes/diff. Just provide me complete final updated code without hidden (@@) lines.
