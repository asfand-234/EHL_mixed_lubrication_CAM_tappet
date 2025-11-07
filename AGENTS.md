Read Python code in text file "ehl_python.txt" which is related to 1D Mixed lubrication Line contact having Unsteady compressible reynolds equation solution. 

PROBLEM:
Reynolds Pressure, asperity pressure and film thickness profiles are not correct.

TASK:

1) Deeply and attentively review every line of the code and run it and analyze the graphs and data. 
2) the load ratio must remain greater than 0.95
2) Improve the reynolds solver so that The reynolds (hydrodynamic) pressure must be dome-like or hertizan-like shape. Which means the data on a non-dimensional x domain must be almost symmetric at center X = 0. So the profile must start near X = -1 goes on symmetricallt X = 1. Which is called contact zone. And away from it pressure must be zero. But pure reynolds physics based not false.
Also before exist there must be a slight cavitation region where a slighly pressure must drop and then rise. 
See the "reference_profile.PNG" it must be like that.

3) Currently  the film thickness magnitude is greater which makes asperity pressure zero which is incorrect. as it is mixed lubrication so its central flat film thickness must be less than 0.1e6 m. The film thickness must be flat/constant along contact zone (central) and exponentially increasing from -1 to left inlet. 
4) The asperity pressure profile must also close to belly shape and must lie on contact zone only and away from it must be zero without non-physical forcing or clipping. It must be physics based.
5)focus on each physics and modify if any inconsistency or non-physical condition exist.
6) For help and reference you can read a sample MATLAB code in text files named:
EHL_00_run_Study_A2_gen, EHL_00_run_Study_A3_gen, EHL_00_run_Study_A_gen, EHL_01_setup_Study_A2_gen, EHL_01_setup_Study_A3_gen, EHL_01_setup_Study_A_gen, EHL_02_mainprocess
If you need any physics, you can exact idea from it.
HARD STRICT RULE:
1) Do not add any window clipping based or non-physical conditions to restrict pressure on contact zone rather you must have to optimize the physics to remain all pressures on contact zone not shrink/thin and not leakong outside contact zone.

TESTING AND ACCEPTANCE:
keep running code, keep analyze graphs and data by yourself, keep studying from relevant literature and keep optimizing entire simulation until all profiles meet the standard. Then generate only ready to paste google colab complete script without missing any line. Do not generate results for me.
