Read Python code in text file "ehl_python.txt" which is related to 1D Mixed lubrication Line contact having Unsteady compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

PROBLEM:
Reynolds Pressure, asperity pressure and film thickness profiles and magnitudes are not correct.

TASK:
Work for only untextured case. 
1) Deeply and attentively review every line of the code and run it and analyze the graphs and data. 
2) the current Reynolds pressure magnitude is incorrectly too large. Improve the reynolds solver so that The reynolds (hydrodynamic) pressure must be close to dome-like or hertizan-like shape. Which means the data on a non-dimensional x domain must be almost symmetric at center X = 0. So the profile must start near X = -1 goes on symmetricallt X = 1. Which is called contact zone. And away from it pressure must be zero. But pure reynolds physics based not false.
Also before exist there must be a slight cavitation region where a slighly pressure must drop and then rise. 
See the "reference_profile.PNG" it must be like that.

3) Currently  the film thickness magnitude is incorrectly greater. as it is mixed lubrication so its central flat film thickness must be less than 0.1e6 m. The film thickness must be flat/constant along contact zone (central) and exponentially increasing from -1 to left inlet. 
4) The asperity pressure must be maximum during CAM nose region and decreasing away from it. So its profile must also close to belly shape and must lie on contact zone only and away from it must be zero without non-physical forcing or clipping. It must be physics based.
5)focus on each physics and modify if any inconsistency or non-physical condition exist.

HARD STRICT RULE:
1) Do not add any window clipping based or non-physical conditions to restrict pressure on contact zone rather you must have to optimize the physics to remain all pressures on contact zone not shrink/thin and not leaking outside contact zone.
2) Do not change formulas in CAM "KINEMATICS".

TESTING AND ACCEPTANCE:
keep running code, keep analyze graphs and data by yourself, keep studying from relevant literature and keep optimizing entire simulation until all profiles meet the standard. Then generate only ready to paste google colab complete script without missing any line. Do not generate results for me.
