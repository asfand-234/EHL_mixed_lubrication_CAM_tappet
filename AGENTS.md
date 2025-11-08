Read Python code in text file "ehl_python.txt" which is related to 1D Mixed lubrication Line contact having Unsteady compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

There is no Time limit for you to complete below task. Deep dive until you achieve the targets.

PROBLEM:
Reynolds Pressure, asperity pressure and film thickness profiles and magnitudes are not correct.

TASK:
RUN the code for only untextured case. 
1) Deeply and attentively review every line of the code and analyze the graphs and data. 
2) the current Reynolds pressure magnitude is incorrectly too large. Improve the reynolds solver so that The reynolds (hydrodynamic) pressure must be close to dome-like or hertizan-like shape. Which means the data on a non-dimensional X domain must be almost symmetric at center X = 0. So the profile must start near X = -1 goes on symmetrically upto X = 1. Which is called contact zone. And away from it pressure must be zero. But pure reynolds physics based not unrealistic clipping.
Also before exit (X<1), there must be a slightly cavitation region where a slighyly pressure must drop and then rise. 
See the "reference_profile.PNG" it must be like that.

3) Currently the film thickness magnitude is incorrectly greater. As it is mixed lubrication so its central flat film thickness must be less than 0.1e6 m. The film thickness must be flat/constant along contact zone (central) and exponentially increasing from -1 to left most inlet. 
4) The asperity pressure must be maximum during CAM nose region (around -35° to 35°) and decreasing away from it. So its profile must also close to belly shape and must lie on contact zone only and away from it must be zero without non-physical forcing or clipping. It must be physics based.
5)focus on each physics and modify if any inconsistency or non-physical condition exist.

HARD STRICT RULE:
1) *DO NOT* simplify anything rather you have to optimize physics based on literature.
1) *DO NOT* add any window clipping based or non-physical conditions to restrict pressure on contact zone rather you must have to improvethe physics to remain all pressures on contact zone not shrink/thin and not leaking outside contact zone.
3) *DO NOT* change formulas in CAM "KINEMATICS".

TESTING AND ACCEPTANCE:
keep running code, keep analyzing graphs and data by yourself, keep studying from relevant literature and keep optimizing entire simulation until all profiles meet the standard. Then generate only ready to paste google colab complete script without missing any line. Do not generate results for me.
