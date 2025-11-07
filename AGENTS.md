Read Python code in text file "ehl_python.txt" which is related to 1D Mixed lubrication Line contact having Unsteady compressible reynolds equation solution. 

PROBLEM:
Reynolds Pressure, asperity pressure and film thickness profiles are not correct.

TASK:
1) Deeply and attentively review every line of the code and run it and analyze the graphs and data. 
2) Modify the code so that The reynolds (hydrodynamic) pressure must be dome-like or hertizan-like shape. Which means the data on a non-dimensional x domain must be almost symmetric at center X = 0. So the profile must start near X = -1 goes on symmetricallt X = 1. Which is called contact zone. Also before exist thete must be a slight cabitation region where a slighly pressure must drop and then rise. 
3) The film thickness must be flat/constant along contact zone (central) and exponentially increasing from -1 to left inlet. And as it is mixed lubrication so its central flat film thickness must be less than 0.1e6 m. 
4) The asperity pressure profile must also close to belly shape and must lie on contact zone only.
5)focus on each physics and modify if any inconsistency or non-physical condition exist.

TESTING AND ACCEPTANCE:
keep running code, keep analyze graphs and data by yourself, keep studying from relevant literature and keep optimizing entire simulation until all profiles meet the standard. Then generate only ready to paste google colab complete script without missing any line. Do not generate results for me.
