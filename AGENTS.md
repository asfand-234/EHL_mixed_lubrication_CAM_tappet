Read and deeply analyze the entire python code in text file "test6666.txt". It is complete study of 1D thermal Transient mixed lubrication Line contact. 
The current code is for a single load and velocity case. 
Task:
Your task is to make the entire study for full cam and flat faced follower cycle. 

1) Now entraining velocity um, sliding velocity Vs, radius of curvature R, and contact load/target load F will be calculated for each cam angle point using kinematics and dynamic equations.

i) Import the cam lift data in text file "updated_lift.text". The cam event range is -84° to 80° cam angles (theta). And lift is in meters. It has discretized into 329 points.
Then calculate the first and second derivative of cam lift with respect to theta.
Then using following relations:
velocity of contact point with respect to cam in x direction
Vc = (rb + Lift + d^2Lift/dtheta^2)*omega
where rb = base circle radius = 18.4e-3 m
omega = amgular velocity = (2*pi*N)/60
N = rpm = 300

Velocity of contact point with respect to the follower in x direction
Vf = omega* d^2Lift/dtheta^2
Entraining velocity
um = (Vf + Vc)/2
Sliding velocity = Vs = Vc - Vf

R = d^2Lift/dtheta^2 + Lift + rb

Contact load 

F = K_spring*(Lift + delta) + (d^2Lift/dtheta^2)*M_eq*omega^2 
K_spring= 7130  N/m
delta = initial spring compression = 1.77e-3 m
M_eq = 0.05733 kg
Hertzian contact length (cam geometrical width) = 7.2e-3 m

Keep all formulas exactly same!.

2) Then link this dynamic data with entire study. The entire study must be time dependent [0 time_step time_end]. time_step and time_end can be designed on the basis of cam cycle (-84° to 80°), squeez term and rpm etc. So that entire simulation must be robust and perfecly aligned with standard literature. 
3) Remove scalar values of R, F, um, Vs so that every physics now must take corresposding value from these kinematic and dynamic calculated data. 
4) Remove all current plot and print commands. Remove all other necessary comments and ambuigity if you observe. And  run the simulation of Transient, thermal mixed lubrication and add plot commands for:
Plot um, Vs, R, F vs cam angles
Non-dimensional reynolds pressure vs X axis (for entire cam cycle in single graph)
Non-dimensional asperity pressure vs X axis (for entire cam cycle in single graph)
Non-dimensional film thickness vs X axis (for entire cam cycle in single graph)

Make sure everything must be perfectly coupled and stable. Deeply study from research articles from web sources.
Optimize/change the numerical solution scheme. Make sure time stepping smoothly working. Make sure sign changes of entraining velocity is correctly linked with flow direction as per standard theoy. And make sure squeez term and load balacing is smoothly dealing with both regions of cam: cam flanks angles (low load, high velocity) and cam nose angles (high load and low velocity).

TESTING CRITERIA:
Run the entire simulation and keep eye on load error, residul error and runtime.
Load error < 1%
Residual error <1e-7
Total cycle runtime <150 seconds
Pressure and film thickness profiles must have correct shape and lying on its contact zone as per standard theory.
If any cam angle steo fail in any above criteria and start finding root cause and keep fixing and keep re-running and observing progress. Do not stop untill all cam angles pass all criteria untill then leep runing entire code.

HARD STRICT RULES:
1) Never change kinematics formula
2) Never change input parameter values.
3) Never implement non-physical and artificial clamps, clippings, scaling and constraints, except for numerical stability.
