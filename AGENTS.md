Read and deeply analyze the python code in text file "test6666.txt". It is study of 1D thermal, transient Mixed LubricationLine contact for cam and flat faced follower.

Problem:
Reynolds pressure is spreading away from its contact zone X [-1 1] and shifted towards left inlet for almost all cam angles. And cavitation spike is missing near exit.
May be the boundary conditions, normalization, initialization, squeez term or h0 iteration is not perfect. You need to deeply analyze, perform some advance diagnostic checks calculations and do research from standard literature to find the root causes and keep fixing. Also, flat region on film thickness profile is also missing for many cam angles.

Task:
Your task is to optimize the every model in the code and its mumerical scheme to meet below pasted criteria.
(Just for a reference i have attached a sample graph in attached figure from a research article, you can analyze how perfectly shaped reynolds profile (blue) and asperity pressure profile (red) and their sum total pressure (black)).

Load balance iterator and total pressure iteraror must so advance and physically correct to deal with state changes in cam cycle: flanks angles (low load, high velocity and radius) and nose angles (high load, low velocity and radius).
Revsite every physics and calculate its values if needed and analyze if it is behaving correct or not. 
Make sure the cavitation point is correctly determined and cavitation algorithem is perfectly working. 
Deeply study from standard literature and sample codes and optimize the non-linearity density/viscosity coupling.
The solver must behave more flexible for transition effect through purely physics base correction. Read MATLAB sample codes available on git.hub.com related to EHL (but my study must remain 1D Transient, thermal, mixed lubrication Line contact).

Also keep optimizing enture study untill all cam angles meet all criteria. Add more necessary checks for printing live and keep eye on it.

Target criteria:

1) Load error <1%
2) residual <1e-6
3) Total cycle runtime <160 seconds [once all correction achieved then do advanced optimization on convergence, initial guesses, time stepping etc and achieve this target.)
4) No negative pressure and negative film thickness.
5) both non-dimensional pressures must have correct magnitude for each cam angle i.e. sum of max. Non dimesnsional reynolds pressure + max. Non dimensional asperity pressure = 1 (with <10% error) 
6) Both reynolds pressure and asperity pressure must lie on its contact zone having correct shape as per standard theory (hertzian like shape having cavitation spike near exit) but without non-physical or artificial clamps, clipping and scaling rather it must be pure physics base.

HARD STRICT RULES:
1) Do not change Kinematics formulas
2) Do not change input fixed geometry,  material and roughness parameters except grid size, iterations and domain and any physical scalat factor value verified from literature.


Keep doing deep research and keep optimizing untill entire study meet all 6 criteria
