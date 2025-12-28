Read and deeply analyze the python code in text file "test6666.txt". It is study of 1D thermal, transient Mixed LubricationLine contact for cam and flat faced follower.

Problem:
The Reynolds pressure and asperity pressure profiles from current code are not correct as per standard profile. Also film thicknes profile showing very worsr and unrealistic behaviour. Also load error and residual error is greater. Also total runtime of cycle is much larger.

Task:
1) You need to deep dive into entire study And make all necessary corrections to make the entire study robust, but accurate.

Deeply analyze every physics, discretization, loop, initialization, coupling, convergence, time stepping, jacobian, velocity reversal, sign changes during cycle, regime change of lubricant from flanks to nos, cavitation  etc. etc. And everything and make necessary fixes as per standard literature and open source sample codes available on git.hub.com.
2) Keep optimizing all physics and numerical scheme and keep running and monitoring results by yourself. Replace or implement any algorithem or scheme which is giving perfect results. You have open choice to implement any model, technique, algorithm, scheme as per standard literature. But entire study must remain 1D Thermal, Transient, Mixed Lubrication Line contact, Camand flat faced follower. 


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
