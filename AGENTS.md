Read and deeply analyze the entire code in text file "test6666.txt" in main branch . It is 1D thermal Transient mixed lubrication Line contact with mass conserving reynolds equation in cam and tappet mechanism.

TESTING CRITERIA:
1)  Run the code and keep eye on load error, residual error especially after steps 118/329 onward. Try to optimize the convergence scheme so that the solver either never "retry: resetting to Hertzian guess" or only retry for 4 or 5 steps (currently more than 27 steps).
2) The non-dimensional asperity pressure mgnitude is small. Fix it without non-physical scaling. See the criteria 6.
3) Keep deep analyzing the code, keep optimizing numerical scheme and  convergence and keep running and monitoring results untill all below criteria are met.

TARGET CRITERIA:
1) Load error <1%
2) Residual <1e-7
3) Sum of max. Non. Dimensionl reynolds pressure and max. Non. Dimensional asperity pressure = 1 (10% error acceptable).
4) Total cycle runtime <160 seconds
5) No negative pressures and film thickness
6) Bother pressures are accurately lie on its contact zone (no shrinkage, spreading) having cavitation spike near exit.

Keep optimizing until all 6 criteria meet. 
You load balance iterator must be capable of dealing with lubricant regime change during cam cycle but pure physics base. 

Hard stricf Rules:
1) Do not change kinematics formula.
2) Do not change input  material and roughness parameter values.
3) Do not implement any non-physical and artificial scaling, factor, clamps, clipping and constraints except for numerical stability.
4) Try to keep grud N > 80 for better accuracy.
