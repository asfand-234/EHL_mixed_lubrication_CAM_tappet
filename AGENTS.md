Read the python code "python_twelve" related to friction torque calculation with surface texturing using 1D mixed lubrication line contact theory in CAM and flat faced follower mechanism.
TASK:
Your task is to achieve following changes in my code,
1) Reynolds equation itself must be discretized/solved non-dimensionally by using standard EHL scaling for its variables like H = (h*R)/a^2, ETA = eta/eta0, RHO = rho/rho0, U = Ve/Vref  etc. 
But when this non-dimensional pressure P is used then it should be dimensionalized with scaling p = p*ph in all other models where p is required like rheological models, Wh, frictions etc. etc. 
2) Remove unnecessary conditions/window's boundness which is creating any ambiguity in reynolds solver. As per standard mixed lubrication line contact theory, pressure profile must lie on contact zone which in non dimensional case it should be on x=-1 to 1. But the contact window should be -4.5 to 3. And pressure must be zero line away from -1 to 1. Make sure no any shrinkage or leakage of reynolds profile occur.
3) In pressure profile plot control section, it must only plot the same non-dimensional pressure vs nondimensional x axis calculated from its physics. 
4) make sure the viscosity is being calculated as per standard theory and its profile is stable and accurate. 
5) Make sure film thickness is currectly updated after every loop so that final film thickness must be as per standard which is a flat line on contact zone (-a to a) and exponential like increasing away from contact zone 
6) Make sure all physics are perfectly aligned and linked with each other like standard simulation.
TESTING:
Once you achieve and sured all above 6 points changes the run the entire code with a single rpm and any single cam angle and visualize the profiles and convergence if any thing lack from standard literature restart improving from point 1 to 6. Once achieved goal then only regenerate entire complete ready to paste code including both physics section and plot control section without any missing/hidden lines.
