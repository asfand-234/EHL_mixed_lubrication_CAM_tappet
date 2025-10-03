Read my "python_one.txt" code and deeply analyze. It has two main sections. Physics section and Plot control sections. Changed are required in Physics section only. This code is about friction torque calculation in CAM and flat follower using 1D mixed lubrication line contact theory.
The cam lift data is "CamAngle_vs_Lift_smooth.txt" and three texture amplitude files are also given in repository. 

TASK:
1) convert the Reynolds equation into non-dimensional form using standard EHL theory scaling. All the imput variablesof Reynolds equation would be first non-dimensionalized before using in Reynolds solver. X = x/b,  where x domain is -4.5*b to 3*b so X  would be -4.5 to 3,

 H = (h*R)/ph, ETA = eta/eta0, RHO= rho/rho0, V = Ve/v_ref (chose a suitable v_ref, that meet all criteria). When you use dimensional pressure in other models like load balance criteria (below discussed), viscosity, density and friction models then, reconvert non-dimensional P into dimensional p using scaling p = P*ph. Where ph = hertzian pressure. 
2) A very simplified reynolds equation was used which is not suitable for CAM tappet mechanism. Include standard DOWSON & HIGGINSON pressure dependent density model correctly.
3) Instead of simple calculation of asperity load wa, implement standard Greenwood and Tripp statistical model and calculate asperity friction Fb.
combined RMS = sigma = 0.2e-6 m
Average asperity tip radius = beta_a = 2e-4 average summit radius =  k = 2e-4 
Asperity distribution per unit area = eta_R = 1.25e9

4) instead of simple load balance criteria now implement mixed lubrication standard load balance criteria i.e. (Wh + wa)- W = 0. From this load balance criteria you will get h0 (initial film thickness). You can take help from "reference_article.pdf" and also from relevant web sources.

IMPORTANT INSTRUCTIONS:
1) The load balance criteria must meet upto least relative error. If criteria does not meet or convergenve issuee occur then:
  i) adjust h0 intial value
  ii) Tune/calibrate above roughness parameter values but the product of (sigma*beta_a*eta_R) must remain about 0.05 and ratio of (sigma/beta_a) must remain about 0.001. 
  iii) implement damping factor and under-relaxation factor. Do not add any unnecessary factors.
  iv) Improve Cavitation model and stabilization in reynolds solution.
v) implement any other necessary change which must be as per relevant standard literature.
2) Make sure The reynolds equation perfectly converged with highest accuracy and has perfect standard profile (like dome shape) on contact zone (-1 to 1) and away from it, the extended window (towards boundaries) pressure must be zero. 
3) The magnitude of non-dimensional reynolds pressure profile remain 0 to 1 as per theory. So make sure you are in right path.

TESTING:
AFTER MAKING ABOVE CHANGES, MUST RUN THE ENTIRE SIMULATION AND OBSERVE THE CONVERGENCE AND LOAD BALANCE ERROR, THE MAGNITUDE OF NO-DIMENSIONAL REYNOLDS PRESSURE VS X AXIS (STAY FROM 0 TO 1 AS PER THEORY). IF NON OF THESE CRITERIA MEET THEN RE-START IMPROVING ALL PHYSICS TO GET DESIRED RESULTS.

DELIVERABLE:
Once you achieved above all tasks and meet the requirements then generate only and only full complete (not missing lines or no elipses) error free updated code (ready to paste for google colab) with my Plot control section (unchanged).
