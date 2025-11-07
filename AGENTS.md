Read more attentively my code files of MATLAB syntax named:
EHL_00_run_Study_A2_gen, EHL_00_run_Study_A3_gen, EHL_00_run_Study_A_gen, EHL_01_setup_Study_A2_gen, EHL_01_setup_Study_A3_gen, EHL_01_setup_Study_A_gen, EHL_02_mainprocess

It is a MATLAB solver for unsteady Elasto-Hydrodynamic Lubrication (EHL) problems, based on the EHL-FBNS algorithm. The solver uses the Finite Volume (FV) discretization of the Reynolds equation incorporating mass-conserving cavitation through the cavity fraction and elastic deformation through the application of the Boundary Element Method (BEM) to an elastic half-space. Shear thinning, Roelands and Dowson-Higginson relations are also embedded in the code. Furthermore, the load-balance equation is considered.
The code consists out of three main scripts which are suppose to be executed consecutively and two optional scripts:
(Somewhat optional): the scripts denoted with EHL_00_run will set up and run severeal EHL solver simulations within one study. The scripts denoted with EHL_01_setup will set up the input information for the EHL solver. (However, if you don't want to use the EHL_00_run scripts, you need to modify the EHL_01_setup script such that all of the variables read from the autorun structure (which is originally initiated in EHL_00_run) are instead defined in EHL_01_setup). The script denoted with EHL_02_mainprocess incorporates the actual EHL solver and computes the solution of the EHL problem.

**TASKS:**
Your Task is to:
1) Deeply read every line, setup and layout of the code.
2) Convert the entire code into perfect smooth Python syntax. Pay attention to use every functions, definitions, etc. Of Python.
3) Your Python code must be only for 1D Mixed lubrication and unsteady.
Also convert it to Line Contact instead of point contact.
It must solve complete standard rheological models of viscosity and density same as the current code is doing.
The reynolds equation must be compressible and mass conserving.
It must solve the same complete cavitation algorithm and stabilization as currently doing.
4) It must solve same accurately surface textures (micro textures) as doing now. Also a switch must be applied to switch off microtexture when i want to run only for smooth surface.
5) It must plot on 4 graphs in 1D separately. 
reynolds (hydrodynamic) pressure vs x axis for untextured case
Film thickness vs x axis for untextured
reynolds (hydrodynamic) pressure vs x axis for Textured case
Film thickness vs x axis for Textured case
6) Assume any parameter values by your own if not available in currenr srcipt.
**TESTING AND ACCEPTANCE CRITERIA:**
Once the above all 5 points achieved fhen run the entire code and analyze the pressure and film thickness data. If the profiles and data 100% meet the standard theory and script is smoothly converging then provide only "Ready to paste google colab complete script".
No need to provide results data.

**HARD STRICT RULES:**
1) DO NOT simplify anything. You have to replicate complete code into python syntax with some modification as above mentioned. 
2) You must go through the relevant literature when needed to make the code perfect.
   
