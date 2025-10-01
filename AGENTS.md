Read text file "MAIN_SCRIPT.txt"  that calculates the percentage reduction of averaged friction torque with respect to untextured. It is about friction reduction due to textures/grooves in CAM and textured shim (inside bucket tappet) using 1D mixed lubrication theory.
And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 
The current script uses a mathemtical model of textures with fixed parameters and compute entire reynolds equation solution and calculate % averaged friction torque reduction. Only 1 parameter is not fixed a_tex (amplitudes) which requires three input data files.

IMPLEMENTATION STRATEGY:

1) First Improve the texture model by making sure the following things must be done and correctly working:
 i) Make sure shift term is correctly working in the script which tells that how much a texture covers/moves a distance per angle to reflect the same movement of CAM. Because instead of moving CAM the mathematical model conveniently move textures at the same way as cam rotates and cross over the textures on shim surface. So not at every cam angle the texture should be appear, because during actual rotation at few cam angles points the CAM would lie outside the texture land (beweeting texture spacing) so at that point the entire texture model and a_texture value must be zero so that only at those cam angle points results should be like smooth case.

ii) and at those cam angle points which bring cam to cross texture, the textures must be within -b to b contact window geometry.. away from  it , they must be restricted. 
iii) number of grooves appear in contact window (x axis) depends upon the contact width 2*b at instant angle. Let say when 2*b is 48e-6 then only 1 texture having width 35e-6 can be appear. 
 
2) Reynolds equation must be professonally stabilized with highest accuracy and precision. And pressure profile must lie within -b to b. With accurate shape. Also, pay attention reynolds solution and profile may not get destoryed due to microtextures effects. 

3) Make sure the elastic deformation is correctly working and linked with film thickness model.
4) First make above changes and then re-generate updated three amplitudes data text files having 5 columns each (first column for cam angle and rest are for RPMs). The data in these files must now arranged after above same texture model settings. For example At those cam angle points the a_tex amplitude data should be absolutely zero (when the cam lie outside texture land). And keep random values in remaining cells by yourself. All values must be less then 6.5e-6 and every cell in every rpm column must have different value. 

 TESTING (IMPORTANT)
once you generate three files. Then take forward step. Import your own generated updated three amplitude files in updated script and plot live for yourself (*DO NOT* generate for me).
a)Texture model htex vs cam angle at 300 rpm and 8%.
b) reynolds pressure p vs x axis (-4.5*b to 3*b) at 300 RPM 8% texture and at one of that single cam angles which brings cam over shim texture. 
c) film thickness vs x axis at 300 RPM 8% texture and at one of that single cam angle which brings cam over shim texture. 
d) print % averager friction torque reduction for all RPMs and all texturea area densities

If you see any inconsistency, error, or ambiguity in results or if you see texture distribution and their effects on any physics is suspecious and lack of physical meaning or away from standard literature.
Then *RESTART FROM STEP 1 TO TESTING*
DELIVERABLES ONLY TO ME:
1)  complete error free executable google colab script having same plots and print lines as above.
2) Three updated amplitude files.
