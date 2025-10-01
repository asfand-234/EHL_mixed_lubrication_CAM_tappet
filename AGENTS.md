Read my python code ''python_code_CAM_TAPPET.txt'' that calculates the percentage of averaged friction torque  in CAM and untextured shim (inside bucket tappet) using 1D mixed lubrication theory.
And a Cam lift data file named "CamAngle_vs_Lift_smooth.txt". 

TASK:
Your task is to introduced correct mathematical model of surface texture/grooves in script.

IMPLEMENTATION STRATEGY:
1) Texture Mathematical model ("htex").
A guassian groove with following fixed parameters:
 texture width= w_texture = 35e-6 m
Center to center texture spacing= d_texture :
For 5% = 0.0016 m
For 8% = 0.0012 m
For 10% = 0.00096 m
x = -X_in*b(theta) to X_out*b(theta)
X_in = -4.5
X_out = 3
b(theta) = a_hertz [hertzian half contact width as function of CAM angle].

Texture movement (along direction of motion, x axis) = Shift: d(shift, t) - Vf(theta) = 0
Vf = follower velocity as function of cam angle.
Texture amplitude = a_texture = 5e-6 m


i) Make sure shift term is correctly working in the script which tells that how much a texture covers/moves a distance per angle to reflect the same movement of CAM. Because instead of moving CAM the mathematical model should conveniently move textures at the same way as cam rotates and cross over the textures on shim surface. So not at every cam angle the texture should be appear, because during actual rotation at few cam angles points the CAM would lie outside the texture land (beweeting texture spacing) so at that point the entire texture model and a_texture value must be zero so that only at those cam angle points results should be like smooth case.

ii) and at those cam angle points which bring cam to cross texture, the textures must be within -b to b contact window geometry andaway from it they must be restricted. 
iii) number of grooves appear in contact window (x axis) depends upon the contact width 2*b at instant angle. Let say when 2*b is 48e-6 then only 1 texture having width 35e-6 can be appear at that instant cam angle and so on.

2) Reynolds equation must be professonally stabilized with highest accuracy and precision. And pressure profile must lie within -b to b. With accurate shape. Also, pay attention reynolds solution and profile may not get destoryed due to microtextures effects. 

3) Make sure the elastic deformation is correctly working and linked with film thickness model.

TESTING:

a)Texture model htex vs cam angle at 300 rpm and 8%.
b) reynolds pressure p vs x axis (-4.5*b to 3*b) at 300 RPM 8% texture and at one of that single cam angles which brings cam over shim texture). 
c) film thickness vs x axis at 300 RPM 8% texture and at one of that single cam angle which brings cam over shim texture. 
d) print % averaged friction torque reduction with respect to untextured for all RPMs and all texture area densities
