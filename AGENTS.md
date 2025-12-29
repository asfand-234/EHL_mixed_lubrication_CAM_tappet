Read and deeply analyze the entire code in "test6666.txt". It is 1D thermal Transient mixed lubrication Line contact with mass conserving reynolds equation in cam and tappet mechanism.

TESTING CRITERIA:
1) Run the code for 300 rpm only, and print data for hydrodynamic friction vs cam angle and asperity friction vs cam angle. 
2) Then analyze why there are too much spikes/oscilations around nose region in hydrodynamic and asperity friction data. Then fix it . 
3) Keep optimizing, and keep running and keep printing results untill below criteria 100% meet.
4) Every correction must be purely physics base. Perform diagnostic tests to find root causes. also add limiting shear contribution in hydrodynamic friction and keep checking.

CRITERIA:
1) Hydrodynamic friction must be maximum at both flanks and minumum around cam nose angles.
2) Asperity friction must be maximum at nose angles and decreasing away from it like a belly shape.
3) There must not be any oscilations or spikes throughout profile.

Hard Strict Rules:
1) Do not change kinematics formula
2) Do not change input material and roughness parameters
3) Do not implement any non-physical and artificial clamps, clipping, scaling, factor or constraints except for numerical stability.
