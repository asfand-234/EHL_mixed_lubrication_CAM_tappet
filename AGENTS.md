Read my "python_six.txt" code and deeply analyze it. It has two main sections. Physics section and Plot control section. This code is about friction torque calculation in CAM and flat follower using 1D mixed lubrication line contact theory.
The cam lift data is "CamAngle_vs_Lift_smooth.txt" and three texture amplitude files are also given in repository

Changes are required in Physics section only. Do not change my plot section.

PROBLEMS:
The current code has two main issues:
1) Asperity load "Wa" values are too small and profile is not correct.
2) The averaged friction Torque values for all RPMs are small and not following target experimental trends.
TARGET VALUES (for averaged friction torque):
RPM       Avg Friction Torque
300         0.08845
500         0.07849
700         0.06598
900         0.06381

And The asperity load must be greater than 10. 

TASKS:
Your tasks as a highly professional engineering and programming agent are,
1) Improve the asperity load calculation model. Make sure it must be as per standard literature. Make sure its scaling and variables are correctly linked and defined. 
2) Tune/calibrate the roughness parameters values but the product of (sigma*beta_a*eta_R) must remain around 0.05 and ratio of (sigma/beta_a) must remain around 0.001.
3) Make sure, hydrodynamic friction and asperity frictions are accurately calculated because, the averaged friction torque values must show decreasing trend from 300 RPM to 900 RPM. Do not change formula of asperity friction rather focus on asperity load formula.
4) Any other physics which need to be adjusted then do it.

CRITERIA:
Keep improving physics and printing results of asperity load and averaged friction torque if results do not meet with target then again start improving tasks from 1 to 4 until you meet at least 85% same target results.

DEIVERABLE:
Once you achieved target and performed final tests then only generate updated complete (without diff markers, missing lines, hidden lines or ellipses), error free ready to paste google colab script to me.
