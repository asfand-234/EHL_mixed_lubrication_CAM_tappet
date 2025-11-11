Do the same task for 110°C case.

Read Python code in text file "ehl_python.txt" which is related to friction torque reduction due to surface texture using 1D Mixed lubrication Line contact having  in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

TASK:
1) Run the code for only 110°C and print percentage averager friction torque reduction with respect to untextured for all RPMs and for all three texture area densities. 
2) See if every value meet the following target values.
        
RPM     5_percent     8_percent    10_percent
300      3.36%               11.96%            1.29%
500      4.52%               11.05%             0.41%
700     4.37%               14.14%           5.2%
900    -7.55%              1.45%           12.28%
The calculated % averaged friction torque must be same at least 80% as above target values. 
3) only tune/calibrate "E_eff" value in htex model for each case to meet the targets. For example start from first case, 300 rpm and 5% texture area density tune E_eff and understand its trend until you meet this 5% target results at 300 rpm then move to next case 300 rpm but at 8% texture area density and so on. 
4) At the end record all 12 sperate values of E_eff  best suited to meet above target results and print all 12 values to me.

5) Do not modify the code and do not touch any other parameter value.

Note: there is no limit of range and type of value you used for E_eff. It can be any value.
