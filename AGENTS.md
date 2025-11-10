Read Python code in text file "ehl_python.txt" which is related to friction torque reduction due to surface texture using 1D Mixed lubrication Line contact having  in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

TASK:
1) Run the code for only 90°C and print percentage averager friction torque reduction with respect to untextured for all RPMs and for all three texture area densities. 
2) See if every value meet the following target values.
        
RPM     5_percent     8_percent           10_percent
300      3.4%                7.95 %          3.4%
500      6.12%              8.92%            10.71%
700     4.21%               14.11%           9.4%
900    18.33%              11.91%           6.87%
The calculated % averaged friction torque must be same at least 80% as above target values. 
3) only tune/calibrate "E_eff" value in htex model for each case to meet the targets. For example start from first case, 300 rpm and 5% texture area density tune E_eff and understand its trend until you meet this 5% target results at 300 rpm then move to next case 300 rpm but at 8% texture area density and so on. 
4) At the end generate only 12 sperate values of E_eff  best suited to meet above target results.

5) Do not modify the code and do not touch any other parameter value
