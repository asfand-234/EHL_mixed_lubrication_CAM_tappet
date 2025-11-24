Read Python code in text file "ehl_python.txt" which is related to average friction torque reduction due to surface textures using 1D Mixed lubrication Line contact having compressible reynolds equation solution in CAM and tappet mechanism.
And cam lift data "updated_lift.txt"

**TASK:**
Calibrate/tune "E_eff" value in htex for each case of rpm and texture area density to meet the following targets % of averaged friction torque reduction with respect to untextured. 
*TARGET RESULTS:*
RPM     5_percent     8_percent             10_percent
300      3.4%               7.95%            3.4%
500      6.12%              8.92%           10.71%
700     4.21%               14.11%           9.4%
900     18.33%             11.91%           6.87%


**IMPLEMENTATION STRATEGY:**
1) For untextured data, keep text files, "untex_Fb_theta.txt" and "untex_Fh_theta.txt" in your path. This will save computational time for calculating % average friction torque reduction.
2) Run for first case, rpm 300 and texture density 5% and see the % average friction torque reduction. Then calibrate only E_eff value and again print the result. If the result meet at least 85% same as its target value then record its E_eff value and move to next case like rpm 300 and texture density 8%. If not meet the target value then keep tuning E_eff value and keep printing results and so on.

3) There is no limit of range of E_eff value it can be of any type. But every case will must have its own E_eff value. So at the end there would be total 12 best suitable E_eff values of each case. 
4) At the end just provide all 12 values of E_eff to me. No need to provide any other thing.

**HARD STRICT RULES:**
1) DO NOT make any changes in the code. 2) Do NOT change any parameter value. Only calibrate E_eff value for each case.
