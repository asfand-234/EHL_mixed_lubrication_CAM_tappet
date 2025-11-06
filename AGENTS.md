Read and analyze every line of python code text file "python_seventeen.txt" which related to friction torque reduction due to surface textures in CAM and flat faced follower using 1D Mixed lubrication Line contact theory.
and CAM lift data file named "updated.lift.txt"

TASK: 
Your task is to calibrate only "flash temperature & traction parameters and stribeck gated texture synergy parameters given at the top of code for ONLY 90° to meet the target values of percentage reduction of Average friction torque with respect to untextured for only 5% texture area density and for four RPMs i.e. 300, 500, 700, 900.

Target Values:
RPM       % AV. friction Torque reduction
300         3.4%
500        6.12%     
700         4.21%
900         18.33%


IMPLEMENTATION STRATEGY:
1) Start first tuning Flash Temperature parameters for 90°C only and for only 5% texture area density and for 300 RPM. Understand the effect of each parameter.
 Once fitted then these must be constant for all RPMs and for all texture area densities. 
Parameters:                   Range Limit:
F0_COVERAGE_TABLE   =        [0-1]
NF_COVERAGE_TABLE   =         [1-3]
CHI_TEX_TABLE       =         [0.2-0.8]
BETA_EDGE_TABLE     =        [1-1.5]
K_EFF_TABLE         =          [20-70]
BETA_ETA_FLASH_TABLE=       [0.015–0.030]
C_LSS_TEX_TABLE     =       [0.06–0.16]
KAPPA_SRR_TABLE     =        [0.1-0.6]
MASK_THR_MULT =             [3-12]
2) When you see the results are slightly better then start tuning SGTS gate parameters. When these once fitted then these Must remain constant for all RPMs and all texture area densities.
SGTS Gate parameters:                 Range Limit
LAM_STAR   =                           [0.8-1.2]
DELTA_LAM  =                           [0.2-0.6]
S0_SGTS    =                           [0.1-0.3]

3) Then Start tuning A_SGTS parameters.
Every RPM must have its own fitted value.
RPM     A_SGTS Range Limit
300        [-0.6 to 0.8]
500        [-0.6 to 0.8]
700        [-0.6 to 0.8]
900        [-0.6 to 0.8]

4) If the results are not 80% close to the target values then re-start again from tuning flash temperature parameter and keep repeating all above 3 steps until you get at least 70% same target results.
5) Once the target achieved then only and only generate most suitable fitted values for all above parameters in summary.

HARD STRICT RULES:
1) You are not allowed to make any change in any line of the code. Also do not change any other parameter values other than above mentioned.
2) Do not use any above parameter value out of the defined ranges
