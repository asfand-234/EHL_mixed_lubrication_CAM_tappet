# Calibration Plan for E_eff Values

This repository includes numerical models in `ehl_python.txt` to evaluate the mixed lubrication line contact scenario.  The steps below outline how to perform that calibration.

## Prerequisites
- Python environment with `numpy`, `pandas`, `scipy`, and `matplotlib` installed.
- Input data files present in the working directory:
  - `updated_lift.txt`
  - `untex_Fb_theta.txt`
  - `untex_Fh_theta.txt`

## Recommended Workflow
1. **Prepare runtime directory**
   - Set `DATA_DIR` (in your execution environment) to point to the folder containing the input files. if you symlink or copy files so that `DATA_DIR` resolves correctly.
2. **Load baseline data**
   - Use the existing functions in `ehl_python.txt` to load cam lift data and untextured friction components. These serve as reference for percentage reduction calculations.
3. **Select a test case**
   - Start with rpm `300` and texture density `5%` as specified. Keep other parameters fixed.
4. **Tune `E_eff` iteratively**
   - Execute the friction torque computation for the selected case.
   - Adjust only `E_eff` between runs (using runtime overrides such as environment variables or parameter injection in a wrapper script) until the resulting percentage reduction is within 85% of the target for that case.
5. **Record calibrated values**
   - Once a case meets the threshold, log the rpm, texture density, and the chosen `E_eff` in a results table. Proceed sequentially through the remaining 11 cases.
6. **Repeat for all cases**
   - Continue the iteration for each combination of rpm (300, 500, 700, 900) and texture area densities (5%, 8%, 10%). Ensure a unique `E_eff` is captured for every case.

## Results Logging Template
Use the following table format to store tuned values (create a separate results file as needed):

| RPM | Texture Density | Target Reduction (%) | Achieved Reduction (%) | Calibrated `E_eff` |
| --- | --------------- | -------------------- | ---------------------- | ------------------ |
| 300 | 5%              | 3.4                  |                        |                    |
| 300 | 8%              | 7.95                 |                        |                    |
| 300 | 10%             | 3.4                  |                        |                    |
| 500 | 5%              | 6.12                 |                        |                    |
| 500 | 8%              | 8.92                 |                        |                    |
| 500 | 10%             | 10.71                |                        |                    |
| 700 | 5%              | 4.21                 |                        |                    |
| 700 | 8%              | 14.11                |                        |                    |
| 700 | 10%             | 9.4                  |                        |                    |
| 900 | 5%              | 18.33                |                        |                    |
| 900 | 8%              | 11.91                |                        |                    |
| 900 | 10%             | 6.87                 |                        |                    |

