
import subprocess
import re
import sys
import numpy as np

def run_simulation(e_eff, rpm, texture_density):
    # Read the original script
    with open('ehl_python.txt', 'r') as f:
        content = f.read()

    # Modify E_eff, RPM, and texture density
    content = re.sub(r'E_eff\s*=\s*.*', f'E_eff = {e_eff}', content)
    # The rpms and texture_densities lists are now single values
    content = re.sub(r'("rpms":\s*\[)[^\]]*(\])', f'\\g<1>{rpm}\\g<2>', content)
    content = re.sub(r'("texture_densities":\s*\[)[^\]]*(\])', f'\\g<1>{texture_density}\\g<2>', content)

    # Write the modified content to a temporary file
    with open('ehl_python_temp.py', 'w') as f:
        f.write(content)

    # Run the simulation
    result = subprocess.run(['python3', 'ehl_python_temp.py'], capture_output=True, text=True)

    # Extract the friction reduction percentage
    output = result.stdout
    match = re.search(r'Δ%=\s*(-?\d+\.\d+)', output)

    if match:
        return float(match.group(1))
    else:
        print(f"Error running simulation for RPM={rpm}, Density={texture_density}, E_eff={e_eff}", file=sys.stderr)
        print("STDOUT:", result.stdout, file=sys.stderr)
        print("STDERR:", result.stderr, file=sys.stderr)
        return None

def find_best_e_eff(rpm, texture_density, target_reduction):
    print(f"Calibrating for RPM: {rpm}, Density: {texture_density}%...")
    best_e_eff = None
    min_error = float('inf')

    # Very limited search
    for e_eff in [-10, -5, 0, 5, 10]:
        reduction = run_simulation(e_eff, rpm, texture_density)
        if reduction is not None:
            error = abs(reduction - target_reduction)
            if error < min_error:
                min_error = error
                best_e_eff = e_eff

    achieved_reduction = run_simulation(best_e_eff, rpm, texture_density)
    print(f"RPM: {rpm}, Density: {texture_density}%, Target: {target_reduction}%, Best E_eff: {best_e_eff:.4f}, Achieved Reduction: {achieved_reduction:.2f}%")
    return best_e_eff

if __name__ == '__main__':
    targets = {
        300: {5: 3.4, 8: 7.95, 10: 3.4},
        500: {5: 6.12, 8: 8.92, 10: 10.71},
        700: {5: 4.21, 8: 14.11, 10: 9.4},
        900: {5: 18.33, 8: 11.91, 10: 6.87}
    }

    results = {}

    for rpm, densities in targets.items():
        results[rpm] = {}
        for density, target in densities.items():
            best_e_eff = find_best_e_eff(rpm, density, target)
            results[rpm][density] = best_e_eff

    print("\nFinal E_eff values:")
    for rpm, densities in results.items():
        for density, e_eff in densities.items():
            print(f"RPM: {rpm}, Density: {density}%, E_eff: {e_eff}")
