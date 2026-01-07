"""Analyze run log to find problematic steps"""

import re

with open('run_balanced.log', 'r') as f:
    lines = f.readlines()

problematic = []
for line in lines:
    if 'Step' in line and 'Load err' in line:
        match = re.search(r'Step (\d+)/\d+ \| Load err=([0-9.e+-]+)', line)
        if match:
            step = int(match.group(1))
            load_err = float(match.group(2))
            if load_err > 0.01:  # 1%
                problematic.append((step, load_err))

print(f"Found {len(problematic)} steps with load error > 1%:")
for step, err in sorted(problematic, key=lambda x: x[1], reverse=True)[:20]:
    print(f"  Step {step}: load error = {err:.4e}")
