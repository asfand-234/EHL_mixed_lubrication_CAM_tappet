# 1D Transient Reynolds Equation Solution for Mixed Lubrication Theory

## Overview

This repository contains a complete implementation of a 1D transient Reynolds equation solver for mixed lubrication theory, specifically designed for CAM-SHIM (bucket tappet) contacts with surface textures. The solver incorporates advanced tribological models and numerical methods to accurately predict lubrication performance and friction reduction.

## Key Features

### 1. **Complete Mixed Lubrication Model**
- **Hydrodynamic Lubrication**: Solves the 1D transient Reynolds equation with mass-conserving cavitation
- **Asperity Contact**: Implements Greenwood-Tripp statistical model for rough surface contact
- **Load Balance**: Ensures total load is carried by combination of hydrodynamic and asperity contact

### 2. **Advanced Fluid Models**
- **Pressure-Viscosity**: Houpert exponential model with pressure-viscosity coefficient
- **Shear-Thinning**: Carreau model for non-Newtonian behavior at high shear rates
- **Eyring Stress**: Limiting shear stress model for boundary lubrication
- **Compressibility**: Dowson-Higginson density-pressure relationship

### 3. **Surface Texture Modeling**
- **Variable Amplitude**: Theta-dependent texture amplitude from experimental data
- **Texture Shift**: Accounts for sliding-induced texture displacement
- **Multiple Densities**: Supports 5%, 8%, and 10% area coverage textures
- **Realistic Profiles**: Gaussian-based texture height distribution

### 4. **Elastic Deformation**
- **Influence Functions**: Calculates elastic deflection using Green's functions
- **Contact Mechanics**: Hertzian contact as initial approximation
- **Coupled Solution**: Iterative coupling between pressure and deformation

### 5. **Robust Numerical Methods**
- **Mass Conservation**: Rusanov flux scheme for content transport
- **Pressure Solver**: Thomas algorithm for tridiagonal systems
- **Adaptive Time Stepping**: CFL-based substep calculation
- **Relaxation**: Stabilized iteration with pressure and film thickness relaxation

## Mathematical Foundation

### Reynolds Equation
The 1D transient Reynolds equation with mass conservation:

```
∂/∂t(ρH) + ∂/∂x(ρHU) = ∂/∂x(ρH³/12η ∂p/∂x)
```

Where:
- `ρ`: Fluid density
- `H`: Film thickness
- `U`: Entraining velocity
- `p`: Pressure
- `η`: Dynamic viscosity

### Mixed Lubrication Closure
Total load balance:
```
W = W_h + W_a
```

Where:
- `W`: Applied load
- `W_h`: Hydrodynamic load (pressure integration)
- `W_a`: Asperity load (Greenwood-Tripp model)

### Film Thickness Equation
```
h(x) = h₀ + x²/(2R) + u_el(x) + h_tex(x)
```

Where:
- `h₀`: Central film thickness (Hamrock-Dowson)
- `R`: Effective radius of curvature
- `u_el(x)`: Elastic deformation
- `h_tex(x)`: Texture contribution

## Code Structure

### Main Class: `Reynolds1DMixedLubrication`

#### Initialization Methods
- `__init__()`: Initialize solver with material and fluid properties
- `_load_cam_data()`: Load and smooth cam profile data
- `_init_greenwood_tripp()`: Create asperity contact lookup tables
- `_load_texture_data()`: Load texture amplitude data files

#### Core Physics Methods
- `calculate_kinematics()`: Compute velocities and forces from cam profile
- `solve_reynolds_at_angle()`: Main solver for single cam angle
- `asperity_pressure_greenwood_tripp()`: Calculate asperity contact pressure
- `elastic_deflection()`: Compute elastic deformation
- `texture_profile()`: Generate texture height profile

#### Fluid Property Methods
- `viscosity_houpert()`: Pressure-dependent viscosity
- `viscosity_carreau()`: Shear-thinning viscosity
- `density_dowson_higginson()`: Pressure-dependent density

#### Analysis Methods
- `analyze_friction_reduction()`: Compare textured vs untextured friction
- `plot_results()`: Visualize solution results
- `run_example()`: Execute example analysis

## Input Data Files

### Required Files
1. **`CamAngle_vs_Lift_smooth.txt`**: Cam profile data (angle vs lift)
2. **`a_texture_data_5pct.txt`**: 5% texture amplitude data
3. **`a_texture_data_8pct.txt`**: 8% texture amplitude data  
4. **`a_texture_data_10pct.txt`**: 10% texture amplitude data

### Data Format
- Cam data: Two columns (angle_deg, lift_m)
- Texture data: Multiple columns (angle_deg, RPM300, RPM500, RPM700, RPM900)

## Usage Examples

### Basic Usage
```python
from reynolds_1d_mixed_lubrication import Reynolds1DMixedLubrication

# Initialize solver
solver = Reynolds1DMixedLubrication(data_dir=".")

# Run example analysis
results, reduction_results = solver.run_example()
```

### Custom Analysis
```python
# Analyze specific operating point
rpm = 500
angle_deg = 0.0
density_key = "5%"

# Calculate kinematics
R, Ve, Vs, W, omega = solver.calculate_kinematics(rpm)
idx = np.argmin(np.abs(solver.th_deg - angle_deg))

# Get texture parameters
dt = np.gradient(solver.th)[idx] / max(omega, 1e-30)
shift = solver.integrate_shift(Vs, omega)
atex = solver.get_texture_amplitude(density_key, rpm)
d_tex = solver.D_TEXTURE[density_key]

# Solve Reynolds equation
results = solver.solve_reynolds_at_angle(
    R[idx], Ve[idx], Vs[idx], W[idx], dt, angle_deg, rpm,
    atex_theta=float(max(atex[idx], 0.0)), 
    shift_theta=shift[idx], d_texture=d_tex
)
```

### Friction Reduction Analysis
```python
# Analyze friction reduction for different RPMs
reduction_results = solver.analyze_friction_reduction("5%", rpms=[300, 500, 700, 900])
print(reduction_results)
```

## Key Results

The solver provides comprehensive output including:

### Solution Variables
- **Pressure Distribution**: `p(x)` - Hydrodynamic pressure field
- **Film Thickness**: `h(x)` - Local film thickness including texture
- **Asperity Pressure**: `p_asp(x)` - Contact pressure from surface roughness

### Performance Metrics
- **Friction Forces**: Hydrodynamic (`Fh`) and boundary (`Fb`) components
- **Load Distribution**: Percentage carried by fluid film vs asperity contact
- **Contact Parameters**: Half-width, maximum pressure, minimum film thickness

### Friction Reduction Analysis
- **Torque Comparison**: Textured vs untextured surfaces
- **Percentage Reduction**: Quantified improvement from surface texturing
- **RPM Dependence**: Performance variation with operating speed

## Physical Validation

The solver includes several validation checks:

1. **Load Balance**: Ensures applied load equals calculated load (< 1% error)
2. **Pressure Positivity**: All pressures remain non-negative
3. **Film Thickness**: All film thicknesses remain positive
4. **Texture Effect**: Measurable impact of surface texturing on friction

## Numerical Parameters

### Grid Resolution
- **Spatial Points**: 171 points in contact region
- **Core Pressure Grid**: 451 points for high accuracy
- **Time Steps**: Adaptive based on CFL condition

### Convergence Control
- **Pressure Iterations**: Maximum 52 iterations per time step
- **Relaxation Factors**: 0.85 (pressure), 0.55 (film thickness)
- **Tolerance**: Implicit through load balance

### Stability Features
- **Smoothing**: Applied to pressure field for stability
- **Positivity**: Enforced for pressure and film thickness
- **Mass Conservation**: Rusanov scheme prevents numerical diffusion

## Applications

This solver is particularly useful for:

1. **Tribological Design**: Optimizing surface textures for friction reduction
2. **Engine Development**: Analyzing cam-tappet lubrication performance
3. **Research**: Understanding mixed lubrication physics
4. **Validation**: Comparing with experimental friction measurements

## Dependencies

- **NumPy**: Numerical computations and array operations
- **Pandas**: Data handling and file I/O
- **Matplotlib**: Results visualization and plotting

## Performance Considerations

- **Computational Cost**: ~1-2 seconds per cam angle on modern hardware
- **Memory Usage**: Moderate (< 100 MB for typical problems)
- **Scalability**: Linear scaling with number of cam angles analyzed

## Future Enhancements

Potential improvements include:

1. **2D Extension**: Full 2D Reynolds equation solution
2. **Thermal Effects**: Temperature-dependent properties
3. **Cavitation Modeling**: Advanced cavitation algorithms
4. **Optimization**: Automatic texture parameter optimization
5. **Parallel Processing**: Multi-core acceleration for full-cycle analysis

## References

The implementation is based on established tribological theory:

1. **Reynolds Equation**: Classical lubrication theory
2. **Greenwood-Tripp Model**: Statistical rough surface contact
3. **Houpert Viscosity**: Pressure-viscosity relationships
4. **Carreau Model**: Non-Newtonian fluid behavior
5. **Hamrock-Dowson**: EHL film thickness formulas

## Conclusion

This complete 1D transient Reynolds equation solver provides a robust, accurate, and efficient tool for analyzing mixed lubrication in textured contacts. The implementation combines advanced physics models with stable numerical methods to deliver reliable predictions of lubrication performance and friction reduction benefits from surface texturing.