# Water Hammer Simulation Comparison

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![License](https://img.shields.io/badge/license-MIT-blue)]()

This project compares different numerical methods for simulating water hammer (hydraulic transient) phenomena in pipelines. It implements Method of Characteristics (MOC) as the reference solution, then compares it against Finite Volume Method (FVM), Lattice Boltzmann Method (LBM), and Smoothed Particle Hydrodynamics (SPH) approaches. Each method simulates instantaneous valve closure in a pipe with friction, providing researchers and engineers with a comprehensive toolkit for hydraulic transient analysis.

## Table of Contents
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Quick Start](#quick-start)
- [Command-line Arguments](#command-line-arguments)
- [Output Artifacts](#output-artifacts)
- [Citation / Reference](#citation--reference)

## Project Structure

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `MOC.py` | Implements the Method of Characteristics reference solution for water hammer simulation | `simulate_instant_valve_closure()` - Main MOC simulation with detailed physical setup<br>`moc_solve()` - Parameterized MOC solver returning comprehensive results dictionary |
| `FVM.py` | Implements Finite Volume Method aligned with MOC for water hammer simulation | `moc_reference()` - Retrieves MOC reference solution and metadata<br>`exact_linear_flux()` - Computes exact linear flux for FVM scheme<br>`run_fvm_aligned()` - Runs FVM simulation aligned with MOC parameters |
| `LBM.py` | Implements Lattice Boltzmann Method tuned for water hammer simulation | `simulate_moc_reference()` - Calls MOC reference and returns head and velocity<br>`simulate_lbm_tuned()` - Runs tuned LBM simulation with wave-like streaming and friction<br>`apply_wave_streaming()` - Performs wave-like streaming of LBM populations<br>`apply_friction_source()` - Applies Darcy-Weisbach friction source term |
| `SPH.py` | Implements Smoothed Particle Hydrodynamics aligned with MOC for water hammer | `cubic_b_spline_w()` - Computes 1D cubic B-spline kernel value<br>`interpolate_shifted_characteristic()` - SPH interpolation of shifted characteristic variables<br>`run_sph_aligned_with_moc()` - Runs SPH simulation aligned with MOC parameters |

## Workflow

1. **Run MOC reference simulation**: `moc_solve()` computes reference solution with physical parameters: L=300m, Hr=70m, N=25, f=0.018, u0=0.1m/s, t_max=20s
2. **Run comparative method (FVM/LBM/SPH)**: Each method imports MOC reference, uses aligned parameters (dx, dt, a), and runs its own simulation
3. **Compute error metrics**: Calculate relative L2 error and max absolute error for valve head history compared to MOC
4. **Generate comparison plot**: Save PNG plot comparing valve head history between method and MOC to png/ directory
5. **Print performance and error statistics**: Output wave speed, time step, error metrics, and elapsed time to console

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run any simulation module:
```bash
python MOC.py
python FVM.py
python LBM.py
python SPH.py
```

Each module can run independently and will generate comparison plots in the `png/` directory.

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| *This project currently uses hardcoded parameters. Each module's main() function can be modified to accept command-line arguments for customization.* | | |

## Output Artifacts

The pipeline produces the following output files:

- `png/MOC_valve_head.png` - Reference MOC solution plot
- `png/fvm_vs_moc_valve_head.png` - FVM vs MOC comparison plot
- `png/lbm_vs_moc_tuned_valve_head.png` - LBM vs MOC comparison plot
- `png/sph_vs_moc_valve_head.png` - SPH vs MOC comparison plot

## Citation / Reference

*If you use this software in your research, please cite the relevant papers for the numerical methods implemented here.*