# Monte Carlo Simulations

In this directory can be found source code for implementations of the Monte Carlo (Metropolis algorithm and Wang-Landau method) simulations of various systems.

## Overview 

In this repository are implementations of Monte Carlo simulations of various physical systems. For details and the instructions on how to run the code, read the `README.md` in the sub-folders.

The simulations are:

- **Metropolis algorithm - GPU**:
  - [MC_MA_IAKL](./MC_MA_IAKL/README.md) - $S=1/2$ Ising antiferromagnet on kagome lattice
  - [MC_MA_SIAKL](./MC_MA_SIAKL/README.md) - $S=1/2$ Stacked Ising antiferromagnet on kagome lattice
  - [MC_MA_SIAKL_GEN_S](./MC_MA_SIAKL_GEN_S/README.md) - Stacked Ising antiferromagnet on kagome lattice with a general spin $S$ (Work in progress)
  - [MC_MA_SIAKL_INFTY](./MC_MA_SIAKL_INFTY/README.md) - $S=\infty$ Stacked Ising antiferromagnet on kagome lattice
- **Wang-Landau & Metropolis - CPU**:
  - [WL_IAKL](./WL_IAKL/README.md) - $S=1/2$ Ising antiferromagnet on kagome lattice
  - [WL_IMHL](./WL_IMHL/README.md) - $S=1/2$ Ising antiferromagnet on honeycomb lattice with next nearest neighbors (Work in progress)
- **Metropolis algorithm - CPU**:
  - [Various Matlab Implementations](./MC_Matlab_various/README.md)

## Used technologies

Part of the codebase is in Matlab and the rest in C++ CUDA. CUB library is used for reductions in CUDA code. There are also convenience scripts written in Python 3 and Matlab.

## Publications

The code in this repository was used to calculate data for the following papers:
1. SEMJAN, M., ŽUKOVIČ, M. “Absence of long-range order in a three-dimensional stacked Ising antiferromagnet on kagome lattice”. _Phys. Lett. A_ (2022) **430**, 127975.
2. SEMJAN, M., ŽUKOVIČ, M. “Magnetocaloric properties of an Ising antiferromagnet on a kagome lattice“. _Acta Phys. Pol. A_ (2020) **137**, 622.
3. SEMJAN, M., ŽUKOVIČ, M. “Absence of long-range order in a general spin-S kagome lattice Ising antiferromagnet“. _Phys. Lett. A_ (2020) **384**, 126615.
4. SEMJAN, M., ŽUKOVIČ, M. “Global Thermodynamic Properties of Complex Spin Systems Calculated from Density of States and Indirectly by Thermodynamic Integration Method”. _EPJ Web Conf_. (2020) **226**, 02019.
