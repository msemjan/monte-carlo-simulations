# Metropolis algorithm in CUDA GPGPU framework 

This is an implementation of Metropolis algorithm for the stacked $S=1/2$ Ising antiferromagnet on the kagome lattice. 

## Prerequisities 

- [make](https://en.wikipedia.org/wiki/Make_(software))
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [CUB](https://nvlabs.github.io/cub/) library - download it, and change the path in `makefile` (in `INCLUDES` variable) 

## Features
 
- Calculates energy and magnetizations
- Calculates and saves mean values of observables (if `CALCULATE_MEANS` macro is defined)
- Saves the final configuration of the lattice, if `SAVE_CONFIGURATION` macro is defined
- Saves time series of observables, if `SAVE_TS` macro is defined
- Random numbers are produced in batch for better performance
- Logic of Metropolis algorithm is separated from the lattice shape (contained in kernels) 

## How to run

1. Install the CUB library, and update the `makefile`
2. Modify the configuration file `includes/config.h` and save the changes
3. Compile with `make all` command
4. Run with `./bin/metropolis`

## Limitations

- Current implementation allows to simulate only $L=8$, $16$, $32$, $48$, $64$, and $96$

## TODO

- [ ] 'Linearize' the lattice
- [ ] Lattice can be allocated on heap
- [ ] Rewrite update kernels, using grid-stride loops
- [ ] Add a possibility to introduce the (selective) dilution of the lattice
- [ ] Some sweeps can be skipped (not saving data) to both de-correlate data and allow for longer simulations
