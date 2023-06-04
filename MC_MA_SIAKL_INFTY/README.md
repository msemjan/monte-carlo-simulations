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

### Using Makefile

1. Install the CUB library, and update the `makefile`
2. Modify the configuration file `includes/config.h` and save the changes
3. Compile with `make all` command
4. Run with `./bin/metropolis`

### Using Python launcher script

If you want to run several simulations in batch, it is preferable to use the `launcher.py` script. You need Python 3 and `chevron` package (Install with `pip install chevron`). To use the Python script:

1. Install the CUB library, Python and Chevron package
2. Modify the save directory (`dir` string at the line 102) in the configuration file template `includes/config.mo` and save the changes
3. Modify values of user parameters in the `launcher.py` if necessary
4. Launch the script with `python3 launcher.py`

## Loading data

There are scripts for loading data in `scripts` directory. 
- `loader.m` for loading time series and calculating thermal averages
- `load_lattice.m` for loading a snapshots
- `sum_layers.m` for drawing snapshots

The code is well commented and self-explanatory. Just make sure to set user parameters correctly and when loading data with `loader.m`, specify which folders should be processed by modifying the cell array in the `working_data.m`.

## Limitations

- Current implementation allows to simulate only $L=8$, $16$, $32$, $48$, and $64$
- The length of the time series is also limited

## TODO

- [ ] 'Linearize' the lattice
- [ ] Lattice can be allocated on heap
- [ ] Rewrite update kernels, using grid-stride loops
- [ ] Some sweeps can be skipped (not saving data) to both de-correlate data and allow for longer simulations
- [ ] Some modification can be made to use a smaller buffer for time series data, and copy data to host on a separate stream. Goal is to do this, while kernels are running (in non-blocking fashion). Then save the data to the disk. The advantage would be that the time series doesn't have to fit into the RAM of the GPU and a longer simulation can be executed
