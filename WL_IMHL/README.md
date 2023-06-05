# Wang-Landau method and Metropolis algorithm

In this folder is an implementation of the Wang-Landau method and Metropolis algorithm for the Ising antiferromagnet on the honeycomb lattice with next nearest neighbors in C++. This is a work in progress, and the code has to be modified to give correct results.

## Running the code

### Manual compilation

Before running, modify the user defined parameters as needed (mainly the `SAVE_DIR`, `J1`, `J2`, `L`, `FLATNESS_CRITERION`, and `FINAL_F`) and save the changes.

You can use `g++` or `clang++`, both should work. The simulations can be compiled with:
```bash
# Wang-Landau method
g++ ./src/wl_honeycomb.cpp -o wl_honeycomb.out

# Run with
./wl_honeycomb.out
```

It's also possible to use the `Makefile`:
```bash
# Compile
make all

# Run with
./wl_honeycomb.out
```

### Modifications of code

You may also need to tweak some settings to achieve desired results, such as changing temperature mesh by rewriting code in `metropolis.cpp`. Before running, definitely change the value of `SAVE_DIR` in `metropolis.cpp` and `wl.cpp` to the directory where you want to save the results.
