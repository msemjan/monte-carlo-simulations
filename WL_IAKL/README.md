# Wang-Landau method and Metropolis algorithm

In this folder is an implementation of the Wang-Landau method and Metropolis algorithm for the Ising antiferromagnet on the kagome lattice in C++. 

## Running the code

### Manual compilation

You can use `g++` or `clang++`, both should work. The simulations can be compiled with:
```bash
# Wang-Landau method
g++ wl.cpp -o wl -D FINAL_F={final value of f} -D L={lattice size} -D FLATNESS_CRITERION={flatness criterion}

# Metropolis algorithm
g++ metropolis.cpp -o ma -D L={lattice size}
```

### Convenience script

Alternatively, you can use a Python 3 script for batch processing:
```bash
# Wang-Landau method
python3 launcher.py

# Metropolis algorithm
python3 launcher_ma.py
```
It will execute several runs of the simulation for various combinations of parameters. These can be changed by rewriting `numCopies`, `L`, `fFinal` and `flatnessCriterion` in `launcher.py` and `launcher_ma.py`. 

### Modifications of code

You may also need to tweak some settings to achieve desired results, such as changing temperature mesh by rewriting code in `metropolis.cpp`. Before running, definitely change the value of `SAVE_DIR` in `metropolis.cpp` and `wl.cpp` to the directory where you want to save the results.
