'''
This script compiles the Metropolis algorithm implemented in C++ (file 
metropolis.cpp) for various of macros using g++ compiler and then runs it. It
can be used to execute the simulation for different lattice sizes a specified
number of times.

===============================================================================
                            THE BEERWARE LICENCE:

This code is supplied as is. Author guarantees nothing. Be smart and use 
at your own risk. As long as you retain this notice you can do whatever you 
want with this stuff. If we meet some day, and you think this stuff is worth 
it, you can buy me a beer in return.

===============================================================================
Made by <marek.semjan@student.upjs.sk>.
'''

import os

# launching parameters
numCopies = 10
L = [16, 32, 68, 76, 84]

for i in range(numCopies):
    for l in L:
        # prepare a command
        cmdStr = f"g++ metropolis.cpp -o ma -D L={l}"
        
        # compilation
        print(cmdStr)
        os.system(cmdStr)

        # runing compiled code
        os.system(f"echo \"{cmdStr}\" >> out.txt")
        os.system("time ./ma >> out.txt")
