'''
This script compiles the Wang-Landau algorithm implemented in C++ (file 
wl.cpp)  for various of macros using clang++ compiler and then runs it.

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

# lunching parameters
numCopies = 10
L = [16, 32, 68, 76, 84]
fFinal = [1.000000001, 1.0000000001, 1.00000000001]

flatnessCriterion = [0.8, 0.9]

for i in range(numCopies):
    for flatness in flatnessCriterion:
        for l in L:
            for f in fFinal:
                # prepare a command
                cmdStr = f"g++ wl.cpp -o wl -D FINAL_F={f:2.15f} -D L={l} -D FLATNESS_CRITERION={flatness}"
                
                # compilation
                print(cmdStr)
                os.system(cmdStr)

                # runing compiled code
                os.system(f"echo \"{cmdStr}\" # >> out.txt")
                os.system("time ./rewl >> out.txt")
