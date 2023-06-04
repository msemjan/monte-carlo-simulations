import os
import chevron


DRY_RUN = False
OUTPUT_BINARY = 'bin/metropolis'
CONFIG_TEMPLATE_FILE = 'include/config.mo'
CONFIG_FILE = "include/config.h"
INCLUDES = "-I./include -I$HOME/cub"
FLAGS = "-std=c++11 -lm -lcurand"
SRC = "./src/metropolis.cu"
LIB = "-L./lib"
CMD = f"nvcc {SRC} {INCLUDES} {LIB} -o {OUTPUT_BINARY} {FLAGS}  && time ./{OUTPUT_BINARY}"

with open(CONFIG_TEMPLATE_FILE, 'r') as f:
    CONFIG_TEMPLATE = f.read()

#  L = 64
N = 1
#DELTA = 0.5
SPIN = "(3.0/2.0)"

# Num sweeps
# (1<<18) =   262 144
# (1<<19) =   524 288
# (1<<20) = 1 048 576
# (1<<21) = 2 097 152
# (1<<22) = 4 194 304

#  for L in [64, 48, 32, 16]:
#  for L in [32]:
#  L = 48
#  for  DELTA in [0.75]:
DELTA = 0.75
for L in [16]:
    for _ in range(N):
        config = {
            'J1': f'(-{DELTA})',
            'J2': f'(1-{DELTA})',
            'L': str(L),
            'NUM_THERM_SWEEPS': str(1 << 20),
            'NUM_SWEEPS': str(1 << 20),
            #  'NUM_TEMP': '100',
            #  'MIN_TEMP': '0.16',
            #  'MAX_TEMP': '0.22',
            #  'NUM_TEMP': '50',
            #  'MIN_TEMP': '0.1800', # Delta = 0.75
            #  'MAX_TEMP': '0.2000', # Delta = 0.75
            #  'MIN_TEMP': '0.2200', # Delta = 0.5
            #  'MAX_TEMP': '0.2400', # Delta = 0.5
            'NUM_TEMP': 40,     # Delta = 0.75, better data 
            #  'MIN_TEMP': 0.15, # Delta = 0.75, better data
            #  'MAX_TEMP': 0.25,   # Delta = 0.75, better data
            'MIN_TEMP': 0.005, # Delta = 0.75, better data 
            'MAX_TEMP': 2.0,   # Delta = 0.75, better data 
            'DELTA_TEMP': '(maxTemperature - minTemperature) / (double)numTemp',
            'INV_TEMP': '//',
            'EXP_TEMP': '//',
            'LIN_TEMP': '',
            'SPIN': SPIN,
        }
        rendered_config = chevron.render(CONFIG_TEMPLATE, config)
        if DRY_RUN:
            print(rendered_config)
            print()

        if not DRY_RUN:
            with open(CONFIG_FILE, 'w') as f:
                f.write(rendered_config)

        print(f'Launching simulation with L={L}')
        print(CMD)
        if not DRY_RUN:
            os.system(CMD)
