import os
import chevron

# User parameters
N       = 5                                      # Number of copies
Ls      = [8, 16, 32, 64]                        # Simulated lattice sizes
Deltas  = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # Simulated values of Delta
DRY_RUN = False                                  # Set to true to output commands to run without running them

# Flags, folders, etc.
OUTPUT_BINARY           = 'bin/metropolis'
CONFIG_TEMPLATE_FILE    = 'include/config.mo'
CONFIG_FILE             = "include/config.h"
INCLUDES                = "-I./include -I$HOME/cub"
FLAGS                   = "-std=c++11 -lm -lcurand"
SRC                     = "./src/metropolis.cu"
LIB                     = "-L./lib"

# Command for compiling and running the simulation
CMD = f"nvcc {SRC} {INCLUDES} {LIB} -o {OUTPUT_BINARY} {FLAGS}  && time ./{OUTPUT_BINARY}"

# Open the file with chevron template
with open(CONFIG_TEMPLATE_FILE, 'r') as f:
    CONFIG_TEMPLATE = f.read()

# Loops for all combinations of parameters
for DELTA in Deltas:
    for L in Ls:
        for _ in range(N):
            # Set configuration of the simulation
            config = {
                'J1': f'(-{DELTA})',
                'J2': f'(1-{DELTA})',
                'L': str(L),
                'NUM_THERM_SWEEPS': str(1 << 20),
                'NUM_SWEEPS': str(1 << 20),
                'NUM_TEMP': 40, 
                'MIN_TEMP': 0.15, 
                'MAX_TEMP': 0.25,
                'DELTA_TEMP': '(maxTemperature - minTemperature) / (double)numTemp',
                'INV_TEMP': '//',
                'EXP_TEMP': '//',
                'LIN_TEMP': '',
            }

            # Render the template
            rendered_config = chevron.render(CONFIG_TEMPLATE, config)

            # Print the configuration, if using dry run setting
            if DRY_RUN:
                print(rendered_config)
                print()

            # Write the configuration into a file, if using dry run setting (this will overwrite the current config.h!)
            if not DRY_RUN:
                with open(CONFIG_FILE, 'w') as f:
                    f.write(rendered_config)

            # Launch the simulation
            print(f'Launching simulation with L={L}')
            print(CMD)
            if not DRY_RUN:
                os.system(CMD)
