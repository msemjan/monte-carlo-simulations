/**
 * This file contains user-defined lunch parameters.
 *
 **/

// Include guard
#ifndef CUDA_MC_CONFIG_H_
#define CUDA_MC_CONFIG_H_

//#define DELTA (1/4.0)
//#define J1 (-DELTA)                         // Intralayer interaction coupling constant
//#define J2 (1-DELTA)                        // Interlayer interaction coupling constant
#define J1 {{J1}}                             // Intralayer interaction coupling constant
#define J2 {{J2}}                             // Interlayer interaction coupling constant
#define L {{L}}                               // Linear lattice size
#define LAYERS {{L}}                          // Number of interacting layers
#define N_XY (L * L)                          // Number of sublattice spins in one layer
#define N (L * L * LAYERS / 2)                // Number of sublattice spins
#define VOLUME (3 * 2 * N * 2)                // Total number of spins
#define LBLOCKS 8                             // Length of a block (x and y direction)
#define ZBLOCKS 1                             // Length of a block (z direction)
#define BBLOCKS 8                             // Block size 
#define RAND_N (32 * VOLUME)                  // Number of random numbers
#define field 0                               // External magnetic field
{{INV_TEMP}}#define USE_INVERSE_TEMPERATURE 1 // Use this for inverse temperature mesh
{{EXP_TEMP}}#define USE_EXP_TEMPERATURE 1     // Use this for exponentially decaying temperature mesh
{{LIN_TEMP}}#define USE_LINEAR_TEMPERATURE 1  // Use this for uniformly distributed temperature mesh
#define A 0.0001                              // Magic number - for calculation of temperature 
#define B 0.26                                // Magic number - for calculation of temperature 
#define SAVE_TS 1                             // Uncomment to save time series
#define SAVE_TEMPERATURES 1                   // Uncomment to save temperatures
//#define SAVE_MEANS 1                          // Uncomment to save mean values
#define SAVE_LAST_CONFIG 1                    // Uncomment to save the snapshot of the last configuration
// #define SAVE_INIT_CONFIG 1                   // Uncomment to save the snapshot of the initial configuration
#define PBC_Z 1                               // Periodic boundary conditions along the z-axis
#define NUM_STREAMS 8                         // Number of streams (DO NOT TOUCH)
#define CALCULATE_CHAIN_ORDER_PARA 1          // Calculate order parameter of linear chains along z-axis
#define CALCULATE_TRIANGLE_ORDER_PARA 1       // Calculate pseudo-order parameter of triangular plaquettes in xy-planes
#define NUM_CHAIN_ORDER_POINTS numTemp        // Length of the array for chain order parameter (DO NOT TOUCH)
//#define GLOBAL_SEED 42                        // Uncomment for a deterministic seed for RNG (Useful for debugging)
//#define USE_BOLTZ_TABLE 1                     // Uncomment to use texture memory table with Boltzmann factors (Not really applicable for S=\infty model
//#define DEBUG 1                               // Uncomment for debug messages
//#define SAVE_LATTICES 100                     // Define how often should lattice snapshots be saved (associated with a performance hit)
//#define CREATE_HEADERS 1                      // Uncomment to add headers into files with data

// Functions to transform a random float ~(0,1) to continuous spins ~(-1,1)
#define p1(x) ((mType)(2*x-1))
#define p2(x) ((mType)(2*x-1))
#define p3(x) ((mType)(2*x-1))
#define p4(x) ((mType)(2*x-1))
#define p5(x) ((mType)(2*x-1))
#define p6(x) ((mType)(2*x-1))

// Typedefs
typedef float spinType;                       // Type of spins
typedef float mType;                          // Magnetization and spins
typedef float chainType;                      // Type of chain order parameter
typedef double eType;                         // Energy
typedef double tType;                         // Temperature
typedef float rngType;                        // RNG generation - numbers

// NOTE: Values of const expressions
// (1<<18) =   262 144
// (1<<19) =   524 288
// (1<<20) = 1 048 576
// (1<<21) = 2 097 152
// (1<<22) = 4 194 304

// Parameters of the simulation
const unsigned int numThermalSweeps = {{NUM_THERM_SWEEPS}};   // Sweeps for thermalization (discarder)
const unsigned int numSweeps = {{NUM_SWEEPS}};                // Number of sweeps that are saved
const size_t numTemp = {{NUM_TEMP}};                          // Number of temperatures 
const tType minTemperature = {{MIN_TEMP}};                    // Minimal temperature
const tType maxTemperature = {{MAX_TEMP}};                    // Maximal temperature
const tType deltaTemperature = {{DELTA_TEMP}};                // Temperature step
const unsigned int boltzL = 2 * 5 * 3;                        // Number of unique Boltzmann factors

// Lunch specifications
dim3 DimBlock(L / LBLOCKS, L / LBLOCKS, LAYERS / (2 * ZBLOCKS));
dim3 DimGrid(LBLOCKS, LBLOCKS, ZBLOCKS);

dim3 DimBlockLinear(N / (LBLOCKS * LBLOCKS), 1, 1);
dim3 DimGridLinear(LBLOCKS * LBLOCKS, 1, 1);

dim3 DimBlockBase(L / BBLOCKS, L / BBLOCKS, 1);
dim3 DimGridBase(BBLOCKS, BBLOCKS, 1);
#define NUM_BLOCKS_BASE (BBLOCKS*BBLOCKS)
#define NUM_GRID_BASE (L*L/NUM_BLOCKS_BASE) 

// Global variables
// Texture memory
texture<float, cudaTextureType1D, cudaReadModeElementType> boltz_tex;


#ifndef USE_BOLTZ_TABLE
__constant__ double d_beta;
#endif
// Boltzman table - works for systems with spin number s = 1/2
std::vector<float> boltz(boltzL);
float *d_boltz;

std::string dir = "/path/to/save/directory/";

cudaStream_t streams[NUM_STREAMS];

#endif  // CUDA_MC_CONFIG_H_
