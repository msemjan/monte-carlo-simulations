/**
 * This file contains user-defined lunch parameters.
 *
 **/

// Include guard
#ifndef CUDA_MC_CONFIG_H_
#define CUDA_MC_CONFIG_H_
#include <cmath>

//#define DELTA (1/4.0)
//#define J1 (-DELTA)               // Intralayer interaction coupling constant
//#define J2 (1-DELTA)              // Interlayer interaction coupling constant
#define J1 {{J1}}                  // Intralayer interaction coupling constant
#define J2 {{J2}}           // Interlayer interaction coupling constant
#define L {{L}}                  // Linear lattice size
#define LAYERS {{L}}             // Number of interacting layers
#define N_XY (L * L)            // Number of sublattice spins in one layer
#define N (L * L * LAYERS / 2)  // Number of sublattice spins
#define VOLUME (3 * 2 * N * 2)  // Total number of spins
#define SPIN {{SPIN}} 
#define LBLOCKS 8               // Lenght of a block (x and y direction)
#define ZBLOCKS 1               // Lenght of a block (z direction)
// #define BBLOCKS 16
#define BBLOCKS 8  
#define RAND_N (32 * VOLUME)   // Number of random numbers
#define field 0                 // External magnetic field
{{INV_TEMP}}#define USE_INVERSE_TEMPERATURE 1
{{EXP_TEMP}}#define USE_EXP_TEMPERATURE 1
{{LIN_TEMP}}#define USE_LINEAR_TEMPERATURE 1
// #define A 0.00005
// #define B 0.12
#define A 0.0001
#define B 0.26
#define SAVE_TS 1
#define SAVE_TEMPERATURES 1
// #define SAVE_MEANS 1
#define SAVE_LAST_CONFIG 1
#define SAVE_INIT_CONFIG 1
#define PBC_Z 1  // Periodic boundary conditions along the z-axis
#define NUM_STREAMS 8
#define CALCULATE_CHAIN_ORDER_PARA 1
#define CALCULATE_TRIANGLE_ORDER_PARA 1
#define NUM_CHAIN_ORDER_POINTS numTemp
// #define GLOBAL_SEED 42
// #define USE_BOLTZ_TABLE 1
//#define DEBUG 1
// #define SAVE_LATTICES 100 // 10000
// #define CREATE_HEADERS 1

// Functions to transform a random float ~(0,1) to spins
#define MULTIPLICITY (float)( ceil(2 * SPIN + 1) )
#define p1(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))
#define p2(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))
#define p3(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))
#define p4(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))
#define p5(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))
#define p6(x) ((mType)((float)( - SPIN + (int)(MULTIPLICITY * x - 0.001)) / SPIN ))

// Typedefs
typedef float spinType;
typedef float mType;      // Magnetization and spins
typedef float chainType;
typedef double eType;   // Energy
typedef double tType;   // Temperature
typedef float rngType;  // RNG generation - numbers
// typedef curandStatePhilox4_32_10_t generatorType;
// RNG generation - generator
// NOTE: Values of const expressions
// (1<<18) =   262 144
// (1<<19) =   524 288
// (1<<20) = 1 048 576
// (1<<21) = 2 097 152
// (1<<22) = 4 194 304

// Parameters of the simulation
const unsigned int numThermalSweeps = {{NUM_THERM_SWEEPS}};  // Sweeps for thermalization
const unsigned int numSweeps = {{NUM_SWEEPS}};         // Number of sweeps
const size_t numTemp = {{NUM_TEMP}};              // 20
const tType minTemperature = {{MIN_TEMP}};
const tType maxTemperature = {{MAX_TEMP}};
const tType deltaTemperature = {{DELTA_TEMP}};
const unsigned int boltzL = 2 * 5 * 3;  // # of unique Boltzman factors

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
// #else
// Boltzman table - works for systems with spin number s = 1/2
std::vector<float> boltz(boltzL);
float *d_boltz;

std::string dir = "/media/semjan/DATA/SIAKL_GEN_S_cuda/";
// std::string dir = "/run/media/tumaak/SAMSUNG/DATA/SIAKL_spoj_cuda/";

cudaStream_t streams[NUM_STREAMS];

#endif  // CUDA_MC_CONFIG_H_
