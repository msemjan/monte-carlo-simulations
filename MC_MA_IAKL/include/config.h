/**
 * This file contains user-defined lunch parameters.
 *
 **/

// Include guard
#ifndef CUDA_MC_CONFIG_H_
#define CUDA_MC_CONFIG_H_

#define J1 -1											// Interaction coupling constant
#define L 64                    	// Linear lattice size
#define N (L*L)                 	// Number of sublattice spins
#define VOLUME (3 * N)          	// Total number of spins
#define LBLOCKS 8               	// Lenght of a block
#define RAND_N (128 * N)        	// Number of random numbers
#define field 0                 	// External magnetic field
#define USE_INVERSE_TEMPERATURE 1 // Comment out to use a uniform step
																	// temperature interval, otherwise an
																	// exponential function with increased
																	// density at low temperatures is used
#define A 0.001										// magic number
#define B 0.205										// another magic number
																	
#define SAVE_DIR "/path/to/save/directory"; // save directory
#define SAVE_TS 1														// uncomment to save time series
#define SAVE_TEMPERATURES 1									// save temperature in a separate file
#define SAVE_MEANS 1												// save mean values
#define SAVE_LAST_CONFIG 1									// save a snapshot of the last configuration
#define DILUTION 0.90												// set a dilution
#define NUM_STREAMS 4												// number of streams, DO NOT TOUCH this!
//#define USE_BOLTZ_TABLE 1									  // use texture memory to save Boltmann factors
//#define DEBUG 1															// write debug messages
//#define SAVE_LATTICES 1											// save snapshots of the system regularly
//#define CREATE_HEADERS 1										// create headers in save files

// Typedefs
typedef int mType;              // Magnetization and spins
typedef double eType;           // Energy
typedef double tType;           // Temperature
typedef float rngType;          // RNG generation - numbers
typedef curandStatePhilox4_32_10_t generatorType;
                                // RNG generation - generator
// NOTE: Values of const expressions
// (1<<18) =   262 144
// (1<<19) =   524 288
// (1<<20) = 1 048 576
// (1<<21) = 2 097 152
// (1<<22) = 4 194 304

// Parameters of the simulation
const unsigned int numThermalSweeps = 1<<19;   // Number of sweeps for thermalization
const unsigned int numSweeps        = 1<<20;   // Number of sweeps that are saved to a file
const tType minTemperature          = 0.0;		 // Minimal temperature - used only when USE_INVERSE_TEMPERATURE is not set
const tType maxTemperature          = 3.0;		 // Maximal temperature	
const tType deltaTemperature        = 0.6;		 // Temperature step	
const size_t numTemp                = 60;			 // Number of temperatures - ignored when USE_INVERSE_TEMPERATURE is not set
const unsigned int boltzL           = 2 * 5;   // Number of unique Boltzman factors

// Lunch specifications
dim3 DimBlock((L * L) / (LBLOCKS * LBLOCKS), 1, 1);
dim3 DimGrid(1 ,LBLOCKS * LBLOCKS, 1);
// dim3 DimBlock(L / LBLOCKS, L / LBLOCKS, 1);
// dim3 DimGrid(LBLOCKS, LBLOCKS, 1);
dim3 DimBlockLinear(L*L / (LBLOCKS*LBLOCKS) , 1, 1);
dim3 DimGridLinear(LBLOCKS*LBLOCKS, 1, 1);

// Global variables
// Texture memory
texture<float, cudaTextureType1D, cudaReadModeElementType> boltz_tex;

#ifndef USE_BOLTZ_TABLE
__constant__ double d_beta;
#else
// Boltzman table - works for systems with spin number s = 1/2
std::vector<float> boltz( boltzL );
float *d_boltz;
#endif

cudaStream_t streams[NUM_STREAMS];

#endif // CUDA_MC_CONFIG_H_
