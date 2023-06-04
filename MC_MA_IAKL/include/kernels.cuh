#ifndef MC_MA_KERNELS_CUH_
#define MC_MA_KERNELS_CUH_

// C/C++ imports
#include <algorithm>
#include <chrono>  // measuring of the time
#include <cmath>
#include <cstdio>  // fread, fwrite, fprintf
#include <cstdlib>
#include <ctime>     // time
#include <fstream>   // C++ type-safe files
#include <iomanip>   // std::setprecision, std::put_time
#include <iostream>  // cin, cout
#include <limits>    // std::numeric_limits
#include <numeric>   // accumulate
#include <random>
#include <sstream>  // for higher precision to_string
#include <string>   // for work with file names
#include <vector>   // data container

// CUB
#include <cub/cub.cuh>  // For parallel reductions

// CUDA specific imports
#include <cuda.h>    // CUDA header
#include <curand.h>  // Parallel Random Number Generators

//#define WIN_OS 1
#define LINUX_OS 1

// Auxiliary header files
#include "auxiliary_functions.h"  // Various Helper functions
#include "fileWrapper.h"
#include "safe_cuda_macros.cuh"
#include "systemSpecific.h"

// Simulation related imports
#include "Quantities.cuh"
#include "config.h"

#define UP(x) ((x!=L-1)*(x+1))
#define DOWN(x) ((x==0)*(L-1) + (x!=0)*(x-1))

__device__ float get_boltzmann_factor(mType s, mType sumNN) {
  #ifdef USE_BOLTZ_TABLE
    return tex1Dfetch( boltz_tex, (s + 1) / 2 + 4 + sumNN );
  #else
    return (float)exp( -d_beta * 2 * s * ( J1 * sumNN + field));
  #endif

}

// Calculates the local energy of lattice
__global__ void energyCalculation(Lattice* d_s) {
    // Thread identification
    unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;

    // Calculation of energy
    d_s->exchangeEnergy[x + L * y] = (-1
            * ( J1 * (eType) d_s->s1[x + L * y]
                            * ( (eType) d_s->s2[x + L * y]
                              + (eType) d_s->s3[x + L * y]
                              + (eType) (d_s->s3[x + DOWN(y) * L]) )
              + J1 * (eType) d_s->s2[x + L * y]
                            * ( (eType) d_s->s3[x + L * y]
                              + (eType) (d_s->s3[UP(x) + DOWN(y) * L])
                              + (eType) (d_s->s1[UP(x) + y * L]))));
}

// Tries to flip each spin of the sublattice 1
__global__ void update1( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
    unsigned short x, y; //, xD, yD;
    // double p;
    mType sumNN;
    mType s1;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    s1    = s->s1[L * y  + x ];
    sumNN = s->s2[L * y  + x ]
          + s->s2[L * y  + DOWN(x)]
          + s->s3[L * y  + x ]
          + s->s3[L * DOWN(y) + x ];

    s->s1[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<get_boltzmann_factor(s1, sumNN)));
}

// Tries to flip each spin of the sublattice 2
__global__ void update2( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
	unsigned short x, y; //, xU, yD;
	// double p;
    mType s2;
    mType sumNN;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    s2    = s->s2[L * y  + x ];
    sumNN = s->s1[L * y  + x ]
          + s->s1[L * y  + UP(x)]
          + s->s3[L * y  + x ]
          + s->s3[L * DOWN(y) + UP(x)];

    s->s2[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<get_boltzmann_factor(s2, sumNN)));
}

// Tries to flip each spin of the sublattice 3
__global__ void update3( Lattice* s
                       , rngType* numbers
                       , unsigned int offset)
{
	unsigned short x, y; //, xD, yU;
	// double p;
    mType s3;
    mType sumNN;

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    s3    = s->s3[L * y  + x ];
    sumNN = s->s1[L * y  + x ]
          + s->s1[L * UP(y) + x ]
          + s->s2[L * y  + x ]
          + s->s2[L * UP(y) + DOWN(x)];

    s->s3[L * y + x] *= 1 - 2*((mType)(numbers[offset + L*y+x]<get_boltzmann_factor(s3, sumNN)));
}

// Updates all sublattices
void update( Lattice* s
           , rngType* numbers
           , unsigned int offset)
{
    update1<<<DimBlock,DimGrid>>>( s, numbers,   0 + offset );
    CUDAErrChk(cudaPeekAtLastError());

    update2<<<DimBlock,DimGrid>>>( s, numbers,   N + offset );
    CUDAErrChk(cudaPeekAtLastError());

    update3<<<DimBlock,DimGrid>>>( s, numbers, 2*N + offset );
    CUDAErrChk(cudaPeekAtLastError());
}

#endif
