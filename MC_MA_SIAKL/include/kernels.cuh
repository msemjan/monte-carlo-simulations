#ifndef CUDA_MC_KERNELS_CUH_
#define CUDA_MC_KERNELS_CUH_

// C/C++ imports
#include <algorithm>
#include <chrono>  // measuring of the time
#include <cmath>
#include <cstdio>  // fread, fwrite, fprintf
#include <cstdlib>
#include <ctime>     // time
#include <fstream>   // C++ type-safe files
#include <iomanip>   // std::put_time
#include <iomanip>   // std::setprecision
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
#define zUP(z) ((z != LAYERS / 2 - 1) * (z + 1))
#define zDOWN(z) ((z == 0) * (LAYERS / 2 - 1) + (z != 0) * (z - 1))

#ifdef USE_BOLTZ_TABLE
  #define get_boltzmann_factor(s, sumNN, sumNL) (tex1Dfetch( boltz_tex, (s + 1) / 2 + (4 + sumNN) + (2 + sumNL)*5 ))
#else    
  #define get_boltzmann_factor(s, sumNN, sumNL) (exp( -d_beta * 2 * s * ( J1 * sumNN + J2* sumNL + field)))
#endif

// Calculate energy
__global__ void energyCalculation(Lattice* d_s) {
  unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned short z = blockDim.z * blockIdx.z + threadIdx.z;

  d_s->exchangeEnergy[N_XY * z + x + L * y] =
      (-1 * (  // Within layer
                J1 * (eType)d_s->s1[N_XY * z + x + L * y] *
                    ((eType)d_s->s2[N_XY * z + L * y + x] +
                     (eType)d_s->s3[N_XY * z + L * y + x] +
                     (eType)d_s->s3[N_XY * z + L * DOWN(y) + x]) +
                J1 * (eType)d_s->s2[N_XY * z + x + L * y] *
                    ((eType)d_s->s3[N_XY * z + L * y + x] +
                     (eType)d_s->s3[N_XY * z + L * DOWN(y) + UP(x)] +
                     (eType)d_s->s1[N_XY * z + L * y + UP(x)])
                // Within other layer
                + J1 * (eType)d_s->s4[N_XY * z + x + L * y] *
                      ((eType)d_s->s5[N_XY * z + L * y + x] +
                       (eType)d_s->s6[N_XY * z + L * y + x] +
                       (eType)d_s->s6[N_XY * z + L * DOWN(y) + x]) +
                J1 * (eType)d_s->s5[N_XY * z + x + L * y] *
                    ((eType)d_s->s6[N_XY * z + L * y + x] +
                     (eType)d_s->s6[N_XY * z + L * DOWN(y) + UP(x)] +
                     (eType)d_s->s4[N_XY * z + L * y + UP(x)])
                // Betweeen layers
                + J2 * (eType)(d_s->s1[N_XY * z + x + L * y] *
                                   d_s->s4[N_XY * z + x + L * y] +
                               d_s->s2[N_XY * z + x + L * y] *
                                   d_s->s5[N_XY * z + x + L * y] +
                               d_s->s3[N_XY * z + x + L * y] *
                                   d_s->s6[N_XY * z + x + L * y]

                               + d_s->s1[N_XY * zUP(z) + x + L * y] *
                                     d_s->s4[N_XY * z + x + L * y] +
                               d_s->s2[N_XY * zUP(z) + x + L * y] *
                                   d_s->s5[N_XY * z + x + L * y] +
                               d_s->s3[N_XY * zUP(z) + x + L * y] *
                                   d_s->s6[N_XY * z + x + L * y])));
}

// Updates of sublattices
__global__ void update1(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  eType sumNN, sumNL;
  mType s1;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s1 = s->s1[N_XY * z + L * y + x];
  sumNN = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * z + L * y + DOWN(x)] +
          s->s3[N_XY * z + L * y + x] + s->s3[N_XY * z + L * DOWN(y) + x];
  sumNL = s->s4[N_XY * z + L * y + x] + s->s4[N_XY * zDOWN(z) + L * y + x];

  s1 = s1 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s1, sumNN, sumNL)));
  s->s1[N_XY * z + L * y + x] = s1;
}

__global__ void update2(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s2;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s2 = s->s2[L * y + x];
  sumNN = s->s1[L * y + x] + s->s1[L * y + UP(x)] + s->s3[L * y + x] +
          s->s3[L * DOWN(y) + UP(x)];
  sumNL = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * zDOWN(z) + L * y + x];

  s2 = s2 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s2, sumNN, sumNL)));
  s->s2[N_XY * z + L * y + x] = s2;
}

__global__ void update3(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s3;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s3 = s->s3[L * y + x];
  sumNN = s->s1[L * y + x] + s->s1[L * UP(y) + x] + s->s2[L * y + x] +
          s->s2[L * UP(y) + DOWN(x)];
  sumNL = s->s6[N_XY * z + L * y + x] + s->s6[N_XY * zDOWN(z) + L * y + x];

  s3 = s3 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s3, sumNN, sumNL)));
  s->s3[N_XY * z + L * y + x] = s3;
}

__global__ void update4(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  eType sumNN, sumNL;
  mType s4;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;
  
  s4 = s->s4[N_XY * z + L * y + x];
  sumNN = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * z + L * y + DOWN(x)] +
          s->s6[N_XY * z + L * y + x] + s->s6[N_XY * z + L * DOWN(y) + x];
  sumNL = s->s1[N_XY * z + L * y + x] + s->s1[N_XY * zUP(z) + L * y + x];

  s4 = s4 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s4, sumNN, sumNL)));
  s->s4[N_XY * z + L * y + x] = s4;
}

__global__ void update5(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s5;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;


  s5 = s->s5[L * y + x];
  sumNN = s->s4[L * y + x] + s->s4[L * y + UP(x)] + s->s6[L * y + x] +
          s->s6[L * DOWN(y) + UP(x)];
  sumNL = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * zUP(z) + L * y + x];

  s5 = s5 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s5, sumNN, sumNL)));
  s->s5[N_XY * z + L * y + x] = s5;
}

__global__ void update6(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s6;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s6 = s->s6[L * y + x];

  sumNN = s->s4[L * y + x] + s->s4[L * UP(y) + x] + s->s5[L * y + x] +
          s->s5[L * UP(y) + DOWN(x)];
  sumNL = s->s3[N_XY * z + L * y + x] + s->s3[N_XY * zUP(z) + L * y + x];

  s6 = s6 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s6, sumNN, sumNL)));
  s->s6[N_XY * z + L * y + x] = s6;
}

void update(Lattice* s, rngType* numbers, unsigned int offset) {
  update1<<<DimBlock, DimGrid>>>(s, numbers, 0 + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update2<<<DimBlock, DimGrid>>>(s, numbers, 1 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update3<<<DimBlock, DimGrid>>>(s, numbers, 2 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update4<<<DimBlock, DimGrid>>>(s, numbers, 3 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update5<<<DimBlock, DimGrid>>>(s, numbers, 4 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update6<<<DimBlock, DimGrid>>>(s, numbers, 5 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
}

#endif
