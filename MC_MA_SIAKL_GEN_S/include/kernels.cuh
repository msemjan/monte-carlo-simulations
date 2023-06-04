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
// #include <curand.h>  // Parallel Random Number Generators

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

// __device__  __forceinline__ float get_boltzmann_factor(mType s, mType sumNN, mType sumNL) {
//   #ifdef USE_BOLTZ_TABLE
//     return tex1Dfetch( boltz_tex, (s + 1) / 2 + (4 + sumNN) + (2 + sumNL)*5 );
//   #else
//     return exp( -d_beta * 2 * s * ( J1 * sumNN + J2* sumNL + field));
//   #endif
//
// }

// __global__ void energyCalculation(Lattice* d_s) {
//   unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
//   unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;
//   unsigned short z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   d_s->exchangeEnergy[N_XY * z + x + L * y] =
//       (-1 * (  // Within layer
//                 J1 * (eType)d_s->s1[N_XY * z + x + L * y] *
//                     ((eType)d_s->s2[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * DOWN(y) + x]) +
//                 J1 * (eType)d_s->s2[N_XY * z + x + L * y] *
//                     ((eType)d_s->s3[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * DOWN(y) + UP(x)] +
//                      (eType)d_s->s1[N_XY * z + L * y + UP(x)])
//                 // Within other layer
//                 + J1 * (eType)d_s->s4[N_XY * z + x + L * y] *
//                       ((eType)d_s->s5[N_XY * z + L * y + x] +
//                        (eType)d_s->s6[N_XY * z + L * y + x] +
//                        (eType)d_s->s6[N_XY * z + L * DOWN(y) + x]) +
//                 J1 * (eType)d_s->s5[N_XY * z + x + L * y] *
//                     ((eType)d_s->s6[N_XY * z + L * y + x] +
//                      (eType)d_s->s6[N_XY * z + L * DOWN(y) + UP(x)] +
//                      (eType)d_s->s4[N_XY * z + L * y + UP(x)])
//                 // Betweeen layers
//                 + J2 * (eType)(d_s->s1[N_XY * z + x + L * y] * d_s->s4[N_XY * z + x + L * y] +
//                                d_s->s2[N_XY * z + x + L * y] * d_s->s5[N_XY * z + x + L * y] +
//                                d_s->s3[N_XY * z + x + L * y] * d_s->s6[N_XY * z + x + L * y]
//
//                                + d_s->s1[N_XY * zUP(z) + x + L * y] * d_s->s4[N_XY * z + x + L * y] +
//                                d_s->s2[N_XY * zUP(z) + x + L * y] * d_s->s5[N_XY * z + x + L * y] +
//                                d_s->s3[N_XY * zUP(z) + x + L * y] * d_s->s6[N_XY * z + x + L * y])));
//   // d_s->exchangeEnergy[x + L * y] = (-1
//   //         * (J1 * (eType) d_s->s1[x + L * y]
//   //                 * ( (eType) d_s->s2[x + L * y]
//   //                   + (eType) d_s->s3[x + L * y]
//   //                   + (eType) (d_s->s3[x + DOWN(y) * L]) )
//   //                 +
//   //                 J1 * (eType) d_s->s2[x + L * y]
//   //                         * ((eType) d_s->s3[x + L * y]
//   //                            + (eType) (d_s->s3[UP(x) + DOWN(y) * L])
//   //                            + (eType) (d_s->s1[UP(x) + y * L]))));
// }

__global__ void energyCalculationAndReduce(Lattice* d_s, eType* blockSum, eType* blockSumChain, mType* blockSumTriangle) {
  unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;

  eType exchangeEnergy = 0;
  eType exchangeEnergyChain = 0;

  #ifdef CALCULATE_TRIANGLE_ORDER_PARA
      mType triangleOrder = 0;
  #endif

  for(unsigned short z = 0; z < LAYERS/2; z++){
    exchangeEnergy +=
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
              ));
    exchangeEnergyChain += (-1 * (
               // Betweeen layers
               + J2 * (eType)(d_s->s1[N_XY * z + x + L * y] * d_s->s4[N_XY * z + x + L * y] +
                 d_s->s2[N_XY * z + x + L * y] * d_s->s5[N_XY * z + x + L * y] +
                 d_s->s3[N_XY * z + x + L * y] * d_s->s6[N_XY * z + x + L * y]

                 + d_s->s1[N_XY * zUP(z) + x + L * y] * d_s->s4[N_XY * z + x + L * y] +
                 d_s->s2[N_XY * zUP(z) + x + L * y] * d_s->s5[N_XY * z + x + L * y] +
                 d_s->s3[N_XY * zUP(z) + x + L * y] * d_s->s6[N_XY * z + x + L * y])));

    #ifdef CALCULATE_TRIANGLE_ORDER_PARA
    triangleOrder += abs( d_s->s1[N_XY * z + L * y + x ] 
                        + d_s->s2[N_XY * z + L * y + x ] 
                        + d_s->s3[N_XY * z + L * y + x ] )
                   + abs( d_s->s4[N_XY * z + L * y + x ] 
                        + d_s->s5[N_XY * z + L * y + x ] 
                        + d_s->s6[N_XY * z + L * y + x ] )
                   + abs( d_s->s1[N_XY * z + L * y + UP(x) ] 
                        + d_s->s2[N_XY * z + L * DOWN(y) + UP(x) ] 
                        + d_s->s3[N_XY * z + L * y + x] )
                   + abs( d_s->s4[N_XY * z + L * y + UP(x)] 
                        + d_s->s5[N_XY * z + L * DOWN(y) + UP(x)] 
                        + d_s->s6[N_XY * z + L * y + x] );
    #endif


  }

  {
    typedef cub::BlockReduce<eType, BBLOCKS , cub::BLOCK_REDUCE_RAKING, BBLOCKS> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    eType sum = BlockReduce(temp_storage).Sum(exchangeEnergy);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0){
      blockSum[blockIdx.y*gridDim.x + blockIdx.x] = sum;
    }
  }

  {
    typedef cub::BlockReduce<eType, BBLOCKS , cub::BLOCK_REDUCE_RAKING, BBLOCKS> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    eType sum = BlockReduce(temp_storage).Sum(exchangeEnergyChain);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0){
      blockSumChain[blockIdx.y*gridDim.x + blockIdx.x] = sum;
    }
  }

  #ifdef CALCULATE_TRIANGLE_ORDER_PARA
  {
    typedef cub::BlockReduce<mType, BBLOCKS , cub::BLOCK_REDUCE_RAKING, BBLOCKS> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    mType sum = BlockReduce(temp_storage).Sum(triangleOrder);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0){
      blockSumTriangle[blockIdx.y*gridDim.x + blockIdx.x] = sum;
    }
  }
      
  #endif
}

#ifdef CALCULATE_CHAIN_ORDER_PARA
__global__ void chainOrderKernel(Lattice *d_s, mType* blockSum){
  unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;
  mType chainOrder1 = 0;
  mType chainOrder2 = 0;
  mType chainOrder3 = 0;
  
  for(unsigned short z = 0; z < LAYERS/2; z++){
    chainOrder1 += d_s->s1[N_XY * z + L * y + x] + d_s->s4[N_XY * z + L * y + x];
    chainOrder2 += d_s->s2[N_XY * z + L * y + x] + d_s->s5[N_XY * z + L * y + x];
    chainOrder3 += d_s->s3[N_XY * z + L * y + x] + d_s->s6[N_XY * z + L * y + x];
  }
  
  typedef cub::BlockReduce<mType, BBLOCKS , cub::BLOCK_REDUCE_RAKING, BBLOCKS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  mType sum = BlockReduce(temp_storage).Sum(abs(chainOrder1) + abs(chainOrder2) + abs(chainOrder3));
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0){
    blockSum[blockIdx.y*gridDim.x + blockIdx.x] = sum;
  }
}
#endif

__global__ void update1(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  eType sumNN, sumNL;
  mType s1, s1_new;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s1 = s->s1[N_XY * z + L * y + x];
  s1_new = p1(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * z + L * y + DOWN(x)] +
    s->s3[N_XY * z + L * y + x] + s->s3[N_XY * z + L * DOWN(y) + x];
  sumNL = s->s4[N_XY * z + L * y + x] + s->s4[N_XY * zDOWN(z) + L * y + x];

  // s1 = s1 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s1, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s1-s1_new)/2.0, sumNN, sumNL);
  s1 = trial*s1_new+(!trial)*s1;
  s->s1[N_XY * z + L * y + x] = s1;
  // s->s1[N_XY * z + L * y + x] = ( 1 - 2* (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s1, sumNN, sumNL)));
}

__global__ void update2(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s2, s2_new;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s2 = s->s2[N_XY * z + L * y + x];
  s2_new = p2(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s1[N_XY * z + L * y + x] + s->s1[N_XY * z + L * y + UP(x)] +
    s->s3[N_XY * z + L * y + x] + s->s3[N_XY * z + L * DOWN(y) + UP(x)];
  sumNL = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * zDOWN(z) + L * y + x];

  // s2 *=
  //     1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s2, sumNN, sumNL)));
  // s->s2[N_XY * z + L * y + x] = s2 * (1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s2, sumNN, sumNL)));
  // s2 = s2 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s2, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s2-s2_new)/2.0, sumNN, sumNL);
  s2 = trial*s2_new+(!trial)*s2;
  s->s2[N_XY * z + L * y + x] = s2;
}

__global__ void update3(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s3, s3_new;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s3 = s->s3[N_XY * z + L * y + x];
  s3_new = p3(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s1[N_XY * z + L * y + x] + s->s1[N_XY * z + L * UP(y) + x] +
    s->s2[N_XY * z + L * y + x] + s->s2[N_XY * z + L * UP(y) + DOWN(x)];
  sumNL = s->s6[N_XY * z + L * y + x] + s->s6[N_XY * zDOWN(z) + L * y + x];

  // s3 *=
  //     1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s3, sumNN, sumNL)));
  // s->s3[N_XY * z + L * y + x] = s3 * (1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s3, sumNN, sumNL)));
  // s3 = s3 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s3, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s3-s3_new)/2.0, sumNN, sumNL);
  s3 = trial*s3_new+(!trial)*s3;
  s->s3[N_XY * z + L * y + x] = s3;
}

__global__ void update4(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  eType sumNN, sumNL;
  mType s4, s4_new;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s4 = s->s4[N_XY * z + L * y + x];
  s4_new = p4(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * z + L * y + DOWN(x)] +
    s->s6[N_XY * z + L * y + x] + s->s6[N_XY * z + L * DOWN(y) + x];
  sumNL = s->s1[N_XY * z + L * y + x] + s->s1[N_XY * zUP(z) + L * y + x];

  // s4 *=
  //     1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s4, sumNN, sumNL)));
  // s->s4[N_XY * z + L * y + x] = s4 * (1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s4, sumNN, sumNL)));

  // s4 = s4 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s4, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s4-s4_new)/2.0, sumNN, sumNL);
  s4 = trial*s4_new+(!trial)*s4;
  s->s4[N_XY * z + L * y + x] = s4;
}

__global__ void update5(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s5, s5_new;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;


  s5 = s->s5[N_XY * z + L * y + x];
  s5_new = p5(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s4[N_XY * z + L * y + x] + s->s4[N_XY * z + L * y + UP(x)] +
    s->s6[N_XY * z + L * y + x] + s->s6[N_XY * z + L * DOWN(y) + UP(x)];
  sumNL = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * zUP(z) + L * y + x];

  // s5 *=
  //     1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s5, sumNN, sumNL)));
  // s->s5[N_XY * z + L * y + x] = s5 * (1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s5, sumNN, sumNL)));

  // s5 = s5 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s5, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s5-s5_new)/2.0, sumNN, sumNL);
  s5 = trial*s5_new+(!trial)*s5;
  s->s5[N_XY * z + L * y + x] = s5;
}

__global__ void update6(Lattice* s, rngType* numbers, unsigned int offset) {
  unsigned short x, y, z;
  mType s6, s6_new;
  eType sumNN, sumNL;

  x = blockDim.x * blockIdx.x + threadIdx.x;
  y = blockDim.y * blockIdx.y + threadIdx.y;
  z = blockDim.z * blockIdx.z + threadIdx.z;

  s6 = s->s6[N_XY * z + L * y + x];
  s6_new = p6(numbers[offset + N_XY * z + L * y + x + N]);

  sumNN = s->s4[N_XY * z + L * y + x] + s->s4[N_XY * z + L * UP(y) + x] +
    s->s5[N_XY * z + L * y + x] + s->s5[N_XY * z + L * UP(y) + DOWN(x)];
  sumNL = s->s3[N_XY * z + L * y + x] + s->s3[N_XY * zUP(z) + L * y + x];

  // s6 *=
  //     1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s6, sumNN, sumNL)));
  // s->s6[N_XY * z + L * y + x] = s6 * (1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s6, sumNN, sumNL)));

  // s6 = s6 * ( 1 - 2 * (numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor(s6, sumNN, sumNL)));
  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s6-s6_new)/2.0, sumNN, sumNL);
  s6 = trial*s6_new+(!trial)*s6;
  s->s6[N_XY * z + L * y + x] = s6;
}

void update(Lattice* s, rngType* numbers, unsigned int offset) {
  update1<<<DimBlock, DimGrid>>>(s, numbers,  0     + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update2<<<DimBlock, DimGrid>>>(s, numbers,  2 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update3<<<DimBlock, DimGrid>>>(s, numbers,  4 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update4<<<DimBlock, DimGrid>>>(s, numbers,  6 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update5<<<DimBlock, DimGrid>>>(s, numbers,  8 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
  update6<<<DimBlock, DimGrid>>>(s, numbers, 10 * N + offset);
  CUDAErrChk(cudaPeekAtLastError());
}

#endif


// /// Calculates the local energy of lattice
// __global__ void energyCalculation(Lattice* d_s) {
//   // Thread identification
//   unsigned short x = blockDim.x * blockIdx.x + threadIdx.x;
//   unsigned short y = blockDim.y * blockIdx.y + threadIdx.y;
//   unsigned short z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Shifts in x and y direction
//   unsigned short yD = (y == 0) * (L - 1) + (y != 0) * (y - 1);  // y - 1
//   unsigned short xU = (x != L - 1) * (x + 1);                   // x + 1
//   unsigned short zU = (z != LAYERS / 2 - 1) * (z + 1);          // z + 1
//
//   // Calculation of energy
//   d_s->exchangeEnergy[N_XY * z + x + L * y] =
//       (-1 * (  // Within layer
//                 J1 * (eType)d_s->s1[N_XY * z + x + L * y] *
//                     ((eType)d_s->s2[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * yD + x]) +
//                 J1 * (eType)d_s->s2[N_XY * z + x + L * y] *
//                     ((eType)d_s->s3[N_XY * z + L * y + x] +
//                      (eType)d_s->s3[N_XY * z + L * yD + xU] +
//                      (eType)d_s->s1[N_XY * z + L * y + xU])
//                 // Within other layer
//                 + J1 * (eType)d_s->s4[N_XY * z + x + L * y] *
//                       ((eType)d_s->s5[N_XY * z + L * y + x] +
//                        (eType)d_s->s6[N_XY * z + L * y + x] +
//                        (eType)d_s->s6[N_XY * z + L * yD + x]) +
//                 J1 * (eType)d_s->s5[N_XY * z + x + L * y] *
//                     ((eType)d_s->s6[N_XY * z + L * y + x] +
//                      (eType)d_s->s6[N_XY * z + L * yD + xU] +
//                      (eType)d_s->s4[N_XY * z + L * y + xU])
//                 // Betweeen layers
//                 + J2 * (eType)(d_s->s1[N_XY * z + x + L * y] *
//                                    d_s->s4[N_XY * z + x + L * y] +
//                                d_s->s2[N_XY * z + x + L * y] *
//                                    d_s->s5[N_XY * z + x + L * y] +
//                                d_s->s3[N_XY * z + x + L * y] *
//                                    d_s->s6[N_XY * z + x + L * y]
//
//                                + d_s->s1[N_XY * zU + x + L * y] *
//                                      d_s->s4[N_XY * z + x + L * y] +
//                                d_s->s2[N_XY * zU + x + L * y] *
//                                    d_s->s5[N_XY * z + x + L * y] +
//                                d_s->s3[N_XY * zU + x + L * y] *
//                                    d_s->s6[N_XY * z + x + L * y])));
//   // d_s->exchangeEnergy[x + L * y] = (-1
//   //         * (J1 * (eType) d_s->s1[x + L * y]
//   //                 * ( (eType) d_s->s2[x + L * y]
//   //                   + (eType) d_s->s3[x + L * y]
//   //                   + (eType) (d_s->s3[x + yD * L]) )
//   //                 +
//   //                 J1 * (eType) d_s->s2[x + L * y]
//   //                         * ((eType) d_s->s3[x + L * y]
//   //                            + (eType) (d_s->s3[xU + yD * L])
//   //                            + (eType) (d_s->s1[xU + y * L]))));
// }
//
// /// Tries to flip each spin of the sublattice 1
// __global__ void update1(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xD, yD, zD;
//   double p;
//   eType sumNN, sumNL;
//   mType s1;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xD = (x == 0) * (L - 1) + (x != 0) * (x - 1);           // x - 1
//   yD = (y == 0) * (L - 1) + (y != 0) * (y - 1);           // y - 1
//   zD = (z == 0) * (LAYERS / 2 - 1) + (z != 0) * (z - 1);  // z - 1
//
//   // Fetching spin
//   s1 = s->s1[N_XY * z + L * y + x];
//
//   // Calculating sum of n.n. in layer
//   sumNN = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * z + L * y + xD] +
//           s->s3[N_XY * z + L * y + x] + s->s3[N_XY * z + L * yD + x];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s4[N_XY * z + L * y + x] + s->s4[N_XY * zD + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s1 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s1[N_XY * z + L * y + x] *=
//       1 - 2 * (mType)(numbers[offset + N_XY * z + L * y + x] < p);
// }
//
// /// Tries to flip each spin of the sublattice 2
// __global__ void update2(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xU, yD, zD;
//   double p;
//   mType s2;
//   eType sumNN, sumNL;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xU = (x != L - 1) * (x + 1);                            // x + 1
//   yD = (y == 0) * (L - 1) + (y != 0) * (y - 1);           // y - 1
//   zD = (z == 0) * (LAYERS / 2 - 1) + (z != 0) * (z - 1);  // z - 1
//
//   // Fetching the spin
//   s2 = s->s2[L * y + x];
//
//   // Calculation sum of n.n. in layer
//   sumNN = s->s1[L * y + x] + s->s1[L * y + xU] + s->s3[L * y + x] +
//           s->s3[L * yD + xU];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * zD + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s2 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s2[N_XY * z + L * y + x] *=
//       1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < p));
// }
//
// /// Tries to flip each spin of the sublattice 3
// __global__ void update3(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xD, yU, zD;
//   double p;
//   mType s3;
//   eType sumNN, sumNL;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xD = (x == 0) * (L - 1) + (x != 0) * (x - 1);           // x - 1
//   yU = (y != L - 1) * (y + 1);                            // y + 1
//   zD = (z == 0) * (LAYERS / 2 - 1) + (z != 0) * (z - 1);  // z - 1
//
//   // Fetching the spin
//   s3 = s->s3[L * y + x];
//
//   // Calculation sum of n.n. in layer
//   sumNN = s->s1[L * y + x] + s->s1[L * yU + x] + s->s2[L * y + x] +
//           s->s2[L * yU + xD];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s6[N_XY * z + L * y + x] + s->s6[N_XY * zD + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s3 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s3[N_XY * z + L * y + x] *=
//       1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < p));
// }
//
// /// Tries to flip each spin of the sublattice 4
// __global__ void update4(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xD, yD, zU;
//   double p;
//   eType sumNN, sumNL;
//   mType s4;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xD = (x == 0) * (L - 1) + (x != 0) * (x - 1);  // x - 1
//   yD = (y == 0) * (L - 1) + (y != 0) * (y - 1);  // y - 1
//   zU = (z != LAYERS / 2 - 1) * (z + 1);          // z + 1
//
//   // Fetching spin
//   s4 = s->s4[N_XY * z + L * y + x];
//
//   // Calculating sum of n.n. in layer
//   sumNN = s->s5[N_XY * z + L * y + x] + s->s5[N_XY * z + L * y + xD] +
//           s->s6[N_XY * z + L * y + x] + s->s6[N_XY * z + L * yD + x];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s1[N_XY * z + L * y + x] + s->s1[N_XY * zU + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s4 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s4[N_XY * z + L * y + x] *=
//       1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < p));
// }
//
// /// Tries to flip each spin of the sublattice 5
// __global__ void update5(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xU, yD, zU;
//   double p;
//   mType s5;
//   eType sumNN, sumNL;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xU = (x != L - 1) * (x + 1);                   // x + 1
//   yD = (y == 0) * (L - 1) + (y != 0) * (y - 1);  // y - 1
//   zU = (z != LAYERS / 2 - 1) * (z + 1);          // z + 1
//
//   // Fetching the spin
//   s5 = s->s5[L * y + x];
//
//   // Calculation sum of n.n. in layer
//   sumNN = s->s4[L * y + x] + s->s4[L * y + xU] + s->s6[L * y + x] +
//           s->s6[L * yD + xU];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s2[N_XY * z + L * y + x] + s->s2[N_XY * zU + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s5 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s5[N_XY * z + L * y + x] *=
//       1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < p));
// }
//
// /// Tries to flip each spin of the sublattice 6
// __global__ void update6(Lattice* s, rngType* numbers, unsigned int offset) {
//   unsigned short x, y, z, xD, yU, zU;
//   double p;
//   mType s6;
//   eType sumNN, sumNL;
//
//   // Thread identification
//   x = blockDim.x * blockIdx.x + threadIdx.x;
//   y = blockDim.y * blockIdx.y + threadIdx.y;
//   z = blockDim.z * blockIdx.z + threadIdx.z;
//
//   // Offsets for n.n.
//   xD = (x == 0) * (L - 1) + (x != 0) * (x - 1);  // x - 1
//   yU = (y != L - 1) * (y + 1);                   // y + 1
//   zU = (z != LAYERS / 2 - 1) * (z + 1);          // z + 1
//
//   // Fetching the spin
//   s6 = s->s6[L * y + x];
//
//   // Calculation sum of n.n. in layer
//   sumNN = s->s4[L * y + x] + s->s4[L * yU + x] + s->s5[L * y + x] +
//           s->s5[L * yU + xD];
//
//   // Calculation sum of n.n. frm other layers
//   sumNL = s->s3[N_XY * z + L * y + x] + s->s3[N_XY * zU + L * y + x];
//
//   // Fetching Boltzman factor
//   p = tex1Dfetch(boltz_tex, (s6 + 1) / 2 + (4 + sumNN) + (2 + sumNL) * 5);
//
//   // Spin update
//   s->s6[N_XY * z + L * y + x] *=
//       1 - 2 * ((mType)(numbers[offset + N_XY * z + L * y + x] < p));
// }
//
// /// Updates all sublattices
// void update(Lattice* s, rngType* numbers, unsigned int offset) {
//   update1<<<DimBlock, DimGrid>>>(s, numbers, 0 + offset);
//   CUDAErrChk(cudaPeekAtLastError());
//
//   update2<<<DimBlock, DimGrid>>>(s, numbers, 1 * N + offset);
//   CUDAErrChk(cudaPeekAtLastError());
//
//   update3<<<DimBlock, DimGrid>>>(s, numbers, 2 * N + offset);
//   CUDAErrChk(cudaPeekAtLastError());
//
//   update4<<<DimBlock, DimGrid>>>(s, numbers, 3 * N + offset);
//   CUDAErrChk(cudaPeekAtLastError());
//
//   update5<<<DimBlock, DimGrid>>>(s, numbers, 4 * N + offset);
//   CUDAErrChk(cudaPeekAtLastError());
//
//   update6<<<DimBlock, DimGrid>>>(s, numbers, 5 * N + offset);
//   CUDAErrChk(cudaPeekAtLastError());
// }
//
// #endif
