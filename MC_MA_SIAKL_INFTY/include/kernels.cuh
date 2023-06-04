#ifndef CUDA_MC_KERNELS_CUH_
#define CUDA_MC_KERNELS_CUH_

// C/C++ imports
#include <algorithm>  // std::for_each
#include <chrono>     // measuring of the time
#include <cmath>      // std::exp
#include <cstdio>     // fread, fwrite, fprintf
#include <cstdlib>    // fprintf
#include <ctime>      // time
#include <fstream>    // C++ type-safe files
#include <iomanip>    // std::put_time
#include <iomanip>    // std::setprecision
#include <iostream>   // cin, cout
#include <limits>     // std::numeric_limits
#include <numeric>    // accumulate
#include <random>     // random numbers
#include <sstream>    // for higher precision to_string
#include <string>     // for work with file names
#include <vector>     // data container

// CUB
#include <cub/cub.cuh>  // For parallel reductions

// CUDA specific imports
#include <cuda.h>       // CUDA header

//#define WIN_OS 1
#define LINUX_OS 1

// Auxiliary header files
#include "auxiliary_functions.h"
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

  bool trial = numbers[offset + N_XY * z + L * y + x] < get_boltzmann_factor((s1-s1_new)/2.0, sumNN, sumNL);
  s1 = trial*s1_new+(!trial)*s1;
  s->s1[N_XY * z + L * y + x] = s1;
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
