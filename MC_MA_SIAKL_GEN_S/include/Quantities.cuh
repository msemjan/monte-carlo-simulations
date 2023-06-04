#ifndef QUANTITIES_CUH_
#define QUANTITIES_CUH_

// C/C++ imports
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
#include <random>
#include <sstream>  // for higher precision to_string
#include <string>   // for work with file names
#include <vector>   // data container

// CUDA specific imports
#include <cuda.h>    // CUDA header
#include <curand.h>  // Parallel Random Number Generators

#include <cub/cub.cuh>  // For parallel reductions

//#define WIN_OS 1
#define LINUX_OS 1

// Auxiliary header files
#include "Lattice.h"
#include "auxiliary_functions.h"  // Various Helper functions
#include "config.h"
#include "fileWrapper.h"
#include "safe_cuda_macros.cuh"
#include "systemSpecific.h"
#include "kernels.cuh"

struct SpinToMag {
  __host__ __device__ __forceinline__ mType operator()( const spinType &s ) const {
    return (mType) s;
  }
};

struct ChainToMag {
  __host__ __device__ __forceinline__ mType operator()( const chainType &s ) const {
    return (mType) s;
  }
};

// Functor for squaring values with normalization to number of spins
template<typename T> struct PerVolume {
  __host__ __device__ __forceinline__ double operator()( const T &a ) const {
    return (double) a / (double) VOLUME;
  }
};

// Functor for squaring values
template<typename T> struct Square {
  __host__ __device__ __forceinline__ double operator()( const T &a ) const {
    return double( a * a );
  }
};

// Functor for squaring values with normalization to number of spins
template<typename T> struct SquarePerVolume {
  __host__ __device__ __forceinline__ double operator()( const T &a ) const {
    return (double) (a * a) / (double) VOLUME ;
  }
};

class Quantities {
  public:
    unsigned int numSweeps, numTemp;
    eType *d_energy, *d_mEnergy, *d_energy_chain, *d_mEnergy_chain, *d_mEnergySq,*d_temp_storage_ee,
          *d_temp_storage_e;  //, d_exchangeEnergy;
    mType *d_m1, *d_m2, *d_m3, *d_m4, *d_m5, *d_m6, *d_mm1, *d_mm2, *d_mm3,
          *d_mm4, *d_mm5, *d_mm6, *d_temp_storage_m1, *d_temp_storage_m2,
          *d_temp_storage_m3, *d_temp_storage_m4, *d_temp_storage_m5, *d_temp_storage_m6,
          *d_temp_storage_s1, *d_temp_storage_s2, *d_temp_storage_s3, *d_temp_storage_s4,
          *d_temp_storage_s5, *d_temp_storage_s6;
    eType *blockSumEnergy, *blockSumEnergyChain;

#ifdef CALCULATE_SQARES
    mType *d_mm1Sq, *d_mm2Sq, *d_mm3Sq, *d_mm4Sq, *d_mm5Sq,
          *d_mm6Sq;
#endif
#ifdef CALCULATE_CHAIN_ORDER_PARA
    mType *d_mcho, *d_temp_storage_cho, *d_temp_storage_chol, *blockSumChain;
    chainType *d_cho;
    size_t temp_storage_bytes_cho, temp_storage_bytes_chol;
#endif

#ifdef CALCULATE_TRIANGLE_ORDER_PARA
    mType *blockSumTriangle, *d_triangle, *d_temp_storage_triangle;
    size_t temp_storage_bytes_triangle;
#endif

    size_t temp_storage_bytes_e, temp_storage_bytes_m, temp_storage_bytes_s,
           temp_storage_bytes_ee;

    /// Constructor
    Quantities(unsigned int numSweeps, unsigned int numTemp, Lattice *d_s) {
      this->numSweeps = numSweeps;
      this->numTemp = numTemp;

      // Allocation of memory
      CUDAErrChk( cudaMalloc((void **)&d_energy, numSweeps * numTemp * sizeof(eType)));
      CUDAErrChk( cudaMalloc((void **)&d_energy_chain, numSweeps * numTemp * sizeof(eType)));
      CUDAErrChk(cudaMalloc((void **)&d_m1, numSweeps * numTemp * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void **)&d_m2, numSweeps * numTemp * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void **)&d_m3, numSweeps * numTemp * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void **)&d_m4, numSweeps * numTemp * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void **)&d_m5, numSweeps * numTemp * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void **)&d_m6, numSweeps * numTemp * sizeof(mType)));
#ifdef SAVE_MEANS
      CUDAErrChk( cudaMalloc((void **)&d_mEnergy, numTemp * sizeof(eType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm1, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm2, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm3, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm4, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm5, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm6, numTemp * sizeof(mType)));
#endif

      CUDAErrChk(cudaMalloc((void**)& blockSumEnergy, NUM_GRID_BASE*sizeof(eType)));
      CUDAErrChk(cudaMalloc((void**)& blockSumEnergyChain, NUM_GRID_BASE*sizeof(eType)));
#ifdef CALCULATE_SQARES
      CUDAErrChk( cudaMalloc((void **)&d_mEnergySq, numTemp * sizeof(eType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm1Sq, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm2Sq, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm3Sq, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm4Sq, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm5Sq, numTemp * sizeof(mType)));
      CUDAErrChk( cudaMalloc((void **)&d_mm6Sq, numTemp * sizeof(mType)));
#endif
#ifdef CALCULATE_CHAIN_ORDER_PARA
      CUDAErrChk(cudaMalloc((void**)& d_cho, NUM_CHAIN_ORDER_POINTS * numSweeps * sizeof(mType)));
#ifdef SAVE_MEANS
      CUDAErrChk(cudaMalloc((void**)& d_mcho, NUM_CHAIN_ORDER_POINTS * sizeof(mType)));
#endif
      CUDAErrChk(cudaMalloc((void**)& blockSumChain, NUM_GRID_BASE*sizeof(chainType)));
      d_temp_storage_cho = NULL;
      d_temp_storage_chol = NULL;
      temp_storage_bytes_cho = 0;
      temp_storage_bytes_chol = 0;
#endif
#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      CUDAErrChk(cudaMalloc((void**)& d_triangle, numTemp * numSweeps * sizeof(mType)));
      CUDAErrChk(cudaMalloc((void**)& blockSumTriangle, NUM_GRID_BASE*sizeof(mType)));
#endif

      // CUB boilerplate
      d_temp_storage_e = NULL;
      d_temp_storage_m1 = NULL;
      d_temp_storage_s1 = NULL;
      d_temp_storage_s2 = NULL;
      d_temp_storage_s3 = NULL;
      d_temp_storage_s4 = NULL;
      d_temp_storage_s5 = NULL;
      d_temp_storage_s6 = NULL;
      d_temp_storage_ee = NULL;
      temp_storage_bytes_e = 0;
      temp_storage_bytes_m = 0;
      temp_storage_bytes_s = 0;
      temp_storage_bytes_ee = 0;

      // CUB preparation
#ifdef SAVE_MEANS
      cub::DeviceReduce::Sum(d_temp_storage_e, temp_storage_bytes_e, d_energy, d_mEnergy, numSweeps);
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_e, temp_storage_bytes_e));

      cub::DeviceReduce::Sum(d_temp_storage_m1, temp_storage_bytes_m, d_m1, d_mm1, numSweeps);
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m1, temp_storage_bytes_m));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m2, temp_storage_bytes_m));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m3, temp_storage_bytes_m));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m4, temp_storage_bytes_m));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m5, temp_storage_bytes_m));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_m6, temp_storage_bytes_m));
#endif
#ifdef CALCULATE_CHAIN_ORDER_PARA
      CUDAErrChk(cudaMalloc((void**)& d_temp_storage_cho, temp_storage_bytes_m));

      cub::DeviceReduce::Sum(d_temp_storage_chol, temp_storage_bytes_chol,
          blockSumChain, d_cho, NUM_GRID_BASE);
      CUDAErrChk(cudaMalloc((void**)& d_temp_storage_chol, temp_storage_bytes_chol));
#endif
#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      d_temp_storage_triangle = NULL;
      temp_storage_bytes_triangle = 0;

      cub::DeviceReduce::Sum(d_temp_storage_triangle, temp_storage_bytes_triangle,
          blockSumTriangle, d_triangle, NUM_GRID_BASE);
      CUDAErrChk(cudaMalloc((void**)& d_temp_storage_triangle, temp_storage_bytes_triangle));
#endif

      SpinToMag op;

      cub::TransformInputIterator<mType, SpinToMag, spinType*> s1( d_s->s1, op);
      cub::DeviceReduce::Sum(d_temp_storage_s1, temp_storage_bytes_s, s1, d_m1, N);
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s1, temp_storage_bytes_s));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s2, temp_storage_bytes_s));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s3, temp_storage_bytes_s));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s4, temp_storage_bytes_s));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s5, temp_storage_bytes_s));
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_s6, temp_storage_bytes_s));

      cub::DeviceReduce::Sum(d_temp_storage_ee, temp_storage_bytes_ee, blockSumEnergy, d_energy, NUM_GRID_BASE);
      CUDAErrChk(cudaMalloc((void **)&d_temp_storage_ee, temp_storage_bytes_ee));

#ifdef DEBUG
      std::cout << "Allocated " << temp_storage_bytes_e << " bytes at adress "
        << temp_storage_bytes_e << " for energy" << std::endl;

      std::cout << "Allocated " << temp_storage_bytes_m << " bytes at adress "
        << temp_storage_bytes_m << " for magnetization" << std::endl;
#endif
    }

    /// Destructor
    ~Quantities() {
      // Deallocation of memory
      if (d_energy) CUDAErrChk(cudaFree(d_energy));
      if (d_energy_chain) CUDAErrChk(cudaFree(d_energy_chain));
      if (d_m1) CUDAErrChk(cudaFree(d_m1));
      if (d_m2) CUDAErrChk(cudaFree(d_m2));
      if (d_m3) CUDAErrChk(cudaFree(d_m3));
      if (d_m4) CUDAErrChk(cudaFree(d_m4));
      if (d_m5) CUDAErrChk(cudaFree(d_m5));
      if (d_m6) CUDAErrChk(cudaFree(d_m6));
      if (d_mEnergy) CUDAErrChk(cudaFree(d_mEnergy));
      if (d_mm1) CUDAErrChk(cudaFree(d_mm1));
      if (d_mm2) CUDAErrChk(cudaFree(d_mm2));
      if (d_mm3) CUDAErrChk(cudaFree(d_mm3));
      if (d_mm4) CUDAErrChk(cudaFree(d_mm4));
      if (d_mm5) CUDAErrChk(cudaFree(d_mm5));
      if (d_mm6) CUDAErrChk(cudaFree(d_mm6));
      if (blockSumEnergy) CUDAErrChk(cudaFree(blockSumEnergy));
      if (blockSumEnergyChain) CUDAErrChk(cudaFree(blockSumEnergyChain));
#ifdef CALCULATE_SQARES
      if (d_mEnergySq) CUDAErrChk(cudaFree(d_mEnergySq));
      if (d_mm1Sq) CUDAErrChk(cudaFree(d_mm1Sq));
      if (d_mm2Sq) CUDAErrChk(cudaFree(d_mm2Sq));
      if (d_mm3Sq) CUDAErrChk(cudaFree(d_mm3Sq));
      if (d_mm4Sq) CUDAErrChk(cudaFree(d_mm4Sq));
      if (d_mm5Sq) CUDAErrChk(cudaFree(d_mm5Sq));
      if (d_mm6Sq) CUDAErrChk(cudaFree(d_mm6Sq));
#endif
      if (d_temp_storage_e) CUDAErrChk(cudaFree(d_temp_storage_e));
      if (d_temp_storage_ee) CUDAErrChk(cudaFree(d_temp_storage_ee));
      if (d_temp_storage_m1) CUDAErrChk(cudaFree(d_temp_storage_m1));
      if (d_temp_storage_m2) CUDAErrChk(cudaFree(d_temp_storage_m2));
      if (d_temp_storage_m3) CUDAErrChk(cudaFree(d_temp_storage_m3));
      if (d_temp_storage_m4) CUDAErrChk(cudaFree(d_temp_storage_m4));
      if (d_temp_storage_m5) CUDAErrChk(cudaFree(d_temp_storage_m5));
      if (d_temp_storage_m6) CUDAErrChk(cudaFree(d_temp_storage_m6));
      if (d_temp_storage_s1) CUDAErrChk(cudaFree(d_temp_storage_s1));
      if (d_temp_storage_s2) CUDAErrChk(cudaFree(d_temp_storage_s2));
      if (d_temp_storage_s3) CUDAErrChk(cudaFree(d_temp_storage_s3));
      if (d_temp_storage_s4) CUDAErrChk(cudaFree(d_temp_storage_s4));
      if (d_temp_storage_s5) CUDAErrChk(cudaFree(d_temp_storage_s5));
      if (d_temp_storage_s6) CUDAErrChk(cudaFree(d_temp_storage_s6));
#ifdef CALCULATE_CHAIN_ORDER_PARA
      if (d_temp_storage_cho) CUDAErrChk(cudaFree(d_temp_storage_cho));
      if (d_temp_storage_chol) CUDAErrChk(cudaFree(d_temp_storage_chol));
      if (d_cho) CUDAErrChk(cudaFree(d_cho));
      if (d_mcho) CUDAErrChk(cudaFree(d_mcho));
      if (blockSumChain) CUDAErrChk(cudaFree(blockSumChain));
#endif
#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      if (d_triangle) CUDAErrChk(cudaFree(d_triangle));
      if (d_temp_storage_triangle) CUDAErrChk(cudaFree(d_temp_storage_triangle));
      if (blockSumTriangle) CUDAErrChk(cudaFree(blockSumTriangle));
#endif
    }

    /// Calculate sublattice magnetizations and energy from lattice configuration
    void getObservables(Lattice *s, unsigned int temp, unsigned int sweep, int temp_chain_order) {
      SpinToMag op;

      cub::TransformInputIterator<mType, SpinToMag, spinType*> s1( s->s1, op);
      cub::TransformInputIterator<mType, SpinToMag, spinType*> s2( s->s2, op);
      cub::TransformInputIterator<mType, SpinToMag, spinType*> s3( s->s3, op);
      cub::TransformInputIterator<mType, SpinToMag, spinType*> s4( s->s4, op);
      cub::TransformInputIterator<mType, SpinToMag, spinType*> s5( s->s5, op);
      cub::TransformInputIterator<mType, SpinToMag, spinType*> s6( s->s6, op);

      // Calculate sublattice magnetizations and internal energy
      cub::DeviceReduce::Sum(d_temp_storage_s1, temp_storage_bytes_s, s1,
          d_m1 + sweep + temp * numSweeps, N, streams[0]);
      cub::DeviceReduce::Sum(d_temp_storage_s2, temp_storage_bytes_s, s2,
          d_m2 + sweep + temp * numSweeps, N, streams[1]);
      cub::DeviceReduce::Sum(d_temp_storage_s3, temp_storage_bytes_s, s3,
          d_m3 + sweep + temp * numSweeps, N, streams[2]);
      cub::DeviceReduce::Sum(d_temp_storage_s4, temp_storage_bytes_s, s4,
          d_m4 + sweep + temp * numSweeps, N, streams[3]);
      cub::DeviceReduce::Sum(d_temp_storage_s5, temp_storage_bytes_s, s5,
          d_m5 + sweep + temp * numSweeps, N, streams[4]);
      cub::DeviceReduce::Sum(d_temp_storage_s6, temp_storage_bytes_s, s6,
          d_m6 + sweep + temp * numSweeps, N, streams[5]);
#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      energyCalculationAndReduce<<<DimBlockBase, DimGridBase, 0, streams[6]>>> (
          s, blockSumEnergy, blockSumEnergyChain, blockSumTriangle);
#else
      energyCalculationAndReduce<<<DimBlockBase, DimGridBase, 0, streams[6]>>> (
          s, blockSumEnergy, NULL);
#endif
      CUDAErrChk(cudaPeekAtLastError());

      // {
      //   std::vector<eType> blockSum(NUM_GRID_BASE);
      //
      //   CUDAErrChk(cudaMemcpy(blockSum.data(), blockSumEnergy, NUM_GRID_BASE*sizeof(eType),
      //         cudaMemcpyDeviceToHost));
      //
      //   std::cout << vector_to_string(blockSum) << "\n\n" << std::endl;
      // }

      cub::DeviceReduce::Sum(d_temp_storage_ee, temp_storage_bytes_ee, blockSumEnergy,
          d_energy + sweep + temp * numSweeps, NUM_GRID_BASE, streams[6]);
      cub::DeviceReduce::Sum(d_temp_storage_ee, temp_storage_bytes_ee, blockSumEnergyChain,
          d_energy_chain + sweep + temp * numSweeps, NUM_GRID_BASE, streams[6]);

#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      cub::DeviceReduce::Sum(d_temp_storage_triangle, temp_storage_bytes_triangle, blockSumTriangle,
          d_triangle + sweep + temp * numSweeps, NUM_GRID_BASE, streams[6]);
#endif

      // {
      //   eType e = 0;
      //   CUDAErrChk(cudaMemcpy(&e, d_energy + sweep + temp * numSweeps, sizeof(eType),
      //         cudaMemcpyDeviceToHost));
      //   std::cout << sweep + temp * numSweeps << " " <<  e / (double) VOLUME << "\n";
      // }

      // cub::DeviceReduce::Sum(d_temp_storage_ee, temp_storage_bytes_ee, s->exchangeEnergy,
      //                        d_energy + sweep + temp * numSweeps, N, streams[6]);
#ifdef CALCULATE_CHAIN_ORDER_PARA
      // if( get_chain_order ) cub::DeviceReduce::Sum(d_temp_storage_chol,
      //                         temp_storage_bytes_chol, s->chainOrder,
      //                         d_cho + sweep + temp * numSweeps, N_XY, streams[7]);

      // if( temp_chain_order >=0 ){
      chainOrderKernel<<<DimBlockBase, DimGridBase, 0, streams[7]>>>(s, blockSumChain);
      CUDAErrChk(cudaPeekAtLastError());

      // cub::DeviceReduce::Sum(d_temp_storage_chol, temp_storage_bytes_chol, blockSumChain,
      //     d_cho + sweep + temp_chain_order * numSweeps, NUM_GRID_BASE, streams[7]);

      cub::DeviceReduce::Sum(d_temp_storage_chol, temp_storage_bytes_chol, blockSumChain,
          d_cho + sweep + temp * numSweeps, NUM_GRID_BASE, streams[7]);

      // chainType ch = 0;
      // // CUDAErrChk(cudaMemcpy(&ch, d_cho + sweep + (temp - NUM_CHAIN_ORDER_POINTS) * numSweeps,
      // //       sizeof(chainType), cudaMemcpyDeviceToHost));
      // std::cout << sweep + temp_chain_order * numSweeps << " "
      //   <<  ch / (double) VOLUME << "\n";
      // }
#endif
    }

    /// Calculates mean values of observables
    void means(unsigned int temp, int temp_chain_order) {
      // Operators
      // PerVolume<mType> M_op;
      PerVolume<eType> E_op;


      // Create an iterator wrapper
      cub::TransformInputIterator<double, PerVolume<eType>, eType *> it_e(
          d_energy + temp * numSweeps, E_op);

      // cub::TransformInputIterator< double
      //                            , PerVolume<mType>
      //                            , mType*>
      //                       it_m1( d_m1 + temp * numSweeps
      //                            , M_op );
      //
      // cub::TransformInputIterator< double
      //                            , PerVolume<mType>
      //                            , mType*>
      //                       it_m2( d_m2 + temp * numSweeps
      //                            , M_op );
      //
      // cub::TransformInputIterator< double
      //                            , PerVolume<mType>
      //                            , mType*>
      //                       it_m3( d_m3 + temp * numSweeps
      //                            , M_op );

#ifdef CALCULATE_SQARES
      SquarePerVolume<mType> sqM_op;
      SquarePerVolume<eType> sqE_op;

      cub::TransformInputIterator<double, SquarePerVolume<eType>, eType *> it_eSq(
          d_energy + temp * numSweeps, sqE_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m1Sq(d_m1 + temp * numSweeps, sqM_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m2Sq(d_m2 + temp * numSweeps, sqM_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m3Sq(d_m3 + temp * numSweeps, sqM_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m4Sq(d_m4 + temp * numSweeps, sqM_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m5Sq(d_m5 + temp * numSweeps, sqM_op);
      cub::TransformInputIterator<double, SquarePerVolume<mType>, mType *>
        it_m6Sq(d_m6 + temp * numSweeps, sqM_op);
#endif
      // Reductions
      cub::DeviceReduce::Sum(d_temp_storage_e, temp_storage_bytes_e, it_e,
          d_mEnergy + temp, numSweeps, streams[0]);
      cub::DeviceReduce::Sum(d_temp_storage_m1, temp_storage_bytes_m
          , d_m1 + temp * numSweeps, d_mm1 + temp, numSweeps, streams[1]);
      cub::DeviceReduce::Sum(d_temp_storage_m2, temp_storage_bytes_m
          , d_m2 + temp * numSweeps, d_mm2 + temp, numSweeps, streams[2]);
      cub::DeviceReduce::Sum(d_temp_storage_m3, temp_storage_bytes_m
          , d_m3 + temp * numSweeps, d_mm3 + temp, numSweeps, streams[3]);
      cub::DeviceReduce::Sum(d_temp_storage_m4, temp_storage_bytes_m
          , d_m4 + temp * numSweeps, d_mm4 + temp, numSweeps, streams[4]);
      cub::DeviceReduce::Sum(d_temp_storage_m5, temp_storage_bytes_m
          , d_m5 + temp * numSweeps, d_mm5 + temp, numSweeps, streams[5]);
      cub::DeviceReduce::Sum(d_temp_storage_m6, temp_storage_bytes_m
          , d_m6 + temp * numSweeps, d_mm6 + temp, numSweeps, streams[6]);
#ifdef CALCULATE_CHAIN_ORDER_PARA
      if(temp_chain_order >= 0) cub::DeviceReduce::Sum(d_temp_storage_cho, temp_storage_bytes_m
          , d_cho + temp * numSweeps
          , d_mcho + temp, numSweeps, streams[7]);
#endif

#ifdef CALCULATE_SQARES
      cub::DeviceReduce::Sum(d_temp_storage_e, temp_storage_bytes_e, it_eSq,
          d_mEnergySq + temp, numSweeps, streams[0]);
      cub::DeviceReduce::Sum(d_temp_storage_m1, temp_storage_bytes_m, it_m1Sq,
          d_mm1Sq + temp, numSweeps, streams[1]);
      cub::DeviceReduce::Sum(d_temp_storage_m2, temp_storage_bytes_m, it_m2Sq,
          d_mm2Sq + temp, numSweeps, streams[2]);
      cub::DeviceReduce::Sum(d_temp_storage_m3, temp_storage_bytes_m, it_m3Sq,
          d_mm3Sq + temp, numSweeps, streams[3]);
      cub::DeviceReduce::Sum(d_temp_storage_m4, temp_storage_bytes_m, it_m4Sq,
          d_mm4Sq + temp, numSweeps, streams[4]);
      cub::DeviceReduce::Sum(d_temp_storage_m5, temp_storage_bytes_m, it_m5Sq,
          d_mm5Sq + temp, numSweeps, streams[5]);
      cub::DeviceReduce::Sum(d_temp_storage_m6, temp_storage_bytes_m, it_m6Sq,
          d_mm6Sq + temp, numSweeps, streams[6]);
#endif

    }

    /// Copies time series from device to host and save them to a given file
    void save_TS_to_file(std::string filename) {
      // Open a file for writing
      std::ofstream ts_file(filename);

      // First line contains number of sweeps and number of temperatures
      // ts_file << numSweeps << " " << numTemp << std::endl;

      // Create host vectors
      std::vector<eType> energy(numSweeps * numTemp);
      std::vector<eType> energyChain(numSweeps * numTemp);
      std::vector<mType> m1(numSweeps * numTemp);
      std::vector<mType> m2(numSweeps * numTemp);
      std::vector<mType> m3(numSweeps * numTemp);
      std::vector<mType> m4(numSweeps * numTemp);
      std::vector<mType> m5(numSweeps * numTemp);
      std::vector<mType> m6(numSweeps * numTemp);

      // Copy data
      CUDAErrChk(cudaMemcpy(energy.data(), d_energy,
            numSweeps * numTemp * sizeof(eType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(energyChain.data(), d_energy_chain,
            numSweeps * numTemp * sizeof(eType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m1.data(), d_m1, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m2.data(), d_m2, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m3.data(), d_m3, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m4.data(), d_m4, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m5.data(), d_m5, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(m6.data(), d_m6, numSweeps * numTemp * sizeof(mType),
            cudaMemcpyDeviceToHost));

#ifdef CALCULATE_CHAIN_ORDER_PARA
      std::vector<mType> cho(NUM_CHAIN_ORDER_POINTS * numSweeps);
      CUDAErrChk(cudaMemcpy(cho.data(), d_cho, numSweeps * NUM_CHAIN_ORDER_POINTS * sizeof(mType),
            cudaMemcpyDeviceToHost));
#endif

#ifdef CALCULATE_TRIANGLE_ORDER_PARA
      std::vector<mType> triangle(numTemp * numSweeps);
      CUDAErrChk(cudaMemcpy(triangle.data(), d_triangle, numTemp * numSweeps * sizeof(mType), cudaMemcpyDeviceToHost));
#endif

      ts_file.precision(30);

      // Save to the file
#ifdef CREATE_HEADERS
      // Create a header
      ts_file << "E "
        << "m1 "
        << "m2 "
        << "m3 "
        << "m4 "
        << "m5 "
        << "m6 "
        // #ifdef CALCULATE_CHAIN_ORDER_PARA
        //         << "cho "
        // #endif
        << std::endl;
#endif

      for (int i = 0; i < energy.size(); i++) {
        ts_file << energy[i] << " " << m1[i] << " " << m2[i] << " " << m3[i]
          << " " << m4[i] << " " << m5[i] << " " << m6[i];
#ifdef CALCULATE_CHAIN_ORDER_PARA
        ts_file  << " " << cho[i];
#endif
#ifdef CALCULATE_TRIANGLE_ORDER_PARA
        ts_file << " " << triangle[i];
#endif
        ts_file << " " << energyChain[i] << std::endl;
      }

      ts_file << std::endl;
      ts_file.close();


      // #ifdef CALCULATE_CHAIN_ORDER_PARA
      //       to_file(filename+"_chain", cho);
      // #endif

    }

    /// Copies means to host and saves them to a given file
    void save_mean_to_file(std::string filename, std::vector<tType> &temperature) {
      // Open a file for writing
      std::ofstream mean_file(filename);

      // First line contains number of temperatures
      // mean_file << numTemp << std::endl;

      // Create host vectors
      std::vector<eType> mEnergy(numTemp);
      std::vector<mType> mm1(numTemp);
      std::vector<mType> mm2(numTemp);
      std::vector<mType> mm3(numTemp);
      std::vector<mType> mm4(numTemp);
      std::vector<mType> mm5(numTemp);
      std::vector<mType> mm6(numTemp);
#ifdef CALCULATE_CHAIN_ORDER_PARA
      std::vector<mType> mcho(numTemp);
#endif

#ifdef CALCULATE_SQARES
      std::vector<eType> mEnergySq(numTemp);
      std::vector<mType> mm1Sq(numTemp);
      std::vector<mType> mm2Sq(numTemp);
      std::vector<mType> mm3Sq(numTemp);
      std::vector<mType> mm4Sq(numTemp);
      std::vector<mType> mm5Sq(numTemp);
      std::vector<mType> mm6Sq(numTemp);
#endif

      // Copy data
      CUDAErrChk(cudaMemcpy(mEnergy.data(), d_mEnergy, numTemp * sizeof(eType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm1.data(), d_mm1, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm2.data(), d_mm2, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm3.data(), d_mm3, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm4.data(), d_mm4, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm5.data(), d_mm5, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm6.data(), d_mm6, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
#ifdef CALCULATE_CHAIN_ORDER_PARA
      CUDAErrChk(cudaMemcpy(mcho.data(), d_mcho, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
#endif
#ifdef CALCULATE_SQARES
      CUDAErrChk(cudaMemcpy(mEnergySq.data(), d_mEnergySq,
            numTemp * sizeof(eType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm1Sq.data(), d_mm1Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm2Sq.data(), d_mm2Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm3Sq.data(), d_mm3Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm4Sq.data(), d_mm4Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm5Sq.data(), d_mm5Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
      CUDAErrChk(cudaMemcpy(mm6Sq.data(), d_mm6Sq, numTemp * sizeof(mType), cudaMemcpyDeviceToHost));
#endif
      mean_file.precision(30);

#ifdef CREATE_HEADERS
      // Create a header
      mean_file << "T "
        << "E "
#ifdef CALCULATE_SQARES
        << "E2 "
#endif
        << "m1 "
        << "m2 "
        << "m3 "
        << "m4 "
        << "m5 "
        << "m6 "
        // #ifdef CALCULATE_CHAIN_ORDER_PARA
        //         << "mcho "
        // #endif
#ifdef CALCULATE_SQARES
        << "m1Sq "
        << "m2Sq "
        << "m3Sq "
        << "m4Sq "
        << "m5Sq "
        << "m6Sq"
#endif
        ;
#endif

      // Save to the file
      for (int i = 0; i < temperature.size(); i++) {
        mean_file << temperature[i] << " " << mEnergy[i] / ((double)numSweeps) << " "
#ifdef CALCULATE_SQARES
          << mEnergySq[i] / ((double)numSweeps) << " "
#endif
          << mm1[i] / ((double)numSweeps) << " "
          << mm2[i] / ((double)numSweeps) << " "
          << mm3[i] / ((double)numSweeps) << " "
          << mm4[i] / ((double)numSweeps) << " "
          << mm5[i] / ((double)numSweeps) << " "
          << mm6[i] / ((double)numSweeps);
        // #ifdef CALCULATE_CHAIN_ORDER_PARA
        //         mean_file << " " << ((i - NUM_CHAIN_ORDER_POINTS >=0) ? mcho[i] / (double)numSweeps : 0);
        // #endif
#ifdef CALCULATE_SQARES
        mean_file << " "
          << mm1Sq[i] / ((double)numSweeps) << " "
          << mm2Sq[i] / ((double)numSweeps) << " "
          << mm3Sq[i] / ((double)numSweeps) << " "
          << mm4Sq[i] / ((double)numSweeps) << " "
          << mm5Sq[i] / ((double)numSweeps) << " "
          << mm6Sq[i] / ((double)numSweeps);
#endif
        mean_file << std::endl;
      }
      mean_file.close();

      // #ifdef CALCULATE_CHAIN_ORDER_PARA
      //       to_file(filename+"_chain", mcho);
      // #endif
    }
};

#endif  // QUANTITIES_CUH_
