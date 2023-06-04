// C/C++ Includes
#include <vector>                   // containers for data
#include <fstream>                  // File IO
#include <iostream>                 // STD IO
#include <sstream>                  // Stringstreams
#include <ctime>                    // for time and random seed
#include <string>                   // C++ strings
#include <numeric>                  // std::accumulate
#include <algorithm>                // std::count_if

// CUDA Includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Helper functions
#include "safe_cuda_macros.cuh"      // Makros for checking CUDA errors
#include "Lattice.h"                 // Lattice class

// Include guard

// C/C++ Includes
#include <vector>                   // containers for data
#include <fstream>                  // File IO
#include <iostream>                 // STD IO
#include <sstream>                  // Stringstreams
#include <ctime>                    // for time and random seed
#include <string>                   // C++ strings
#include <numeric>                  // std::accumulate
#include <algorithm>                // std::count_if

// CUDA Includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Helper functions
#include "safe_cuda_macros.cuh"      // Makros for checking CUDA errors
#include "Lattice.h"                 // Lattice class
#include "config.h"

// Include guard
#ifndef MC_MA_AUXILIARY_FUNCTIONS_H_
#define MC_MA_AUXILIARY_FUNCTIONS_H_

struct UniSpins {
  __host__ __device__ __forceinline__ mType operator()(const float& x) const {
    return (mType)(x <= 0.5 ? -1.0 : +1.0);
  }
};

#ifdef DILUTION
struct DilSpins {
  __host__ __device__ __forceinline__ mType operator()(const float& x) const {
    return (mType)(0 + (x <= (1 - DILUTION)) * ((x<=(1-DILUTION)/2.0)?-1.0:+1.0 ));
    // return (mType)(x <= 0.5 ? -1.0 : +1.0);
  }
};
#endif


// Converts a std::vector to a string
template<typename T>
std::string vector_to_string(std::vector<T>& v){
    std::stringstream ss;
    bool first = true;
    for (auto it = v.begin(); it != v.end(); it++){
        if (!first)
            ss << ", ";
        ss << *it;
        first = false;
    }
    return ss.str();
}

// Class that holds a curand randnom number generator - its states and random 
// numbers. When the class goes out of scope, it automatically deallocates 
// device memory.
class RNG{
    public:
        unsigned int n;      // Number of random numbers
        unsigned long long seed;
        float* d_rand;     // Pointer to array of random numbers
        curandGenerator_t gen;

        // Constructor - takes care of curand setup
        RNG( unsigned int n, unsigned long long globalSeed ) 
            : n{n}
            , seed{globalSeed}
        {
            // Allocate memory
            CUDAErrChk(cudaMalloc( (void**)& d_rand, this->n*sizeof(float) ));
            
            // Create curand generator
            CURAND_CALL(curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_PHILOX4_32_10 ));

            // Set seed
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed( gen, seed ));
        }
        
        ~RNG(){
            // Deallocate memory
            if( this->d_rand ) CUDAErrChk(cudaFree( this->d_rand  ));
            if( this->gen    ) CURAND_CALL(curandDestroyGenerator(gen));
        }

        // Generates a new batch of n random numbers
        void generate(){
            CURAND_CALL(curandGenerateUniform(gen, d_rand, n));
        }

        // Returns a current batch of random numbers in a form of string
        std::string to_string(){
            // Copy data to host
            std::vector<float> v( this->n );
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost )); 
            
            return vector_to_string(v);
        }

        // Copies random numbers from device to a given host memory 
        // std::vector
        void to_vector( std::vector<float>& v ){
            if( v.size() < this->n ){
                std::cout << "Warning: Output vector is too small! Resizing..." 
                          << std::endl;
                v.resize( this->n );
            }

            // Copy data to host
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost)); 
        }

        // Returns a copy of current random numbers in a form of a std::vector 
        std::vector<float> to_vector(){
            // Copy data to host
            std::vector<float> v(this->n);
            CUDAErrChk(cudaMemcpy( v.data() 
                                 , d_rand 
                                 , this->n * sizeof(float) 
                                 , cudaMemcpyDeviceToHost)); 

            return v;
        }
};

// Kernel that takes random numbers, transforms them with functors/lambdas/??
// and then compies them into lattice
template<typename P1, typename P2, typename P3>
__global__ void init_lattice_kernel( Lattice* d_s
                                   , float* d_rand
                                   , P1 p1
                                   , P2 p2
                                   , P3 p3 ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

     d_s->s1[idx] = (mType) p1( d_rand[idx       ] );
     d_s->s2[idx] = (mType) p2( d_rand[idx +   N ] );
     d_s->s3[idx] = (mType) p3( d_rand[idx + 2*N ] );

}

void init_lattice( Lattice* d_s, float* d_rand ){
    #ifdef DEBUG
    std::cout << "generating random configuration" << std::endl;
    #endif
    
  // Lambda generating +1 and -1 randomly
  // auto uni_spin = [] __device__ ( mType x )
  //         {return (mType)(x <= 0.5 ? (mType)(-1.0) : (mType)(+1.0));};
  UniSpins uni_spin;

  #ifndef DILUTION
    // initialization - fill the lattice with random spins
    init_lattice_kernel<<<DimBlockLinear, DimGridLinear>>>
                                              ( d_s
                                              , d_rand
                                              , uni_spin
                                              , uni_spin
                                              , uni_spin );
  #else 
  DilSpins dil_spin;
    // initialization - fill the lattice with random spins
    init_lattice_kernel<<<DimBlockLinear, DimGridLinear>>>
                                              ( d_s
                                              , d_rand
                                              , uni_spin
                                              , uni_spin
                                              , dil_spin );
  #endif

}



void config_to_file( std::string filename
                   , unsigned int latticeSize
                   , unsigned int numSweeps
                   , unsigned int numTemperatures )
{
    std::ofstream configFile( filename );

    configFile << latticeSize       << " "
               << numSweeps         << " "
               << numTemperatures   << " "
               << std::endl;

    configFile.close();
}

template<typename T>
void to_file( std::string filename, std::vector<T> &vec )
{
    std::ofstream file( filename );

    std::for_each( vec.begin()
                 , vec.end() 
                 , [&](T h){
                    file << h << " ";
                 });

    file.close();
}

#endif // CUDA_MC_AUXILIARY_FUNCTIONS_H_
