// C/C++ Includes
#include <algorithm>  // std::count_if
#include <ctime>      // for time and random seed
#include <fstream>    // File IO
#include <iostream>   // STD IO
#include <numeric>    // std::accumulate
#include <sstream>    // Stringstreams
#include <string>     // C++ strings
#include <vector>     // containers for data

// CUDA Includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Helper functions
#include "Lattice.h"             // Lattice class
#include "safe_cuda_macros.cuh"  // Makros for checking CUDA errors

// Include guard

// C/C++ Includes
#include <algorithm>  // std::count_if
#include <ctime>      // for time and random seed
#include <fstream>    // File IO
#include <iostream>   // STD IO
#include <numeric>    // std::accumulate
#include <sstream>    // Stringstreams
#include <string>     // C++ strings
#include <vector>     // containers for data

// CUDA Includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Helper functions
#include "Lattice.h"  // Lattice class
#include "config.h"
#include "safe_cuda_macros.cuh"  // Makros for checking CUDA errors

// Include guard
#ifndef CUDA_MC_AUXILIARY_FUNCTIONS_H_
#define CUDA_MC_AUXILIARY_FUNCTIONS_H_

// Converts a std::vector to a string
template <typename T>
std::string vector_to_string(std::vector<T>& v) {
  std::stringstream ss;
  bool first = true;
  for (auto it = v.begin(); it != v.end(); it++) {
    if (!first) ss << ", ";
    ss << *it;
    first = false;
  }
  return ss.str();
}

template <typename T>
std::string array_to_string(T* t, unsigned int length) {
  std::stringstream ss;
  // bool first = true;
  for (int i = 0; i < length; i++) {
    ss << t[i] << "\n";
  }
  return ss.str();
}

void to_file(std::string filename, std::string s) {
  std::ofstream file(filename);
  file << s << std::endl;
  file.close();
}

// Class that holds a curand randnom number generator - its states and random
// numbers. When the class goes out of scope, it automatically deallocates
// device memory.
class RNG {
 public:
  unsigned int n;           // Number of random numbers
  unsigned long long seed;  // Random number seed
  float* d_rand;            // Pointer to array of random numbers
  curandGenerator_t gen;    // CuRand generator

  // Constructor - takes care of curand setup
  RNG(unsigned int n, unsigned long long globalSeed) : n{n}, seed{globalSeed} {
    // Allocate memory
    CUDAErrChk(cudaMalloc((void**)&d_rand, this->n * sizeof(float)));

    // Create curand generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    // Set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  }

  ~RNG() {
    // Deallocate memory
    if (this->d_rand) CUDAErrChk(cudaFree(this->d_rand));
    if (this->gen) CURAND_CALL(curandDestroyGenerator(gen));
  }

  // Generates a new batch of n random numbers
  void generate() { CURAND_CALL(curandGenerateUniform(gen, d_rand, n)); }

  // Returns a current batch of random numbers in a form of string
  std::string to_string() {
    // Copy data to host
    std::vector<float> v(this->n);
    CUDAErrChk(cudaMemcpy(v.data(), d_rand, this->n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return vector_to_string(v);
  }

  // Copies random numbers from device to a given host memory
  // std::vector
  void to_vector(std::vector<float>& v) {
    if (v.size() < this->n) {
      std::cout << "Warning: Output vector is too small! Resizing..."
                << std::endl;
      v.resize(this->n);
    }

    // Copy data to host
    CUDAErrChk(cudaMemcpy(v.data(), d_rand, this->n * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Returns a copy of current random numbers in a form of a std::vector
  std::vector<float> to_vector() {
    // Copy data to host
    std::vector<float> v(this->n);
    CUDAErrChk(cudaMemcpy(v.data(), d_rand, this->n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return v;
  }
};

// Kernel that takes random numbers, transforms them with functors/lambdas/??
// and then compies them into lattice
__global__ void init_lattice_kernel(Lattice* d_s, float* d_rand) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  d_s->s1[idx] = p1(d_rand[idx]);
  d_s->s2[idx] = p2(d_rand[idx + N]);
  d_s->s3[idx] = p3(d_rand[idx + 2 * N]);
  d_s->s4[idx] = p4(d_rand[idx + 3 * N]);
  d_s->s5[idx] = p5(d_rand[idx + 4 * N]);
  d_s->s6[idx] = p6(d_rand[idx + 5 * N]);
}

void init_lattice(Lattice* d_s, float* d_rand) {
#ifdef DEBUG
  std::cout << "generating random configuration" << std::endl;
#endif

  // Initialization - Fill the lattice with random spins
  init_lattice_kernel<<<DimBlockLinear, DimGridLinear>>>(d_s, d_rand);
}

void config_to_file(std::string filename, unsigned int latticeSize,
                    unsigned int layers, unsigned int numSweeps,
                    unsigned int numTemperatures) {
  std::ofstream configFile(filename);

  configFile << latticeSize << " " << layers << " " << numSweeps << " "
             << numTemperatures << " " << std::endl;

  configFile.close();
}

template <typename T>
void to_file(std::string filename, std::vector<T>& vec) {
  std::ofstream file(filename);

  std::for_each(vec.begin(), vec.end(), [&](T h) { file << h << "\n"; });

  file.close();
}

#endif  // CUDA_MC_AUXILIARY_FUNCTIONS_H_
