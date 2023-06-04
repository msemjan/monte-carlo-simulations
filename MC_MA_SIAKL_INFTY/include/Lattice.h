#ifndef CUDA_MC_LATTICE_H_
#define CUDA_MC_LATTICE_H_

#include <cuda.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <random>

#include "config.h"
#include "safe_cuda_macros.cuh"

// Class that stores the lattice
class Lattice {
 public:
  spinType s1[N];
  spinType s2[N];
  spinType s3[N];
  spinType s4[N];
  spinType s5[N];
  spinType s6[N];
  __device__ __host__ Lattice() {}
};

void to_file(Lattice* s, std::string filename) {
  std::ofstream latticeFile(filename);

  for (int i = 0; i < N; i++) {
    latticeFile << (mType)s->s1[i] << " " << (mType)s->s2[i] << " " << (mType)s->s3[i] << " "
                << (mType)s->s4[i] << " " << (mType)s->s5[i] << " " << (mType)s->s6[i] << " "
                << std::endl;
  }

  latticeFile.close();
}

void print_lattice(Lattice* s) {
  for (int i = 0; i < N; i++) {
    std::cout << s->s1[i] << " " << s->s2[i] << " " << s->s3[i] << " "
              << s->s4[i] << " " << s->s5[i] << " " << s->s6[i] << " "
              << std::endl;
  }
}

void from_file(Lattice* s, std::string filename){
  std::ifstream latticeFile(filename);
  if(!latticeFile.is_open()) throw std::runtime_error("Could not open lattice file");

  std::string line;
  mType s1, s2, s3, s4, s5, s6;
  int i = 0;

  // Read data line by line
  while(std::getline(latticeFile, line)){
    std::stringstream ss(line);
    ss >> s1 >> s2 >> s3 >> s4 >> s5 >> s6;
    s->s1[i] = s1;
    s->s2[i] = s2;
    s->s3[i] = s3;
    s->s4[i] = s4;
    s->s5[i] = s5;
    s->s6[i] = s6;
    i++;
  }

  latticeFile.close();
}

void cpu_init_lattice(Lattice *s){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> uniR(0.0, 1.0);

  for(int i = 0; i < N; i++){
      s->s1[i] = 2*(mType)(2*uniR(mt)) - 1;
      s->s2[i] = 2*(spinType)(2*uniR(mt)) - 1;
      s->s3[i] = 2*(spinType)(2*uniR(mt)) - 1;
      s->s4[i] = 2*(spinType)(2*uniR(mt)) - 1;
      s->s5[i] = 2*(spinType)(2*uniR(mt)) - 1;
      s->s6[i] = 2*(spinType)(2*uniR(mt)) - 1;
  }
}

void fm_init_lattice(Lattice *s){
  for(int i = 0; i < N; i++){
      s->s1[i] = (spinType)1;
      s->s2[i] = (spinType)1;
      s->s3[i] = (spinType)1;
      s->s4[i] = (spinType)1;
      s->s5[i] = (spinType)1;
      s->s6[i] = (spinType)1;
  }
}

// Calculates table of Boltzman factors for all possible combinations of
// local spin values
void generate_Boltzman_factors(double beta) {
  // Generation of Boltzman factors
  for (int idx1 = 0; idx1 <= 2; idx1 += 2) {
    for (int idx2 = 0; idx2 <= 8; idx2 += 2) {
      for (int idx3 = 0; idx3 <= 4; idx3 += 2) {
        boltz[idx1 / 2 + idx2 + idx3 * 5] =
            exp(-beta * 2 * (idx1 - 1) *
                (J1 * (idx2 - 4) + J2 * (idx3 - 2) + field));
      }
    }
  }

  // Copy Boltzman factors to device
  CUDAErrChk(cudaMemcpy(d_boltz, boltz.data(), boltzL * sizeof(float),
                        cudaMemcpyHostToDevice));
}

#endif  // CUDA_MC_LATTICE_H_
