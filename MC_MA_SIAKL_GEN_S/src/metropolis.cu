/*
 * Modification of my implementation of Metropolis Algorithm in CUDA
 * for Ising antiferromagnet on Kagome Lattice. This version enables to
 * simulate a system consisting of several interacting layers. a system
 * consisting of several interacting layers.
 *
 * COMPILE WITH:
 * nvcc metropolis.cu -I$HOME/cub -lm -lcurand -o metropolis -arch=sm_75
 * --expt-extended-lambda For debugging add -G -Xcompiler -rdynamic flags.
 *
 *  Created on: 02.10.2020
 *      Author: <marek.semjan@student.upjs.sk>
 */

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
#include "kernels.cuh"

/*
o--x-->
|
y
|
V
     |          /    |          /    |         /
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----
     |       /       |       /       |       /
     |     /         |     /         |     /
     |   /           |   /           |   /
     | /             | /             | /
     s3              s3              s3
    /|              /|              /|              /
  /  |            /  |            /  |            /
     |          /    |          /    |          /
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----
     |       /       |       /       |       /
     |     /         |     /         |     /
     |   /           |   /           |   /
     | /             | /             | /
     s3              s3              s3
    /|              /|              /|              /
  /  |            /  |            /  |            /
     |          /    |          /    |          /
---- s1 ---- s2 ---- s1 ---- s2 ---- s1 ---- s2 ----
     |       /       |       /       |       /

                    Fig. 1
 Kagome lattice and it's sublattices s1, s2 and s3.
*/

// =========================================================================
//                                  Main
// =========================================================================
int main() {
  // Time for logging
  std::time_t logTime;
  std::time(&logTime);
  std::tm tm = *std::gmtime(&logTime);
  char buffer[80];

  // Reset device
  cudaDeviceReset();

  // Select device
  cudaSetDevice(0);

  // Randomize random numbers
#ifndef GLOBAL_SEED
  srand(time(NULL));
#else
  srand(GLOBAL_SEED);
#endif

  // Getting PID of the simulation
  mySys::myPidType pid = mySys::getPid();

  // Random number generator
  int globalSeed = (int)(rand() % std::numeric_limits<int>::max());
#ifdef DEBUG
  std::cout << "Calling constructor" << std::endl;
#endif
  RNG generator(RAND_N, globalSeed);

  // Creating folder for files
  strftime(buffer, 80, "%F", &tm);

  std::string folderName =
      "SIAKL_MC_" + std::to_string(L) + "x" + std::to_string(L) + "x" +
      std::to_string(LAYERS) + "_TMIN_" + std::to_string(minTemperature) +
      "_TMAX_" + std::to_string(maxTemperature) + "_dT_" +
      std::to_string(deltaTemperature) + "_MCS_" + std::to_string(numSweeps) +
      "_J1_" + std::to_string(J1) + "_J2_" + std::to_string(J2) + "_F_" +
      std::to_string(field) + "_Start_" + std::string(buffer) + "PID" +
      std::to_string(pid) + "SEED" + std::to_string(globalSeed) + "/";

  mySys::mkdir(dir + folderName);

  // Creating file names
  std::string logFileName = dir + folderName + "log";
  std::string meansFileName = dir + folderName + "means";
  std::string tsFileName = dir + folderName + "ts";
  std::string tempFileName = dir + folderName + "temp";
  std::string simFileName = dir + folderName + "sim_config";
  std::string latticeFilename = dir + folderName + "lattice";

  // Opening a log file
  std::ofstream logFile(logFileName.c_str());

  // Logging something
  logFile << "   Monte Carlo - Metropolis in CUDA - SIAKL\n"
          << "======================================================\n"
          << "Lattice size:\t" << L << "x" << L << "x" << LAYERS
          << "\nJ1:\t\t\t\t" << J1 << "\nJ2:\t\t\t\t" << J2 << "\nThermal: \t\t"
          << numThermalSweeps << "\nSweeps: \t\t" << numSweeps
          << "\nMinTemp:\t\t" << minTemperature << "\nMaxTemp:\t\t"
          << maxTemperature << "\ndTemp:\t\t\t" << deltaTemperature
          << "\nSEED:\t\t\t" << globalSeed << "\nMy PID:\t\t\t" << pid
          << "\nPrototyp v1.0\n"
          << "======================================================\n"
          << "[" << std::put_time(&tm, "%F %T") << "] Simulation started..."
          << std::endl;

  logFile.precision(12);

  // Lattice preparation
  Lattice s;              // Lattice
  Lattice* ss_host = &s;  // Pointer to the lattice
  Lattice* d_s;           // Device pointer to lattice

  // Allocate lattice
  CUDAErrChk(cudaMalloc((void**)&d_s, sizeof(Lattice)));

  // Prepare lattice
  generator.generate();
  int offset = 0;
  init_lattice(d_s, generator.d_rand);
  // increament offse
  offset = (offset + 1) % (RAND_N / (VOLUME));

#ifdef SAVE_INIT_CONFIG
  // Copy lattice to device
  CUDAErrChk(cudaMemcpy(ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));

  // Save lattice
  to_file(ss_host, dir + folderName + "init_lattice");
#endif

  for(int i = 0; i < NUM_STREAMS; i++){
     cudaStreamCreate(&streams[i]); 
  }

  // Preparation of temperatures
  std::vector<double> temperature;
  #ifdef USE_INVERSE_TEMPERATURE 
  temperature.push_back( std::numeric_limits<double>::infinity() );
  for( int i = 1; i < numTemp; i++ )
  {
      temperature.push_back( 1 / ( A*std::pow( 2, B*(i-1) ) ));
  }
  #endif

  #ifdef USE_LINEAR_TEMPERATURE
  for( tType t = maxTemperature; t >= minTemperature; t -= deltaTemperature )
    temperature.push_back( t );
  #endif

  #ifdef USE_EXP_TEMPERATURE
  // temperature.push_back(maxTemperature);
  for( int i = 0; i < numTemp; i++ )
  {
      // temperature.push_back( temperature[0] - i * deltaTemperature );
      temperature.push_back( maxTemperature * std::pow( deltaTemperature, i ) );
  }
  #endif
  
  // Creating object for recording values of observable quantites
  Quantities q(numSweeps, temperature.size(), d_s);

#ifdef USE_BOLTZ_TABLE
  // Prepare array for Bolzman factors
  CUDAErrChk(cudaMalloc((void**)&d_boltz, boltzL * sizeof(float)));
  CUDAErrChk(cudaBindTexture(NULL, boltz_tex, d_boltz, boltzL * sizeof(float)));
#endif

  try {
    double beta;

    // Temperature loop
    for (int tempCounter = 0; tempCounter < temperature.size(); tempCounter++) {
      beta = 1 / temperature[tempCounter];

      #ifdef USE_BOLTZ_TABLE
      generate_Boltzman_factors( beta );
      #else
      cudaMemcpyToSymbol(d_beta, &beta, sizeof(double), 0, cudaMemcpyHostToDevice);
      #endif

      // Loop over sweeps - thermalization
      for (int sweep = 0; sweep < numThermalSweeps; sweep++) {
        // #ifdef DEBUG
        //     std::cout << "THERMALIZATION: S" << sweep << " T: " <<
        //     tempCounter << std::endl;
        // #endif

        // Generate random numbers
        if (offset == 0) generator.generate();

        // Launch kernels :)
        update(d_s, generator.d_rand, offset * VOLUME);
        offset = (offset + 1) % (RAND_N / (VOLUME));
      }

      // Loop over sweeps - recording quantities
      for (int sweep = 0; sweep < numSweeps; sweep++) {
        // #ifdef DEBUG
        //     std::cout << "Sampling: S:" << sweep << " T: " << tempCounter <<
        //     std::endl;
        // #endif

        // Generate random numbers
        if (offset == 0) generator.generate();

        // Launch kernels :)
        update(d_s, generator.d_rand, offset * VOLUME);
        offset = (offset + 1) % (RAND_N / (VOLUME));

        // Calculate energy
        // energyCalculation<<<DimBlock, DimGrid>>>(d_s);
        // energyCalculationAndReduce<<<DimBlockBase, DimGridBase>>> (d_s, q.d_energy);

        // #ifdef CALCULATE_CHAIN_ORDER_PARA
        // if( temperature[tempCounter] <= 3.0 ) chainOrderKernel<<<DimBlockChO, DimGridChO>>>(d_s);
        // #endif

        // Calculate observables
        q.getObservables(d_s, tempCounter, sweep, tempCounter - NUM_CHAIN_ORDER_POINTS );

        
          
#ifdef SAVE_LATTICES
        if (sweep % SAVE_LATTICES == 0 && tempCounter >= temperature.size() - 5 ) {
          // Copy lattice to device
          CUDAErrChk(cudaMemcpy(ss_host, d_s, sizeof(Lattice),
                                cudaMemcpyDeviceToHost));

          // Save lattice
          to_file(ss_host, dir + folderName + "lattice" + std::to_string(sweep / SAVE_LATTICES) );
        }
#endif
      }

#ifdef SAVE_MEANS
      // Calculate means for current temperature
      q.means(tempCounter, tempCounter - NUM_CHAIN_ORDER_POINTS >= 0);
#endif

      // Logging stuff
      std::time(&logTime);
      tm = *std::gmtime(&logTime);
      logFile << std::put_time(&tm, "[%F %T] ")
              << "Finished  another loop. Last beta = " << beta << std::endl;

    }  // end of temperature loop

  } catch (std::exception& e) {
    // Comrades made mistake and things went wrong, if we are here!
    std::cout << "Exception occured!\n" << e.what() << std::endl;
    std::cin.ignore();
    std::cin.ignore();
  }

  // Save simulation parameters to a file
  config_to_file(simFileName, L, LAYERS, numSweeps, numTemp);

#ifdef SAVE_TEMPERATURES
  to_file(tempFileName, temperature);
#endif

#ifdef SAVE_TS
  // Save time series
  q.save_TS_to_file(tsFileName);
#endif

#ifdef SAVE_MEANS
  // Save means
  q.save_mean_to_file(meansFileName, temperature);
#endif

#ifdef SAVE_LAST_CONFIG
  // Copy lattice to device
  CUDAErrChk(cudaMemcpy(ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));

  // Save lattice
  to_file(ss_host, latticeFilename);
#endif

  // Last log
  logFile << std::put_time(&tm, "[%F %T] ") << "Simulation is finished."
          << std::endl;

  // Closing files
  logFile.close();

  // Clean up
#ifdef USE_BOLTZ_TABLE
  CUDAErrChk(cudaUnbindTexture(boltz_tex));
  CUDAErrChk(cudaFree(d_boltz));
#endif
  CUDAErrChk(cudaFree(d_s));

  return 0;
}
