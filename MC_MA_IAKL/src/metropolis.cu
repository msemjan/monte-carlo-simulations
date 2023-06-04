/*
 * A port of my original C++ implementation of Metropolis algorithm
 * for Ising antiferromagnet on Kagome Lattice into CUDA.
 *
 * COMPILE WITH:
 * nvcc metropolis.cu -I$HOME/cub -lm -lcurand -o metropolis -arch=sm_75 --expt-extended-lambda
 * For debugging add -G -Xcompiler -rdynamic flags.
 *
 *  Created on: 14.10.2019
 *      Author: <marek.semjan@student.upjs.sk>
 */


// C/C++ imports
#include <algorithm>    // std::for_each
#include <chrono>       // measuring of the time
#include <cmath>        // std::exp
#include <cstdio>       // fread, fwrite, fprintf
#include <cstdlib>      // fprintf
#include <ctime>        // time
#include <fstream>      // C++ type-safe files
#include <iomanip>      // std::put_time
#include <iomanip>      // std::setprecision
#include <iostream>     // cin, cout
#include <limits>       // std::numeric_limits
#include <numeric>      // accumulate
#include <random>       // random number generator
#include <sstream>      // for higher precision to_string
#include <string>       // for work with file names
#include <vector>       // data container

// CUB
#include <cub/cub.cuh>  // For parallel reductions

// CUDA specific imports
#include <cuda.h>       // CUDA header
#include <curand.h>     // Parallel Random Number Generators

//#define WIN_OS 1      // Use when running on Windows (not tested)
#define LINUX_OS 1      // Use when running on Linux

// Auxiliary header files
#include "auxiliary_functions.h"
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
    std::time( &logTime );
    std::tm tm = *std::gmtime( &logTime );
    char buffer[80];

    // Reset device
    cudaDeviceReset();

    // Select device
    cudaSetDevice(0);

    // Randomize random numbers
    srand(time( NULL ));

    // Getting PID of the simulation
    mySys::myPidType pid = mySys::getPid();

    // Random number generator
    int globalSeed = (int)( rand() % std::numeric_limits<int>::max() );
    #ifdef DEBUG
    std::cout << "Calling constructor" << std::endl;
    #endif
    RNG generator( RAND_N, globalSeed );

    // Creating folder for files
    strftime( buffer, 80, "%F", &tm );
    std::string dir        = SAVE_DIR;
    #ifdef DILUTION
                           + std::string("diluted_")
    #endif
                           + "cuda/";

    std::string folderName = "IAKL_"
    #ifndef DILUTION
                           + std::string("METRO_2D_")
    #else
                           + std::string("diluted_METRO_2D_DIL_")
                           + std::to_string( DILUTION )
    #endif
                           + std::to_string( L )
                           + "x"
                           + std::to_string( L )
                           + "_TMIN_"
                           + std::to_string( minTemperature )
                           + "_TMAX_"
                           + std::to_string( maxTemperature )
                           + "_dT_"
                           + std::to_string( deltaTemperature )
                           + "_MCS_"
                           + std::to_string( numSweeps )
                           + "_J_"
                           + std::to_string( J1 )
                           + "_F_"
                           + std::to_string( field )
                           + "_Start_"
                           + std::string( buffer )
                           + "PID"
                           + std::to_string( pid )
                           + "SEED"
                           + std::to_string( globalSeed )
    #ifdef USE_INVERSE_TEMPERATURE
                           + "_inverse"
    #endif
                           + "/";

    mySys::mkdir( dir + folderName );

    // Creating file names
    std::string logFileName      = dir + folderName + "log.txt";
    std::string meansFileName    = dir + folderName + "means.txt";
    std::string tsFileName       = dir + folderName + "ts.txt";
    std::string tempFileName     = dir + folderName + "temp.txt";
    std::string simFileName      = dir + folderName + "sim_config.txt";
    std::string latticeFilename  = dir + folderName + "lattice.txt";

    // Opening a log file
    std::ofstream logFile( logFileName.c_str() );

    // Logging something
    logFile << "   Monte Carlo - Metropolis in CUDA - Kagome lattice\n"
            << "======================================================\n"
            << "Lattice size:\t"
            << L
            << "x"
            << L
            << "\nJ:\t\t\t\t"
            << J1
            << "\nThermal: \t\t"
            << numThermalSweeps
            << "\nSweeps: \t\t"
            << numSweeps
            << "\nMinTemp:\t\t"
            << minTemperature
            << "\nMaxTemp:\t\t"
            << maxTemperature
            << "\ndTemp:\t\t\t"
            << deltaTemperature
            << "\nSEED:\t\t\t"
            << globalSeed
            << "\nMy PID:\t\t\t"
            << pid
            << "\nPrototyp v1.0\n"
            << "======================================================\n"
            << "["
            << std::put_time( &tm, "%F %T" )
            << "] Simulation started..."
            << std::endl;

    logFile.precision( 12 );

    // Lattice preparation
    Lattice s;				      // Lattice
    Lattice* ss_host = &s;	// Pointer to the lattice
    Lattice* d_s;			      // Device pointer to lattice

    // Allocate lattice
    CUDAErrChk(cudaMalloc( (void**)&d_s, sizeof(Lattice)) );

    // Prepare lattice
    generator.generate();
    int offset = 0;
    init_lattice( d_s, generator.d_rand );

    for(int i = 0; i < NUM_STREAMS; i++){
       cudaStreamCreate(&streams[i]);
    }

    // increament offset
    offset = ( offset + 1 ) % ( RAND_N / VOLUME );

    #ifdef DEBUG
    // Copy init config of the lattice and save it
    std::cout << "Saving the initial lattice configuration" << std::endl;
    CUDAErrChk(cudaMemcpy( ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));
    to_file( ss_host, dir + folderName + "init_lattice.txt" );
    #endif

    // Preparation of temperatures
    std::vector<double> temperature;
    #ifndef USE_INVERSE_TEMPERATURE
    // temperature.push_back(maxTemperature);
    for( int i = 0
       ; i < numTemp
       ; i++ )
    {
        // temperature.push_back( temperature[0] - i * deltaTemperature );
        temperature.push_back( maxTemperature * std::pow( deltaTemperature, i ) );
    }
    #else
    temperature.push_back( std::numeric_limits<double>::infinity() );
    for( int i = 1
       ; i < numTemp
       ; i++ )
    {
        temperature.push_back( 1 / ( A*std::pow( 2, B*(i-1) ) ));
    }
    #endif

    // Creating object for recording values of observable quantites
    Quantities q( numSweeps, temperature.size(), d_s );

    // Prepare array for Bolzman factors
    CUDAErrChk(cudaMalloc( (void**)& d_boltz, boltzL*sizeof(float)) );
    CUDAErrChk(cudaBindTexture( NULL
                              , boltz_tex
                              , d_boltz
                              , boltzL * sizeof(float) ));

    try {
        double beta;

        // Temperature loop
        for(int tempCounter = 0
           ; tempCounter < temperature.size()
           ; tempCounter++ )
        {
            beta = 1/temperature[tempCounter];

            #ifdef USE_BOLTZ_TABLE
            generate_Boltzman_factors( beta );
            #else
            cudaMemcpyToSymbol(d_beta, &beta, sizeof(double), 0, cudaMemcpyHostToDevice);
            #endif


            // Loop over sweeps - thermalization
            for( int sweep = 0; sweep < numThermalSweeps; sweep++ ){
                // Generate random numbers
                if( offset == 0 )
                    generator.generate();

                // Lunch kernels :)
                update( d_s, generator.d_rand, offset * VOLUME );
                offset = ( offset + 1 ) % ( RAND_N / VOLUME );
            }

            // Loop over sweeps - recording quantities
            for( int sweep = 0; sweep < numSweeps; sweep++ ){
                // Generate random numbers
                if( offset == 0 )
                    generator.generate();

                // Lunch kernels :)
                update( d_s, generator.d_rand, offset * volume );
                offset = ( offset + 1 ) % ( RAND_N / VOLUME );

                // Calculate energy
                energyCalculation<<<DimBlock, DimGrid>>>( d_s );

                // Calculate observables
                q.getObservables( d_s, tempCounter, sweep );

                #ifdef SAVE_LATTICES
                if( sweep%1000==0 ) {
                    // Copy lattice to device
                    CUDAErrChk(cudaMemcpy( ss_host, d_s, sizeof(Lattice),
                                           cudaMemcpyDeviceToHost));

                    // Save lattice
                    to_file( ss_host, dir + folderName + "lattice"
                           + std::to_string( sweep/1000 ) + ".txt" );
                }
                #endif
            }

            // Calculate means for current temperature
            q.means( tempCounter );

            // Logging stuff
            std::time( &logTime );
            tm = *std::gmtime( &logTime );
            logFile << std::put_time( &tm, "[%F %T] " )
                    << "Finished  another loop. Last beta = "
                    << beta
                    << std::endl;

        } // end of temperature loop

    } catch (std::exception& e) {
        // Comrades made mistake and things went wrong, if we are here!
        std::cout << "Exception occured!\n" << e.what() << std::endl;
        std::cin.ignore();
        std::cin.ignore();
    }

    // Save simulation parameters to a file
    config_to_file( simFileName, L, numSweeps, numTemp );

    #ifdef SAVE_TEMPERATURES
    to_file( tempFileName, temperature );
    #endif

    #ifdef SAVE_TS
    // Save time series
    q.save_TS_to_file( tsFileName );
    #endif

    #ifdef SAVE_MEANS
    // Save means
    q.save_mean_to_file( meansFileName, temperature );
    #endif

    #ifdef SAVE_LAST_CONFIG
    // Copy lattice to device
    CUDAErrChk(cudaMemcpy( ss_host, d_s, sizeof(Lattice), cudaMemcpyDeviceToHost));

    // Save lattice
    to_file( ss_host, latticeFilename );
    #endif

    // Last log
    logFile << std::put_time( &tm, "[%F %T] " )
            << "Simulation is finished."
            << std::endl;

    // Closing files
	logFile.close();

    // Clean up
    CUDAErrChk(cudaUnbindTexture( boltz_tex ));
    CUDAErrChk(cudaFree( d_s ));
    CUDAErrChk(cudaFree( d_boltz ));

	return 0;
}
