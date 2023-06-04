/*
 * C++ implementation of Metropolis algorithm
 * of Ising antiferromagnet on Kagome Lattice.
 *
 * In this version time series are not saved.
 *
 *  Created on: 15.07.2019
 *      Author: Marek Semjan
 */

#include <algorithm> // std::for_each
#include <chrono>    // measuring of the time
#include <cmath>     // std:exp
#include <cstdio>    // fread, fwrite, fprintf
#include <cstdlib>   // fprintf
#include <ctime>     // time
#include <fstream>   // C++ type-safe files
#include <iomanip>   // std::put_time
#include <iomanip>   // std::setprecision
#include <iostream>  // std::cin, std::cout
#include <limits>
#include <numeric> // std::accumulate
#include <random>  // random numbers
#include <sstream> // for higher precision to_string
#include <string>  // for work with file names
#include <vector>  // data container
// #define WIN_OS 1                // Uncomment when running in CygWin
#define LINUX_OS 1          // Uncomment when running on Linux
#include "fileWrapper.h"    // C++ wrapper for C FILE
#include "systemSpecific.h" // custum functions for getting PID and creating directory
// #include "Observables.h"

#define SAVE_DIR "/home/user/DATA/"
#define SAVE_FREQUENCY 100000 // Data is dumped after each SAVE_FREQUENCY sweeps
#define COORDINATION_NUMBER 4 // Number of the nearest neighbors
#define J1 -1                 // Interaction strength
#define L 16                  // Linear size of the sublattice
#define N (L * L)             // Total number of spins of sublattice
#define RAND_N (3 * N)        // Total number of spins
#define CONST_A 0.001         // Magic number used in creating distribution of the temperature mesh
#define CONST_B 0.8           // Magic number used in creating distribution of the temperature mesh
// #define DEBUG 1
// #define BIN_SAVES 1                         // (Un)comment to save to (binary) text file

#define field 0 // External field

#ifdef DEBUG
//    #include <set>
//    #include "plotter.h"
#endif

// Typedefs
typedef int mType;       // type for magnetization
typedef double eType;    // type for energy
typedef double tType;    // type for temperature
typedef double meanType; // type for mean values

// Parameters of the simulation
const unsigned int numThermalSweeps = 100000; // Sweeps for thermalization
const unsigned int numSweeps = 400000;        // Number of sweeps
const tType minTemperature = 0;               // Minimal simulated temperature (this is kind of misleading, check how the temperature is calculated below)
const tType maxTemperature = 15.0;            // Maximal simulated temperature
const tType deltaTemperature = 0.1;           // Temperature step

// Class that stores the lattice
class Lattice {
public:
    mType s1[N];
    mType s2[N];
    mType s3[N];
    Lattice()
    {
    }
};

// Class that stores the actual values of quantities, as well as their means
class Quantities {
public:
    eType energy, energySq;
    mType m1, m2, m3, m1Sq, m2Sq, m3Sq;
    meanType mEnergy, mEnergySq, mm1, mm2, mm3, mm1Sq, mm2Sq, mm3Sq;

    // Constructor - does nothing
    Quantities()
    {
    }

    // Sets mean values to actual value of the quantity / numSweeps
    void startRecording()
    {
        mEnergy = 0;   // energy / (meanType) numSweeps;
        mEnergySq = 0; // energySq / (meanType) numSweeps;
        mm1 = 0;       // m1 / (meanType) numSweeps;
        mm2 = 0;       // m2 / (meanType) numSweeps;
        mm3 = 0;       // m3 / (meanType) numSweeps;
        mm1Sq = 0;     // m1Sq / (meanType) numSweeps;
        mm2Sq = 0;     // m2Sq / (meanType) numSweeps;
        mm3Sq = 0;     // m3Sq / (meanType) numSweeps;
    }

    // Sets initial values of quantities
    void setInit(eType energy, mType m1, mType m2, mType m3)
    {
        this->energy = energy;
        this->energySq = energy * energy;
        this->m1 = m1;
        this->m2 = m2;
        this->m3 = m3;
        this->m1Sq = m1 * m1;
        this->m2Sq = m2 * m2;
        this->m3Sq = m3 * m3;
        this->startRecording();
    }

    // Changes values after a successful MC trial
    void doFlip(eType dE, mType dM1, mType dM2, mType dM3)
    {
        energy += dE;
        m1 += dM1;
        m2 += dM2;
        m3 += dM3;

        energySq = energy * energy;
        m1Sq = m1 * m1;
        m2Sq = m2 * m2;
        m3Sq = m3 * m3;
    }

    // Update mean values
    void updateMeans()
    {
        mEnergy += energy / (meanType)numSweeps;
        mEnergySq += energySq / (meanType)numSweeps;
        mm1 += m1 / (meanType)numSweeps;
        mm2 += m2 / (meanType)numSweeps;
        mm3 += m3 / (meanType)numSweeps;
        mm1Sq += m1Sq / (meanType)numSweeps;
        mm2Sq += m2Sq / (meanType)numSweeps;
        mm3Sq += m3Sq / (meanType)numSweeps;
    }

    std::string means_to_string()
    {
        std::ostringstream streamObj;
        streamObj.precision(32);

        streamObj << mEnergy << "\t"
                  << mm1 << "\t"
                  << mm2 << "\t"
                  << mm3 << "\t"
                  << mEnergySq << "\t"
                  << mm1Sq << "\t"
                  << mm2Sq << "\t"
                  << mm3Sq;

        return streamObj.str();
    }
};

#ifdef DEBUG
unsigned long int accRate = 0;
unsigned long int totalNumSteps = 0;
#endif

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

// Prototypes of host functions
int getIndexOfEnergy(eType energy);
#ifdef DEBUG
void energyCalculation(Lattice* s_dev, Quantities* q);
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path);
#endif
void initialization(Lattice* s, Quantities* q);
void energyCalculation(Lattice* s_dev, Quantities* q);
void simulate(Lattice* s, Quantities* q, bool thermalization);
void update1(Lattice* s, Quantities* q, bool thermalization);
void update2(Lattice* s, Quantities* q, bool thermalization);
void update3(Lattice* s, Quantities* q, bool thermalization);

// Random numbers generation
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> uniR(0.0, 1.0);

// Boltzman table - works for systems with spin number s = 1/2
std::vector<double> boltz(2 * 5, 0.0);

int main()
{
    // Time for logging
    std::time_t logTime;
    std::time(&logTime);
    std::tm tm = *std::gmtime(&logTime);
    char buffer[80];

    // Randomize random numbers
    srand(time(NULL));

    // Getting PID of the simulation
    mySys::myPidType pid = mySys::getPid();

    // Creating folder for files - This is so ugly, I am sorry 😔
    strftime(buffer, 80, "%F_%T", &tm);
    std::string dir = SAVE_DIR;
    std::string folderName = "Kagome_METRO_2D_" + std::to_string(L) + "x" + std::to_string(L) + "_TMIN_" + std::to_string(minTemperature) + "_TMAX_" + std::to_string(maxTemperature) + "_dT_" + std::to_string(deltaTemperature) + "_MCS_" + std::to_string(numSweeps) + "_J_" + std::to_string(J1) + "_F_" + std::to_string(field) + "_Start_" + std::string(buffer) + "PID" + std::to_string(pid) + "/";
    mySys::mkdir((dir + folderName));

    // Creating file names
    std::string logFileName = dir + folderName + "log.txt";
    std::string thermoDataFileName = dir + folderName + "thermo.txt";

    // Opening a log file
    std::ofstream logFile(logFileName.c_str());
    std::ofstream thermoFile(thermoDataFileName.c_str());
    fileWrapper latticeFile(dir + folderName + "lattice.dat", "w+b");

    // Logging info about the simulation
    logFile << "   Monte Carlo - Metropolis - Kagome lattice" << std::endl;
    logFile << "======================================================" << std::endl;

    logFile << "Lattice size:\t\t" << L << "x" << L << "\nJ:\t\t\t" << J1 << "\nSweeps: \t\t" << numSweeps << "\nMinTemp:\t\t" << minTemperature << "\nMaxTemp:\t\t" << maxTemperature << "\ndTemp:\t\t\t" << deltaTemperature << "\nConstA:\t\t\t" << CONST_A << "\nConstB:\t\t\t" << CONST_B << "\nMy PID:\t\t\t" << pid << "\nPrototyp v1.0\n"
                                                                                                                                                                                                                                                                                                                              "======================================================\n"
            << "[" << std::put_time(&tm, "%F %T") << "] Simulation started..." << std::endl;

    // Increase the precision of the cout
    logFile.precision(12);
    thermoFile.precision(32);

    Lattice s;             // Lattice
    Lattice* ss_host = &s; // Pointer to the lattice

    // Creating a header
    thermoFile << "beta\tmE\tmm1\tmm2\tmm3\tmESq\tmm1Sq\tmm2Sq\tmm3Sq" << std::endl;

    // Creating object for recording values of observable quantites
    Quantities q;

    // GNUPlot thingy for ploting lattice... I am leaving it as a reference, but don't use it
    // #ifdef DEBUG
    //// Energy time series
    // std::vector<eType> energyTS;
    // energyTS.resize(numSweeps+numThermalSweeps);

    //// Ehm... A counter, for counting.
    // int counter = 0;

    //// File for saving the energyTS
    // std::ofstream eTSFile((dir + folderName + "ets.txt").c_str());

    //// Drawing for debug purposes
    // FILE* e_pipe = popen("gnuplot -persist", "w");  // For plotting energy
    // FILE* s_pipe = popen("gnuplot -persist", "w");  // For plotting graph
    //
    // fprintf(e_pipe, "set terminal wxt size 350,262 enhanced font 'Verdana,10' persist\n");
    // fprintf(s_pipe, "set terminal wxt size 350,262 enhanced font 'Verdana,10' persist\n");
    //
    // bool test = true;
    // #endif

    try {
        // Preparation of temperatures
        std::vector<double> inverseTemperature;
        inverseTemperature.push_back(0);
        for (int i = 1; i <= std::floor((maxTemperature - minTemperature) / deltaTemperature); i++) {
            inverseTemperature.push_back(
                CONST_A * std::exp(CONST_B * (minTemperature + deltaTemperature * i)));
        }

        // Initial spin configuration and calculation of initial energy
        initialization(ss_host, &q);

        double beta;

        // Temperature loop
        for (int tempCounter = 0; tempCounter <= inverseTemperature.size();
             tempCounter++) {
            beta = inverseTemperature[tempCounter];

            // Generation of Boltzman factors
            for (int idx1 = 0; idx1 <= 2; idx1 += 2) {
                for (int idx2 = 0; idx2 <= 8; idx2 += 2) {
                    boltz[idx1 / 2 + idx2] = exp(-beta * 2 * (idx1 - 1) * (J1 * (idx2 - 4) + field));
                }
            }

            // Loop over sweeps - thermalization
            for (int sweep = 0; sweep < numThermalSweeps; sweep++) {
                simulate(ss_host, &q, false);

#ifdef DEBUG
                std::cout << "e " << q.energy / (3 * N) << "\tmE " << q.mEnergy / (3 * N) << "\tdmE " << q.energy / (3 * N) / numSweeps
                          << std::endl;

// Another part of code used for ploting lattice using GNU Plot
// energyTS[counter] = q.energy/(3*N);
// eTSFile << counter << " " << energyTS[counter] << std::endl;

// if(counter++%50==0){
//     std::cout << "Iter " << counter << std::endl;
//     fprintf(e_pipe, "%s", ("plot '" + dir + folderName +
//                 "ets.txt'\n").c_str());

//    makeSnapshot(ss_host, s_pipe, (dir+folderName));
//}
#endif
            }

            q.startRecording();

            // Loop over sweeps - recording quantities
            for (int sweep = 0; sweep < numSweeps; sweep++) {
                simulate(ss_host, &q, true);
                q.updateMeans();

#ifdef DEBUG
                std::cout << "e " << q.energy / (3 * N) << "\tmE " << q.mEnergy / (3 * N) << "\tdmE " << q.energy / (3 * N) / numSweeps
                          << std::endl;
// energyTS[counter] = q.energy/(3*N);
// eTSFile << counter << " " << energyTS[counter] << std::endl;

// if(counter++%50==0){
//     std::cout << "Iter " << counter << std::endl;
//     fprintf(e_pipe, "%s", ("plot '" + dir + folderName +
//                 "ets.txt'\n").c_str());

//    makeSnapshot(ss_host, s_pipe, (dir+folderName));
//}
#endif
            }

            // Saving data after each temperature dependance is done
            thermoFile << beta << "\t" << q.means_to_string() << std::endl;

            // It's rewind time!
            latticeFile.rewind();

            // Save the lattice
            latticeFile.writeAndFlush(ss_host, sizeof(Lattice));

            // Logging stuff
            std::time(&logTime);
            tm = *std::gmtime(&logTime);
            logFile << std::put_time(&tm, "[%F %T] ") << "Finished  another loop. Last beta = " << beta
                    << std::endl;

        } // end of temperature loop

    } catch (std::exception& e) {
        std::cout << "Exception occured!\n"
                  << e.what() << std::endl;
        std::cin.ignore();
        std::cin.ignore();
    }

#ifdef DEBUG
// Closing files and pipes
// eTSFile.close();
// if(e_pipe!=NULL) fclose(e_pipe);
// if(s_pipe!=NULL) fclose(s_pipe);
#endif

    // Closing files
    logFile.close();
    latticeFile.close();
    thermoFile.close();

    return 0;
}

// =========================================================================
//                                  Functions
// =========================================================================

///
/// Makes a snapshot of the lattice, saves the configurations of up and down
/// spins to files "up.txt" and "down.txt", respectively. And at the end it
/// plots snapshot via pipe to the gnuplot.
///
/// Uncomment code above to enable plotting of lattice during the simulation
///
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path)
{
    std::ofstream up(path + "up.txt", std::ios::out);
    std::ofstream down(path + "down.txt", std::ios::out);

    int x, y;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            // s1
            x = 4 * (j + 1) - 2 * (i + 1);
            y = -2 * (i + 1);
            if (s_dev->s1[i * L + j] == -1) {
                down << x << " " << y << std::endl;
            } else {
                up << x << " " << y << std::endl;
            }
            // s2
            x = 4 * (j + 1) + 2 - 2 * (i + 1);
            y = -2 * (i + 1);
            if (s_dev->s2[i * L + j] == -1) {
                down << x << " " << y << std::endl;
            } else {
                up << x << " " << y << std::endl;
            }
            // s3
            x = 4 * (j + 1) - (2 * (i + 1) + 1);
            y = -2 * (i + 1) - 1;
            if (s_dev->s3[i * L + j] == -1) {
                down << x << " " << y << std::endl;
            } else {
                up << x << " " << y << std::endl;
            }
        }
    }
    up.close();
    down.close();

    fprintf(gnuplotPipe, "%s", ("plot \"" + (path + "up.txt") + "\" with circles" + " linecolor rgb \"#ff0000\" fill solid,\\\n" + "\"" + (path + "down.txt") + "\" with circles linecolor rgb " + "\"#0000ff\"" + " fill solid\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set xrange [-" + std::to_string(2 * L) + ":" + std::to_string(4.5 * L) + "]\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set yrange [-" + std::to_string(3 * L) + ":" + std::to_string(0) + "]\n").c_str());
    fflush(gnuplotPipe);
}

/// Initiates lattice - hot start
void initialization(Lattice* s, Quantities* q)
{
    int m1 = 0, m2 = 0, m3 = 0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            s->s1[x + L * y] = 2 * (int)(2 * uniR(mt)) - 1;
            s->s2[x + L * y] = 2 * (int)(2 * uniR(mt)) - 1;
            s->s3[x + L * y] = 2 * (int)(2 * uniR(mt)) - 1;

            m1 += s->s1[x + L * y];
            m2 += s->s2[x + L * y];
            m3 += s->s3[x + L * y];
        }
    }

    energyCalculation(s, q);
    q->setInit(q->energy, m1, m2, m3);
}

/// Calculates the energy of lattice and stores it in a variable
/// given by the pointer
void energyCalculation(Lattice* s_dev, Quantities* q)
{
    eType energy = 0;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            // Shifts in x and y direction
            unsigned short yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
            unsigned short xU = (x + 1 == L) ? 0 : (x + 1);        // x + 1

            // Calculation of energy
            energy += (-1
                * (J1 * (eType)s_dev->s1[x + L * y]
                        * ((eType)s_dev->s2[x + L * y]
                            + (eType)s_dev->s3[x + L * y]
                            + (eType)(s_dev->s3[x + yD * L]))
                    + J1 * (eType)s_dev->s2[x + L * y]
                        * ((eType)s_dev->s3[x + L * y]
                            + (eType)(s_dev->s3[xU + yD * L])
                            + (eType)(s_dev->s1[xU + y * L]))));
        }
    }

    q->energy = energy;
}

/// Tries to flip each spin of the lattice
void simulate(Lattice* s, Quantities* q, bool thermalization)
{
    update1(s, q, thermalization);
    update2(s, q, thermalization);
    update3(s, q, thermalization);
#ifdef DEBUG
    totalNumSteps += 3 * N;
#endif
}

/// Tries to flip each spin of the sublattice 1
void update1(Lattice* s, Quantities* q, bool thermalization)
{
    unsigned short xD, yD;
    double p;
    eType dE, sumNN;
    mType s1;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
            xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
            yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1

            s1 = s->s1[L * y + x];
            sumNN = s->s2[L * y + x] + s->s2[L * y + xD] + s->s3[L * y + x]
                + s->s3[L * yD + x];
            dE = 2 * J1 * s1 * sumNN;

            p = boltz[(s1 + 1) / 2 + 4 + sumNN];

            if (uniR(mt) < p) {
                s->s1[L * y + x] *= -1;
                q->doFlip(dE, -2 * s1, 0, 0);

#ifdef DEBUG
                accRate++;
#endif
            }

            // if(thermalization)
            //     q->updateMeans();
        }
    }
}

/// Tries to flip each spin of the sublattice 2
void update2(Lattice* s, Quantities* q, bool thermalization)
{
    unsigned short xU, yD;
    double p;
    mType s2;
    eType dE, sumNN;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {

            xU = (x + 1 == L) ? 0 : (x + 1);        // x + 1
            yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1

            s2 = s->s2[L * y + x];
            sumNN = s->s1[L * y + x] + s->s1[L * y + xU] + s->s3[L * y + x]
                + s->s3[L * yD + xU];
            dE = 2 * J1 * s2 * sumNN;

            p = boltz[(s2 + 1) / 2 + 4 + sumNN];

            if (uniR(mt) < p) {
                s->s2[L * y + x] *= -1;
                q->doFlip(dE, 0, -2 * s2, 0);

#ifdef DEBUG
                accRate++;
#endif
            }

            // if(thermalization)
            //     q->updateMeans();
        }
    }
}

/// Tries to flip each spin of the sublattice 3
void update3(Lattice* s, Quantities* q, bool thermalization)
{
    unsigned short xD, yU;
    double p;
    mType s3;
    eType dE, sumNN;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {

            xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
            yU = (y + 1 == L) ? 0 : (y + 1);        // y + 1

            s3 = s->s3[L * y + x];
            sumNN = s->s1[L * y + x] + s->s1[L * yU + x] + s->s2[L * y + x]
                + s->s2[L * yU + xD];

            dE = 2 * J1 * s3 * sumNN;

            p = boltz[(s3 + 1) / 2 + 4 + sumNN];

            if (uniR(mt) < p) {
                s->s3[L * y + x] *= -1;
                q->doFlip(dE, 0, 0, -2 * s3);

#ifdef DEBUG
                accRate++;
#endif
            }

            // if(thermalization)
            //     q->updateMeans();
        }
    }
}
