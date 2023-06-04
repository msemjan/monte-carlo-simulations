/*
 * C++ implementation of Replica Exchange Wang-Landau algorithm
 * of Ising antiferromagnet on Kagome Lattice.
 *
 *  Created on: Mar 11, 2019
 *      Author: tumaak
 */

//#include "stdafx.h"
#include <fstream>							// C++ type-safe files
#include <iostream>                         // cin, cout
#include <sstream>                          // precision for std::to_string
#include <string>                           // for work with file names
#include <vector>                           // data container
#include <numeric>                          // accumulate
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>                           // measuring of the time
#include <iomanip>                  		// std::put_time
#include <ctime>                           	// time
#include <cstdio>                           // fread, fwrite, fprintf
#include <cstdlib>
#include <random>
#include <iomanip>                           // std::setprecision
//#define WIN_OS 1
#define LINUX_OS 1
#include "systemSpecific.h"
#include "fileWrapper.h"

#define SAVE_DIR "/home/user/DATA/"
#define SAVE_FREQUENCY 100000
#define COORDINATION_NUMBER	4				// Number of the nearest neighbors
#define J1 -1                               // Interaction strength
//#define FLATNESS_CRITERION 0.8              // Flatness criterion
//#define FINAL_F 1.000000001                // Final value of modification factor f
//#define L 16                               // Linear size of the sublattice
#define N (L*L)                             // Total number of spins of sublattice
#define RAND_N (3 * N)                      // Total number of spins
#define E_MAX (+(N * 3) * 2)
#define E_MIN (-2 * (N * 3) / 3)
#define E_INC (+4)
//#define DEBUG 1
//#define BIN_SAVES 1                         // (Un)comment to save to (binary) text file
#define SAVE_ENERGIES 1                     // Uncomment to save all possible energies into a text file
//#define CALCULATE_TD                      // (Un)comment to calculate thermodynamics

#ifdef DEBUG
    #include <set>
//    #include "plotter.h"
#endif

typedef int mType;
typedef int eType;
typedef double tType;
typedef unsigned long histType;
typedef double gType;

// Class that stores the lattice
class Lattice {
public:
	mType s1[N];
	mType s2[N];
	mType s3[N];
	Lattice() {
	}
};

const unsigned int minNumMCSteps = 500;    // Minimal number of MC steps
const tType minTemperature = 0.11;
const tType maxTemperature = 3.00;
const tType deltaTemperature = 0.01;
#ifdef DEBUG
std::set<eType> setE;
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
void energyCalculation(Lattice* s_dev, eType* energy);
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path);
#endif
bool isFlatHistogram(std::vector<histType>& hist);
void initialization(Lattice* s, eType *energy);
void energyCalculation(Lattice* s_dev, eType* energy);
void simulate(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType *energy, std::vector<eType>& energies, gType log_f);
void update1(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType *energy, std::vector<eType>& energies, gType log_f);
void update2(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType *energy, std::vector<eType>& energies, gType log_f);
void update3(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType *energy, std::vector<eType>& energies, gType log_f);
void isFlatHistogramPrint(std::vector<histType>& hist, std::ofstream& file);

// Random numbers generation
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> uniR(0.0,1.0);


int main() {
    // Time for logging
    std::time_t logTime;
    std::time(&logTime);
    std::tm tm = *std::gmtime(&logTime);

    // Randomize random numbers
    srand(time(NULL));

    // Getting PID of the simulation
    mySys::myPidType pid = mySys::getPid();

    // Creating nice string for FINAL_F with proper precision
    std::ostringstream streamObj;
    streamObj << FINAL_F;
    std::string finalFStr = streamObj.str();


    // Creating folder for files
    std::string dir = SAVE_DIR;
    std::string folderName = "Kagome_WL_2D_" + std::to_string(L) + "x" +
        std::to_string(L) + "_TMIN_" + std::to_string(minTemperature) + 
        "_TMAX_" + std::to_string(maxTemperature) + "_dT_" + 
        std::to_string(deltaTemperature) + "_MCS_" + 
        std::to_string(minNumMCSteps) + "_J_" + std::to_string(J1) + 
        "_FLAT_" + std::to_string(FLATNESS_CRITERION) + "_FIN_F_" + 
        finalFStr + "PID" + std::to_string(pid) + "/";
    mySys::mkdir((dir + folderName));

    // Creating file names
    std::string logFileName = dir + folderName + "log.txt";
    std::string thermoDataFileName = dir + folderName + "thermo.txt";

    // Opening a log file
    std::ofstream logFile(logFileName.c_str());
    #ifdef BIN_SAVES
    fileWrapper histFile(dir + folderName + "hist.dat", "w+b");
    fileWrapper gFile(dir + folderName + "g.dat", "w+b");
    #else
    std::ofstream histFile(dir + folderName + "hist.txt", std::ios::out);
    std::ofstream gFile(dir + folderName + "g.txt", std::ios::out);
    histFile.precision(30);
    gFile.precision(30);
    #endif
    fileWrapper latticeFile(dir + folderName + "lattice.dat", "w+b");

    // Logging something
    logFile << "   Monte Carlo - Wang-Landau - Kagome lattice" << std::endl;
    logFile << "======================================================" << 
            std::endl;

    logFile << "Lattice size:\t\t" << L << "x" << L <<
        "\nJ:\t\t\t" << J1 << "\nMin MCS: \t\t" << minNumMCSteps <<
        "\nMinTemp:\t\t" << minTemperature <<
        "\nMaxTemp:\t\t" << maxTemperature << 
        "\ndTemp:\t\t\t" << deltaTemperature << 
        "\nFLATNESS:\t\t" << FLATNESS_CRITERION <<
        "\nFINAL F:\t\t" << finalFStr <<
        "\nMy PID:\t\t\t" << pid << "\nProduction 2.0\n"
        "======================================================\n" <<
        "[" << std::put_time(&tm, "%F %T") << "] Simulation started..." << 
        std::endl;

    logFile.precision(12);

    std::vector<eType> possibleEnergyValues; // Vector with all possible energy values
    std::vector<gType> g;     				 // The density of states g(E)
    std::vector<histType> hist;  				 // Histogram in E space

    Lattice s;				// Lattice
    Lattice* ss_host = &s;	// Pointer to the lattice

    // Generating all possible values of energy
    //possibleEnergyValues.push_back(E_MIN);
    for (eType i = E_MIN; i <= E_MAX - 8; i+=E_INC) {
        possibleEnergyValues.push_back(i);
    }
    possibleEnergyValues.push_back(E_MAX);
    //possibleEnergyValues.push_back(E_MAX);
    //std::cout << "Num possible energies " << possibleEnergyValues.size() <<
    //          " " << 2 + (E_MAX - E_MIN - 16) / E_INC <<   std::endl;

    #ifdef SAVE_ENERGIES
    {
        // Saving all possible energies into the file
        std::ofstream saveEnergyFile(dir + folderName + "e.txt");
        for(int i = 0; i < possibleEnergyValues.size(); i++)
            saveEnergyFile << possibleEnergyValues[i] << "\n";
        saveEnergyFile.close();

        // exit(1);
    }
    #endif

    // Setting initial values of DOS g(E) = 1 for each E
    g.resize(possibleEnergyValues.size());
    std::fill(g.begin(), g.end(), 1);

    // Resizing the histogram
    hist.resize(possibleEnergyValues.size());


    #ifdef DEBUG
    // Drawing out histogram and g(E) for debug purposes
    FILE* g_pipe = popen("gnuplot -persist", "w");  // For plotting g(E)
    FILE* h_pipe = popen("gnuplot -persist", "w");  // For histogram
    FILE* a_pipe = popen("gnuplot -persist", "w");  // For acceptance rate
    FILE* s_pipe = popen("gnuplot -persist", "w");
    bool test = true;
    #endif

    try {
        unsigned int numMCSteps;             // Number of Monte Carlo steps
        gType f = (gType)std::exp(1);        // The modification factor
        gType log_f = (gType)std::log(f);    // Logarithm of the modification factor
        eType energy = 0;                    // Energy
        while (f > FINAL_F) {
            // Initializing histogram in E space
            std::fill(hist.begin(), hist.end(), 1);

            numMCSteps = 0;

            // Initial spin configuration and calculation of initial energy
            initialization(ss_host, &energy);

            // Monte Carlo loop
            while (numMCSteps < minNumMCSteps || !isFlatHistogram(hist)) {
                simulate(ss_host, g, hist, &energy, possibleEnergyValues, 
                         log_f);
                numMCSteps++;
                
                if(numMCSteps % SAVE_FREQUENCY == 0){
                    // It's rewind time!
                    latticeFile.rewind();
                    
                    #ifdef BIN_SAVES
                    // It's rewind time!
                    histFile.rewind();
                    gFile.rewind();

                    // Save the data in binary files
                    histFile.writeAndFlush(hist.data(), 
                                          sizeof(histType)*hist.size());
                    gFile.writeAndFlush(g.data(), sizeof(gType)*g.size());
                    #else
                    // It's rewind time!
                    histFile.close();
                    gFile.close();
                    histFile.open(dir + folderName + "hist.txt", std::ios::out|
                            std::ios::trunc);
                    gFile.open(dir + folderName + "g.txt", std::ios::out | 
                            std::ios::trunc);
                    

                    // Save the data to text files
                    std::for_each(hist.begin(), 
                                  hist.end(), 
                                  [&](histType h){
                                    histFile << h << std::endl;
                                  });
                    std::for_each(g.begin(), 
                                  g.end(), 
                                  [&](gType g){
                                    gFile << g << std::endl;
                                  });

                    // Flushing fstreams
                    histFile.flush();
                    gFile.flush();
                    #endif

                    // Save the lattice
                    latticeFile.writeAndFlush(ss_host, sizeof(Lattice));
                }

                #ifdef DEBUG
                test = !isFlatHistogram(hist);
                if (numMCSteps % 50 == 0) {
                    std::cout << "TOT_MCS: " << totalNumSteps << " numMCS: "
                        << numMCSteps << " Flatness: " 
                        << test << " MinSteps? " << 
                        (numMCSteps < minNumMCSteps) <<  std::endl;
                    std::ofstream gTestFile(dir + folderName + "gTest.txt");
                    std::ofstream hTestFile(dir + folderName + "hTest.txt");
                    std::ofstream aTestFile(dir + folderName + "aTest.txt", 
                            std::ios::app);

                    for(int i = 0; i < g.size(); i++){
                        gTestFile << possibleEnergyValues[i] << " " << g[i] 
                            << "\n";
                        hTestFile << possibleEnergyValues[i] << " " << hist[i] 
                            << "\n";
                    }

                    aTestFile << accRate/(long double)totalNumSteps << "\n";
                    

                    gTestFile.flush();
                    hTestFile.flush();
                    aTestFile.flush();

                    fprintf(g_pipe,"%s", ("plot '" + dir + folderName + 
                                "gTest.txt'\n").c_str());
                    fprintf(h_pipe,"%s", ("plot '" + dir + folderName + 
                                "hTest.txt' with boxes\n").c_str());
                    fprintf(a_pipe,"%s", ("plot '" + dir + folderName + 
                                "aTest.txt'\n").c_str());
                    fprintf(g_pipe,"set title 'ln(g)'\n");
                    fprintf(h_pipe,"set title 'h(E)'\n");
                    fprintf(a_pipe,"set title 'Acceptance rate'\n");
                    
                    makeSnapshot(ss_host, s_pipe, (dir+folderName)); 

                    gTestFile.close();
                    hTestFile.close();
                    aTestFile.close();
                }
                #endif
            }

            // It's rewind time!
            latticeFile.rewind();
            
            #ifdef BIN_SAVES
            // It's rewind time!
            histFile.rewind();
            gFile.rewind();

            // Save the data in binary files
            histFile.writeAndFlush(hist.data(), 
                                  sizeof(histType)*hist.size());
            gFile.writeAndFlush(g.data(), sizeof(gType)*g.size());
            #else
            // It's rewind time!
            histFile.close();
            gFile.close();
            histFile.open(dir + folderName + "hist.txt", std::ios::out | 
                    std::ios::trunc);
            gFile.open(dir + folderName + "g.txt", std::ios::out | 
                    std::ios::trunc);

            // Save the data to text files
            std::for_each(hist.begin(), 
                          hist.end(), 
                          [&](histType h){
                            histFile << h << std::endl;
                          });
            std::for_each(g.begin(), 
                          g.end(), 
                          [&](gType g){
                            gFile << g << std::endl;
                          });

            // Flushing fstreams
            histFile.flush();
            gFile.flush();
            #endif
            
            // Save the data in a binary file
            latticeFile.writeAndFlush(ss_host, sizeof(Lattice));

            // Modification of factor f
            f = sqrt(f);
            log_f = log(f);

            // Logging stuff
            std::time(&logTime);
            tm = *std::gmtime(&logTime);
            logFile << std::put_time(&tm, "[%F %T] ") <<
                "Finished  another loop. Now f = " << f
                << std::endl;


        } // End of f-modifying while cyclus

        #ifdef CALCULATE_TD
        //
        // Calculation of thermodynamic 
        //

        std::vector<gType> DOS;    	// Density of states
        //DOS.resize(g.size());
        for (int i = 0; i < g.size(); i++)
            DOS.push_back(0.0);
        //std::fill(DOS.begin(), DOS.end(), 0.0);

        // Normalization of DOS - Sum of all states is equal to 2^N
        gType minG = std::accumulate(g.begin(), g.end(), 
            std::numeric_limits<gType>::max(),
            [=](gType x, gType y) {return std::min(x,y); });
        gType sumG = std::accumulate(g.begin(), g.end(), 0.0,
            [=](gType x, gType y) {return x + std::exp(y-minG); });
        std::transform(g.begin(), g.end(), DOS.begin(),
            [=](gType i) {return std::exp(i-minG) / sumG; });

        std::vector<tType> T;				// Vector of temperatures
        for (tType temperature = minTemperature; temperature <= maxTemperature;
            temperature += deltaTemperature) {
            T.push_back(temperature);
        }
        std::vector<gType> Z(T.size());	// Partition function
        std::vector<gType> U(T.size());	// Internal energy
        std::vector<gType> C(T.size());	// Specific heat
        std::vector<gType> V(T.size());	// Energetic fourth-order cumulant

        // Calculation of partial function
        for (unsigned int i = 0; i < Z.size(); i++) {
            for (unsigned int j = 0; j < DOS.size(); j++) {
                Z[i] += DOS[j] * std::exp(-possibleEnergyValues[j] / T[i]);
            }
        }

        // Calculation of thermodynamic quantities
        gType tmp1, tmp2, tmp3;
        for (unsigned int i = 0; i < Z.size(); i++) {
            tmp2 = 0;
            tmp3 = 0;
            for (unsigned int j = 0; j < DOS.size(); j++) {
                tmp1 = DOS[j] * std::exp(-possibleEnergyValues[j] / T[i]);
                U[i] += possibleEnergyValues[j] * tmp1 / Z[i];
                tmp2 += std::pow(possibleEnergyValues[j], 2)*tmp1 / Z[i];
                tmp3 += std::pow(possibleEnergyValues[j], 4)*tmp1 / Z[i];
            }
            C[i] = (tmp2 - std::pow(U[i], 2)) / std::pow(T[i], 2);
            V[i] = 1 - tmp3 / (3 * std::pow(tmp2, 2));
        }

        // Saving data

        std::fstream dataFile(thermoDataFileName, std::ios::out);
        // Writeout dos
        for (int i = 0; i < DOS.size(); i++)
            dataFile << DOS[i] << std::endl;

        dataFile << "\nSUM G = " << sumG << std::endl;

        // Creating a header
        dataFile << "i\tZ[i]\tU[i]\tC[i]\tV[i]" << std::endl;
        for (unsigned int i = 0; i < Z.size(); i++) {
            dataFile << i << "\t" << Z[i] << "\t" << U[i] << "\t"
                << C[i] << "\t" << V[i] << std::endl;
        }

        #endif // CALCULATE_TD

        // It's rewind time!
        latticeFile.rewind();
        
        #ifdef BIN_SAVES
        // It's rewind time!
        histFile.rewind();
        gFile.rewind();

        // Save the data in binary files
        histFile.writeAndFlush(hist.data(), 
                              sizeof(histType)*hist.size());
        gFile.writeAndFlush(g.data(), sizeof(gType)*g.size());
        #else
        // It's rewind time!
        histFile.close();
        gFile.close();
        histFile.open(dir + folderName + "hist.txt", std::ios::out | 
                std::ios::trunc);
        gFile.open(dir + folderName + "g.txt", std::ios::out | 
                std::ios::trunc);

        // Save the data to text files
        std::for_each(hist.begin(), 
                      hist.end(), 
                      [&](histType h){
                        histFile << h << std::endl;
                      });
        std::for_each(g.begin(), 
                      g.end(), 
                      [&](gType g){
                        gFile << g << std::endl;
                      });

        // Flushing fstreams
        histFile.flush();
        gFile.flush();
        #endif
        
        // Save the data in a binary file
        latticeFile.writeAndFlush(ss_host, sizeof(Lattice));

        #ifdef CALCULATE_TD
        // Closing file
        dataFile.close();
        #endif  // CALCULATE_TD
    } catch (std::exception& e) {
        std::cout << "Exception occured!\n" << e.what() << std::endl;
        std::cin.ignore();
        std::cin.ignore();
    }

    #ifdef DEBUG
    if(g_pipe != NULL) fclose(g_pipe);
    if(h_pipe != NULL) fclose(h_pipe);
    if(a_pipe != NULL) fclose(a_pipe);
    if(s_pipe != NULL) fclose(s_pipe);

    // Writes out possible energies collected in the set and also ones
    // in the possibleEnergyValues. At the end writes, if all values are
    // in the array
    std::string energyFileName = dir + folderName + "energie.txt";
    std::fstream eFile(energyFileName, std::ios::out);

    // Print out all energies
    std::for_each(setE.begin(), 
                  setE.end(), 
                  [&](eType e){
                    eFile << e << std::endl;
                  });

    eFile << " =================================== " << std::endl;

    // Check, if all energies that were measured are really in the list
    // of all possible energies. If not record them in the file.
    std::for_each(possibleEnergyValues.begin(),
    		      possibleEnergyValues.end(),
                  [&](eType e){
    	          if(setE.find(e)==setE.end()){ 
                    eFile << "Energia " << e << "nie je v mnozine!" 
                    <<std::endl;
                  }});

    eFile.close();
    #endif

	// Closing files
	logFile.close();
    latticeFile.close();
    gFile.close();
    histFile.close();

	return 0;
}

// =========================================================================
//                                  Functions
// =========================================================================

// Returns a index corresponding to the given energy in the vector
// of possible energies
int getIndexOfEnergy(eType energy) {
    #ifdef DEBUG
    setE.insert(energy);    
    #endif
/*
    if (energy < E_MIN) {
        std::string msg = "Energia mensia ako najmensia! Jej hodnota je " +
            std::to_string(energy) + " a zodpovedajuci index by bol " +
            std::to_string(((energy - E_MIN) / E_INC)) + ".\n";
        //throw std::exception();
    	throw std::out_of_range(msg);
    }

    if (E_MAX < energy) {
        std::string msg = "Energia vacsia ako najvacsia! Jej hodnota je " +
            std::to_string(energy) + " a zodpovedajuci index by bol " +
            std::to_string(((energy - E_MIN) / E_INC)) + ".\n";
//        throw std::exception();
        throw std::out_of_range(msg);
    }*/
    if (energy == E_MAX)
        return (int)((E_MAX - E_MIN) / E_INC) - 1;
    return (int)((energy - E_MIN) / E_INC);
}

#ifdef DEBUG
// Returns a index corresponding to the given energy in the vector
// of possible energies
int getIndexOfEnergy(eType energy, std::vector<eType>& possibleEnergies) {
    setE.insert(energy);    

    if (energy < E_MIN) {
        std::string msg = "Energia mensia ako najmensia! Jej hodnota je " +
            std::to_string(energy) + " a zodpovedajuci index by bol " +
            std::to_string(((energy - E_MIN) / E_INC)) + ".\n";
        //throw std::exception();
    	throw std::out_of_range(msg);
    }

    if (E_MAX < energy) {
        std::string msg = "Energia vacsia ako najvacsia! Jej hodnota je " +
            std::to_string(energy) + " a zodpovedajuci index by bol " +
            std::to_string(((energy - E_MIN) / E_INC)) + ".\n";
//        throw std::exception();
        throw std::out_of_range(msg);
    }
    
    int idx;

    if (energy == E_MAX){
        idx = (int)((E_MAX - E_MIN) / E_INC) - 1;
        if(energy!=possibleEnergies[idx])
            throw std::runtime_error("Indexy energie su zle: idx = E_MAX!");
        return idx;
    }

    idx = (int)((energy - E_MIN) / E_INC);
    if(energy!=possibleEnergies[idx])
        throw std::runtime_error("Indexy energie su zle: idx = " + 
                std::to_string(idx));

    return idx;
}

//
// Makes a snapshot of the lattice, saves the configurations of up and down 
// spins to files "up.txt" and "down.txt", respectively. And at the end it 
// plots snapshot via pipe to the gnuplot.
//
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path){
    std::ofstream up(path+"up.txt", std::ios::out);
    std::ofstream down(path+"down.txt", std::ios::out);
    
    int x, y;

    for(int i = 0; i < L; i++){
        for(int j = 0; j < L; j++){
            // s1
            x = 4*(j+1) - 2*(i+1);
            y = -2*(i+1);
            if(s_dev->s1[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s2
            x = 4*(j+1) + 2 - 2*(i+1);
            y = -2*(i+1);
            if(s_dev->s2[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s3
            x = 4*(j+1)-(2*(i+1)+1);
            y = -2*(i+1)-1;
            if(s_dev->s3[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
        }
    }
    up.close();
    down.close();

    fprintf(gnuplotPipe,"%s", ("plot \"" + (path+"up.txt") + "\" with circles"+
           " linecolor rgb \"#ff0000\" fill solid,\\\n" +
           "\"" + (path+"down.txt") + "\" with circles linecolor rgb " + 
           "\"#0000ff\"" + " fill solid\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set xrange [-"+std::to_string(2*L)+":"+
                std::to_string(4.5*L)+"]\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set yrange [-"+std::to_string(3*L)+":"+
                std::to_string(0)+"]\n").c_str());
    fflush(gnuplotPipe);

}

#endif

//
// Checks wether a given histogram hist satisfies the flatness criterion.
//
bool isFlatHistogram(std::vector<histType>& hist) {
	// Calculating the average of histogram
	histType mean = std::accumulate(hist.begin(), hist.end(), 0)
			/ (histType) hist.size();

	// Histogram is not flat, if for some value of energy E, distance
	// from mean value is greater than FLATNESS_CRITERION
	for (unsigned int i = 0; i < hist.size(); i++) {
    if(std::abs(hist[i] - mean) > (1 - FLATNESS_CRITERION) * mean) {
			return false;
		}
	}

	return true;
}

//
// Checks wether a given histogram hist satisfies the flatness criterion.
//
void isFlatHistogramPrint(std::vector<histType>& hist, std::ofstream& file) {
    // Calculating the average of histogram
    gType mean = std::accumulate(hist.begin(), hist.end(), 0)
        / (gType)hist.size();
    file << " ======================================= " << std::endl;
    file << "mean is " << mean << std::endl;
    // Histogram is not flat, if for some value of energy E, distance
    // from mean value is greater than FLATNESS_CRITERION
    for (unsigned int i = 0; i < hist.size(); i++) {
        file << hist[i] << "\t" << std::abs(hist[i] - mean) / mean << "\t" <<
         (std::abs(hist[i] - mean) / mean > FLATNESS_CRITERION) << std::endl;
    }
}

// Initiates lattice - hot start
void initialization(Lattice* s, eType *energy) {
	for (int x = 0; x < L; x++) {
		for (int y = 0; y < L; y++) {
			s->s1[x + L * y] = 2 * (int)(2*uniR(mt)) - 1;
			s->s2[x + L * y] = 2 * (int)(2*uniR(mt)) - 1;
			s->s3[x + L * y] = 2 * (int)(2*uniR(mt)) - 1;
		}
	}

	energyCalculation(s, energy);
}

// Calculates the energy of lattice and stores it in a variable
// given by the pointer
void energyCalculation(Lattice* s_dev, eType* energy) {
	*energy = 0;
	for (int x = 0; x < L; x++) {
		for (int y = 0; y < L; y++) {
			// Shifts in x and y direction
			unsigned short yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
			unsigned short xU = (x + 1 == L) ? 0 : (x + 1);        // x + 1

			// Calculation of energy
			*energy += (-1
					* (J1 * (eType) s_dev->s1[x + L * y]
							* ((eType) s_dev->s2[x + L * y]
							   + (eType) s_dev->s3[x + L * y]
							   + (eType) (s_dev->s3[x + yD * L]))
							+
							J1 * (eType) s_dev->s2[x + L * y]
									* ((eType) s_dev->s3[x + L * y]
									   + (eType) (s_dev->s3[xU + yD * L])
									   + (eType) (s_dev->s1[xU + y * L]))));
		}
	}
}

// Tries to flip each spin of the lattice
void simulate(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType *energy, std::vector<eType>& energies, gType log_f) {
	update1(s, g, hist, energy, energies, log_f);
	update2(s, g, hist, energy, energies, log_f);
	update3(s, g, hist, energy, energies, log_f);
    #ifdef DEBUG
    totalNumSteps += 3*N;
    #endif
}


// Tries to flip each spin of the sublattice 1
void update1(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType *energy, std::vector<eType>& energies, gType log_f) {
	unsigned short xD, yD;
	gType p;
    eType dE, sumNN;
	int gi, gf;
    for (int y = 0; y < L; y++) {
	    for (int x = 0; x < L; x++) {
            xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
			yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1

			sumNN = s->s2[L * y + x] + s->s2[L * y + xD] + s->s3[L * y + x]
					+ s->s3[L * yD + x];
			dE = 2 * J1 * s->s1[L * y + x] * sumNN;
            #ifndef DEBUG
			gi = getIndexOfEnergy(*energy);
			gf = getIndexOfEnergy(*energy + dE);
            #endif
            #ifdef DEBUG
			gi = getIndexOfEnergy(*energy, energies);
			gf = getIndexOfEnergy(*energy + dE, energies);
            #endif

			p = std::exp(g[gi] - g[gf]);

			if (uniR(mt) < p) {
				s->s1[L * y + x] *= -1;
				*energy = *energy + dE;
				g[gf] += log_f;
				hist[gf]++;
                #ifdef DEBUG
                accRate++;
                #endif
			} else {
				g[gi] += log_f;
				hist[gi]++;
			}
		}
	}
}

// Tries to flip each spin of the sublattice 2
void update2(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType *energy, std::vector<eType>& energies, gType log_f) {
	unsigned short xU, yD;
	gType p;
    eType dE, sumNN;
	int gi, gf;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {
		
			xU = (x + 1 == L) ? 0 : (x + 1); // x + 1
			yD = (y - 1 == -1) ? (L - 1) : (y - 1); // y - 1
            
			sumNN = s->s1[L * y + x] + s->s1[L * y + xU] + s->s3[L * y + x]
					+ s->s3[L * yD + xU];
			dE = 2 * J1 * s->s2[L * y + x] * sumNN;
            #ifndef DEBUG
			gi = getIndexOfEnergy(*energy);
			gf = getIndexOfEnergy(*energy + dE);
            #endif
            #ifdef DEBUG
			gi = getIndexOfEnergy(*energy, energies);
			gf = getIndexOfEnergy(*energy + dE, energies);
            #endif

			p = std::exp(g[gi] - g[gf]);

			if (uniR(mt) < p) {
				s->s2[L * y + x] *= -1;
				*energy = *energy + dE;
				g[gf] += log_f;
				hist[gf]++;
                #ifdef DEBUG
                accRate++;
                #endif
			} else {
				g[gi] += log_f;
				hist[gi]++;
			}
		}
	}
}

// Tries to flip each spin of the sublattice 3
void update3(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType *energy, std::vector<eType>& energies, gType log_f) {
	unsigned short xD, yU;
	gType p;
    eType dE, sumNN;
	int gi, gf;
    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) {

			xD = (x - 1 == -1) ? (L - 1) : (x - 1); // x - 1
			yU = (y + 1 == L) ? 0 : (y + 1); // y + 1

			sumNN = s->s1[L * y + x] + s->s1[L * yU + x] + s->s2[L * y + x]
					+ s->s2[L * yU + xD];

			dE = 2 * J1 * s->s3[L * y + x] * sumNN;
            
            #ifndef DEBUG
			gi = getIndexOfEnergy(*energy);
			gf = getIndexOfEnergy(*energy + dE);
            #endif
            #ifdef DEBUG
			gi = getIndexOfEnergy(*energy, energies);
			gf = getIndexOfEnergy(*energy + dE, energies);
            #endif

			p = std::exp(g[gi] - g[gf]);

			if (uniR(mt) < p) {
				s->s3[L * y + x] *= -1;
				*energy = *energy + dE;
				g[gf] += log_f;
				hist[gf]++;
                #ifdef DEBUG
                accRate++;
                #endif
			} else {
				g[gi] += log_f;
				hist[gi]++;
			}
		}
	}
}
