/*
 * C++ implementation of Replica Exchange Wang-Landau algorithm
 * of Ising antiferromagnet with the second neighbors on Honeycomb lattice.
 *
 *  Created on: Oct 30, 2019
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
#include <ranges>
//#define WIN_OS 1
#define LINUX_OS 1
#include "systemSpecific.h"
#include "fileWrapper.h"

#define SAVE_FREQUENCY 100000
#define FREQ_HIST_TEST 1000  
//#define COORDINATION_NUMBER	4				// Number of the nearest neighbors
#define J1 -1                               // Interaction strength
#define J2 -1
#define L 16                              // Linear size of the sublattice
#define N (L*L)                             // Total number of spins of sublattice
#define RAND_N (2 * N)                      // Total number of spins
#define FLATNESS_CRITERION 0.8                // Flatness criterion
#define FINAL_F 1.000000001                    // Final value of modification factor f1.0000000001
// #define DEBUG 1
//#define BIN_SAVES 1                         // (Un)comment to save to (binary) text file
#define SAVE_ENERGIES 1                     // Uncomment to save all possible energies into a text file
//#define CALCULATE_TD                      // (Un)comment to calculate thermodynamics

#ifdef DEBUG
    #include <set>
//    #include "plotter.h"
#endif

#define UP(x) ((x!=L-1)*(x+1))
#define DOWN(x) ((x==0)*(L-1) + (x!=0)*(x-1))

// Typedefs
typedef int mType;
typedef int eType;
typedef double tType;
typedef unsigned long long histType;
typedef double gType;


// Global variables
const unsigned int minNumMCSteps = 3500;    // Minimal number of MC steps
const int numIntervals = 150;               // Number of intervals
int lenInterval  = 5;                       // Lenght of each interval
const int pre = 2;                          // Lenght of intersection
const eType E_MIN = -3*N;
const eType E_MAX = 9*N;
const eType E_INC = 2;
const tType minTemperature = 0.11;
const tType maxTemperature = 3.00;
const tType deltaTemperature = 0.01;
#ifdef DEBUG
std::set<eType> setE;
std::string pathForSnapshots;
unsigned int snapshotCounter = 0;
unsigned long int accRate = 0;
unsigned long int totalNumSteps = 0;
unsigned long int finalTotalNumSteps = std::pow(10,9);
bool abortingTheMission = false;
#endif

// Class that stores the lattice
class Lattice {
public:
	mType s1[N];
	mType s2[N];
	Lattice() {
	}
    #ifdef DEBUG
    ~Lattice(){
        std::cout << "Destroying the lattice..." << std::endl;

        // Writes out possible energies collected in the set and also ones
        // in the possibleEnergyValues. At the end writes, if all values are
        // in the array
        std::string energyFileName = "/run/media/tumaak/SAMSUNG/DATA/wl_honeycomb/energie.txt";
        std::fstream eFile(energyFileName, std::ios::out);
        
        eFile << "We encountered these energies: " << std::endl;

        // Print out all energies
        std::for_each(setE.begin(), 
                      setE.end(), 
                      [&](eType e){
                        eFile << e << std::endl;
                      });

        eFile.close();
        }
    #endif
};



/*
 Energy spectrum is -3*(2*N):1:3*(2*N).

            s1(i-1,j)          s1(i-1,j+1)
                   .          .
                     .      .
        s2(i,j-1) --- s1(i,j) --- s2(i,j)
                  .  .   |   .  .
         .          .    |    .     .
 s1(i,j-1)  s1(i+1,j-1)  |  s1(i+1,j)    s1(i,j+1)
                      s2(i+1,j-1) 



                      s1(i-1,j+1) 
 s2(i,j-1)  s2(i-1,j)    |  s2(i-1,j+1)    s2(i,j+1)
          .      .       |      .        .
               .   .     |   .     .
         s1(i,j) ---   s2(i,j) --- s1(i,j+1)
                    .         .
                  .             .
            s2(i+1,j-1)      s2(i+1,j)


                    Fig. 1
 Interactions between sublattices s1 and s2. Dashed is J1, dotted J2.
*/

// Prototypes of host functions
int getIndexOfEnergy(eType energy, std::vector<eType>& possibleEnergies);
#ifdef DEBUG
void energyCalculation(Lattice* s_dev, eType* energy);
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path);
void saveSnapshot(Lattice* s_dev, std::string path);
#endif
bool isFlatHistogram(std::vector<histType>& hist);
void initialization(Lattice* s, eType *energy);
void initialization_saf(Lattice* s, eType *energy);
void initialization_fm(Lattice* s, eType *energy);
void energyCalculation(Lattice* s_dev, eType* energy);
void simulate(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f);
void update1(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f);
void update2(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
		eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f);
void isFlatHistogramPrint(std::vector<histType>& hist, std::ofstream& file);
void find_state_in_interval(Lattice *s, eType e_min, eType e_max, eType* energy);
std::vector<gType> glueDOSEstimate(std::vector<std::vector<gType>>& gi);

template<typename T> T sq(T x){ return x*x;};
template <class T> int binSearch(std::vector<T>& arr, T what);
template<typename T> void to_file(std::string filename, std::vector<T> v);
template<typename T> void to_file(std::string filename, std::vector<std::vector<T>> v);
template<typename T> void to_file(std::string filename, T* v, size_t len);
template<typename T> double average(std::vector<T>& vec);
void to_file(Lattice* s, std::string filename);
void calculateTD(std::string filename, tType T_min, tType T_max, tType dT, 
    std::vector<eType>& possibleEnergies, std::vector<gType> g);

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
    std::string dir = "/run/media/tumaak/SAMSUNG/DATA/wl_honeycomb/";
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
    std::string histFilename = dir + folderName + "hist.txt";
    std::string gFilename = dir + folderName + "g.txt";
    std::string latticeFilename = dir + folderName + "lattice.dat";

    #ifdef DEBUG
    pathForSnapshots = dir + folderName + "snaps/";
    mySys::mkdir(pathForSnapshots);
    #endif

    // Opening a log file
    std::ofstream logFile(logFileName.c_str());
    // #ifdef BIN_SAVES
    // fileWrapper histFile(dir + folderName + "hist.dat", "w+b");
    // fileWrapper gFile(dir + folderName + "g.dat", "w+b");
    // #else
    // std::ofstream histFile(dir + folderName + "hist.txt", std::ios::out);
    // std::ofstream gFile(dir + folderName + "g.txt", std::ios::out);
    // histFile.precision(30);
    // gFile.precision(30);
    // #endif


    // Logging something
    logFile << "   Monte Carlo - Wang-Landau - Honeycomb lattice" << std::endl;
    logFile << "======================================================" << 
            std::endl;

    logFile << "Lattice size:\t\t" << L << "x" << L <<
        "\nJ:\t\t\t" << J1 << "\nMin MCS: \t\t" << minNumMCSteps <<
        "\nMinTemp:\t\t" << minTemperature <<
        "\nMaxTemp:\t\t" << maxTemperature << 
        "\ndTemp:\t\t\t" << deltaTemperature << 
        "\nFLATNESS:\t\t" << FLATNESS_CRITERION <<
        "\nFINAL F:\t\t" << finalFStr <<
        "\nMy PID:\t\t\t" << pid << "\nPrototype 1.0\n"
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
    // Equivalent to [-3*N/2 -3*N/2+6:2:9*N/2-18] in matlab
    
    possibleEnergyValues.push_back(E_MIN);
    int len_pe = ((E_MAX - 18) - (E_MIN + 6))/E_INC + 2;
    int counter = 0;
    for (eType i = E_MIN + 6; i <= E_MAX - 18; i+=E_INC) {
      if( //counter < 12 ||
          counter == len_pe - 15 || // (len_pe - 16 < counter && counter < len_pe - 13) ||
          (len_pe - 13 <= counter && counter <= len_pe - 12) ||
          counter == len_pe - 10 || //(len_pe - 11 < counter && counter < len_pe - 9 ) ||
          (len_pe -  9 < counter && counter < len_pe - 2 )
        ){
          counter++;
          continue;
        }


      possibleEnergyValues.push_back(i);
      counter++;
    }
    possibleEnergyValues.push_back(E_MAX);
    #ifdef SAVE_ENERGIES
    to_file(dir + folderName + "possibleEnergies.txt", possibleEnergyValues);
    #endif

    // Spliting of the energy range into intervals
    lenInterval = (int)std::ceil(possibleEnergyValues.size()/(double)numIntervals)-1;
    std::vector<std::vector<eType>> energyInterval(numIntervals);
    for(int i = 0; i < energyInterval.size(); i++){
      energyInterval[i].resize(lenInterval+pre);
      if(i==0){
        std::copy(possibleEnergyValues.begin(), 
                  possibleEnergyValues.begin()+lenInterval+pre, 
                  energyInterval[i].begin()));    
      } if(i==energyInterval.size()-1){
        std::copy(possibleEnergyValues.end()-lenInterval-pre,
                  possibleEnergyValues.end(),
                  energyInterval[i].begin());    
      } else {
        std::copy(possibleEnergyValues.begin()+(i-1)*lenInterval, 
                  possibleEnergyValues.begin()+i*lenInterval+pre, 
                  energyInterval[i].begin()));    
      }
    }

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

        for(int interval = 0; interval < numIntervals; interval++){
          gType f = (gType)std::exp(1);        // The modification factor
          gType log_f = (gType)std::log(f);    // Logarithm of the modification factor
          eType energy = 0;                    // Energy

          // Initial spin configuration and calculation of initial energy
          if(interval < 50){
            // Super-antiferromagnet
            initialization_saf(ss_host, &energy);
          }else{
            // Random state
            initialization(ss_host, &energy);
          }


          while (f > FINAL_F) {
              #ifdef DEBUG
              if(abortingTheMission){
                  std::cout << "aborting, too many iterations\n" << "Did " << totalNumSteps << " steps\n";
                  break;
              }
              totalNumSteps = 0;
              #endif
              // Initializing histogram in E space
              std::fill(hist.begin(), hist.end(), 1);

              numMCSteps = 0;

              // Monte Carlo loop
              while (true) {
                  if((numMCSteps > minNumMCSteps && numMCSteps % FREQ_HIST_TEST==0 ) && isFlatHistogram(hist)) 
                    break;

                  #ifdef DEBUG
                  if(totalNumSteps>finalTotalNumSteps){
                      abortingTheMission = true;
                      std::cout << "total " << totalNumSteps << " / " << finalTotalNumSteps << "\n";
                      break;
                  }
                  #endif

                  simulate(ss_host, g, hist, &energy, possibleEnergyValues, log_f);
                  numMCSteps++;

                  if(numMCSteps % SAVE_FREQUENCY == 0){
                      // Saving files
                      to_file(histFilename, hist);
                      to_file(gFilename, g);
                      to_file(ss_host, latticeFilename);
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
              
              // #ifdef BIN_SAVES
              // // It's rewind time!
              // histFile.rewind();
              // gFile.rewind();
              //
              // // Save the data in binary files
              // histFile.writeAndFlush(hist.data(),
              //                       sizeof(histType)*hist.size());
              // gFile.writeAndFlush(g.data(), sizeof(gType)*g.size());
              // #else
              // // It's rewind time!
              // histFile.close();
              // gFile.close();
              // histFile.open(dir + folderName + "hist.txt", std::ios::out |
              //         std::ios::trunc);
              // gFile.open(dir + folderName + "g.txt", std::ios::out |
              //         std::ios::trunc);
              //
              // // Save the data to text files
              // std::for_each(hist.begin(),
              //               hist.end(),
              //               [&](histType h){
              //                 histFile << h << std::endl;
              //               });
              // std::for_each(g.begin(),
              //               g.end(),
              //               [&](gType g){
              //                 gFile << g << std::endl;
              //               });
              //
              // // Flushing fstreams
              // histFile.flush();
              // gFile.flush();
              // #endif
              //
              // // Save the data in a binary file
              // to_file(ss_hosti, latticeFilename);

              to_file(histFilename, hist);
              to_file(gFilename, g);
              to_file(ss_host, latticeFilename);

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
        } // End of loop through the intervals


        #ifdef CALCULATE_TD

        #endif // CALCULATE_TD

        to_file(histFilename, hist);
        to_file(gFilename, g);
        to_file(ss_host, latticeFilename);
        // It's rewind time!
        // latticeFile.rewind();
        
        // #ifdef BIN_SAVES
        // It's rewind time!
        // histFile.rewind();
        // gFile.rewind();

        // Save the data in binary files
        // histFile.writeAndFlush(hist.data(),
        //                       sizeof(histType)*hist.size());
        // gFile.writeAndFlush(g.data(), sizeof(gType)*g.size());
        // #else
        // It's rewind time!
        // histFile.close();
        // gFile.close();
        // histFile.open(dir + folderName + "hist.txt", std::ios::out |
        //         std::ios::trunc);
        // gFile.open(dir + folderName + "g.txt", std::ios::out |
        //         std::ios::trunc);

        // Save the data to text files
        // std::for_each(hist.begin(),
        //               hist.end(),
        //               [&](histType h){
        //                 histFile << h << std::endl;
        //               });
        // std::for_each(g.begin(),
        //               g.end(),
        //               [&](gType g){
        //                 gFile << g << std::endl;
        //               });
        //
        // // Flushing fstreams
        // histFile.flush();
        // gFile.flush();
        // #endif
        
        // Save the data in a binary file
        // latticeFile.writeAndFlush(ss_host, sizeof(Lattice));

            
        #ifdef CALCULATE_TD
        // Closing file
        dataFile.close();
        #endif  // CALCULATE_TD
    } catch (std::exception& e) {
        std::cout << "Exception occured!\n" << e.what() << std::endl;
        #ifdef DEBUG
        makeSnapshot(ss_host, s_pipe, (dir+folderName)); 
        saveSnapshot(ss_host, pathForSnapshots);
        #endif
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


    std::cout << "finishing normally\n";

	// Closing files
	logFile.close();
  // gFile.close();
  // histFile.close();

	return 0;
}

// =========================================================================
//                                  Functions
// =========================================================================

/// Returns a index corresponding to the given energy in the vector
/// of possible energies
int getIndexOfEnergy(eType energy, std::vector<eType>& possibleEnergies) {
    auto idx = binSearch(possibleEnergies, energy) ;
    
    if(idx == -1)
      throw std::out_of_range("Energy is out of range!: " + std::to_string(energy));

    return idx;
}

///
/// Makes a snapshot of the lattice, saves the configurations of up and down 
/// spins to files "up.txt" and "down.txt", respectively. And at the end it 
/// plots snapshot via pipe to the gnuplot.
///
void makeSnapshot(Lattice* s_dev, FILE* gnuplotPipe, std::string path){
    //std::cerr << "Snopshots are not implemented yet." << std::endl;

    std::ofstream up(path+"up.txt", std::ios::out);
    std::ofstream down(path+"down.txt", std::ios::out);
    
    int x, y;

    for(int i = 0; i < L; i++){
        for(int j = 0; j < L; j++){
            // s1
            x = 2*j + i;
            y = -2*i-1;
            if(s_dev->s1[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s2
            x = 2*j + i + 1;
            y = -2*i;
            if(s_dev->s2[i*L+j]==-1){
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
    fprintf(gnuplotPipe, "%s", ("set xrange [-"+std::to_string(3)+":"+
                std::to_string(3.5*L)+"]\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set yrange [-"+std::to_string(3*L)+":"+
                std::to_string(3)+"]\n").c_str());
    fflush(gnuplotPipe);

}

#ifdef DEBUG
///
/// Creates and saves a snapshot similarly to makeSnapshot(), but doesn't 
/// plot it. This function requires a globa counter named snapshotCounter
/// for its correct function.
///
void saveSnapshot(Lattice* s_dev, std::string path){
    std::ofstream up(path+"up"+std::to_string(snapshotCounter)+".txt", std::ios::out);
    std::ofstream down(path+"down"+std::to_string(snapshotCounter)+".txt", std::ios::out);
    snapshotCounter++;
    
    int x, y;

    for(int i = 0; i < L; i++){
        for(int j = 0; j < L; j++){
            // s1
            x = 2*j + i;
            y = -2*i-1;
            if(s_dev->s1[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s2
            x = 2*j + i + 1;
            y = -2*i;
            if(s_dev->s2[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
        }        
    }
    up.close();
    down.close();
}
#endif
///
/// Checks wether a given histogram hist satisfies the flatness criterion.
///
bool isFlatHistogram(std::vector<histType>& hist) {
	// Calculating the average of histogram
	// double mean = std::accumulate(hist.begin(), hist.end(), 0)
	//     / (double) hist.size();
  double mean = average(hist);

	// Histogram is not flat, if for some value of energy E, distance
	// from mean value is greater than FLATNESS_CRITERION
	for (unsigned int i = 0; i < hist.size(); i++) {
		//if (std::abs(hist[i] - mean) / mean > FLATNESS_CRITERION) {
    if(std::abs(hist[i] - mean) > (1-FLATNESS_CRITERION) * mean) {
    // if(FLATNESS_CRITERION * hist[i] > mean){
    // if(FLATNESS_CRITERION * hist[i] > mean){
			return false;
		}
	}

	return true;
}

///
/// Checks wether a given histogram hist satisfies the flatness criterion.
///
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

/// Initiates lattice - Hot start
void initialization(Lattice* s, eType *energy) {
  for (int x = 0; x < L; x++) {
    for (int y = 0; y < L; y++) {
      s->s1[x + L * y] = 2 * (int)(2*uniR(mt)) - 1;
      s->s2[x + L * y] = 2 * (int)(2*uniR(mt)) - 1;
    }
  }

	energyCalculation(s, energy);
}

/// Initiates lattice - Ferromagnet
void initialization_fm(Lattice* s, eType *energy) {
  for (int x = 0; x < L; x++) {
    for (int y = 0; y < L; y++) {
      s->s1[x + L * y] = 1;
            s->s2[x + L * y] = 1;
    }
  }

	energyCalculation(s, energy);
}

/// Initiates lattice - Super-Antiferromagnet 
void initialization_saf(Lattice* s, eType *energy) {
  for (int x = 0; x < L; x++) {
    for (int y = 0; y < L; y++) {
      s->s1[x + L * y] = (x+y)%2==0 ? +1 : -1; 
      s->s2[x + L * y] = (x+y)%2==0 ? -1 : +1;
    }
  }
	energyCalculation(s, energy);
}

/// Calculates the energy of lattice and stores it in a variable
/// given by the pointer
void energyCalculation(Lattice* s_dev, eType* energy) {
	*energy = 0;
  eType sumNN, sumNNN;
	for (int x = 0; x < L; x++) {
		for (int y = 0; y < L; y++) {
      *energy -= J1 * ( s_dev->s1[L * y + x] * s_dev->s2[L * DOWN(y) + x]
                      + s_dev->s1[L * y + x] * s_dev->s2[L * y + x]
                      + s_dev->s1[L * UP(y) + DOWN(x)] * s_dev->s2[L * y + x]
                      )
               + J2 * ( s_dev->s1[L * y + x] * s_dev->s1[L * DOWN(y) + x]
                      + s_dev->s1[L * y + x] * s_dev->s1[L * y + DOWN(x)]
                      + s_dev->s1[L * y + x] * s_dev->s1[L * UP(y) + DOWN(x)]
                      + s_dev->s2[L * y + x] * s_dev->s2[L * DOWN(y) + x]
                      + s_dev->s2[L * y + x] * s_dev->s2[L * y + DOWN(x)]
                      + s_dev->s2[L * y + x] * s_dev->s2[L * UP(y) + DOWN(x)]
                      );
		}
	}
}

/// Tries to flip each spin of the lattice
void simulate(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f) {
	update1(s, g, hist, e_min, e_max, energy, energies, log_f);
	update2(s, g, hist, e_min, e_max, energy, energies, log_f);
    #ifdef DEBUG
    totalNumSteps += 2*N;
    #endif
}


/// Tries to flip each spin of the sublattice 1
void update1(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f) {
	gType p;
  eType dE, sumNN, sumNNN;
	int gi, gf;
  for (int y = 0; y < L; y++) {
    for (int x = 0; x < L; x++) {
			sumNN  = s->s2[L * y  + x ] 
             + s->s2[L * DOWN(y)  + x] 
             + s->s2[L * DOWN(y) + UP(x)];

      sumNNN = s->s1[L * DOWN(y)  + x]
             + s->s1[L * DOWN(y)  + UP(x)]
             + s->s1[L * y + DOWN(x) ]
             + s->s1[L * y + UP(x)]
             + s->s1[L * UP(y) + DOWN(x)]
             + s->s1[L * UP(y) + x ];

			dE = 2 * s->s1[L * y + x] * ( J1 * sumNN + J2 * sumNNN );
		
      if(*energy + dE < e_min || e_max < *energy + dE)
        return;

      gi = getIndexOfEnergy(*energy, energies);
			gf = getIndexOfEnergy(*energy + dE, energies);

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

/// Tries to flip each spin of the sublattice 2
void update2(Lattice* s, std::vector<gType>& g, std::vector<histType>& hist,
	eType e_min, eType e_max, eType *energy, std::vector<eType>& energies, gType log_f) {
	gType p;
  eType dE, sumNN, sumNNN;
	int gi, gf;
  for (int y = 0; y < L; y++) {
    for (int x = 0; x < L; x++) {
			sumNN  = s->s1[L * y  + x ] 
             + s->s1[L * UP(y) + x] 
             + s->s1[L * UP(y) + DOWN(x)];

      sumNNN = s->s2[L * DOWN(y)  + x]
             + s->s2[L * DOWN(y)  + UP(x)]
             + s->s2[L * y + DOWN(x)]
             + s->s2[L * y + UP(x)]
             + s->s2[L * UP(y) + x]
             + s->s2[L * UP(y) + DOWN(x)];

			dE = 2 * s->s2[L * y + x] * ( J1 * sumNN + J2 * sumNNN );

      if(*energy + dE < e_min || e_max < *energy + dE)
        return;

			gi = getIndexOfEnergy(*energy, energies);
			gf = getIndexOfEnergy(*energy + dE, energies);

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

template <class T>
int binSearch(std::vector<T>& arr, T what) {
  int low = 0;
  int high = arr.size() - 1;
  while (low <= high) {
    int mid = (low + high) / 2;
    if (arr[mid] > what)
      high = mid - 1;
    else if (arr[mid] < what)
      low = mid + 1;
    else
      return mid;
  }
  return -1; // indicate not found 
}

template<typename T>
void to_file(std::string filename, std::vector<T> v){
  std::ofstream f(filename, std::ios::out);
  f.precision(30);
  for (auto& it : v) {
      f << it << std::endl;
  }
  f.close();
}

template<typename T>
void to_file(std::string filename, std::vector<std::vector<T>> v){
  std::ofstream f(filename, std::ios::out);
  f.precision(30);
  for(int i = 0; i < v[0].size(); i++){
    for(int j = 0; j < v.size(); j++){
      f <<  v[j][i] << " ";
    }
    f << std::endl;
  }
}

template<typename T>
void to_file(std::string filename, T* v, size_t len){
  std::ofstream f(filename, std::ios::out);
  f.precision(30);
  for (int i = 0; i < len; i++) {
      f << v[i] << std::endl;
  }
  f.close();
}

void to_file(Lattice* s, std::string filename) {
  std::ofstream latticeFile(filename);
  latticeFile.precision(30);
  for (int i = 0; i < N; i++) {
    latticeFile << s->s1[i] << " " << s->s2[i] << std::endl;
  }

  latticeFile.close();
}

std::vector<gType> glueDOSEstimate(std::vector<std::vector<gType>> &gi){
  std::vector<gType> gtot;    

  // Gule parts together
  gtot.reserve(numIntervals*lenInterval);
  std::copy(gi[0].begin(), gi[0].end()-pre/2, std::back_inserter(gtot));
  for(int i = 1; i < numIntervals; i++){
    double delta = *(gtot.end()-1) - gi[i][0]; //pre/2];
    std::cout << i << ": " << *(gtot.end()-1) << " - " << gi[i][pre/2] << " = "  << delta << std::endl;
    // auto gii = std::ranges::subrange(gi[i].begin()+pre/2, gi[i].end()-pre/2) |
    //            std::views::transform([=](double x){return x+delta;});
    auto gii = std::ranges::subrange(gi[i].begin()+pre/2, gi[i].end()-pre/2) | 
               std::views::transform([=](double x){return x+delta;});
    if(i < numIntervals - 1){
      std::copy(gii.begin(), gii.end(), std::back_inserter(gtot));
    }else{
      std::copy(gii.begin()+pre, gii.begin() + lenInterval, std::back_inserter(gtot));
    }
  }

  return &gtot;
}

void calculateTD(std::string filename, tType T_min, tType T_max, tType dT, 
    std::vector<eType>& possibleEnergyValues, std::vector<gType>& g)
{
  std::vector<gType> DOS(g.size());    	// Density of states

  
  gType maxG = std::accumulate(g.begin(), g.end(), 
      std::numeric_limits<gType>::min(),
      [=](gType x, gType y) {return std::max(x,y); });

  gType logSumExp = std::log(std::accumulate(g.begin(), g.end(), 0.0,
        [=](gType x, gType y) {return x + std::exp(y-maxG);}));

  std::transform(g.begin(), g.end(), DOS.begin(),
      [=](gType i) {return i - logSumExp + N*std::log(2);});


  std::vector<tType> T;				// Vector of temperatures
  for (tType temperature = T_min; temperature <= T_max; temperature += dT) 
      T.push_back(temperature);

  std::vector<gType> Z(T.size());	// Partition function
  std::vector<gType> U(T.size());	// Internal energy
  std::vector<gType> C(T.size());	// Specific heat
  std::vector<gType> V(T.size());	// Energetic fourth-order cumulant

  // Calculation of partial function
  for (unsigned int i = 0; i < Z.size(); i++) {
    for (unsigned int j = 0; j < DOS.size(); j++) {
      gType maxExp = std::accumulate(DOS.begin(), DOS.end(), 
        std::numeric_limits<gType>::min(),
        [=](gType x, gType y) {return std::max(x,y-possibleEnergyValues[j]/T[i]); });
      Z[i] += std::exp(DOS[i]-possibleEnergyValues[j] / T[i] - maxExp);
    }
  }

  // Calculation of thermodynamic quantities
  gType tmp1, tmp2, tmp3;
  for (unsigned int i = 0; i < Z.size(); i++) {
      tmp2 = 0;
      tmp3 = 0;
      for (unsigned int j = 0; j < DOS.size(); j++) {
          gType maxExp = std::accumulate(DOS.begin(), DOS.end(), 
            std::numeric_limits<gType>::min(),
            [=](gType x, gType y) {return std::max(x,y-possibleEnergyValues[j]/T[i]); });
          tmp1 = std::exp(DOS[j]-possibleEnergyValues[j] / T[i] - maxExp);
          U[i] += possibleEnergyValues[j] * tmp1 / Z[i];
          tmp2 += std::pow(possibleEnergyValues[j], 2)*tmp1 / Z[i];
          tmp3 += std::pow(possibleEnergyValues[j], 4)*tmp1 / Z[i];
      }
      C[i] = (tmp2 - std::pow(U[i], 2)) / sq(T[i]);
      V[i] = 1 - tmp3 / (3 * sq(tmp2));
  }

  // Saving data

  std::fstream dataFile(filename, std::ios::out);
  // Writeout dos
  for (int i = 0; i < DOS.size(); i++)
      dataFile << DOS[i] << std::endl;

  // dataFile << "\nSUM G = " << sumG << std::endl;

  // Creating a header
  dataFile << "i\tZ[i]\tU[i]\tC[i]\tV[i]" << std::endl;
  for (unsigned int i = 0; i < Z.size(); i++) {
      dataFile << i << "\t" << Z[i] << "\t" << U[i] << "\t"
          << C[i] << "\t" << V[i] << std::endl;
  }

}

template<typename T>
double average(std::vector<T>& vec){
  double mean = 0;
  
  for(int i = 0; i < vec.size(); i++){
      mean += (vec[i] - mean )/(i+2);
  }

  return mean;
}

/// Randomly flip spins, until we find the interval
void find_state_in_interval(Lattice *s, eType e_min, eType e_max, eType* energy){
  // If we are in the interval, do nothing
  if( *energy < e_min || e_max < *energy ) return;

  // Well, clearly we are not in the interval, so we have to do the work :/
  unsigned int x, y, i;
  eType dE, sumNN, sumNNN;

  // Flip spins randomly
  while( true ){
    x = uniR(mt) * L; 
    y = uniR(mt) * L;

    sumNN  = s->s2[L * y  + x ] 
           + s->s2[L * DOWN(y)  + x] 
           + s->s2[L * DOWN(y) + UP(x)];

    sumNNN = s->s1[L * DOWN(y)  + x]
           + s->s1[L * DOWN(y)  + UP(x)]
           + s->s1[L * y + DOWN(x) ]
           + s->s1[L * y + UP(x)]
           + s->s1[L * UP(y) + DOWN(x)]
           + s->s1[L * UP(y) + x ];

    dE = 2 * s->s1[L * y + x] * ( J1 * sumNN + J2 * sumNNN );
    *energy = *energy + dE;
    s->s1[L * y + x] *= -1;

    if( *energy < e_min || e_max < *energy ) return;

    x = uniR(mt) * L; 
    y = uniR(mt) * L;
    sumNN  = s->s1[L * y  + x ] 
           + s->s1[L * UP(y) + x] 
           + s->s1[L * UP(y) + DOWN(x)];

    sumNNN = s->s2[L * DOWN(y)  + x]
           + s->s2[L * DOWN(y)  + UP(x)]
           + s->s2[L * y + DOWN(x)]
           + s->s2[L * y + UP(x)]
           + s->s2[L * UP(y) + x]
           + s->s2[L * UP(y) + DOWN(x)];

    dE = 2 * s->s2[L * y + x] * ( J1 * sumNN + J2 * sumNNN );
    *energy = *energy + dE;
    s->s2[L * y + x] *= -1;

    if( *energy < e_min || e_max < *energy ) return;

    i +=2; 
    // if(i%1000==0)
    //   std::cout << i << " " << *energy << std::endl;
  }
}
