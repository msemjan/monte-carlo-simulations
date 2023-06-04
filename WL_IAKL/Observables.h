#ifndef OBSERVABLE_H
#define OBSERVABLE_H

#include <fstream>      // Work with files
#include <vector>       // Containers to store things
#include <algorithm>    // accumulate, for_each

typedef double eType;   // Type for energy
typedef double tType;   // Type for temperature
typedef int mType;      // Type for magnetization
typedef double meType;  // Type for mean values of energy
typedef double mmType;  // Type for mean values of magnetization

/**
 * Class for storing values of the observable quatities.
 *
 */
class Observable{
    public:
    //std::vector<tType> beta;            // inverse temperature
    std::vector<eType> e;               // internal energy
    std::vector<mType> m1, m2, m3;      // sublattice magnetization
    
    std::vector<meType> meanE, meanESq;               // mean values of energy
    std::vector<mmType> meanMSq;                      // mean value of sqare of total magnetization
    std::vector<mmType> meanM1, meanM2, meanM3;       // mean values of sublattice magnetizations
    std::vector<mmType> meanM1Sq, meanM2Sq, meanM3Sq; // mean values of squares of sublattice magnetization

    /*
     * Constructor - allocates memory for the arrays storing observables
     *
     * @param numSweeps number of sweeps
     * @param numTemperatures number of temperatures
     */
    Observable(unsigned int numSweeps, unsigned int numTemperatures){
        // Resizing
        e.resize(numSweeps);
        m1.resize(numSweeps);
        m2.resize(numSweeps);
        m3.resize(numSweeps);
        
        meanE.resize(numTemperatures);
        meanESq.resize(numTemperatures);
        meanM1.resize(numTemperatures);
        meanM2.resize(numTemperatures);
        meanM3.resize(numTemperatures);
        meanM1Sq.resize(numTemperatures);
        meanM2Sq.resize(numTemperatures);
        meanM3Sq.resize(numTemperatures);
        meanMSq.resize(numTemperatures);

        // Filling with zeros
        std::fill(meanE.begin(), meanE.end(), (meType)0);
        std::fill(meanESq.begin(), meanESq.end(), (meType)0);
        std::fill(meanM1.begin(), meanM1.end(), (mmType)0);
        std::fill(meanM2.begin(), meanM2.end(), (mmType)0);
        std::fill(meanM3.begin(), meanM3.end(), (mmType)0);
        std::fill(meanM1Sq.begin(), meanM1Sq.end(), (mmType)0);
        std::fill(meanM2Sq.begin(), meanM2Sq.end(), (mmType)0);
        std::fill(meanM3Sq.begin(), meanM3Sq.end(), (mmType)0);
        std::fill(meanMSq.begin(), meanMSq.end(), (mmType)0);
    }

    /*
     * Sets observables obtained at a given sweep and temperatures
     *
     * @param sweep the sweep at which the values were obtained
     * @param temp the temperature at which the values were obtained
     * @param e the energy 
     * @param m1 the magnetization of the 1th sublattice
     * @param m2 the magnetization of the 2nd sublattice
     * @param m3 the magnetization of the 3rn sublattice
     */
    void setValues(unsigned int sweep, unsigned int temp, eType e, mType m1, mType m2, mType m3){
        // Setting values
        this->e[sweep]  = e;
        this->m1[sweep] = m1;
        this->m2[sweep] = m2;
        this->m3[sweep] = m3;

        // Calculating mean values
        meanE[temp]  += (meType)e  / this->e.size();
        meanM1[temp] += (mmType)m1 / this->m1.size();
        meanM2[temp] += (mmType)m2 / this->m2.size();
        meanM3[temp] += (mmType)m3 / this->m3.size();

        // Calculating mean values ofsquares
        meanESq[temp]  += (meType)(e*e)   / this->e.size();
        meanM1Sq[temp] += (mmType)(m1*m1) / this->m1.size();
        meanM2Sq[temp] += (mmType)(m2*m2) / this->m2.size();
        meanM3Sq[temp] += (mmType)(m3*m3) / this->m3.size();
        meanMSq[temp] +=  (mmType)(m1+m2+m3)*(m1+m2+m3) / this->m1.size();
    }

    /*
     * Saves the data to files in a given folder
     *
     * @param folder2savePath folder to which save files with data (must exist) 
     */
    void saveData(std::string folder2savePath){
        // Opening files
        std::ofstream eFile(folder2savePath  + "/e.txt", std::ios::app);
        std::ofstream m1File(folder2savePath + "/m1.txt", std::ios::app);
        std::ofstream m2File(folder2savePath + "/m2.txt", std::ios::app);
        std::ofstream m3File(folder2savePath + "/m3.txt", std::ios::app);
        
        // Writing data to the file
        eFile.precision(20);
        std::for_each(e.begin(), 
                      e.end(), 
                      [&](eType e){
                        eFile << e << "\t";
                      });
        eFile << std::endl;

        m1File.precision(20);
        std::for_each(m1.begin(), 
                      m1.end(), 
                      [&](mType m){
                        m1File << m << "\t";
                      });
        m1File << std::endl;

        m2File.precision(20);
        std::for_each(m2.begin(), 
                      m2.end(), 
                      [&](mType m){
                        m2File << m << "\t";
                      });
        m2File << std::endl;


        m3File.precision(20);
        std::for_each(m3.begin(), 
                      m3.end(), 
                      [&](mType m){
                        m3File << m << "\t";
                      });
        m3File << std::endl;

        // Closing files
        eFile.close();
        m1File.close();
        m2File.close();
        m3File.close();
    }

    /*
     * Save the data in a binary format in a given directory (NOT IMPLEMENTED)
     *
     * @param folder2savePath folder to which save files with data (must exist) 
     */
    void saveBinData(std::string folder2savePath){
    
    }
   
    /*
     * Generates a header for calculated mean values
     *
     * @param location folder in which the means file with the header will be created
     */
    void makeMeanHeader(std::string location){
        // Opening the file
        std::ofstream meansFile(location+"/means.txt", std::ios::app);
        
        // Writing the data
        meansFile << "t\tE\tE^2\tM1\tM2\tM3\tM\tM1^2\tM2^2\tM3^2\tM^2" <<
            std::endl;

        // Closing the file
        meansFile.close();
    }

    /*
     * Saves the mean obtained during a simulation
     *
     * @param location folder in which the data will be saved
     * @param t temperature
     * @param temp index of the temperature
     */
    void saveMeanData(std::string location, tType t, unsigned int temp){
        // Opening the file
        std::ofstream meansFile(location+"/means.txt", std::ios::app);
        
        // Writing the data
        meansFile.precision(20);
        meansFile << t      << "\t" 
            << meanE[temp]  << "\t" 
            << meanESq[temp]<< "\t" 
            << meanM1[temp] << "\t" 
            << meanM2[temp] << "\t" 
            << meanM3[temp] << "\t" 
            << meanM1[temp] + meanM2[temp] + meanM3[temp] << "\t"
            << meanM1Sq[temp] << "\t" 
            << meanM2Sq[temp] << "\t" 
            << meanM3Sq[temp] << "\t"
            << meanMSq[temp]  << std::endl;

        // Closing the file
        meansFile.close();
    }
};
#endif
