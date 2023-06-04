/**
 *  Provides simple pipe interface for communication with the gnuplot.
 *
 */

#ifndef PLOTTER_H
#define PLOTTER_H
#include <cstdio>
#include <vector>
#include <string>

class GnuplotPipe{
    FILE* pipe;
    FILE* tmp;
public:
    GnuplotPipe(){
        pipe = popen("gnuplot -persist", "w");
        if(pipe==NULL){
            fprinf(stdErr, "Pipeline to gnuplot failed to open!\n");
            throw std::exception();
        }
        tmp = fopen("tmp.txt", "w");
        if(tmp==NULL){
            fprintf(stdErr, "Failed to open temporary file!\n");
            throw std::exception();
        }
    }

    GnuplotPipe(std::string& tmpFileName){
        pipe = popen("gnuplot -persist", "w");
        if(pipe==NULL){
            fprinf(stdErr, "Pipeline to gnuplot failed to open!\n");
            throw std::exception();
        }
        tmp = fopen(tmpFileName.c_str(), "w");
        if(tmp==NULL){
            fprintf(stdErr, "Failed to open temporary file!\n");
            throw std::exception();
        }
    }

    ~GnuplotPipe(){
        if(pipe!=NULL){
            pclose(pipe);
        }
        if(tmp!=NULL){
            fclose(tmp);
        }
    }    
    
    void sendCommand(std::string& s){
        if(pipe!=NULL){
            fprintf(pipe,s.c_str());
        }else{
            fprinf(stdErr, "Pipeline to gnuplot failed to open!\n");
            throw std::exception();
        }
    }

    void clear(){
        if(pipe!=NULL){
            fprintf(pipe,"clear");
        }else{
            fprinf(stdErr, "Pipeline to gnuplot failed to open!\n");
            throw std::exception();
        }
    }

    template<typename T>
    void sendData(std::vector<T>& ydata){
        if(tmp!=NULL){
            rewind(tmp);
            for(int i = 0; i < ydata.size(); i++){
                fprintf(tmp,"%d %f\n", i, ydata[i]);
            }
            fflush(tmp);
        }
    }

    template<typename T>
    void sendData(std::vector<T>& xdata, std::vector<T>& ydata){
        if(xdata.size()!=ydata.size()){
            fprintf(strErr,
                "WARNING: xdata and ydata must have same length! Plotting is skipped\n");
            return;
        }
        if(tmp!=NULL){
            rewind(tmp);
            for(int i = 0; i < ydata.size(); i++){
                fprintf(tmp,"%f %f\n", xdata[i], ydata[i]);
            }
            fflush(tmp);
        }
    }
}


#endif
