/*
 * fileWrapper.h
 *
 *  Created on: Mar 13, 2019
 *      Author: M. Semjan
 */

#ifndef FILEWRAPPER_H_
#define FILEWRAPPER_H_

#include <string>
#include <cstdio>

class fileWrapper {
	FILE* f;
public:
	/*
	 * Default constructor 
	 *
	 * @param fileName the filename (with full path) to which the handler points
	 * @param flags a list of flags for opening the file that fopen()) accepts
	 */
	fileWrapper(std::string fileName, std::string flags){
		f = std::fopen(fileName.c_str(), flags.c_str());
	}

	/*
	 * Rewind to the beginning of the file
	 */
	void rewind(){
		if(f!=NULL){
			std::rewind(f);
		}
	}

	/*
	 * Write data into the file
	 *
	 * @param data a pointer to the data
	 * @param size size of the data in bytes
	 */
	void write(const void* data, int size){
		std::fwrite(data,size,1,f);
	}

	/*
	 * Flush everything into the file
	 */
	void flush(){
		std::fflush(f);
	}

	/*
	 * Write data into the file and flush
	 *
	 * @param data a pointer to the data
	 * @param size size of the data in bytes
	 */
	void writeAndFlush(const void* data, int size){
		std::fwrite(data,size,1,f);
		std::fflush(f);
	}

	/*
	 * Read data from the file
	 *
	 * @param data a pointer to memory where the read data will be stored
	 * @param size size of the read data in bytes
	 */
	void read(void* data, int size){
		std::fread(data, size, 1, f);
	}

	/*
	 * Close the file. This function is called automatically, when fileWrapper
	 * goes out of scope to prevent memory leaks.
	 */
	void close(){
		if(this->is_opened()){
			std::fclose(f);
			f=NULL;
		}
	}

	/*
	 * Check if the file is still opened.
	 */
	bool is_opened(){
		return f!=NULL;
	}
	
	/*
	 * Destructor - checks if the files is closed and closes it if not.
	 */
	~fileWrapper(){
		this->close();
	}
};

#endif /* FILEWRAPPER_H_ */
