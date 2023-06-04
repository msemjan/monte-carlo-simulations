/*
 * systemSpecific.h
 *
 *  Created on: Mar 13, 2019
 *      Author: Marek Semjan
 */

#ifndef SYSTEMSPECIFIC_H_
#define SYSTEMSPECIFIC_H_
#include <string>
#ifdef WIN_OS
	#include <Windows.h>
  #include "stdafx.h"
  #include <cstdlib>
#endif
#ifdef LINUX_OS
	#include <unistd.h>
#endif

namespace mySys {
	#ifdef LINUX_OS
	typedef pid_t myPidType;
	
	/*
	 * This function returns the PID of a running process
	 */
	myPidType getPid(){
		return getpid();
	}

	/*
	 * This function creates a directory with a given name
	 */
	void mkdir(std::string name){
		system(("mkdir -p " + name).c_str());
	}
	#endif // LINUX_OS

	#ifdef WIN_OS
	typedef DWORD myPidType;

	/*
	 * This function returns the PID of a running process
	 */
	myPidType getPid(){
		return GetCurrentProcessId();
	}

	/*
	 * This function creates a directory with a given name
	 */
	void mkdir(std::string name) {
    std::system(("mkdir "+ name).c_str());
	}
	#endif // WIN_OS


}

#endif /* SYSTEMSPECIFIC_H_ */
