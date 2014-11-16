/* Copyright (c) 2009, Choon Hui Teo and S V N Vishwanathan
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 *
 * Authors      : Choon Hui Teo (ChoonHui.Teo@anu.edu.au)
 *                S V N Vishwanathan (vishy@stat.purdue.edu)
 *
 * Created      : (08/01/2007)
 *
 * Last Updated : (28/04/2009)
 */


#ifndef _TIMER_HPP_
#define _TIMER_HPP_

// #ifndef _MSC_VER
// #include <gmkl-config.h>
// #endif

// #include <time.h>
// #if GMKL_HAVE_SYS_TIME_H
// #include <sys/time.h>
// #else
// #include <time.h>
// #endif
// #if GMKL_HAVE_SYS_PARAM_H
// #include <sys/param.h>
// // #else
// // #include <param.h>
// #endif

// #if GMKL_HAVE_SYS_TIMES_H
// #include <sys/times.h>
// // #else
// // #include <times.h>
// #endif

// #if GMKL_HAVE_SYS_TYPES_H
// #include <sys/types.h>
// // #else
// // #include <types.h>
// #endif

// #if defined(CLK_TCK)
// #define TIMES_TICKS_PER_SEC double(CLK_TCK)
// #elif defined(_SC_CLK_TCK)
// #define TIMES_TICKS_PER_SEC double(sysconf(_SC_CLK_TCK))
// #elif defined(HZ)
// #define TIMES_TICKS_PER_SEC double(HZ)
// #endif

// Keep track of CPU and wall-clock time (in seconds) of program segments

// include different files based on the OS we are working with 

#if _WIN32||_WIN64
#include <windows.h>
#elif __linux__
#include <ctime>
#include <stdio.h>
#else 
#include <sys/time.h>
#endif

class Timer {

private:

  long long  _start_cpu;    // CPU time at start of stopwatch 
  
  double _multiplier;
  
  long long get_count(void);
  
  double get_multiplier(void);

public:

  int  num_calls;        // number of intervals

  // CPU time 
  double total_cpu;     // total 
  double max_cpu;       // longest recorded interval
  double min_cpu;       // shortest recorded interval
  double last_cpu;      // last recorded interval 
  
  Timer();             
  virtual ~Timer(){}   
  
  void   start();          // start stopwatch
  void   stop();           // stop stopwatch
  void   reset();          // reset 
  
  double avg_cpu() const { return total_cpu/num_calls; } 
  
};

#endif
