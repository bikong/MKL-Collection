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


#ifndef _TIMER_CPP_
#define _TIMER_CPP_

#include <limits>
#include <stdexcept>
#include <sstream>
#include <cassert>

#include "timer.hpp"

#ifdef _MSC_VER
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif


Timer::Timer()
  :_start_cpu(-1),
   num_calls(0),
   total_cpu(0),
   max_cpu(-std::numeric_limits<double>::max()),
   min_cpu(std::numeric_limits<double>::max()),
   last_cpu(std::numeric_limits<double>::max()){
  _multiplier = get_multiplier();
}

double Timer::get_multiplier(void){
#if _WIN32||_WIN64
  LARGE_INTEGER qpfreq;
  QueryPerformanceFrequency(&qpfreq);
  return 1.0/(double)qpfreq.QuadPart;
#elif __linux__
  return 1E-9;
#else 
  return 1E-6;
#endif 
}

long long Timer::get_count(void){
#if _WIN32||_WIN64
  LARGE_INTEGER qpcnt;
  QueryPerformanceCounter(&qpcnt);
  return qpcnt.QuadPart;
#elif __linux__
  struct timespec ts;
  int status = clock_gettime( CLOCK_REALTIME, &ts );
  assert(!status);
  return static_cast<long long>(1000000000UL)*static_cast<long long>(ts.tv_sec) + static_cast<long long>(ts.tv_nsec);
#else 
  struct timeval tv;
  int status = gettimeofday(&tv, NULL);
  assert(!status);
  return static_cast<long long>(1000000)*static_cast<long long>(tv.tv_sec) + static_cast<long long>(tv.tv_usec);
#endif 
}

void Timer::start(){
  _start_cpu = get_count();
  return;
}


void Timer::stop(){
  
  if (_start_cpu < 0){
    std::stringstream os;
    os << "Need to call start() before calling stop()" << std::endl;
		throw std::runtime_error(os.str());
  }
  
  long long _end_cpu = get_count();
  last_cpu = (_end_cpu - _start_cpu)*_multiplier;
  
  _start_cpu = -1;
  total_cpu += last_cpu; 
  max_cpu = std::max(max_cpu, last_cpu);
  min_cpu = std::min(min_cpu, last_cpu);
  
  num_calls++;
  return;
}

void Timer::reset(){
  num_calls = 0;
  _start_cpu = -1;
  total_cpu = 0;
  max_cpu = -std::numeric_limits<double>::max();
  min_cpu = std::numeric_limits<double>::max();
  return; 
}

#endif
