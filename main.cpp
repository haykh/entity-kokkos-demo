#include "timer.h"

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

using plog_t = plog::ColorConsoleAppender<plog::FuncMessageFormatter>;

#include <Kokkos_Core.hpp>

#define Lambda KOKKOS_LAMBDA

#define HostExeSpace Kokkos::OpenMP
#define HostMemSpace Kokkos::HostSpace

#ifndef GPU
#  define AccelExeSpace Kokkos::OpenMP
#  define AccelMemSpace Kokkos::HostSpace
#else
#  define AccelExeSpace Kokkos::Cuda
#  define AccelMemSpace Kokkos::CudaSpace
#endif

template<typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

using NTTRange = Kokkos::RangePolicy<AccelExeSpace>;

#include <cstddef>
#include <iostream>

auto main() -> int {
  plog_t console_appender;
  plog::init(plog::verbose, &console_appender);

  using real_t = double;
  Kokkos::initialize();
  {
  }
  Kokkos::finalize();
  return 0;
}
