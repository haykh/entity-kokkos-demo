#include "timer.h"

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

using plog_t = plog::ColorConsoleAppender<plog::FuncMessageFormatter>;

#include <Kokkos_Core.hpp>

#define Lambda KOKKOS_LAMBDA


#if !defined(GPU) && defined(OMP)
#  define AccelExeSpace Kokkos::OpenMP
#  define AccelMemSpace Kokkos::HostSpace
#elif defined(GPU)
#  define AccelExeSpace Kokkos::Cuda
#  define AccelMemSpace Kokkos::CudaSpace
#else
#  define AccelExeSpace Kokkos::Serial
#  define AccelMemSpace Kokkos::HostSpace
#endif

#define HostMemSpace Kokkos::HostSpace
#if defined (OMP)
#  define HostExeSpace Kokkos::OpenMP
#else
#  define HostExeSpace Kokkos::Serial
#endif

template<typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

using NTTRange = Kokkos::RangePolicy<AccelExeSpace>;

using NTT3DRange = Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>;

#include <cstddef>
#include <iostream>

using index_t = std::size_t;

auto main() -> int {
  plog_t console_appender;
  plog::init(plog::verbose, &console_appender);

  using real_t = double;
  Kokkos::initialize();
  {
    int niter {100};
    NTTArray<double***> A("A", 200, 200, 200);
    NTTArray<double***> B("B", 200, 200, 200);

    ntt::Timer timer1("init");
    ntt::Timer timer2("compute");
    ntt::Timer timer3("reduce");

    timer1.start();
    Kokkos::parallel_for("init", 
      NTT3DRange({0, 0, 0}, {200, 200, 200}),
      Lambda (index_t i, index_t j, index_t k) {
        A(i,j,k) = static_cast<double>(i + j + k);
        B(i,j,k) = -static_cast<double>(i + j + k) * 0.5 / niter;
      }
    );
    timer1.stop();

    timer2.start();
    for (index_t n{0}; n < niter; ++n) {
      Kokkos::parallel_for("compute",
        NTT3DRange({0, 0, 0}, {200, 200, 200}),
        Lambda (index_t i, index_t j, index_t k) {
          A(i,j,k) = A(i,j,k) + 2.0 * B(i,j,k);
        }
      );
    }
    timer2.stop();

    double sum{0.0};
    timer3.start();
    Kokkos::parallel_reduce("reduce",
      NTT3DRange({0, 0, 0}, {200, 200, 200}),
      Lambda (index_t i, index_t j, index_t k, double & s) {
        s += A(i,j,k);
      }, sum
    );
    timer3.stop();
    
    timer1.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer2.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer3.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    std::cout << "SUM: " << sum << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
