#include "timer.h"

#include <Kokkos_Core.hpp>

#define Lambda KOKKOS_LAMBDA

//#if !defined(GPU) && defined(OMP)
//#  define AccelExeSpace Kokkos::OpenMP
//#  define AccelMemSpace Kokkos::HostSpace
//#elif defined(GPU)
//#  define AccelExeSpace Kokkos::Cuda
//#  define AccelMemSpace Kokkos::CudaSpace
//#else
//#  define AccelExeSpace Kokkos::Serial
//#  define AccelMemSpace Kokkos::HostSpace
//#endif

//#define HostMemSpace Kokkos::HostSpace
//#if defined (OMP)
//#  define HostExeSpace Kokkos::OpenMP
//#else
//#  define HostExeSpace Kokkos::Serial
//#endif

//template<typename T>
//using NTTArray = Kokkos::View<T, AccelMemSpace>;

//using NTTRange = Kokkos::RangePolicy<AccelExeSpace>;

//using NTT3DRange = Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>;

template<typename T>
using NTTArray = Kokkos::View<T>;
using NTTRange = Kokkos::RangePolicy<>;
using NTT3DRange = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

#include <cstddef>
#include <iostream>
#include <vector>

using index_t = std::size_t;

auto main() -> int {
  using real_t = double;
  Kokkos::initialize();
  {
    int niter {100};
    int tiles[] = {0, 0, 0};
    int size[] = {400, 400, 400};

    NTTArray<double***> A("A", size[0], size[1], size[2]);
    NTTArray<double***> B("B", size[0], size[1], size[2]);

    ntt::Timer timer1("init");
    ntt::Timer timer2("compute");
    ntt::Timer timer3("reduce");

    timer1.start();
    Kokkos::parallel_for("init", 
      NTT3DRange({0, 0, 0}, size, tiles),
      Lambda (index_t i, index_t j, index_t k) {
        A(i,j,k) = static_cast<double>(i + j + k);
        B(i,j,k) = -static_cast<double>(i + j + k) * 0.5 / static_cast<double>(niter);
      }
    );
    timer1.stop();

    timer2.start();
    for (index_t n{0}; n < niter; ++n) {
      double coeff = 1.0 / static_cast<double>(niter);
      Kokkos::parallel_for("compute",
        NTT3DRange({0, 0, 0}, size, tiles),
        Lambda (index_t i, index_t j, index_t k) {
          A(i,j,k) = A(i,j,k) + 2.0 * B(i,j,k) + coeff;
        }
      );
    }
    timer2.stop();

    double sum{0.0};
    timer3.start();
    Kokkos::parallel_reduce("reduce",
      NTT3DRange({0, 0, 0}, size, tiles),
      Lambda (index_t i, index_t j, index_t k, double & s) {
        s += A(i,j,k) / static_cast<double>(size[0] * size[1] * size[2]);
      }, sum
    );
    timer3.stop();
    
    timer1.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer2.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer3.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";

    unsigned long int sz = size[0] * size[1] * size[2];
    double Gbytes = 1.0e-9 * static_cast<double>( sizeof(double) * (sz * niter * 2) );
    std::cout << "Bandwith : " << Gbytes / timer2.getElapsedIn(ntt::second) << " [GB/s]\n";
    std::cout << (std::abs(sum - 1.0) < 1.0e-8 ? "Test Passed" : "ERROR") << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
