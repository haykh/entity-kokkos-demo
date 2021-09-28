#include "timer.h"

#include <Kokkos_Core.hpp>

#define Lambda KOKKOS_LAMBDA

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
    int size[] = {8000000};

    NTTArray<double*> A("A", size[0]);
    NTTArray<double*> B("B", size[0]);

    ntt::Timer timer1("init");
    ntt::Timer timer2("compute");
    ntt::Timer timer3("reduce");

    timer1.start();
    Kokkos::parallel_for("init", 
      NTTRange(0, size[0]),
      Lambda (index_t i) {
        A(i) = static_cast<double>(i);
        B(i) = -static_cast<double>(i) * 0.5 / static_cast<double>(niter);
      }
    );
    timer1.stop();

    timer2.start();
    for (index_t n{0}; n < niter; ++n) {
      double coeff = 1.0 / static_cast<double>(niter);
      Kokkos::parallel_for("compute",
        NTTRange(0, size[0]),
        Lambda (index_t i) {
          A(i) = A(i) + 2.0 * B(i) + coeff;
        }
      );
    }
    timer2.stop();

    double sum{0.0};
    timer3.start();
    Kokkos::parallel_reduce("reduce",
      NTTRange(0, size[0]),
      Lambda (index_t i, double & s) {
        s += A(i) / static_cast<double>(size[0]);
      }, sum
    );
    timer3.stop();
    
    timer1.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer2.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer3.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";

    double Gbytes = 1.0e-9 * double( sizeof(double) * (size[0] * niter * 2) );
    std::cout << "Bandwith : " << Gbytes / timer2.getElapsedIn(ntt::second) << " [GB/s]\n";
    std::cout << (std::abs(sum - 1.0) < 1.0e-8 ? "Test Passed" : "ERROR") << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
