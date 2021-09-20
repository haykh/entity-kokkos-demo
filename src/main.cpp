#include "timer.h"

#ifdef KOKKOS
#  include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <iostream>

auto main() -> int {
  Kokkos::initialize();
  {
    int N = 10000000;
    double value = 16.695311, dvalue = 0.0001;
    auto Sum = [=](const std::size_t i, double &sum) {
      sum += 1.0 / (static_cast<double>(i) + 1.0);
    };

    timer::Timer timer1("kokkos");
    timer::Timer timer2("serial");

    double sum1 {0.0};
    timer1.start();
    Kokkos::parallel_reduce(N, Sum, sum1);
    timer1.stop();

    double sum2 {0.0};
    timer2.start();
    for (std::size_t i{0}; i < N; ++i) {
      Sum(i, sum2);
    }
    timer2.stop();
  }
  Kokkos::finalize();
  return 0;
}
