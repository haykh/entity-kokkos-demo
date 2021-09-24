#include "timer.h"

#ifdef KOKKOS
#  include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <iostream>

auto main() -> int {
  using real_t = float;
  using memspace = Kokkos::CudaSpace;
  // using memspace = Kokkos::HostSpace;

  Kokkos::initialize();
  {
  std::size_t N{10000000};
  Kokkos::View<real_t*, memspace> A("a", N);
  Kokkos::View<real_t*, memspace> B("b", N);

  // init array
  timer::Timer timer("kokkos");
  timer.start();

  Kokkos::parallel_for("init", N,
    KOKKOS_LAMBDA (const std::size_t n) {
      A(n) = 1.1;
      B(n) = 2.2;
    }
  );

  Kokkos::parallel_for("add", N,
    KOKKOS_LAMBDA (const std::size_t n) {
      A(n) = A(n) + B(n);
    }
  );

  real_t sum{0.0};
  Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA (const std::size_t n, real_t &sum) {
      sum += A(n);
    }, sum
  );
  timer.stop();
  timer.printElapsed(timer::millisecond);
  std::cout << sum / static_cast<float>(N) << "\n";

  }
  Kokkos::finalize();
  return 0;
}

// // **** example 1
// int N = 10000000;
// double value = 16.695311, dvalue = 0.0001;
// auto Sum = [=](const std::size_t i, double &sum) {
//   sum += 1.0 / (static_cast<double>(i) + 1.0);
// };
//
// timer::Timer timer1("kokkos");
// timer::Timer timer2("serial");
//
// double sum1 {0.0};
// timer1.start();
// Kokkos::parallel_reduce(N, Sum, sum1);
// timer1.stop();
//
// double sum2 {0.0};
// timer2.start();
// for (std::size_t i{0}; i < N; ++i) {
//   Sum(i, sum2);
// }
// timer2.stop();
//
// timer1.printElapsed(timer::millisecond);
// timer2.printElapsed(timer::millisecond);
