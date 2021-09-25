#include "timer.h"

#include <Kokkos_Core.hpp>
#define KL KOKKOS_LAMBDA
#define AccelMemSpace Kokkos::HostSpace
#define oneDArray(T) Kokkos::View<T*, AccelMemSpace>

#include <cstddef>
#include <iostream>

auto main() -> int {
  using real_t = double;
  //using memspace = Kokkos::CudaSpace;
  // using memspace = Kokkos::HostSpace;

  //template<typename T>
  //using oneDArray_t = Kokkos::View<T*, memspace>;
  using index_t = const std::size_t;

  Kokkos::initialize();
  {
  std::size_t N{10000000};
  oneDArray(real_t) A("a", N);
  oneDArray(real_t) B("b", N);

  // init array
  timer::Timer timer("kokkos");
  timer.start();

  Kokkos::parallel_for("init", N,
    KL (index_t n) {
      A(n) = 1.1;
      B(n) = 2.2;
    }
  );

  Kokkos::parallel_for("add", N,
    KL (index_t n) {
      A(n) = A(n) + B(n);
    }
  );

  real_t sum{0.0};
  Kokkos::parallel_reduce("sum", N,
    KL (index_t n, real_t &sum) {
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
