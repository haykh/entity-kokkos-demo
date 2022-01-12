#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>

auto main() -> int {
  Kokkos::initialize();
  {
    int N {100000};
    Kokkos::View<float*, Kokkos::HostSpace> evens {"evens", N};
    Kokkos::View<float*, Kokkos::HostSpace> odds {"odds", N};
    
    // filling arrays
    Kokkos::parallel_for(
      "filling",
      Kokkos::RangePolicy<Kokkos::OpenMP>({0}, {N}),
      KOKKOS_LAMBDA (const int i) {
        evens(i) = 1.0 / (4.0 * (i + 1) * (i + 1));
        odds(i) = 1.0 / ((2.0 * (i + 1) - 1.0) * (2.0 * (i + 1) - 1.0));
      }
    );

    Kokkos::View<float*, Kokkos::HostSpace> sums {"sums", N};
    Kokkos::parallel_for(
      "summing",
      Kokkos::RangePolicy<Kokkos::OpenMP>({0}, {N}),
      KOKKOS_LAMBDA (const int i) {
        sums(i) = evens(i) + odds(i);
      }
    );

    float sum {0.0};
    Kokkos::parallel_reduce(
      "sum",
      Kokkos::RangePolicy<Kokkos::OpenMP>({0}, {N}),
      KOKKOS_LAMBDA (const int i, float& s) {
        s += sums(i);
      }, sum
    );

    std::cout << std::sqrt(sum * 6.0) << "\n";
  }
  Kokkos::finalize();

  return 0;
}
