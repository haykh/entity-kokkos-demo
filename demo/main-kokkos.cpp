#include "global.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>

auto main() -> int {
  Kokkos::initialize();
  {
    int N {100000};
    ntt::NTTArray<real_t*> evens{"evens", N};
    ntt::NTTArray<real_t*> odds{"odds", N};
    
    // filling arrays
    Kokkos::parallel_for(
      "filling",
      ntt::NTTRange<ntt::Dimension::ONE_D>({0}, {N}),
      Lambda (const int i) {
        evens(i) = 1.0 / SQR(2.0 * (i + 1));
        odds(i) = 1.0 / SQR(2.0 * (i + 1) - 1.0);
      }
    );

    ntt::NTTArray<real_t*> sums{"odds", N};
    Kokkos::parallel_for(
      "summing",
      ntt::NTTRange<ntt::Dimension::ONE_D>({0}, {N}),
      Lambda (const int i) {
        sums(i) = evens(i) + odds(i);
      }
    );

    real_t sum {0.0};
    Kokkos::parallel_reduce(
      "sum",
      ntt::NTTRange<ntt::Dimension::ONE_D>({0}, {N}),
      Lambda (const int i, double& s) {
        s += sums(i);
      }, sum
    );

    std::cout << std::sqrt(sum * 6.0) << "\n";
  }
  Kokkos::finalize();

  return 0;
}
