#include "global.h"

#include <Kokkos_Core.hpp>

auto main() -> int {
  Kokkos::initialize();
  {
    ntt::NTTArray<real_t*> evens{"evens", 1000};
    ntt::NTTArray<real_t[1000]> odds{"odds"};
    
    // filling arrays
    Kokkos::parallel_for(
      "filling",
      ntt::NTTRange<ntt::Dimension::ONE_D>({0}, {1000}),
      Lambda (int i) {
      }
    );
  }
  Kokkos::finalize();

  return 0;
}
