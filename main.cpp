#include "global.h"

#include <Kokkos_Core.hpp>

auto main() -> int {
  Kokkos::initialize();
  {
    NTTArray<real_t*> evens{1000}
    NTTArray<real_t*> odds{1000};
  }
  Kokkos::finalize();

  return 0;
}
