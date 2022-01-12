#include "global.h"

#include <Kokkos_Core.hpp>

auto main() -> int {
  Kokkos::initialize();
  {
  }
  Kokkos::finalize();

  return 0;
}
