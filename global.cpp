#include "global.h"

#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
#include <cassert>
#include <string>
#include <iomanip>

namespace ntt {

  auto stringifySimulationType(SimulationType sim) -> std::string {
    switch (sim) {
    case SimulationType::PIC:
      return "PIC";
    case SimulationType::FORCE_FREE:
      return "FF";
    case SimulationType::MHD:
      return "MHD";
    default:
      return "N/A";
    }
  }

  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::OPEN:
      return "Open";
    case BoundaryCondition::USER:
      return "User";
    case BoundaryCondition::COMM:
      return "Communicate";
    default:
      return "N/A";
    }
  }

  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string {
    switch (pusher) {
    case ParticlePusher::BORIS:
      return "Boris";
    case ParticlePusher::VAY:
      return "Vay";
    case ParticlePusher::PHOTON:
      return "Photon";
    default:
      return "N/A";
    }
  }

  template <>
  auto NTTRange<Dimension::ONE_D>(const int (&i1)[1], const int (&i2)[1]) -> RangeND<Dimension::ONE_D> {
    return Kokkos::RangePolicy<AccelExeSpace>(static_cast<range_t>(i1[0]), static_cast<range_t>(i2[0]));
  }

  template <>
  auto NTTRange<Dimension::TWO_D>(const int (&i1)[2], const int (&i2)[2]) -> RangeND<Dimension::TWO_D> {
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
      {static_cast<range_t>(i1[0]), static_cast<range_t>(i1[1])},
      {static_cast<range_t>(i2[0]), static_cast<range_t>(i2[1])});
  }
  template <>
  auto NTTRange<Dimension::THREE_D>(const int (&i1)[3], const int (&i2)[3]) -> RangeND<Dimension::THREE_D> {
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
      {static_cast<range_t>(i1[0]), static_cast<range_t>(i1[1]), static_cast<range_t>(i1[2])},
      {static_cast<range_t>(i2[0]), static_cast<range_t>(i2[1]), static_cast<range_t>(i2[2])});
  }

} // namespace ntt

