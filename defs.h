#include "timer.h"
#include <Kokkos_Core.hpp>

#define Lambda    KOKKOS_LAMBDA
#define Inline    KOKKOS_INLINE_FUNCTION

#if !defined(GPUENABLED) && defined(OMPENABLED)
#  define AccelExeSpace Kokkos::OpenMP
#  define AccelMemSpace Kokkos::HostSpace
#elif defined(GPUENABLED)
#  define AccelExeSpace Kokkos::Cuda
#  define AccelMemSpace Kokkos::CudaSpace
#else
#  define AccelExeSpace Kokkos::Serial
#  define AccelMemSpace Kokkos::HostSpace
#endif

#define HostMemSpace Kokkos::HostSpace
#if defined(OMPENABLED)
#  define HostExeSpace Kokkos::OpenMP
#else
#  define HostExeSpace Kokkos::Serial
#endif

#ifdef SINGLE_PRECISION
using real_t = float;
inline constexpr float ONE {1.0f};
#else
using real_t = double;
inline constexpr double ONE {1.0};
#endif

using range_t = Kokkos::RangePolicy<AccelExeSpace>::member_type;

template <typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

template <typename T>
using NTTAtomicArray = Kokkos::View<T, AccelMemSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;


using ntt_1drange_t = Kokkos::RangePolicy<AccelExeSpace>;
using ntt_2drange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>;
using ntt_3drange_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>;

auto NTT1DRange(const std::vector<long int>&) -> ntt_1drange_t;
auto NTT1DRange(const long int&, const long int&) -> ntt_1drange_t;
auto NTT2DRange(const std::vector<long int>&, const std::vector<long int>&) -> ntt_2drange_t;
auto NTT3DRange(const std::vector<long int>&, const std::vector<long int>&) -> ntt_3drange_t;

constexpr double C_DOUBLE_ABS_EPSILON{1e-12};
constexpr double C_DOUBLE_REL_EPSILON{1e-8};
constexpr float C_FLOAT_ABS_EPSILON{1e-6f};
constexpr float C_FLOAT_REL_EPSILON{1e-8f};

template<typename T>
auto numbersAreEqual(T a, T b, T absEpsilon, T relEpsilon) -> bool {
  double diff{std::abs(a - b)};
  if (diff <= absEpsilon) { return true; }
  a = std::abs(a);
  b = std::abs(b);
  T min {std::min(a, b)};
  a -= min;
  b -= min;
  return (diff <= (std::max(std::abs(a), std::abs(b)) * relEpsilon));
}

auto numbersAreEqual(float a, float b) -> bool {
  return numbersAreEqual(a, b, C_FLOAT_ABS_EPSILON, C_FLOAT_REL_EPSILON);
}
auto numbersAreEqual(double a, double b) -> bool {
  return numbersAreEqual(a, b, C_DOUBLE_ABS_EPSILON, C_DOUBLE_REL_EPSILON);
}
