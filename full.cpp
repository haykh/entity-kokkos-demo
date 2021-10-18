#include "defs.h"
#include <Kokkos_Random.hpp>

#include <iostream>
#include <iomanip>

using size_t = unsigned long int;

#define N_GHOSTS 2

struct Particles {
  NTTArray<real_t*> ux, uy, uz;
  NTTArray<real_t*> x, y, z;
  size_t npart;
  Particles(size_t npart) :
    npart{npart},
    ux{"Ux", npart}, uy{"Uy", npart}, uz{"Uz", npart},
    x{"X", npart}, y{"Y", npart}, z{"Z", npart} {}

  auto loopAll() const -> ntt_1drange_t {
    return ntt_1drange_t(0, npart);
  }
};
struct Fields {
  NTTArray<real_t**> jx, jy, jz;

  size_t sx, sy;
  real_t xmin, xmax, ymin, ymax;
  real_t dxy;
  Fields(size_t sx, size_t sy) :
    sx{sx}, sy{sy},
    jx{"Jx", sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS},
    jy{"Jy", sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS},
    jz{"Jz", sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS}
    {}
  auto loopAll() const -> ntt_2drange_t {
    return ntt_2drange_t({0, 0}, {sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS});
  }
};

Inline auto xTOi(const Fields& flds, const real_t& x) -> size_t {
  return N_GHOSTS + static_cast<size_t>(flds.sx * (x - flds.xmin) / (flds.xmax - flds.xmin));
}
Inline auto yTOj(const Fields& flds, const real_t& y) -> size_t {
  return N_GHOSTS + static_cast<size_t>(flds.sy * (y - flds.ymin) / (flds.ymax - flds.ymin));
}

template<class RandPool>
struct GenRandomParticles {
  RandPool rand_pool;
  using gen_type = typename RandPool::generator_type;

  Particles prtls;
  Fields flds;

  GenRandomParticles(const Fields& flds_, const Particles& prtls_, RandPool rand_pool_)
   : flds{flds_}, prtls{prtls_}, rand_pool{rand_pool_} {}

  Inline void operator() (const size_t p) const {
    gen_type rgen = rand_pool.get_state();
    auto rnd {Kokkos::rand<gen_type, real_t>::draw(rgen, flds.xmin, flds.xmax)};
    prtls.x(p) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, flds.ymin, flds.ymax);
    prtls.y(p) = rnd;
    prtls.z(p) = 0.0;

    real_t gamma { 10.0 };
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.ux(p) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.uy(p) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.uz(p) = rnd;

    rand_pool.free_state(rgen);
  }
};

struct DepositStep {
  Fields flds;
  Particles prtls;
  real_t weighted_charge {1.0};
  real_t dt;
  real_t dx;

  DepositStep(const Fields& flds_, const Particles& prtls_)
    : flds{flds_}, prtls{prtls_},
      dx{(flds.xmax - flds.xmin) / flds.sx},
      dt{0.4 * (flds.xmax - flds.xmin) / flds.sx}
    {}

  Inline void operator() (const size_t p) const {
    real_t gamma_inv {1.0 / std::sqrt(1.0 + prtls.ux(p) * prtls.ux(p) + prtls.uy(p) * prtls.uy(p) + prtls.uz(p) * prtls.uz(p))};
    real_t x1 {prtls.x(p) - dt * prtls.ux(p) * gamma_inv};
    real_t y1 {prtls.y(p) - dt * prtls.uy(p) * gamma_inv};
    real_t z1 {prtls.z(p) - dt * prtls.uz(p) * gamma_inv};

    auto i1 {xTOi(flds, x1)};
    auto j1 {yTOj(flds, y1)};
    auto i2 {xTOi(flds, prtls.x(p))};
    auto j2 {yTOj(flds, prtls.y(p))};

    real_t xr = std::min(
                  dx * static_cast<real_t>(std::min(i1, i2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(i1, i2)),
                    0.5 * (x1 + prtls.x(p))
                  )
                );
    real_t yr = std::min(
                  dx * static_cast<real_t>(std::min(j1, j2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(j1, j2)),
                    0.5 * (y1 + prtls.y(p))
                  )
                );
    real_t zr = std::min(1.0, std::max(static_cast<real_t>(0), 0.5 * (z1 + prtls.z(p))));

    real_t Wx1 {0.5 * (x1 + xr) - dx * static_cast<real_t>(i1)};
    real_t Wy1 {0.5 * (y1 + yr) - dx * static_cast<real_t>(j1)};
    real_t Wx2 {0.5 * (prtls.x(p) + xr) - dx * static_cast<real_t>(i2)};
    real_t Wy2 {0.5 * (prtls.y(p) + yr) - dx * static_cast<real_t>(j2)};
    real_t onemWx1 {1.0 - Wx1};
    real_t onemWy1 {1.0 - Wy1};
    real_t onemWx2 {1.0 - Wx2};
    real_t onemWy2 {1.0 - Wy2};

    real_t Fx1 {-(xr - x1) * weighted_charge};
    real_t Fy1 {-(yr - y1) * weighted_charge};
    real_t Fz1 {-(zr - z1) * weighted_charge};
    real_t Fx2 {-(prtls.x(p) - xr) * weighted_charge};
    real_t Fy2 {-(prtls.y(p) - yr) * weighted_charge};
    real_t Fz2 {-(prtls.z(p) - zr) * weighted_charge};

    Kokkos::atomic_add(&flds.jx(i1  , j1  ), Fx1 * onemWy1);
    Kokkos::atomic_add(&flds.jx(i1  , j1+1), Fx1 * Wy1);

    Kokkos::atomic_add(&flds.jy(i1  , j1  ), Fy1 * onemWx1);
    Kokkos::atomic_add(&flds.jy(i1+1, j1  ), Fy1 * Wx1);

    Kokkos::atomic_add(&flds.jx(i2  , j2  ), Fx2 * onemWy2);
    Kokkos::atomic_add(&flds.jx(i2  , j2+1), Fx2 * Wy2);

    Kokkos::atomic_add(&flds.jy(i2  , j2  ), Fy2 * onemWx2);
    Kokkos::atomic_add(&flds.jy(i2+1, j2  ), Fy2 * Wx2);

    Kokkos::atomic_add(&flds.jz(i1  , j1  ), Fz1 * onemWx1 * onemWy1);
    Kokkos::atomic_add(&flds.jz(i1+1, j1  ), Fz1 * Wx1 * onemWy1);
    Kokkos::atomic_add(&flds.jz(i1  , j1+1), Fz1 * onemWx1 * Wy1);
    Kokkos::atomic_add(&flds.jz(i1+1, j1+1), Fz1 * Wx1 * Wy1);

    Kokkos::atomic_add(&flds.jz(i2  , j2  ), Fz2 * onemWx2 * onemWy2);
    Kokkos::atomic_add(&flds.jz(i2+1, j2  ), Fz2 * Wx2 * onemWy2);
    Kokkos::atomic_add(&flds.jz(i2  , j2+1), Fz2 * onemWx2 * Wy2);
    Kokkos::atomic_add(&flds.jz(i2+1, j2+1), Fz2 * Wx2 * Wy2);
  }
};

void Init(const Fields& flds, const Particles& prtls) {
  using RandPoolType = typename Kokkos::Random_XorShift64_Pool<>;
  RandPoolType rand_pool(53748573);
  Kokkos::parallel_for("init_prtls",
    prtls.loopAll(),
    GenRandomParticles<RandPoolType>(flds, prtls, rand_pool)
  );
}

auto Reduce(const Fields& flds) -> std::tuple<real_t, real_t, real_t> {
  real_t sumx {0.0}, sumy {0.0}, sumz {0.0};
  Kokkos::parallel_reduce("add_x",
    flds.loopAll(),
    Lambda(const size_t i, const size_t j, real_t& sum) {
    sum += flds.jx(i, j);
  }, sumx);
  Kokkos::parallel_reduce("add_y",
    flds.loopAll(),
    Lambda(const size_t i, const size_t j, real_t& sum) {
    sum += flds.jy(i, j);
  }, sumy);
  Kokkos::parallel_reduce("add_z",
    flds.loopAll(),
    Lambda(const size_t i, const size_t j, real_t& sum) {
    sum += flds.jz(i, j);
  }, sumz);

  return {sumx, sumy, sumz};
}

void Reset(const Fields& flds) {
  Kokkos::deep_copy(flds.jx, 0.0);
  Kokkos::deep_copy(flds.jy, 0.0);
  Kokkos::deep_copy(flds.jz, 0.0);
}

void Deposit(const Fields& flds, const Particles& prtls) {
  Kokkos::parallel_for("deposit",
    prtls.loopAll(),
    DepositStep(flds, prtls)
  );
}

void DepositSerial(const Fields& flds, const Particles& prtls) {
  DepositStep dep(flds, prtls);
  for (size_t p {0}; p < prtls.npart; ++p) {
    dep(p);
  }
}

auto main() -> int {
  Kokkos::initialize();
  {
    std::cout << ".... init\n";

    ntt::Timer t_deposit_1("Parallel deposit [atomic, simple]");
    ntt::Timer t_deposit_serial("Serial deposit");

    int SX { 256 }, SY { 256 };
    int ppc = 128;
    size_t NPART { static_cast<size_t>(SX * SY * ppc) };
    int n_iter { 10 };

    Particles myparticles(NPART);
    Fields myfields(256, 256);
    myfields.xmin = -1.0;
    myfields.xmax = 1.0;
    myfields.ymin = -1.0;
    myfields.ymax = 1.0;
    myfields.dxy = (myfields.xmax - myfields.xmin) / myfields.sx;

    // init
    Init(myfields, myparticles);

    DepositSerial(myfields, myparticles);
    auto [rx0, ry0, rz0] = Reduce(myfields);
    Reset(myfields);

    // deposit
    for (int i {0}; i < n_iter; ++i) {
      {
        t_deposit_1.start();
        Deposit(myfields, myparticles);
        t_deposit_1.stop();
      }
      {
        auto [rx, ry, rz] = Reduce(myfields);
        if (!numbersAreEqual(rx, rx0, 1e-8, 1e-6) ||
            !numbersAreEqual(ry, ry0, 1e-8, 1e-6) ||
            !numbersAreEqual(rz, rz0, 1e-8, 1e-6)) {
          std::cout << "PARALLEL TEST FAILED\n";
          std::cout << std::setprecision(17) << rx << " : " << ry << " : " << rz << std::endl;
          std::cout << std::setprecision(17) << rx0 << " : " << ry0 << " : " << rz0 << std::endl;
        }
      }

      Reset(myfields);

      {
        t_deposit_serial.start();
        DepositSerial(myfields, myparticles);
        t_deposit_serial.stop();
      }
      {
        auto [rx, ry, rz] = Reduce(myfields);
        if (!numbersAreEqual(rx, rx0, 1e-8, 1e-6) ||
            !numbersAreEqual(ry, ry0, 1e-8, 1e-6) ||
            !numbersAreEqual(rz, rz0, 1e-8, 1e-6)) {
          std::cout << "SERIAL TEST FAILED\n";
        }
      }

      Reset(myfields);
    }

    t_deposit_serial.printElapsed(ntt::millisecond);
    std::cout << "\n";
    t_deposit_1.printElapsed(ntt::millisecond);
    std::cout << " [x" << t_deposit_serial.getElapsedIn(ntt::millisecond) / t_deposit_1.getElapsedIn(ntt::millisecond) << "]";
    std::cout << "\n";

    std::cout << ".... fin\n";
  }
  Kokkos::finalize();
  return 0;
}
