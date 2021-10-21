#ifndef AUX_H
#define AUX_H

#include "objects_re.h"

Inline auto xTOi(const Fields& flds, const real_t& x) -> size_t {
  return N_GHOSTS + static_cast<size_t>(flds.sx * (x - flds.xmin) / (flds.xmax - flds.xmin));
}
Inline auto yTOj(const Fields& flds, const real_t& y) -> size_t {
  return N_GHOSTS + static_cast<size_t>(flds.sy * (y - flds.ymin) / (flds.ymax - flds.ymin));
}

void InitHost(const Fields& flds, const Particles& prtls) {
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
    prtls.prtls(p, p_x) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, flds.ymin, flds.ymax);
    prtls.prtls(p, p_y) = rnd;
    prtls.prtls(p, p_z) = 0.0;

    real_t gamma { 10.0 };
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.prtls(p, p_ux) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.prtls(p, p_uy) = rnd;
    rnd = Kokkos::rand<gen_type, real_t>::draw(rgen, -gamma, gamma);
    prtls.prtls(p, p_uz) = rnd;

    rand_pool.free_state(rgen);
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

struct DepositStep_Host {
  Fields flds;
  Particles prtls;
  real_t weighted_charge {1.0};
  real_t dt;
  real_t dx;

  typename Kokkos::View<real_t**>::HostMirror jx_m, jy_m, jz_m;
  typename Kokkos::View<real_t*[6]>::HostMirror prtls_m;

  DepositStep_Host(const Fields& flds_, const Particles& prtls_)
    : flds{flds_}, prtls{prtls_},
      dx{(flds.xmax - flds.xmin) / flds.sx},
      dt{0.4 * (flds.xmax - flds.xmin) / flds.sx},
      jx_m {Kokkos::create_mirror_view(flds.jx)},
      jy_m {Kokkos::create_mirror_view(flds.jy)},
      jz_m {Kokkos::create_mirror_view(flds.jz)},
      prtls_m {Kokkos::create_mirror_view(prtls.prtls)} {
    Kokkos::deep_copy (jx_m, flds.jx);
    Kokkos::deep_copy (jy_m, flds.jy);
    Kokkos::deep_copy (jz_m, flds.jz);

    Kokkos::deep_copy (prtls_m, prtls.prtls);
  }
  ~DepositStep_Host() {
    Kokkos::deep_copy (flds.jx, jx_m);
    Kokkos::deep_copy (flds.jy, jy_m);
    Kokkos::deep_copy (flds.jz, jz_m);

    Kokkos::deep_copy (prtls.prtls, prtls_m);
  }

  Inline void operator() (const size_t p) const {
    real_t gamma_inv {1.0 / std::sqrt(1.0 + prtls_m(p, p_ux) * prtls_m(p, p_ux) + prtls_m(p, p_uy) * prtls_m(p, p_uy) + prtls_m(p, p_uz) * prtls_m(p, p_uz))};
    real_t x1 {prtls_m(p, p_x) - dt * prtls_m(p, p_ux) * gamma_inv};
    real_t y1 {prtls_m(p, p_y) - dt * prtls_m(p, p_uy) * gamma_inv};
    real_t z1 {prtls_m(p, p_z) - dt * prtls_m(p, p_uz) * gamma_inv};

    auto i1 {xTOi(flds, x1)};
    auto j1 {yTOj(flds, y1)};
    auto i2 {xTOi(flds, prtls_m(p, p_x))};
    auto j2 {yTOj(flds, prtls_m(p, p_y))};

    real_t xr = std::min(
                  dx * static_cast<real_t>(std::min(i1, i2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(i1, i2)),
                    0.5 * (x1 + prtls_m(p, p_x))
                  )
                );
    real_t yr = std::min(
                  dx * static_cast<real_t>(std::min(j1, j2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(j1, j2)),
                    0.5 * (y1 + prtls_m(p, p_y))
                  )
                );
    real_t zr = std::min(1.0, std::max(static_cast<real_t>(0), 0.5 * (z1 + prtls_m(p, p_z))));

    real_t Wx1 {0.5 * (x1 + xr) - dx * static_cast<real_t>(i1)};
    real_t Wy1 {0.5 * (y1 + yr) - dx * static_cast<real_t>(j1)};
    real_t Wx2 {0.5 * (prtls_m(p, p_x) + xr) - dx * static_cast<real_t>(i2)};
    real_t Wy2 {0.5 * (prtls_m(p, p_y) + yr) - dx * static_cast<real_t>(j2)};
    real_t onemWx1 {1.0 - Wx1};
    real_t onemWy1 {1.0 - Wy1};
    real_t onemWx2 {1.0 - Wx2};
    real_t onemWy2 {1.0 - Wy2};

    real_t Fx1 {-(xr - x1) * weighted_charge};
    real_t Fy1 {-(yr - y1) * weighted_charge};
    real_t Fz1 {-(zr - z1) * weighted_charge};
    real_t Fx2 {-(prtls_m(p, p_x) - xr) * weighted_charge};
    real_t Fy2 {-(prtls_m(p, p_y) - yr) * weighted_charge};
    real_t Fz2 {-(prtls_m(p, p_z) - zr) * weighted_charge};

    Kokkos::atomic_add(&jx_m(i1  , j1  ), Fx1 * onemWy1);
    Kokkos::atomic_add(&jx_m(i1  , j1+1), Fx1 * Wy1);

    Kokkos::atomic_add(&jy_m(i1  , j1  ), Fy1 * onemWx1);
    Kokkos::atomic_add(&jy_m(i1+1, j1  ), Fy1 * Wx1);

    Kokkos::atomic_add(&jx_m(i2  , j2  ), Fx2 * onemWy2);
    Kokkos::atomic_add(&jx_m(i2  , j2+1), Fx2 * Wy2);

    Kokkos::atomic_add(&jy_m(i2  , j2  ), Fy2 * onemWx2);
    Kokkos::atomic_add(&jy_m(i2+1, j2  ), Fy2 * Wx2);

    Kokkos::atomic_add(&jz_m(i1  , j1  ), Fz1 * onemWx1 * onemWy1);
    Kokkos::atomic_add(&jz_m(i1+1, j1  ), Fz1 * Wx1 * onemWy1);
    Kokkos::atomic_add(&jz_m(i1  , j1+1), Fz1 * onemWx1 * Wy1);
    Kokkos::atomic_add(&jz_m(i1+1, j1+1), Fz1 * Wx1 * Wy1);

    Kokkos::atomic_add(&jz_m(i2  , j2  ), Fz2 * onemWx2 * onemWy2);
    Kokkos::atomic_add(&jz_m(i2+1, j2  ), Fz2 * Wx2 * onemWy2);
    Kokkos::atomic_add(&jz_m(i2  , j2+1), Fz2 * onemWx2 * Wy2);
    Kokkos::atomic_add(&jz_m(i2+1, j2+1), Fz2 * Wx2 * Wy2);
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
    real_t gamma_inv {1.0 / std::sqrt(1.0 + prtls.prtls(p, p_ux) * prtls.prtls(p, p_ux) + prtls.prtls(p, p_uy) * prtls.prtls(p, p_uy) + prtls.prtls(p, p_uz) * prtls.prtls(p, p_uz))};
    real_t x1 {prtls.prtls(p, p_x) - dt * prtls.prtls(p, p_ux) * gamma_inv};
    real_t y1 {prtls.prtls(p, p_y) - dt * prtls.prtls(p, p_uy) * gamma_inv};
    real_t z1 {prtls.prtls(p, p_z) - dt * prtls.prtls(p, p_uz) * gamma_inv};

    auto i1 {xTOi(flds, x1)};
    auto j1 {yTOj(flds, y1)};
    auto i2 {xTOi(flds, prtls.prtls(p, p_x))};
    auto j2 {yTOj(flds, prtls.prtls(p, p_y))};

    real_t xr = std::min(
                  dx * static_cast<real_t>(std::min(i1, i2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(i1, i2)),
                    0.5 * (x1 + prtls.prtls(p, p_x))
                  )
                );
    real_t yr = std::min(
                  dx * static_cast<real_t>(std::min(j1, j2) + 1),
                  std::max(
                    static_cast<real_t>(dx * std::max(j1, j2)),
                    0.5 * (y1 + prtls.prtls(p, p_y))
                  )
                );
    real_t zr = std::min(1.0, std::max(static_cast<real_t>(0), 0.5 * (z1 + prtls.prtls(p, p_z))));

    real_t Wx1 {0.5 * (x1 + xr) - dx * static_cast<real_t>(i1)};
    real_t Wy1 {0.5 * (y1 + yr) - dx * static_cast<real_t>(j1)};
    real_t Wx2 {0.5 * (prtls.prtls(p, p_x) + xr) - dx * static_cast<real_t>(i2)};
    real_t Wy2 {0.5 * (prtls.prtls(p, p_y) + yr) - dx * static_cast<real_t>(j2)};
    real_t onemWx1 {1.0 - Wx1};
    real_t onemWy1 {1.0 - Wy1};
    real_t onemWx2 {1.0 - Wx2};
    real_t onemWy2 {1.0 - Wy2};

    real_t Fx1 {-(xr - x1) * weighted_charge};
    real_t Fy1 {-(yr - y1) * weighted_charge};
    real_t Fz1 {-(zr - z1) * weighted_charge};
    real_t Fx2 {-(prtls.prtls(p, p_x) - xr) * weighted_charge};
    real_t Fy2 {-(prtls.prtls(p, p_y) - yr) * weighted_charge};
    real_t Fz2 {-(prtls.prtls(p, p_z) - zr) * weighted_charge};

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

void Deposit(const Fields& flds, const Particles& prtls) {
  Kokkos::parallel_for("deposit",
    prtls.loopAll(),
    DepositStep(flds, prtls)
  );
}

void DepositSerial(const DepositStep_Host& dep) {
  for (size_t p {0}; p < dep.prtls.npart; ++p) {
    dep(p);
  }
}

struct MoveStep_Host {
  Fields flds;
  Particles prtls;
  real_t weighted_charge {1.0};
  real_t dt;
  real_t dx;

  typename Kokkos::View<real_t**>::HostMirror jx_m, jy_m, jz_m;
  typename Kokkos::View<real_t*[6]>::HostMirror prtls_m;

  MoveStep_Host(const Fields& flds_, const Particles& prtls_)
    : flds{flds_}, prtls{prtls_},
      dx{(flds.xmax - flds.xmin) / flds.sx},
      dt{0.4 * (flds.xmax - flds.xmin) / flds.sx},
      jx_m {Kokkos::create_mirror_view(flds.jx)},
      jy_m {Kokkos::create_mirror_view(flds.jy)},
      jz_m {Kokkos::create_mirror_view(flds.jz)},
      prtls_m {Kokkos::create_mirror_view(prtls.prtls)}
  {
    Kokkos::deep_copy (jx_m, flds.jx);
    Kokkos::deep_copy (jy_m, flds.jy);
    Kokkos::deep_copy (jz_m, flds.jz);

    Kokkos::deep_copy (prtls_m, prtls.prtls);
  }
  ~MoveStep_Host() {
    Kokkos::deep_copy (flds.jx, jx_m);
    Kokkos::deep_copy (flds.jy, jy_m);
    Kokkos::deep_copy (flds.jz, jz_m);

    Kokkos::deep_copy (prtls.prtls, prtls_m);
  }

  Inline void operator() (const size_t p) const {
    real_t gamma_inv {1.0 / std::sqrt(1.0 + prtls_m(p, p_ux) * prtls_m(p, p_ux) + prtls_m(p, p_uy) * prtls_m(p, p_uy) + prtls_m(p, p_uz) * prtls_m(p, p_uz))};
    real_t x1 {prtls_m(p, p_x) + dt * prtls_m(p, p_ux) * gamma_inv};
    real_t y1 {prtls_m(p, p_y) + dt * prtls_m(p, p_uy) * gamma_inv};
    real_t z1 {prtls_m(p, p_z) + dt * prtls_m(p, p_uz) * gamma_inv};

    auto i1 {xTOi(flds, x1)};
    auto j1 {yTOj(flds, y1)};
    auto i2 {xTOi(flds, prtls_m(p, p_x))};
    auto j2 {yTOj(flds, prtls_m(p, p_y))};

    prtls_m(p, p_x) = x1 + prtls_m(p, p_ux);
    prtls_m(p, p_x) = y1 + prtls_m(p, p_uy);

    //ux_m(p) += jy_m(1, 2);
    //uy_m(p) += jz_m(2, 2);
    //uz_m(p) += jx_m(1, 1);
    prtls_m(p, p_ux) += jy_m(i1, j2);
    prtls_m(p, p_uy) += jz_m(i2, j2);
    prtls_m(p, p_uz) += jx_m(i1, j1);
  }
};

struct MoveStep {
  Fields flds;
  Particles prtls;
  real_t dt;
  real_t dx;

  MoveStep(const Fields& flds_, const Particles& prtls_)
    : flds{flds_}, prtls{prtls_},
      dx{(flds.xmax - flds.xmin) / flds.sx},
      dt{0.4 * (flds.xmax - flds.xmin) / flds.sx}
    {}

  Inline void operator() (const size_t p) const {
    real_t gamma_inv {1.0 / std::sqrt(1.0 + prtls.prtls(p, p_ux) * prtls.prtls(p, p_ux) + prtls.prtls(p, p_uy) * prtls.prtls(p, p_uy) + prtls.prtls(p, p_uz) * prtls.prtls(p, p_uz))};
    real_t x1 {prtls.prtls(p, p_x) + dt * prtls.prtls(p, p_ux) * gamma_inv};
    real_t y1 {prtls.prtls(p, p_y) + dt * prtls.prtls(p, p_uy) * gamma_inv};
    real_t z1 {prtls.prtls(p, p_z) + dt * prtls.prtls(p, p_uz) * gamma_inv};

    auto i1 {xTOi(flds, x1)};
    auto j1 {yTOj(flds, y1)};
    auto i2 {xTOi(flds, prtls.prtls(p, p_x))};
    auto j2 {yTOj(flds, prtls.prtls(p, p_y))};

    prtls.prtls(p, p_x) = x1 + prtls.prtls(p, p_ux);
    prtls.prtls(p, p_y) = y1 + prtls.prtls(p, p_uy);

    //prtls.ux(p) += flds.jy(1, 2);
    //prtls.uy(p) += flds.jz(2, 2);
    //prtls.uz(p) += flds.jx(1, 1);
    if ((i1 < 0) || (i1 >= 260) || 
        (i2 < 0) || (i2 >= 260) || 
        (j1 < 0) || (j1 >= 260) || 
        (j2 < 0) || (j2 >= 260)) {
      std::cout << "ERRR " << i1 << " " << i2 << " " << j1 << " " << j2 << "\n";
    } else {
      prtls.prtls(p, p_ux) += flds.jy(i1, j2);
      prtls.prtls(p, p_uy) += flds.jz(i2, j2);
      prtls.prtls(p, p_uz) += flds.jx(i1, j1);
    }
  }
};
void Reset(const Fields& flds) {
  Kokkos::deep_copy(flds.jx, 0.0);
  Kokkos::deep_copy(flds.jy, 0.0);
  Kokkos::deep_copy(flds.jz, 0.0);
}

void Move(const Fields& flds, const Particles& prtls) {
  Kokkos::parallel_for("move",
    prtls.loopAll(),
    MoveStep(flds, prtls)
  );
}

void MoveSerial(const MoveStep_Host& mov) {
  for (size_t p {0}; p < mov.prtls.npart; ++p) {
    mov(p);
  }
}

#endif
