#ifndef OBJECTS_H
#define OBJECTS_H

#include "defs.h"

enum ParticleVar {
  p_ux = 0, p_uy, p_uz,
  p_x, p_y, p_z
};

enum FieldVar {
  f_jx = 0, f_jy, f_jz
};

struct Particles {
  NTTArray<real_t*[6]> prtls;
  size_t npart;
  Particles(size_t npart) :
    npart{npart}, prtls{"prtl_vars", npart} {}

  auto loopAll() const -> ntt_1drange_t {
    return ntt_1drange_t(0, npart);
  }
};
struct Fields {
  NTTArray<real_t**[3]> flds;

  size_t sx, sy;
  real_t xmin, xmax, ymin, ymax;
  real_t dxy;
  Fields(size_t sx, size_t sy) :
    sx{sx}, sy{sy},
    flds{"Jx", sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS}
    {}
  auto loopAll() const -> ntt_2drange_t {
    return ntt_2drange_t({0, 0}, {sx + 2 * N_GHOSTS, sy + 2 * N_GHOSTS});
  }
};

#endif
