#ifndef OBJECTS_H
#define OBJECTS_H

#include "defs.h"

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

#endif
