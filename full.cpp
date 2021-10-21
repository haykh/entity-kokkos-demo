#include "defs.h"
#include "objects.h"
#include "aux.h"

#include <args.hxx>

#include <stdexcept>
#include <iostream>
#include <iomanip>

auto main(int argc, char **argv) -> int {
  args::ArgumentParser parser("Kokkos `entity` performance test");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  int ppc {12};
  int SX { 256 }, SY { 256 };
  int n_iter {10};
  args::ValueFlag<int> ppc_parse(parser, "integer", "Particles per cell", {'p', "ppc"});
  args::ValueFlag<int> sx_parse(parser, "integer", "Size in x", {'x', "sx"});
  args::ValueFlag<int> sy_parse(parser, "integer", "Size in y", {'y', "sy"});
  args::ValueFlag<int> niter_parse(parser, "integer", "Number of iterations", {'n', "niter"});
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  if (ppc_parse) ppc = args::get(ppc_parse);
  if (sx_parse) SX = args::get(sx_parse);
  if (sy_parse) SY = args::get(sy_parse);
  if (niter_parse) n_iter = args::get(niter_parse);

  std::cout << "using: [" << SX << " x " << SY << "] with " << ppc << " ppc and " << n_iter << " iterations\n";

  Kokkos::initialize();
  {
    std::cout << ".... init\n";

    ntt::Timer t_deposit_1("Parallel deposit [atomic, simple]");
    ntt::Timer t_move_1("Parallel move [simple]");
    ntt::Timer t_deposit_serial("Serial deposit");
    ntt::Timer t_move_serial("Serial move");

    size_t NPART { static_cast<size_t>(SX * SY * ppc) };

    std::cout << "Memory: " << NPART * 8 * 6 / 1e9 << " G\n";

    Particles myparticles(NPART);
    Fields myfields(SX, SY);
    myfields.xmin = -1.0;
    myfields.xmax = 1.0;
    myfields.ymin = -1.0;
    myfields.ymax = 1.0;
    myfields.dxy = (myfields.xmax - myfields.xmin) / myfields.sx;

    std::uniform_real_distribution<real_t> unif(0.0, 1.0);
    std::default_random_engine re;
    Kokkos::View<real_t*, HostMemSpace> px_init("px", myparticles.npart);
    Kokkos::View<real_t*, HostMemSpace> py_init("py", myparticles.npart);
    Kokkos::View<real_t*, HostMemSpace> pz_init("pz", myparticles.npart);
    Kokkos::View<real_t*, HostMemSpace> pux_init("pux", myparticles.npart);
    Kokkos::View<real_t*, HostMemSpace> puy_init("puy", myparticles.npart);
    Kokkos::View<real_t*, HostMemSpace> puz_init("puz", myparticles.npart);

    real_t gamma { 10.0 };
    for (int i {0}; i < myparticles.npart; ++i) {
      px_init(i) = myfields.xmin + (myfields.xmax - myfields.xmin) * unif(re);
      py_init(i) = myfields.ymin + (myfields.ymax - myfields.ymin) * unif(re);
      pux_init(i) = 2.0 * (unif(re) - 0.5) * gamma;
      puy_init(i) = 2.0 * (unif(re) - 0.5) * gamma;
      puz_init(i) = 2.0 * (unif(re) - 0.5) * gamma;
    }
    Kokkos::deep_copy (myparticles.x, px_init);
    Kokkos::deep_copy (myparticles.y, py_init);
    Kokkos::deep_copy (myparticles.z, pz_init);
    Kokkos::deep_copy (myparticles.ux, pux_init);
    Kokkos::deep_copy (myparticles.uy, puy_init);
    Kokkos::deep_copy (myparticles.uz, puz_init);

    // init
    //Init(myfields, myparticles);

    {
      DepositStep_Host dep(myfields, myparticles);
      DepositSerial(dep);
    }
    auto [rx0, ry0, rz0] = Reduce(myfields);
    std::cout << rx0 << " " << ry0 << " " << rz0 << "\n";
    Reset(myfields);
    Kokkos::deep_copy (myparticles.x, px_init);
    Kokkos::deep_copy (myparticles.y, py_init);
    Kokkos::deep_copy (myparticles.z, pz_init);
    Kokkos::deep_copy (myparticles.ux, pux_init);
    Kokkos::deep_copy (myparticles.uy, puy_init);
    Kokkos::deep_copy (myparticles.uz, puz_init);

    // deposit
    for (int i {0}; i < n_iter; ++i) {
      {
        t_deposit_1.start();
        Deposit(myfields, myparticles);
        t_deposit_1.stop();
      }
      {
        t_move_1.start();
        Move(myfields, myparticles);
        t_move_1.stop();
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
      Kokkos::deep_copy (myparticles.x, px_init);
      Kokkos::deep_copy (myparticles.y, py_init);
      Kokkos::deep_copy (myparticles.z, pz_init);
      Kokkos::deep_copy (myparticles.ux, pux_init);
      Kokkos::deep_copy (myparticles.uy, puy_init);
      Kokkos::deep_copy (myparticles.uz, puz_init);

      {
        DepositStep_Host dep(myfields, myparticles);
        t_deposit_serial.start();
        DepositSerial(dep);
        t_deposit_serial.stop();
      }
      {
        MoveStep_Host mov(myfields, myparticles);
        t_move_serial.start();
        MoveSerial(mov);
        t_move_serial.stop();
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
      Kokkos::deep_copy (myparticles.x, px_init);
      Kokkos::deep_copy (myparticles.y, py_init);
      Kokkos::deep_copy (myparticles.z, pz_init);
      Kokkos::deep_copy (myparticles.ux, pux_init);
      Kokkos::deep_copy (myparticles.uy, puy_init);
      Kokkos::deep_copy (myparticles.uz, puz_init);
    }

    t_deposit_serial.printElapsed(ntt::millisecond);
    std::cout << "\n";
    t_deposit_1.printElapsed(ntt::millisecond);
    std::cout << " [x" << t_deposit_serial.getElapsedIn(ntt::millisecond) / t_deposit_1.getElapsedIn(ntt::millisecond) << "]";
    std::cout << "\n\n";

    t_move_serial.printElapsed(ntt::millisecond);
    std::cout << "\n";
    t_move_1.printElapsed(ntt::millisecond);
    std::cout << " [x" << t_move_serial.getElapsedIn(ntt::millisecond) / t_move_1.getElapsedIn(ntt::millisecond) << "]";
    std::cout << "\n\n";
    std::cout << ".... fin\n";
  }
  Kokkos::finalize();
  return 0;
}
