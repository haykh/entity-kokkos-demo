#include "timer.h"

#include <cstddef>
#include <iostream>
#include <vector>

auto main() -> int {
  using real_t = double;
  {
    int niter {100};
    int size[] = {8000000};

    auto A = new double[size[0]];
    auto B = new double[size[0]];

    ntt::Timer timer1("init");
    ntt::Timer timer2("compute");
    ntt::Timer timer3("reduce");

    timer1.start();
    std::size_t i;
    #pragma omp parallel for
    for (i = 0; i < size[0]; ++i) {
      A[i] = static_cast<double>(i);
      B[i] = -static_cast<double>(i) * 0.5 / static_cast<double>(niter);
    }
    timer1.stop();

    timer2.start();
    for (std::size_t n{0}; n < niter; ++n) {
      double coeff = 1.0 / static_cast<double>(niter);
      #pragma omp parallel for
      for (i = 0; i < size[0]; ++i) {
        A[i] = A[i] + 2.0 * B[i] + coeff;
      }
    }
    timer2.stop();

    double sum{0.0};
    timer3.start();
    #pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < size[0]; ++i) {
      sum += A[i] / static_cast<double>(size[0]);
    }
    timer3.stop();
    
    timer1.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer2.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";
    timer3.printElapsed(std::cout, ntt::millisecond);
    std::cout << "\n";

    double Gbytes = 1.0e-9 * double( sizeof(double) * (size[0] * niter * 2) );
    std::cout << "Bandwith : " << Gbytes / timer2.getElapsedIn(ntt::second) << " [GB/s]\n";
    std::cout << (std::abs(sum - 1.0) < 1.0e-8 ? "Test Passed" : "ERROR") << std::endl;
  }
  return 0;
}
