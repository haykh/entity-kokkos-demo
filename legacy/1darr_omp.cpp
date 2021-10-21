#include <cstddef>
#include <iostream>
#include <vector>

#include <omp.h>

auto main() -> int {
  using real_t = double;
  {
    int niter {10};
    int size[] = {1000000};

    auto A = new double[size[0]];
    auto B = new double[size[0]];

    auto timer1 = omp_get_wtime();
    std::size_t i;
#pragma omp parallel for
    for (i = 0; i < size[0]; ++i) {
      A[i] = static_cast<double>(i);
      B[i] = -static_cast<double>(i) * 0.5 / static_cast<double>(niter);
    }
    timer1 = omp_get_wtime() - timer1;

    double timer2{0.0};
    for (std::size_t n{0}; n < niter; ++n) {
      double coeff = 1.0 / static_cast<double>(niter);
      auto timer2_ = omp_get_wtime();
#pragma omp parallel for
      for (i = 0; i < size[0]; ++i) {
        A[i] += 2.0 * (B[i] - B[i - 1] + 2.0 * B[i + 1]) + coeff;
      }
      timer2_ = omp_get_wtime() - timer2_;
      timer2 += timer2_;
    }

    double sum{0.0};
    auto timer3 = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < size[0]; ++i) {
      sum += A[i] / static_cast<double>(size[0]);
    }
    timer3 = omp_get_wtime() - timer3;
    
    std::cout << "sum: " << sum << "\n";
    std::cout << "init " << timer1 << "\n";
    std::cout << "upd " << timer2 << "\n";
    std::cout << "reduce " << timer3 << "\n";
  }
  return 0;
}
