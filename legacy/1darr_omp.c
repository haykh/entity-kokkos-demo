#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  {
    int niter;
    int size;

    double *A;
    double *B;
    int i, n;

    double coeff;
    double timer1, timer2, timer2_, timer3;
    double sum;

    niter = 10;
    size = 1000000;

    A = (double *)malloc(size * sizeof(double));
    B = (double *)malloc(size * sizeof(double));

    timer1 = omp_get_wtime();
#pragma omp parallel for
    for (i = 0; i < size; ++i) {
      A[i] = (double)(i);
      B[i] = -(double)(i) * 0.5 / (double)(niter);
    }
    timer1 = omp_get_wtime() - timer1;

    coeff = 1.0 / (double)(niter);
    timer2 = 0.0;
    for (n = 0; n < niter; ++n) {
      timer2_ = omp_get_wtime();
#pragma omp parallel for
      for (i = 1; i < size - 1; ++i) {
        A[i] += 2.0 * (B[i] - B[i - 1] + 2.0 * B[i + 1]) + coeff;
      }
      timer2_ = omp_get_wtime() - timer2_;
      timer2 += timer2_;
    }

    sum = 0.0;
    timer3 = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < size; ++i) {
      sum += A[i] / (double)(size);
    }
    timer3 = omp_get_wtime() - timer3;
    
    printf("sum %f\n", sum);
    printf("init %f\n", timer1);
    printf("upd %f\n", timer2);
    printf("reduce %f\n", timer3);
  }
  return 0;
}
