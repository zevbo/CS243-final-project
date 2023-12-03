#include "msl.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// df(g(x))/dx = df(g(x))/dg(x) * dg(x)/dx
// d(x - e)^2/dx = d(x - e)^2/d(x - e) = 2(x - e) =

double msl_loss(int sz, double *output, double *expected) {
  double sum = 0;
  for (int i = 0; i < sz; i++) {
    double d = (output[i] - expected[i]);
    sum += d * d;
  }
  double loss = sum / sz;
  return loss;
}
double *msl_grad(int sz, double *output, double *expected) {
  double *grad = (double *)malloc(sz * sizeof(double));
  for (int i = 0; i < sz; i++) {
    grad[i] = 2 * (output[i] - expected[i]);
    assert(!isbadf(grad[i]));
  }
  return grad;
}