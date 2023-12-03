#include "linear.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int foo(int *A, int n) {
  unsigned sum = 0;
#pragma clang loop vectorize(enable)
  for (int i = 0; i < n; ++i)
    sum += A[i] + 5;
  return sum;
}

void Linear::apply(F_TY *input) {
  F_TY *output = this->val;
  size_t osize = this->output_size;
  size_t isize = this->input_size;
  for (int i = 0; i < osize; i++) {
    // assert(!isbadf(this->bias[i]));
    F_TY *w_on = this->weights + i * isize;
    F_TY r = 0;
#pragma clang loop vectorize(enable)
    for (int j = 0; j < isize; j++) {
      r += w_on[j] * input[j];
      // assert(!isinf(w_on[j]));
      // assert(!isinf(input[j]));
      // assert(!isbadf(r));
    }
    output[i] = r + this->bias[i];
    // assert(!isbadf(output[i]));
  }
}

void Linear::zero_grad() {
  memset(this->grad, 0, this->output_size * sizeof(double));
  memset(this->weight_grad, 0,
         this->input_size * this->output_size * sizeof(double));
}

double *Linear::grad_ptr() { return this->grad; }

F_TY *Linear::output() { return this->val; }

void Linear::update_input_grad(F_TY *input, double *input_grad) {
  for (int i = 0; i < this->output_size; i++) {
    F_TY *w_on = this->weights + i * this->input_size;
    double *w_grad_on = this->weight_grad + i * this->input_size;
    double g = this->grad[i];
    for (int j = 0; j < this->input_size; j++) {

      if (input_grad != NULL) {
        input_grad[j] += g * w_on[j];
        // assert(!isbadf(input_grad[j]));
      }
      w_grad_on[j] += g * input[j];
      // assert(!isbadf(w_grad_on[j]));
    }
  }
}

void Linear::step(double step_size) {
  size_t s = this->input_size * this->output_size;
  for (int i = 0; i < s; i++) {
    this->weights[i] -= step_size * this->weight_grad[i];
    // assert(!isbadf(this->weights[i]));
  }
  for (int i = 0; i < this->output_size; i++) {
    this->bias[i] -= step_size * this->grad[i];
  }
}

Linear::Linear(int input_size, int output_size, double min_weight,
               double max_weight, double min_bias, double max_bias) {
  this->input_size = input_size;
  this->output_size = output_size;
  this->weights = (F_TY *)malloc(input_size * output_size * sizeof(F_TY));
  this->weight_grad =
      (double *)malloc(input_size * output_size * sizeof(double));
  this->bias = (F_TY *)malloc(output_size * sizeof(F_TY));
  this->val = (F_TY *)malloc(output_size * sizeof(F_TY));
  this->grad = (double *)malloc(output_size * sizeof(double));
  for (int i = 0; i < input_size * output_size; i++) {
    this->weights[i] = rand_f() * (max_weight - min_weight) + min_weight;
  }
  for (int i = 0; i < output_size; i++) {
    this->bias[i] = rand_f() * (max_bias - min_bias) + min_bias;
  }
}
