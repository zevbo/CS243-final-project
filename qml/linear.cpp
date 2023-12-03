#include "linear.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Linear::apply(double *input) {
  double *output = this->val;
  for (int i = 0; i < this->output_size; i++) {
    assert(!isbadf(this->bias[i]));
    double r = 0;
    double *w_on = this->weights + i * this->input_size;
    for (int j = 0; j < this->input_size; j++) {
      r += w_on[j] * input[j];
      assert(!isinf(w_on[j]));
      assert(!isinf(input[j]));
      assert(!isbadf(r));
    }
    output[i] = r + this->bias[i];
    assert(!isbadf(output[i]));
  }
}

void Linear::zero_grad() {
  memset(this->grad, 0, this->output_size * sizeof(double));
  memset(this->weight_grad, 0,
         this->input_size * this->output_size * sizeof(double));
}

double *Linear::grad_ptr() { return this->grad; }

double *Linear::output() { return this->val; }

void Linear::update_input_grad(double *input, double *input_grad) {
  for (int i = 0; i < this->output_size; i++) {
    double *w_on = this->weights + i * this->input_size;
    double *w_grad_on = this->weight_grad + i * this->input_size;
    double g = this->grad[i];
    for (int j = 0; j < this->input_size; j++) {

      if (input_grad != NULL) {
        input_grad[j] += g * w_on[j];
        assert(!isbadf(input_grad[j]));
      }
      w_grad_on[j] += g * input[j];
      assert(!isbadf(w_grad_on[j]));
    }
  }
  if (input_grad != NULL) {
    printf("Input grad: ");
    for (int j = 0; j < this->input_size; j++) {
      printf("%f, ", input_grad[j]);
    }
    printf("\n");
  }
}

void Linear::step(double step_size) {
  size_t s = this->input_size * this->output_size;
  for (int i = 0; i < s; i++) {
    this->weights[i] -= step_size * this->weight_grad[i];
    assert(!isbadf(this->weights[i]));
  }
  for (int i = 0; i < this->output_size; i++) {
    this->bias[i] -= step_size * this->grad[i];
  }
}

Linear::Linear(int input_size, int output_size, double min_weight,
               double max_weight, double min_bias, double max_bias) {
  this->input_size = input_size;
  this->output_size = output_size;
  this->weights = (double *)malloc(input_size * output_size * sizeof(double));
  this->weight_grad =
      (double *)malloc(input_size * output_size * sizeof(double));
  this->bias = (double *)malloc(output_size * sizeof(double));
  this->val = (double *)malloc(output_size * sizeof(double));
  this->grad = (double *)malloc(output_size * sizeof(double));
  for (int i = 0; i < input_size * output_size; i++) {
    this->weights[i] = rand_f() * (max_weight - min_weight) + min_weight;
  }
  for (int i = 0; i < output_size; i++) {
    this->bias[i] = rand_f() * (max_bias - min_bias) + min_bias;
  }
}
