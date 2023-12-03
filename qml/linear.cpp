#include "linear.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Linear::apply(double *input) {
  double *output = this->val;
  for (int i = 0; i < this->output_size; i++) {
    double r = 0;
    double *w_on = this->weights + i * this->input_size;
    for (int j = 0; j < this->input_size; j++) {
      r += w_on[j] * input[j];
    }
    output[i] = r + this->bias[i];
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
      }
      w_grad_on[j] += g * input[j];
    }
  }
}

void Linear::step(double step_size) {
  size_t s = this->input_size * this->output_size;
  for (int i = 0; i < s; i++) {
    this->weights[i] -= step_size * this->weight_grad[i];
  }
  for (int i = 0; i < this->output_size; i++) {
    this->bias[i] -= step_size * this->grad[i];
  }
}