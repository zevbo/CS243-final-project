#include "relu.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Relu::apply(double *input) {
  double *output = this->val;
  for (int i = 0; i < this->size; i++) {
    double v = input[i];
    output[i] = v * (v > 0);
  }
}

void Relu::zero_grad() {
  memset(this->grad, 0, this->output_size * sizeof(double));
}

double *Relu::grad_ptr() { return this->grad; }

double *Relu::output() { return this->val; }

void Relu::update_input_grad(double *input, double *input_grad) {
  for (int i = 0; i < this->input_size; i++) {
    input_grad[i] = this->val[i] > 0 ? this->grad[i] : 0;
  }
}

void Relu::step(double step_size) {}

Relu::Relu(int size) {
  this->input_size = size;
  this->output_size = size;
  this->size = size;
  this->val = (double *)malloc(size * sizeof(double));
  this->grad = (double *)malloc(size * sizeof(double));
}
