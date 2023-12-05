#include "tanh.hpp"
#include "utils.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Tanh::apply(F_TY *input) {
  F_TY *output = this->val;
  for (int i = 0; i < this->size; i++) {
    F_TY v = input[i];
    output[i] = (F_TY)(tanh(v / width) * this->mag);
  }
}

void Tanh::zero_grad() {
  memset(this->grad, 0, this->output_size * sizeof(double));
}

double *Tanh::grad_ptr() { return this->grad; }

F_TY *Tanh::output() { return this->val; }

void Tanh::update_input_grad(F_TY *input, double *input_grad) {
  for (int i = 0; i < this->input_size; i++) {
    F_TY v = input[i];
    double t = tanh(v / this->width);
    double grad_m = (1 - t * t) * this->mag / this->width;
    input_grad[i] = this->grad[i] * grad_m;
  }
}

void Tanh::step(double step_size) {}

Tanh::Tanh(int size, double width, double mag) {
  this->input_size = size;
  this->output_size = size;
  this->size = size;
  this->width = width;
  this->mag = mag;
  this->val = (F_TY *)malloc(size * sizeof(F_TY));
  this->grad = (double *)malloc(size * sizeof(double));
}
