#include "model.hpp"
#include "layer.hpp"
#include "msl.hpp"
#include <algorithm>
#include <vector>

void Model::backwards(double *input, double *loss_grad) {
  Layer last_layer = this->layers[this->layers.size() - 1];
  memcpy(last_layer.grad_ptr(), loss_grad,
         last_layer.output_size * sizeof(double));
  for (int i = this->layers.size() - 1; i >= 0; i--) {
    Layer layer = this->layers[i];
    if (i > 0) {
      Layer input_layer = this->layers[i - 1];
      layer.update_input_grad(input_layer.output(), input_layer.grad_ptr());
    } else {
      layer.update_input_grad(input, NULL);
    }
  }
}

void Model::step(double step_size) {
  for (Layer layer : this->layers) {
    layer.step(step_size);
  }
}

void Model::zero_grad() {
  for (Layer layer : this->layers) {
    layer.zero_grad();
  }
}

double *Model::forwards(double *input) {
  double *values_on = input;
  for (Layer layer : this->layers) {
    layer.apply(values_on);
    values_on = layer.output();
  }
  return values_on;
}

void Model::train_on_input(double *input, double *correct_output, double lr) {
  this->zero_grad();
  double *output = forwards(input);
  int output_sz = this->layers[this->layers.size() - 1].output_size;
  double *loss_grad = msl_grad(output_sz, output, correct_output);
  this->backwards(input, loss_grad);
  this->step(lr);
}