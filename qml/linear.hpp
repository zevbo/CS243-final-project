#ifndef LINEAR_H
#define LINEAR_H

#include "layer.hpp"
struct Linear : Layer {
public:
  void zero_grad();
  void apply(double *input);
  double *output();
  void update_input_grad(double *input, double *input_grad);
  double *grad_ptr();
  void step(double step_size);

private:
  double *const weights;
  double *const bias;
  double *const val;
  double *const grad; // equal to the bias grad
  double *const weight_grad;
};

#endif