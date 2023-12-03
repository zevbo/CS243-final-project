#ifndef LINEAR_H
#define LINEAR_H

#include "layer.hpp"
#include <string.h>
struct Linear : public Layer {
public:
  void zero_grad() override;
  void apply(double *input) override;
  double *output() override;
  void update_input_grad(double *input, double *input_grad) override;
  double *grad_ptr() override;
  void step(double step_size) override;
  Linear(int input_size, int output_size, double min_weight, double max_weight,
         double min_bias, double max_bias);

private:
  double *weights;
  double *bias;
  double *val;
  double *grad; // equal to the bias grad
  double *weight_grad;
};

#endif