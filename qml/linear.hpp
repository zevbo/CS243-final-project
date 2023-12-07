#ifndef LINEAR_H
#define LINEAR_H

#define RESIDUAL_TY int
#define MAX_RESIDUAL 1024

#include "layer.hpp"
#include <string.h>
struct Linear : public Layer {
public:
  void zero_grad() override;
  void apply(F_TY *input) override;
  F_TY *output() override;
  void update_input_grad(F_TY *input, double *input_grad) override;
  double *grad_ptr() override;
  void step(double step_size) override;
  Linear(int input_size, int output_size, double min_weight, double max_weight,
         double min_bias, double max_bias);
  F_TY *weights;
  F_TY *bias;

private:
  F_TY *val;
  RESIDUAL_TY *weight_residuals;
  RESIDUAL_TY *bias_residuals;
  double *grad; // equal to the bias grad
  double *weight_grad;
};

#endif