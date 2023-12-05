#ifndef RELU_H
#define RELU_H

#include "layer.hpp"
#include <string.h>
struct Relu : public Layer {
public:
  void zero_grad() override;
  void apply(F_TY *input) override;
  F_TY *output() override;
  void update_input_grad(F_TY *input, double *input_grad) override;
  double *grad_ptr() override;
  void step(double step_size) override;
  Relu(int size);
  int size;

private:
  F_TY *val;
  double *grad;
};

#endif