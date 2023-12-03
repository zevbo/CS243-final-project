#ifndef RELU_H
#define RELU_H

#include "layer.hpp"
#include <string.h>
struct Relu : public Layer {
public:
  void zero_grad() override;
  void apply(double *input) override;
  double *output() override;
  void update_input_grad(double *input, double *input_grad) override;
  double *grad_ptr() override;
  void step(double step_size) override;
  Relu(int size);
  int size;

private:
  double *val;
  double *grad;
};

#endif