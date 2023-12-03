#ifndef LAYER_H
#define LAYER_H
#include <string.h>
struct Layer {
public:
  int input_size;
  int output_size;

  virtual void zero_grad(){};
  virtual void apply(double *input) {}
  virtual double *output() { return NULL; }
  virtual void update_input_grad(double *input, double *input_grad) {}
  virtual double *grad_ptr() { return NULL; }
  virtual void step(double step_size) {}
};

#endif