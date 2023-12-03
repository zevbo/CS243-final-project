#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <string.h>

#define F_TY int8_t

struct Layer {
public:
  int input_size;
  int output_size;

  virtual void zero_grad(){};
  virtual void apply(F_TY *input) {}
  virtual F_TY *output() { return NULL; }
  virtual void update_input_grad(F_TY *input, double *input_grad) {}
  virtual double *grad_ptr() { return NULL; }
  virtual void step(double step_size) {}
};

#endif