#ifndef LAYER_H
#define LAYER_H
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define QUANTIZE 0

#if QUANTIZE
#define F_TY int
#define W_TY int8_t
#else
#define F_TY double
#define W_TY double
#endif

#if QUANTIZE
#define IFQUANTIZE(f1, f2)                                                     \
  { f1 }
#else
#define IFQUANTIZE(f1, f2)                                                     \
  { f2 }
#endif

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