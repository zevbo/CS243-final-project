#ifndef LAYER_H
#define LAYER_H

struct Layer {
public:
  const int input_size;
  const int output_size;

  virtual void zero_grad();
  virtual void apply(double *input);
  virtual double *output();
  virtual void update_input_grad(double *input, double *input_grad);
  virtual double *grad_ptr();
  virtual void step(double step_size);
};

#endif