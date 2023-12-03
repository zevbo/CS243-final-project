#ifndef MODEL_H
#define MODEL_H

#include "layer.hpp"
#include <vector>

struct Model {
  std::vector<Layer *> layers;
  void train_on_input(double *input, double *correct_output, double lr);
  double *forwards(double *input);

protected:
  void backwards(double *input, double *loss_grad);
  void step(double step_size);
  void zero_grad();
};

#endif