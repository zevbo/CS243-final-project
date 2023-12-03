#ifndef MODEL_H
#define MODEL_H

#include "layer.hpp"
#include <vector>

struct Model {
  std::vector<Layer *> layers;
  void train_on_input(double *input, double *correct_output, double lr);
  F_TY *forwards(double *input);

protected:
  F_TY *qforwards(F_TY *input);
  void backwards(F_TY *input, double *loss_grad);
  void step(double step_size);
  void zero_grad();
};

#endif