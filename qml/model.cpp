#include "model.hpp"
#include "layer.hpp"
#include "msl.hpp"
#include "utils.hpp"
#include <algorithm>
#include <vector>

void Model::backwards(F_TY *input, double *loss_grad) {
  Layer *last_layer = this->layers[this->layers.size() - 1];
  memcpy(last_layer->grad_ptr(), loss_grad,
         last_layer->output_size * sizeof(double));
  for (int i = this->layers.size() - 1; i >= 0; i--) {
    Layer *layer = this->layers[i];
    if (i > 0) {
      Layer *input_layer = this->layers[i - 1];
      layer->update_input_grad(input_layer->output(), input_layer->grad_ptr());
    } else {
      layer->update_input_grad(input, NULL);
    }
  }
}

void Model::step(double step_size) {
  for (Layer *layer : this->layers) {
    layer->step(step_size);
  }
}

void Model::zero_grad() {
  for (Layer *layer : this->layers) {
    layer->zero_grad();
  }
}

F_TY *convert_input(Model *model, double *input) {
  F_TY *real_input =
      (F_TY *)malloc(model->layers[0]->input_size * sizeof(F_TY));
  for (int i = 0; i < model->layers[0]->input_size; i++) {
    real_input[i] = (F_TY)input[i];
  }
  return real_input;
}

double *Model::forwards(double *input) {
  F_TY *real_input = convert_input(this, input);
  F_TY *res = this->qforwards(real_input);
  free(real_input);
  IFQUANTIZE(
      {
        int output_sz = this->layers[this->layers.size() - 1]->output_size;
        double *double_output = (double *)malloc(sizeof(double) * output_sz);
        for (int i = 0; i < output_sz; i++) {
          double_output[i] = (double)res[i];
        }
        return double_output;
      },
      return res;)
}

F_TY *Model::qforwards(F_TY *input) {
  F_TY *values_on = input;
  for (Layer *layer : this->layers) {
    layer->apply(values_on);
    values_on = layer->output();
  }
  return values_on;
}

std::pair<int, int> Model::train_on_input(double *input, double *correct_output,
                                          double lr) {
  this->zero_grad();
  F_TY *real_input = convert_input(this, input);
  size_t t1 = microtime();
  F_TY *output = qforwards(real_input);
  size_t t2 = microtime();
  int output_sz = this->layers[this->layers.size() - 1]->output_size;
  // printf("F %f, %f [loss %f]\n", output[0], correct_output[0],
  //        msl_loss(output_sz, output, correct_output));
  double *double_output = (double *)malloc(sizeof(double) * output_sz);
  for (int i = 0; i < output_sz; i++) {
    double_output[i] = (double)output[i];
  }
  double *loss_grad = msl_grad(output_sz, double_output, correct_output);
  // printf("Grad: ");
  // for (int i = 0; i < output_sz; i++) {
  //   printf("%f, ", loss_grad[i]);
  // }
  // printf("\n");
  this->backwards(real_input, loss_grad);
  free(loss_grad);
  this->step(lr);
  size_t t3 = microtime();
  free(real_input);
  free(double_output);
  return std::pair<int, int>(t2 - t1, t3 - t2);
}