#include "linear.hpp"
#include "model.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Relu {
  int size;
  double *grad;
} Relu;

void apply_relu_inplace(Relu layer, double *input) {
  for (int i = 0; i < layer.size; i++) {
    input[i] *= input[i] >= 0;
  }
}

// learn: 1, 1, 0, 0

void test_training() {
  Model md;
  md.layers = std::vector<Layer *>();
  int input_size = 2;
  Linear *l = new Linear(input_size, 1, -1, 1, -1, 1);
  md.layers.push_back(l);
  double *input = (double *)malloc(input_size * sizeof(double));
  double real_weights[] = {0.3, -0.8};
  double real_bias = -0.1;
  int num_trains = 1000;
  printf("Starting...\n");
  fflush(stdout);
  for (int i = 0; i < num_trains; i++) {
    printf("Weights: %f, %f. Bias: %f\n", l->weights[0], l->weights[1],
           l->bias[0]);
    for (int j = 0; j < input_size; j++) {
      input[j] = rand_f();
    }
    double correct_output = real_bias;
    for (int j = 0; j < input_size; j++) {
      correct_output += real_weights[j] * input[j];
    }
    printf("Trying with input: %f, %f\n", input[0], input[1]);
    md.train_on_input(input, &correct_output, 0.1);
  }
}

int main() { test_training(); }