#include "linear.hpp"
#include "model.hpp"
#include "relu.hpp"
#include "tanh.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// learn: 1, 1, 0, 0
// z = 0.3x - 0.8y - 0.1
// z = -0.99x - 0.37 ...

#define MAX(a, b) (a > b ? a : b)

static double function(double *input) {
  return (2 * input[0] / MAX(0.5, input[4]) + input[1] * sqrt(input[3]) -
          input[2] * 0.5 - 0.1 * input[3]);
}

static void fill_input(double *input, int input_size) {
  for (int j = 0; j < input_size; j++) {
    input[j] = rand_f() * 5;
  }
}

static double calc_loss(Model model, int trials) {
  int input_size = model.layers[0]->input_size;
  double *input = (double *)malloc(input_size * sizeof(double));
  double total_loss = 0;
  // printf("Results: ");
  for (int i = 0; i < trials; i++) {
    fill_input(input, input_size);
    double correct_output = function(input);
    double r = model.forwards(input)[0];
    // if (i < 10) {
    //   printf("%f[%f], ", r, correct_output);
    // }
    total_loss += (r - correct_output) * (r - correct_output);
  }
  // printf("\n");
  return total_loss / trials;
}

static void test_input(Model model) {
  double input[] = {0.3971, 0.7544, 0.5695, 0.4388, 0.6387};
  double correct_output = function(input);
  double r = model.forwards(input)[0];
  printf("Test %f: %f, %f\n", r, correct_output,
         (r - correct_output) * (r - correct_output));
}

void print_linear_layer(Linear *l) {
  printf("Linear. Weights:  ");
  for (int i = 0; i < l->output_size * l->input_size; i++) {
    printf("%f, ", l->weights[i]);
  }
  printf(". Bias: ");
  for (int i = 0; i < l->output_size; i++) {
    printf("%f, ", l->bias[i]);
  }
  printf("\n");
}

void run_benchmark1() {
  Model md;
  int input_size = 5;
  double weight_mag = 0.5;
  double bias_mag = 0.5;
  Linear *l1 =
      new Linear(input_size, 1, -weight_mag, weight_mag, -bias_mag, bias_mag);
  l1->weights[0] = 0.23043269;
  l1->weights[1] = -0.19739035;
  l1->weights[2] = -0.08669749;
  l1->weights[3] = 0.20990819;
  l1->weights[4] = -0.42102337;
  l1->bias[0] = 0.2682017;
  md.layers = std::vector<Layer *>{l1};
  double *input = (double *)malloc(input_size * sizeof(double));
  int num_trains = 1000;
  int num_val = 100;
  double lr = 0.01;
  print_linear_layer(l1);
  test_input(md);
  printf("Loss at start: %f\n", calc_loss(md, num_val));
  for (int i = 0; i < num_trains; i++) {
    fill_input(input, input_size);
    double correct_output = function(input);
    // printf("Trying with input: %f, %f\n", input[0], input[1]);
    md.train_on_input(input, &correct_output, lr);
  }
  printf("Loss at end: %f\n", calc_loss(md, num_val));
}