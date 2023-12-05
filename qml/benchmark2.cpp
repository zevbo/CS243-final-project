#include "benchmark2.hpp"
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

void run_benchmark2() {
  Model md;
  int input_size = 5;
  int l1_size = 10;
  double weight_mag = 1;
  double bias_mag = 1;
  Linear *l1 = new Linear(input_size, l1_size, -weight_mag, weight_mag,
                          -bias_mag, bias_mag);
  Relu *r1 = new Relu(l1_size);
  Linear *l2 =
      new Linear(l1_size, 1, -weight_mag, weight_mag, -bias_mag, bias_mag);
  md.layers = std::vector<Layer *>{l1, r1, l2};
  double *input = (double *)malloc(input_size * sizeof(double));
  int num_trains = 10000;
  int num_val = 1000;
  double lr = 0.001;
  // print_linear_layer(l1);
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