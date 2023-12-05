#include "benchmark3.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "relu.hpp"
#include "tanh.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double calc_loss(Model model, int trials) {
  int input_size = model.layers[0]->input_size;
  double *input = (double *)malloc(input_size * sizeof(double));
  double total_loss = 0;
  // printf("Results: ");
  for (int i = 0; i < trials; i++) {
    // double r = model.forwards(input)[0];
    // if (i < 10) {
    //   printf("%f[%f], ", r, correct_output);
    // }
    // total_loss += (r - correct_output) * (r - correct_output);
  }
  // printf("\n");
  return total_loss / trials;
}

void run_benchmark3() {
  Model md;
  int input_size = 28 * 28;
  int l1_size = 128;
  int output_size = 10;
  double weight_mag = 1;
  double bias_mag = 1;
  Linear *l1 = new Linear(input_size, l1_size, -weight_mag, weight_mag,
                          -bias_mag, bias_mag);
  Relu *r1 = new Relu(l1_size);
  Linear *l2 = new Linear(l1_size, output_size, -weight_mag, weight_mag,
                          -bias_mag, bias_mag);
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