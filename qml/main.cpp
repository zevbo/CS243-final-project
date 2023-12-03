#include "linear.hpp"
#include "model.hpp"
#include "relu.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// learn: 1, 1, 0, 0
// z = 0.3x - 0.8y - 0.1
// z = -0.99x - 0.37 ...

#define MAX(a, b) (a > b ? a : b)

double function(double *input) {
  return 2 * input[0] / MAX(0.5, input[4]) + input[1] * input[3] -
         input[3] * input[2] * 0.5 - 0.1 * input[4];
}

double calc_loss(Model model, int trials) {
  int input_size = model.layers[0]->input_size;
  double *input = (double *)malloc(input_size * sizeof(double));
  double total_loss = 0;
  for (int i = 0; i < trials; i++) {
    input[i] = rand_f();
    double correct_output = function(input);
    double r = model.forwards(input)[0];
    total_loss += (r - correct_output) * (r - correct_output);
  }
  return total_loss / trials;
}

void test_training() {
  Model md;
  int input_size = 5;
  int l1_size = 200;
  Linear *l1 = new Linear(input_size, l1_size, -1, 1, -1, 1);
  Relu *r1 = new Relu(l1_size);
  Linear *l2 = new Linear(l1_size, 1, -1, 1, -1, 1);
  md.layers = std::vector<Layer *>{l1, r1, l2};
  double *input = (double *)malloc(input_size * sizeof(double));
  int num_trains = 10000;
  int num_val = 1000;
  double total_loss = 0;
  double lr = 0.01;
  printf("Loss at start: %f\n", calc_loss(md, num_val));
  size_t c1 = 0;
  size_t c2 = 0;
  size_t t1 = microtime();
  for (int i = 0; i < num_trains; i++) {
    for (int j = 0; j < input_size; j++) {
      input[j] = rand_f();
    }
    double correct_output = function(input);
    // printf("Trying with input: %f, %f\n", input[0], input[1]);
    std::pair<int, int> t = md.train_on_input(input, &correct_output, lr);
    c1 += t.first;
    c2 += t.second;
  }
  size_t t2 = microtime();
  printf("Loss at end: %f\n", calc_loss(md, num_val));
  printf("Total train time: %d microseconds. Breakdown: %d, %d\n", t2 - t1, c1,
         c2);
}

int main() { test_training(); }