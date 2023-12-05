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

double function(double *input) {
  return 2 * input[0] / MAX(0.5, input[4]) + input[1] * sqrt(input[3]) -
         input[2] * 0.5 - 0.1 * input[3];
}

void fill_input(double *input, int input_size) {
  for (int j = 0; j < input_size; j++) {
    input[j] = rand_f();
  }
}

double calc_loss(Model model, int trials) {
  int input_size = model.layers[0]->input_size;
  double *input = (double *)malloc(input_size * sizeof(double));
  double total_loss = 0;
  for (int i = 0; i < trials; i++) {
    fill_input(input, input_size);
    double correct_output = function(input);
    double r = model.forwards(input)[0];
    total_loss += (r - correct_output) * (r - correct_output);
  }
  return total_loss / trials;
}

void test_training() {
  Model md;
  int input_size = 5;
  int l1_size = 1000;
  int l2_size = 10;
  Linear *l1 = new Linear(input_size, l1_size, -1, 1, -1, 1);
  Tanh *r1 = new Tanh(l1_size, 20, 5);
  Linear *l2 = new Linear(l1_size, l2_size, -1, 1, -1, 1);
  Tanh *r2 = new Tanh(l2_size, 20, 5);
  Linear *l3 = new Linear(l2_size, 1, -1, 1, -1, 1);
  md.layers = std::vector<Layer *>{l1, r1, l2, r2, l3};
  double *input = (double *)malloc(input_size * sizeof(double));
  int num_trains = 200;
  int lr_decay_index = num_trains / 2;
  int num_val = 100;
  double lr = 0.01;
  double decay_lr = 0.001;
  printf("Loss at start: %f\n", calc_loss(md, num_val));
  size_t c1 = 0;
  size_t c2 = 0;
  size_t t1 = microtime();
  for (int i = 0; i < num_trains; i++) {
    fill_input(input, input_size);
    double correct_output = function(input);
    // printf("Trying with input: %f, %f\n", input[0], input[1]);
    std::pair<int, int> t = md.train_on_input(
        input, &correct_output, i < lr_decay_index ? lr : decay_lr);
    c1 += t.first;
    c2 += t.second;
  }
  size_t t2 = microtime();
  printf("Loss at end: %f\n", calc_loss(md, num_val));
  size_t t3 = microtime();
  printf("Calc loss time: %lu\n", t3 - t2);
  printf("Total train time: %zu microseconds. Breakdown: %zu, %zu\n", t2 - t1,
         c1, c2);
}

#define TY int
void stupid_benchmark() {
  int num_numbers = 1000;
  TY p[num_numbers];
  for (int i = 0; i < num_numbers; i++) {
    p[i] = (TY)(rand_f() * 100 - 50);
  }
  TY p2[num_numbers];
  for (int i = 0; i < num_numbers; i++) {
    p2[i] = p[i];
  }

  size_t t1 = microtime();
  TY sum = 0;
  for (int i = 0; i < num_numbers; i++) {
    for (int j = 0; j < num_numbers; j++) {
      sum += p[i];
      sum += p[j];
      sum -= p[i];
      sum -= p[j];
    }
  }
  size_t t2 = microtime();
  printf("Total time: %lu. Sum %f\n", t2 - t1, (float)sum);
}

int main() {
  // stupid_benchmark();
  test_training();
}