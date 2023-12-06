#include "benchmark3.hpp"
#include "linear.hpp"
#include "model.hpp"
#include "msl.hpp"
#include "relu.hpp"
#include "tanh.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

std::vector<std::pair<double *, double *>> read_in_data(char *file,
                                                        bool stop_early) {
  std::vector<std::pair<double *, double *>> all_data;
  char *og_data = read_full_file(file);
  char *data = og_data;
  while (true) {
    double *d = (double *)malloc(28 * 28 * sizeof(double));
    int v;
    for (int i = 0; i < 28 * 28 + 1; i++) {
      char *data_on = data;
      while (*data != ',' && *data != '|') {
        data++;
      }
      if (*data == '|') {
        return all_data;
      }
      *data = '\0';
      data++;
      double f = atof(data_on);
      if (i == 28 * 28) {
        v = (int)f;
      } else {
        d[i] = f * 128;
      }
    }
    double *y = (double *)calloc(10, sizeof(double));
    y[v] = 50;
    // printf("V: %d\n", v);
    all_data.push_back(std::pair<double *, double *>(d, y));
    if (stop_early && all_data.size() > 10) {
      return all_data;
    }
  }
}

static double calc_loss(Model model,
                        std::vector<std::pair<double *, double *>> val_data) {
  double total_loss = 0;
  int total_correct = 0;
  // printf("Results: ");
  for (std::pair<double *, double *> val_pair : val_data) {
    // printf("Trying with input: %f, %f\n", input[0], input[1]);
    double *r = model.forwards(val_pair.first);
    double loss = msl_loss(10, r, val_pair.second);
    total_loss += loss;
    int highest_prob = 0;
    int guess = -1;
    int correct = -2;
    for (int i = 0; i < 10; i++) {
      if (r[i] > highest_prob) {
        guess = i;
        highest_prob = r[i];
      }
      if (val_pair.second[i] > 0) {
        correct = i;
      }
    }
    total_correct += guess == correct;
  }
  printf("In calc loss, %f percent were correct\n",
         (double)total_correct / val_data.size());
  return total_loss / val_data.size();
}

void run_benchmark3() {
  Model md;
  int input_size = 28 * 28;
  int l1_size = 128;
  int output_size = 10;
  double weight_mag = 2;
  double bias_mag = 0;
  Linear *l1 = new Linear(input_size, l1_size, -weight_mag, weight_mag,
                          -bias_mag, bias_mag);
  Tanh *r1 = new Tanh(l1_size, 5, 5);
  Linear *l2 = new Linear(l1_size, output_size, -weight_mag, weight_mag,
                          -bias_mag, bias_mag);
  md.layers = std::vector<Layer *>{l1, r1, l2};
  double *input = (double *)malloc(input_size * sizeof(double));
  int num_trains = 10000;
  int num_val = 1000;
  double lr = 0.0001;
  // print_linear_layer(l1);
  std::vector<std::pair<double *, double *>> val_data =
      read_in_data("../data/mnist/val", false);
  std::vector<std::pair<double *, double *>> train_data =
      read_in_data("../data/mnist/train", false);
  printf("Read in data\n");
  printf("Loss at start: %f\n", calc_loss(md, val_data));
  size_t t1 = microtime();
  //   test_input(md);
  int num_empochs = 10;
  for (int i = 0; i < num_empochs; i++) {
    for (std::pair<double *, double *> train_pair : train_data) {
      // printf("Trying with input: %f, %f\n", input[0], input[1]);
      md.train_on_input(train_pair.first, train_pair.second, lr);
      // printf("Did train. Loss %f\n", calc_loss(md, val_data));
    }
    printf("Finished epoch %d\n", i);
    printf("Loss is %f\n", calc_loss(md, val_data));
    lr *= 0.7;
  }
  size_t t2 = microtime();
  printf("Loss at end: %f\n", calc_loss(md, val_data));
  printf("Trained on data in %lu microseconds\n", t2 - t1);
  //   printf("Loss at end: %f\n", calc_loss(md, num_val));
}