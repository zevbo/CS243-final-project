#include "sys/timeb.h"
#include "time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Linear {
  double *weights;
  double *bias;
  int input_size;
  int output_size;
} Linear;

double *apply_linear(Linear layer, double *input) {
  double *output = (double *)malloc(sizeof(double) * layer.output_size);
  for (int i = 0; i < layer.output_size; i++) {
    double r = 0;
    double *w_on = layer.weights + i * layer.input_size;
    for (int j = 0; j < layer.input_size; j++) {
      r += w_on[j] * input[j];
    }
    output[i] = r + layer.bias[i];
  }
  return output;
}

typedef struct Relu {
  int size;
} Relu;

void apply_relu_inplace(Relu layer, double *input) {
  for (int i = 0; i < layer.size; i++) {
    input[i] *= input[i] >= 0;
  }
}
