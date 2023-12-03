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