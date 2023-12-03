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

typedef struct Embedding {
  double *embeddings;
  int input_space;
  int ouput_size;
} Embedding;

double *apply_embeddings(Embedding layer, int *input, int num_inputs) {
  double *output =
      (double *)malloc(num_inputs * layer.ouput_size * sizeof(double));

  for (int i = 0; i < num_inputs; i++) {
    fflush(stdout);
    memcpy(output + i * layer.ouput_size,
           layer.embeddings + input[i] * layer.ouput_size,
           layer.ouput_size * sizeof(double));
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

typedef struct LeiserModel {
  Embedding embedding;
  Linear l1;
  Relu r1;
  Linear l2;
} LeiserModel;

#define BOARD_SIZE 64
float apply_leiser_model(LeiserModel lm, int *board) {
  int i = 0;
  double *embedded = apply_embeddings(lm.embedding, board, BOARD_SIZE);
  printf("P1 %f\n", embedded[i]);
  double *l1_output = apply_linear(lm.l1, embedded);
  printf("P2 %f\n", l1_output[i]);
  apply_relu_inplace(lm.r1, l1_output);
  printf("P3 %f\n", l1_output[i]);
  double *l2_output = apply_linear(lm.l2, l1_output);
  printf("P4 %f\n", l2_output[i]);
  float f = l2_output[0];
  free(embedded);
  free(l1_output);
  free(l2_output);
  return f;
}

long get_file_size(FILE *f) {
  fseek(f, 0L, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0L, SEEK_SET);
  return sz;
}

char *read_full_file(char *file_name) {
  FILE *f = fopen(file_name, "r");
  long file_size = get_file_size(f);
  char *file_str = (double *)malloc((file_size + 1) * sizeof(char));
  fread(file_str, 1, file_size, f);
  file_str[file_size] = '\0';
  fclose(f);
  return file_str;
}

double *load_array(int height, int width, char *data_file) {
  char *og_data = read_full_file(data_file);
  char *data = og_data;
  double *result = (double *)malloc(height * width * sizeof(double));
  int i_on = 0;
  char *data_on;
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
#define CASE (*data != ' ' && *data != '\n' && *data != '\0')
      while (!CASE) {
        data++;
      }
      data_on = data;
      while (CASE) {
        data++;
      }
      *data = '\0';
      result[i_on] = atof(data_on);
      i_on++;
    }
  }
  free(og_data);
  return result;
}

Linear build_linear(int input_size, int output_size, char *weights_file,
                    char *bias_file) {
  double *weights = load_array(output_size, input_size, weights_file);
  double *bias = load_array(output_size, 1, bias_file);
  return (Linear){.bias = bias,
                  .weights = weights,
                  .input_size = input_size,
                  .output_size = output_size};
}

Embedding build_embedding(int input_space, int output_size, char *emb_file) {
  double *embeddings = load_array(output_size, input_space, emb_file);
  return (Embedding){.embeddings = embeddings,
                     .input_space = input_space,
                     .ouput_size = output_size};
}

int board_size = 64;
LeiserModel build_model() {
  int num_pieces = 9;
  int emb_size = 4;
  int l1_size = 128;
  Embedding emb = build_embedding(num_pieces, emb_size, "emb.txt");
  Linear l1 = build_linear(board_size * emb_size, l1_size, "l1_weight.txt",
                           "l1_bias.txt");
  Relu r1 = {.size = l1_size};
  Linear l2 = build_linear(l1_size, 1, "l2_weight.txt", "l2_bias.txt");
  return (LeiserModel){.embedding = emb, .l1 = l1, .l2 = l2, .r1 = r1};
}

size_t microtime() {
  // struct timeb t;
  // ftime(&t);
  struct timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000 + t.tv_nsec / 1000;
  // return (size_t)t.millitm + t.time * 1000;
}

int main() {
  LeiserModel model = build_model();
  double *double_board = load_array(1, board_size, "example.txt");
  size_t t1 = microtime();
  int *board = (int *)malloc(board_size * sizeof(int));
  for (int i = 0; i < board_size; i++) {
    board[i] = (int)double_board[i];
    printf("%d[%d], ", board[i], i);
  }
  printf("\n");
  float f = apply_leiser_model(model, board);
  size_t t2 = microtime();
  printf("value: %f [%lu]\n", f, t2 - t1);
}