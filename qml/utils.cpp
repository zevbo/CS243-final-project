#include "utils.hpp"
#include "stdlib.h"
#include "sys/timeb.h"
#include "time.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t microtime() {
  // struct timeb t;
  // ftime(&t);
  struct timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000 + t.tv_nsec / 1000;
  // return (size_t)t.millitm + t.time * 1000;
}

float rand_f() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

bool isbadf(double d) { return isnan(d) || isinf(d) || abs(d) > 10000000; }

long get_file_size(FILE *f) {
  fseek(f, 0L, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0L, SEEK_SET);
  return sz;
}

char *read_full_file(char *file_name) {
  FILE *f = fopen(file_name, "r");
  long file_size = get_file_size(f);
  char *file_str = (char *)malloc((file_size + 1) * sizeof(char));
  fread(file_str, 1, file_size, f);
  file_str[file_size] = '\0';
  fclose(f);
  return file_str;
}