#include "utils.hpp"
#include "stdlib.h"
#include "sys/timeb.h"
#include "time.h"
#include <math.h>

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