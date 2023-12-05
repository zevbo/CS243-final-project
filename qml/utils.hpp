#ifndef UTILS_H
#define UTILS_H

#include <assert.h>
#include <stdint.h>

uint64_t microtime();
float rand_f();
bool isbadf(double d);

#define tassert assert

#endif