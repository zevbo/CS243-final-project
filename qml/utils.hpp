#ifndef UTILS_H
#define UTILS_H

#include <assert.h>
#include <math.h>
#include <stdint.h>

uint64_t microtime();
/** Returns random float between 0 and 1 */
float rand_f();
bool isbadf(double d);
static inline int quantize(float f) { return (int)roundf(f); }

#define tassert assert

#endif