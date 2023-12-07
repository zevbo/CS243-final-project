#include "linear.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int foo(int *A, int n) {
  unsigned sum = 0;
  for (int i = 0; i < n; ++i)
    sum += A[i] + 5;
  return sum;
}

void Linear::apply(F_TY *input) {
  F_TY *output = this->val;
  size_t osize = this->output_size;
  size_t isize = this->input_size;
  for (int i = 0; i < osize; i++) {
    tassert(!isbadf(this->bias[i]));
    F_TY *w_on = this->weights + i * isize;
    F_TY r = 0;
    for (int j = 0; j < isize; j++) {
      r += w_on[j] * (input[j]);
      tassert(!isinf(w_on[j]));
      tassert(!isinf(input[j]));
      tassert(!isbadf(r));
    }
    output[i] = r + this->bias[i];
    tassert(!isbadf(output[i]));
  }
}

void Linear::zero_grad() {
  memset(this->grad, 0, this->output_size * sizeof(double));
  memset(this->weight_grad, 0,
         this->input_size * this->output_size * sizeof(double));
}

double *Linear::grad_ptr() { return this->grad; }

F_TY *Linear::output() { return this->val; }

void Linear::update_input_grad(F_TY *input, double *input_grad) {
  for (int i = 0; i < this->output_size; i++) {
    F_TY *w_on = this->weights + i * this->input_size;
    double *w_grad_on = this->weight_grad + i * this->input_size;
    double g = this->grad[i];
    for (int j = 0; j < this->input_size; j++) {

      if (input_grad != NULL) {
        input_grad[j] += g * w_on[j];
        // if (isbadf(input_grad[j])) {
        //   printf("Bad stuff: %f, %f, %f", input_grad[j], g, w_on[j]);
        // }
        tassert(!isbadf(input_grad[j]));
      }
      w_grad_on[j] += g * input[j];
      tassert(!isbadf(w_grad_on[j]));
    }
  }
}

inline void update_with_residual(RESIDUAL_TY *res, F_TY *real, double inc) {
  IFQUANTIZE(
      {
        double curr_val = (float)(*res) / MAX_RESIDUAL;
        double new_val = curr_val + inc;
        F_TY real_diff = (F_TY)new_val;
        F_TY final_new_val = *real + real_diff;
        final_new_val = MIN(MAX(final_new_val, MIN_TY), MAX_TY);
        *real = final_new_val;
        new_val -= real_diff;
        *res = (RESIDUAL_TY)(new_val * MAX_RESIDUAL);
      },
      { *res += inc; })
  // IFQUANTIZE(
  //     {
  //       *res += inc;
  //       F_TY w_diff = (F_TY)*res;
  //       *real += w_diff;
  //       *res -= w_diff;
  //       // this->weights[i] -= (F_TY)step_size * this->weight_grad[i];
  //     },
  //     { this->weights[i] -= (F_TY)step_size * this->weight_grad[i]; })
}

void Linear::step(double step_size) {
  size_t s = this->input_size * this->output_size;
  // printf("Linear weight stuff %p: ", this);
  for (int i = 0; i < s; i++) {
    update_with_residual(&this->weight_residuals[i], &this->weights[i],
                         -step_size * this->weight_grad[i]);
    // IFQUANTIZE(
    //     {
    //       this->weight_residuals[i] -= step_size * this->weight_grad[i];
    //       F_TY w_diff = (F_TY)this->weight_residuals[i];
    //       this->weights[i] += w_diff;
    //       this->weight_residuals[i] -= w_diff;
    //       // this->weights[i] -= (F_TY)step_size * this->weight_grad[i];
    //     },
    //     { this->weights[i] -= (F_TY)step_size * this->weight_grad[i]; })

    tassert(!isbadf(this->weights[i]));
  }
  // printf("\n");
  for (int i = 0; i < this->output_size; i++) {
    update_with_residual(&this->bias_residuals[i], &this->bias[i],
                         -step_size * this->grad[i]);
    // IFQUANTIZE(
    //     {
    //       this->bias_residuals[i] -= step_size * this->grad[i];
    //       F_TY r_diff = (F_TY)this->bias_residuals[i];
    //       this->bias[i] += (F_TY)r_diff;
    //       this->bias_residuals[i] -= r_diff;
    //       // this->bias[i] -= (F_TY)(step_size * this->grad[i]);
    //     },
    //     { this->bias[i] -= (F_TY)(step_size * this->grad[i]); })
  }
}

Linear::Linear(int input_size, int output_size, double min_weight,
               double max_weight, double min_bias, double max_bias) {
  this->input_size = input_size;
  this->output_size = output_size;
  this->weights = (F_TY *)malloc(input_size * output_size * sizeof(F_TY));
  this->weight_residuals =
      (RESIDUAL_TY *)calloc(input_size * output_size, sizeof(RESIDUAL_TY));
  this->weight_grad =
      (double *)malloc(input_size * output_size * sizeof(double));
  this->bias = (F_TY *)malloc(output_size * sizeof(F_TY));
  this->val = (F_TY *)malloc(output_size * sizeof(F_TY));
  this->grad = (double *)malloc(output_size * sizeof(double));
  this->bias_residuals =
      (RESIDUAL_TY *)calloc(output_size, sizeof(RESIDUAL_TY));
  for (int i = 0; i < input_size * output_size; i++) {
    this->weights[i] = rand_f() * (max_weight - min_weight) + min_weight;
  }
  for (int i = 0; i < output_size; i++) {
    this->bias[i] = rand_f() * (max_bias - min_bias) + min_bias;
  }
}
