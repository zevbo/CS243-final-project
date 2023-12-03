#ifndef MSL_H
#define MSL_H

double msl_loss(int sz, double *output, double *expected);
double *msl_grad(int sz, double *output, double *expected);

#endif