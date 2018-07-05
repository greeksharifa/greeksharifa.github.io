#pragma once

#include <complex>
#include <cmath>
#include <algorithm>

#include "re_define.h"

typedef complex<double> base;

void fft(vector<base> &a, bool inv) {
  int n = (int)a.size();
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    while (!((j ^= bit) & bit)) bit >>= 1;
    if (i < j) swap(a[i], a[j]);
  }
  for (int i = 1; i < n; i <<= 1) {
    double x = inv ? M_PI / i : -M_PI / i;
    base w = { cos(x), sin(x) };
    for (int j = 0; j < n; j += i << 1) {
      base th = { 1, 0 };
      for (int k = 0; k < i; k++) {
        base tmp = a[i + j + k] * th;
        a[i + j + k] = a[j + k] - tmp;
        a[j + k] += tmp;
        th *= w;
      }
    }
  }
  if (inv) {
    for (int i = 0; i < n; i++) {
      a[i] /= n;
    }
  }
}

void multiply(vector<base> &a, vector<base> &b) {
  int n = (int)max(a.size(), b.size());
  int i = 0;
  while ((1 << i) < (n << 1)) i++;
  n = 1 << i;
  a.resize(n);
  b.resize(n);
  fft(a, false);
  fft(b, false);
  for (int i = 0; i < n; i++) {
    a[i] *= b[i];
  }
  fft(a, true);
}
