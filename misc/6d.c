#include <stdio.h>
#define X 22
#define Y 20
#define Z 24

double swap_diff(int *r, int *c) {
  if (*r < *c) {
    int tmp = *c;
    *c = *r;
    *r = tmp;
  }
  int r_ = *r;
  int c_ = *c;
  return ((double)r_) / c_;
}

int main() {
  int size6d[6] = {X, Y, Z, 2, 3, 2};
  double min = 10000000;

  // 5 combination
  for (int i = 0; i < 6; i++) {
    int R = size6d[i];
    int C = 1;
    for (int j = 0; j < 6; j++)
      if (i != j) C *= size6d[j];

    double diff = swap_diff(&R, &C);
    if (diff < min) {
      min = diff;
      printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
    }
  }

  // 4 combination
  for (int i = 0; i < 6; i++) {
    for (int j = i + 1; j < 6; j++) {
      int R = size6d[i] * size6d[j];
      int C = 1;
      for (int k = 0; k < 6; k++)
        if (i != k && j != k) C *= size6d[k];

      double diff = swap_diff(&R, &C);
      if (diff < min) {
        min = diff;
        printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
      }
    }
  }

  // 3 combination
  for (int i = 0; i < 6; i++) {
    for (int j = i + 1; j < 6; j++) {
      for (int k = j + 1; k < 6; k++) {
        int R = size6d[i] * size6d[j] * size6d[k];
        int C = 1;
        for (int m = 0; m < 6; m++)
          if (i != m && j != m && k != m) C *= size6d[m];

        double diff = swap_diff(&R, &C);
        if (diff < min) {
          min = diff;
          printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
        }
      }
    }
  }

  return 0;
}
