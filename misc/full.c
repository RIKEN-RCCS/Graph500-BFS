#include <stdio.h>

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
  int s[] = {2, 2, 2, 2, 2, 2,
             2, 2, 3, 3, 3, 23};  // 2^8 x 3^3 x 23 = 24 x 23 x 24 x 2 x 3 x 2
  double min = 10000000;

  for (int i = 0; i < 12; i++) {
    int R = s[i];
    int C = 1;
    for (int j = 0; j < 12; j++)
      if (i != j) C *= s[j];

    double diff = swap_diff(&R, &C);
    if (diff < min) {
      min = diff;
      printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
    }
  }

  for (int i = 0; i < 12; i++) {
    for (int j = i + 1; j < 12; j++) {
      int R = s[i] * s[j];
      int C = 1;
      for (int k = 0; k < 12; k++)
        if (i != k && j != k) C *= s[k];

      double diff = swap_diff(&R, &C);
      if (diff < min) {
        min = diff;
        printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
      }
    }
  }

  for (int i = 0; i < 12; i++) {
    for (int j = i + 1; j < 12; j++) {
      for (int k = j + 1; k < 12; k++) {
        int R = s[i] * s[j] * s[k];
        int C = 1;
        for (int m = 0; m < 12; m++)
          if (i != m && j != m && k != m) C *= s[m];

        double diff = swap_diff(&R, &C);
        if (diff < min) {
          min = diff;
          printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
        }
      }
    }
  }

  for (int i = 0; i < 12; i++) {
    for (int j = i + 1; j < 12; j++) {
      for (int k = j + 1; k < 12; k++) {
        for (int m = k + 1; m < 12; m++) {
          int R = s[i] * s[j] * s[k] * s[m];
          int C = 1;
          for (int n = 0; n < 12; n++)
            if (i != n && j != n && k != n && m != n) C *= s[n];

          double diff = swap_diff(&R, &C);
          if (diff < min) {
            min = diff;
            printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
          }
        }
      }
    }
  }

  for (int i = 0; i < 12; i++) {
    for (int j = i + 1; j < 12; j++) {
      for (int k = j + 1; k < 12; k++) {
        for (int m = k + 1; m < 12; m++) {
          for (int n = m + 1; n < 12; n++) {
            int R = s[i] * s[j] * s[k] * s[m] * s[n];
            int C = 1;
            for (int p = 0; p < 12; p++)
              if (i != p && j != p && k != p && m != p && n != p) C *= s[p];

            double diff = swap_diff(&R, &C);
            if (diff < min) {
              min = diff;
              printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < 12; i++) {
    for (int j = i + 1; j < 12; j++) {
      for (int k = j + 1; k < 12; k++) {
        for (int m = k + 1; m < 12; m++) {
          for (int n = m + 1; n < 12; n++) {
            for (int p = n + 1; p < 12; p++) {
              int R = s[i] * s[j] * s[k] * s[m] * s[n] * s[p];
              int C = 1;
              for (int q = 0; q < 12; q++)
                if (i != q && j != q && k != q && m != q && n != q && p != q)
                  C *= s[q];

              double diff = swap_diff(&R, &C);
              if (diff < min) {
                min = diff;
                printf("diff = %.2f, R x C = %d x %d\n", diff, R, C);
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
