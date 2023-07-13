#include <stdio.h>
#include <stdlib.h>

int main() {
  int X = 1, Y = 8, Z = 3, divs = 2;
  int max_id = X * Y;
  int DY = Y / divs;

  printf(
      "Dimension (R x C) = %d x %d is mapped to (X x Y x Z) = %d x %d x %d\n",
      Z * divs, X * DY, X, Y, Z);

  for (int r = 0; r < X * Y * Z; r++) {
    int x = r % X;
    //    int y  = (r / X) % Y;
    int y = (r % (X * Y)) / X;
    int z = r / (X * Y);
    int rank_2dc = -1;
    int rank_2dr = r / (X * Y) + (r % (X * Y)) / (X * DY) * Z;
    if (x == 0 && y % DY == 0)
      rank_2dc = 0;
    else if (x == 0)
      rank_2dc = X * DY - (y % DY);
    else if ((y % DY) % 2 == 0)
      rank_2dc = (X - 1) * (y % DY) + x;
    else
      rank_2dc = (X - 1) * (y % DY) + (X - x);

    printf("(x, y, z, r, c) = (%d,%d,%d,%d,%d)\n", x, y, z, rank_2dr, rank_2dc);
  }
  return 0;
}
