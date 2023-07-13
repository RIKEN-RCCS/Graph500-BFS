#include <stdio.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <cstdlib>

static void parse_row_dims(bool* rdim, const char* input) {
  memset(rdim, 0x00, sizeof(bool) * 6);
  while (*input) {
    switch (*(input++)) {
      case 'x':
        rdim[0] = true;
        break;
      case 'y':
        rdim[1] = true;
        break;
      case 'z':
        rdim[2] = true;
        break;
      case 'a':
        rdim[3] = true;
        break;
      case 'b':
        rdim[4] = true;
        break;
      case 'c':
        rdim[5] = true;
        break;
    }
  }
}

static void convert_rank_for_mesh(int& c_rank, int c_rank_x, int c_rank_y,
                                  int c_size_x, int c_size_y) {
  if (c_rank_x == 0 && c_rank_y == 0) {
    c_rank = 0;
  } else if (c_rank_x == 0) {
    c_rank = c_size_x * c_size_y - c_rank_y;
  } else {
    c_rank -= (c_rank_y / 2) * 2;
  }
}

static void compute_rank(std::vector<int>& ss, std::vector<int>& rs,
                         int& size_x, int& size_y, int& rank_x, int& rank_y,
                         int& r, int& s) {
  int size = 1;
  int rank = 0;
  for (int i = ss.size() - 1; i >= 0; --i) {
    if (rank % 2) {
      rank = rank * ss[i] + (ss[i] - 1 - rs[i]);
    } else {
      rank = rank * ss[i] + rs[i];
    }
    size *= ss[i];
  }
  size_x = ss[0];
  size_y = size / size_x;
  rank_x = rs[0];
  rank_y = rank / size_x;
  r = rank;
  s = size;
}

int main() {
  const char* tofu_6d = getenv("TOFU_6D");  // R-axis. e.g. TOFU_6D=yz
  bool rdim[6] = {0};
  parse_row_dims(rdim, tofu_6d);

  int rank6d[6] = {0, 0, 0, 0, 0, 0};
  int size6d[6] = {3, 3, 4, 2, 3, 2};
  std::vector<int> ss_r, rs_r, ss_c, rs_c;
  printf("X Y Z a b c = %d %d %d %d %d %d\n", size6d[0], size6d[1], size6d[2],
         size6d[3], size6d[4], size6d[5]);
  for (rank6d[0] = 0; rank6d[0] < size6d[0]; rank6d[0]++) {
    for (rank6d[1] = 0; rank6d[1] < size6d[1]; rank6d[1]++) {
      for (rank6d[2] = 0; rank6d[2] < size6d[2]; rank6d[2]++) {
        for (rank6d[3] = 0; rank6d[3] < size6d[3]; rank6d[3]++) {
          for (rank6d[4] = 0; rank6d[4] < size6d[4]; rank6d[4]++) {
            for (rank6d[5] = 0; rank6d[5] < size6d[5]; rank6d[5]++) {
              for (int i = 0; i < 6; i++) {
                if (rdim[i]) {
                  ss_r.push_back(size6d[i]);
                  rs_r.push_back(rank6d[i]);
                } else {
                  ss_c.push_back(size6d[i]);
                  rs_c.push_back(rank6d[i]);
                }
              }

              if (ss_r.back() != 1 && ss_r.back() % 2 != 0) {
                printf(
                    "Last dimension of R must be multiple of 2. But it is "
                    "%d.\n",
                    ss_r.back());
                exit(1);
              } else if (ss_c.back() != 1 && ss_c.back() % 2 != 0) {
                printf(
                    "Last dimension of C must be multiple of 2. But it is "
                    "%d,\n",
                    ss_c.back());
                exit(1);
              }

              int r_size_x, r_size_y, r_rank_x, r_rank_y, r_rank, r_size;
              int c_size_x, c_size_y, c_rank_x, c_rank_y, c_rank, c_size;
              compute_rank(ss_c, rs_c, r_size_x, r_size_y, r_rank_x, r_rank_y,
                           r_rank, r_size);
              compute_rank(ss_r, rs_r, c_size_x, c_size_y, c_rank_x, c_rank_y,
                           c_rank, c_size);
              if (ss_r.size() != 1)
                convert_rank_for_mesh(c_rank, c_rank_x, c_rank_y, c_size_x,
                                      c_size_y);

              printf("[%2d] r_x:%2d r_y:%2d\n", c_rank, c_rank_x, c_rank_y);
              ss_r.clear();
              rs_r.clear();
              ss_c.clear();
              rs_c.clear();
            }
          }
        }
      }
    }
  }
  return 0;
}
