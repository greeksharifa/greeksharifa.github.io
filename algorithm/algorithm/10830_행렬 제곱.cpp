#include <cstdio>
#include "matrix.h"

#define mod 1000

int main_10830() {
    int m, N;
    scanf("%d%d", &m, &N);

    vvi original = vvi(m, vi(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &original[i][j]);

    matrix_power_N(original, N, mod, true);
    return 0;
}
