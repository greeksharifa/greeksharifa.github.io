#include <vector>

#include "re_define.h"
#include "bit_library.h"


vvi mat_mul(vvi matrix_A, vvi matrix_B, int mod) {
    int m = matrix_A.size();
    vvi ret(m, vi(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
              ret[i][j] += ((ll)matrix_A[i][k] * matrix_B[k][j]) % mod;
              ret[i][j] %= mod;
            }
        }
    }
    return ret;

}

vvi matrix_power_N(vvi matrix, int N, int mod, bool print) {
    int m = matrix.size(), cnt;
    vvi original = matrix;
    vvi ret = vvi(m, vi(m));
    for (int i = 0; i < m; i++)
        ret[i][i] = 1;
    pi tmp = bit_reverse(N);
    N = tmp.first, cnt = tmp.second;
    while (cnt--) {
        ret = mat_mul(ret, ret, mod);
        if (N & 1) {
            ret = mat_mul(ret, original, mod);
        }
        N >>= 1;
    }
    if (print) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++)
                printf("%d ", ret[i][j]);
            puts("");
        }
    }
    return ret;
}
