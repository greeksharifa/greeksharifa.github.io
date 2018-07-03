#include <vector>

#include "re_define.h"
#include "bit_library.h"


class Matrix {
public:
    int size, mod;
    vvi element, origin;

    Matrix(int size, int* arr, int _mod) :size(size), mod(_mod) {
        origin = element = vvi(size, vi(size, 0));
        for (int i = 0; i < size; i++)
            element[i][i] = 1;
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                origin[i][j] = arr[i*size+j];
    }

    void multiply(vvi other) {
        vvi ret(size, vi(size, 0));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    ret[i][j] += ((ll)element[i][k] * other[k][j]) % mod;
                    ret[i][j] %= mod;
                }
            }
        }
        element = ret;
    }

    void power_K(int K) {
        K = bit_reverse(K);
        while (K) {
            multiply(element);
            if (K & 1) {
                multiply(origin);
            }
            K >>= 1;
        }
    }

    void print() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                printf("%d ", element[i][j]);
            puts("");
        }
    }
};
