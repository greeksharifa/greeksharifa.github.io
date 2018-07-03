#include <cstdio>
#include "matrix.h"

int main() {
    printf("%d\n", bit_reverse(11));
    int arr[2 * 2] = { 1,1,1,0 };
    Matrix m = Matrix(2, arr, 1e9);
    // m.element[1][1] = 0;
    m.power_K(7);
    m.print();
    return 0;
}
