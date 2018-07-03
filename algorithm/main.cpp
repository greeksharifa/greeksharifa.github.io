#include <cstdio>
#include "matrix.h"

int main() {
    matrix m = matrix(2, 1000000000, 2);
    m.element[1][1]=0;
    m.multiply_K(7);
    m.print();
    printf("hjk");
    return 0;
}