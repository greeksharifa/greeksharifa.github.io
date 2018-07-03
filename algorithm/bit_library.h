#include <cstdio>

int bit_reverse(int n){
    int ret = 0;
    while(n){
        ret <<= 1;
        ret |= n&1;
        n >>= 1;
    }
    return ret;
}