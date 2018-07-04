#pragma once

#include "re_define.h"

pi bit_reverse(int n) {
    int ret = 0, cnt = 0;
    while (n) {
        ++cnt;
        ret <<= 1;
        ret |= n & 1;
        n >>= 1;
    }
    return mp(ret, cnt);
}
