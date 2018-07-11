---
layout: post
title: 고속 푸리에 변환(Fast Fourier Theorem, FFT). 큰 수의 곱셈
author: YouWon
categories: [Algorithm & Data Structure]
tags: [FFT]
---

## 참조

분류 | URL
-------- | --------
문제 | [큰 수 곱셈](https://www.acmicpc.net/problem/13277)
응용 문제 | [koosaga BOJ FFT 문제집](https://www.acmicpc.net/workbook/view/824)
[참조 라이브러리](https://greeksharifa.github.io/algorithm%20&%20data%20structure/2018/07/07/algorithm-library/) | [sharifa_header.h](https://github.com/greeksharifa/ps_code/blob/master/library/sharifa_header.h), [bit_library.h](https://github.com/greeksharifa/ps_code/blob/master/library/bit_library.h)
이 글에서 설명하는 라이브러리 | [fft.h](https://github.com/greeksharifa/ps_code/blob/master/library/fft.h)


--- 

## 개요

### 시간복잡도: $ O(N log N) $
### 공간복잡도: $ O(N) $
- N은 두 수열의 길이의 max값이다. 
- FFT는 convolution을 빠르게 해 주는 것이지만, PS에서는 거의 대부분 곱셈을 빠르게 하기 위해 쓰인다.

이 글에서는 FFT(고속 푸리에 변환)을 설명한다.  
이론적인 부분에 대한 자세한 설명은 [topology-blog](http://topology-blog.tistory.com/6)에 잘 되어 있으므로 생략한다.

---

## 알고리즘
  
큰 수의 곱셈을 수행할 때 FFT의 개략적인 설명은 다음과 같이 적어 두었다.

1. 각 수열을 먼저 reverse시킨다. $ O(N) $
2. 각 수열에 푸리에 변환을 적용한다. $ O(N log N) $
3. 푸리에 변환을 적용하면 convolution을 단순 곱셈으로 변환시킬 수 있으므로, 2의 결과물을 element-wise 곱셈을 시킨다. $ O(N) $
4. 3의 결과물을 푸리에 역변환을 시킨다. $ O(N log N) $
5. 1에서 reverse를 시켰으므로, 다시 reverse를 시켜준다. $ O(N) $

FFT의 핵심 부분은 2~4번 부분이다. 1번과 5번은 우리가 수를 쓸 때 앞부분에 큰 자리 수를 적기 때문에 필요하다.


## 구현

다음과 같다. FFT 구현과 출력 함수만 정의되어 있으므로, 따로 설명하진 않겠다.

다만 주의할 점이 하나 있는데, 출력 함수에서 곱하는 두 수가 0인 경우 예외 처리를 해주어야 한다.

```cpp
/*
fft 함수는 http://topology-blog.tistory.com/6 블로그를 참조한 것입니다.
*/

#pragma once

#include "sharifa_header.h"
#include "bit_library.h"

typedef complex<double> base;

void fft(vector<base> &a, bool inv) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        while (!((j ^= bit) & bit)) bit >>= 1;
        if (i < j) swap(a[i], a[j]);
    }
    for (int i = 1; i < n; i <<= 1) {
        double x = (inv ? 1 : -1) * M_PI / i;
        base w = { cos(x), sin(x) };
        for (int j = 0; j < n; j += i << 1) {
            base th(1);
            for (int k = 0; k < i; k++) {
                base tmp = a[i + j + k] * th;
                a[i + j + k] = a[j + k] - tmp;
                a[j + k] += tmp;
                th *= w;
            }
        }
    }
    if (inv) {
        for (int i = 0; i < n; i++) a[i] /= n;
    }
}

vector<int> multiply(vector<int> &A, vector<int> &B) {
    vector<base> a(A.begin(), A.end());
    vector<base> b(B.begin(), B.end());
    int n = power_of_2_ge_than(max(a.size(), b.size())) * 2;

    a.resize(n);	b.resize(n);
    fft(a, false);	fft(b, false);

    for (int i = 0; i < n; i++)
        a[i] *= b[i];
    fft(a, true);

    vector<int> ret(n);
    for (int i = 0; i < n; i++)
        ret[i] = (int)round(a[i].real());
    return ret;
}
```

## BOJ 13277(큰 수 곱셈) 문제 풀이

문제: [BOJ 13277(큰 수 곱셈)](https://www.acmicpc.net/problem/13277)

풀이: [BOJ 13277(큰 수 곱셈) 문제 풀이](https://greeksharifa.github.io/ps/2018/07/08/PS-13277/) 
