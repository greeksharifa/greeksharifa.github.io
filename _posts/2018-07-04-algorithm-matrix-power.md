---
layout: post
title: 행렬의 N 거듭제곱 빠르게 구하기
author: YouWon
categories: [Algorithm & Data Structure]
tags: [Matrix Power]
---

## 참조

분류 | URL
-------- | --------
문제 | [행렬 제곱](https://www.acmicpc.net/problem/10830)
응용 문제 | [스포일러 1](https://www.acmicpc.net/problem/2749)
[참조 라이브러리](https://greeksharifa.github.io/algorithm%20&%20data%20structure/2018/07/07/algorithm-library/) | [sharifa_header.h](https://github.com/greeksharifa/ps_code/blob/master/library/sharifa_header.h), [bit_library.h](https://github.com/greeksharifa/ps_code/blob/master/library/bit_library.h)
이 글에서 설명하는 라이브러리 | [matrix.h](https://github.com/greeksharifa/ps_code/blob/master/library/matrix.h)

--- 

## 개요

### 시간복잡도: $ O(M^3 log N) $
### 공간복잡도: $ O(M^2) $
- M은 행렬의 크기, N은 거듭제곱할 수를 나타낸다. 물론 행렬은 정사각행렬이다.

이 글에서는 행렬의 N 거듭제곱을 빠르게 구하는 방법을 설명한다. 
사실 행렬의 N승은 [정수의 N 거듭제곱 빠르게 구하기](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)을 구하는 것과 근본적으로 동일하다.  
다만 단순 정수의 곱이 아닌 행렬곱을 사용할 뿐이다.

---

## 알고리즘

먼저 예를 하나 들어보자. 행렬 $A$의 11승을 구하고 싶다고 하자. 어떻게 계산하는 것이 빠르겠는가?

단순무식한 방법을 적자면, $A^2$부터 구하면 된다. 그리고 $A^3$을 구한다. ... 마지막으로 $A^{11}$을 구한다.  
그러면 계산량은? 행렬곱 10번이다.

*나쁘지 않은데?* 라고 생각한다면, N이 크면 당연히 **시간 초과**임을 생각하라. N이 10억쯤 한다면? 10억 번을 1초 안에 계산할 수 있겠는가?  
[문제](https://www.acmicpc.net/problem/10830)처럼 N(문제에서는 B이다.)이 $10^{11}$이나 한다면?

당연히 이 방법으로는 어림도 없다. 그럼 조금 더 좋은 방법을 생각해야 한다.

이제 $A^{11} = A \cdot A \cdot A \cdot A \cdot A \cdot A \cdot A \cdot A \cdot A \cdot A \cdot A$를 조금 다르게 써 보자.

$$ A^{11} = (A^5)^2 \cdot A $$

만약에 여러분이, $A^5$를 알고 있다고 하자. 그러면 계산을 몇 번이나 해야 할까?  
행렬곱은 딱 두 번 뿐이다.

조금 전 단순무식하게 구할 때를 떠올려보자. $A^5$로부터 $A^{11}$을 구하는 것은 행렬곱 연산이 6번이나 필요했다.  
그러나 지금은 단 두 번만에 해결이 된다.

이제 $A^5$를 구하는 방법을 생각해보자. 다음을 생각할 수 있을 것이다.

$$ A^5 = (A^2)^2 \cdot A $$

$$ A^2 = (A)^2 $$

그러면 이것을 어떻게 코드로 옮길 것인가?  
고려할 것이 몇 가지 있다. 하나씩 살펴보자.

- A의 N승을 위의 방법으로 구할 때 반드시 고려해야 하는 부분은, 현재 구하고자 하는 N이 짝수인지 홀수인지이다.  
  - 만약 홀수라면, 다음과 같다. $ A^{2k+1} = (A^k)^2 \cdot A$이다.  
  - 만약 짝수라면, 조금 더 간단하다. $ A^{2k} = (A^k)^2 $이다.
- $A^N$을 구하기 전에, 먼저 2진수로 나타내 본다. N=11인 경우, N=$1011_2$이다.
- $A^{11}$을 종이에 쓰고 천천히 생각해보라. 다음 두 가지 중 맞는 것은 무엇인가? N=$1011_2$이다.
  - 가장 끝자리 비트(LSB)부터 고려하여 위의 거듭제곱 알고리즘을 따른다.
  - 가장 앞자리 비트(MSB)부터 고려하여 위의 거듭제곱 알고리즘을 따른다.
  - 답은 MSB부터 고려하는 것이다. N=11로 놓고 종이에 써보면, MSB를 고려하는 것은 $A^{11}$을 구하지만, LSB를 고려하는 것은 $A^{13}$을 구하게 될 것이다. 
  - 이것이 바로 [matrix.h](https://github.com/greeksharifa/ps_code/blob/master/library/matrix.h)에서 `bit_reverse` 함수를 사용하는 이유이다.
  - 한 가지 더 주의할 점은, 비트 반전만 해서는 안된다. 100이 001로 바뀌어 그냥 1이 되기 때문이다. 따라서 자리수를 기억해 두어야 한다.

이제 $A^{11}$는 다음과 같은 순서로 구하면 된다는 것을 알 수 있을 것이다. 11=$1011_2$임을 기억하라.  
물론 $A^0 = 1$이다.

이진수 | 식
-------- | :--------
1 | $ (A^0)^2 \cdot A = A^1 $
0 | $ (A^1)^2 = A^2 $
1 | $ (A^2)^2 \cdot A = A^5 $
1 | $ (A^5)^2 \cdot A = A^{11} $


조금 더 복잡한 예를 들어보겠다. 46=$101110_2$이다.

이진수 | 식
-------- | :--------
1 | $ (A^0)^2 \cdot A = A^1 $
0 | $ (A^1)^2 = A^2 $
1 | $ (A^2)^2 \cdot A = A^5 $
1 | $ (A^5)^2 \cdot A = A^{11} $
1 | $ (A^{11})^2 \cdot A = A^{23} $
0 | $ (A^{23})^2 = A^{46} $

이진수로 나타냈을 때 해당 자리가 1이면 제곱한 후 A를 추가로 곱하고, 0이면 그냥 제곱만 하면 된다.

행렬의 거듭제곱은 아주 복잡하지는 않다. 헷갈린다면 [정수의 N 거듭제곱 빠르게 구하기](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)을 참조하라.

## 구현

거듭제곱이 구현된 행렬 클래스는 다음과 같다. 필자의 편의를 위해, `re_define.h`에 `#define`을 활용한 많은 단축 선언들을 사용했다. 

```cpp
#include "sharifa_header.h"
#include "bit_library.h"

vector<vector<int> > mat_mul(vector<vector<int> > matrix_A, vector<vector<int> > matrix_B, int mod) {
    int m = matrix_A.size();
    vector<vector<int> > ret(m, vector<int>(m));
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

vector<vector<int> > matrix_power_N(vector<vector<int> > matrix, int N, int mod, bool print) {
    int m = matrix.size(), len = binary_len(N);
    vector<vector<int> > original = matrix;
    vector<vector<int> > ret = vector<vector<int> >(m, vector<int>(m));
    for (int i = 0; i < m; i++)
        ret[i][i] = 1;
    
	N = bit_reverse(N);
    while (len--) {
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
```

## 문제 풀이

사용법은 어렵지 않다. 행렬을 2차원 벡터로 만든다.  
그리고 행렬을 N승을 취한 후, `print` 인자를 `true`로 주어 `matrix_power_N` 함수를 호출하면 문제는 풀린다.

```cpp
#include "../library/matrix.h"

#define mod 1000

int main_10830() {
    int m, N;
    scanf("%d%d", &m, &N);

    vector<vector<int> > original = vector<vector<int> >(m, vector<int>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &original[i][j]);

    matrix_power_N(original, N, mod, true);
    return 0;
}
```

***주의: 이 코드를 그대로 복붙하여 채점 사이트에 제출하면 당연히 틀린다. 못 믿겠다면, 저런 헤더 파일이 채점 사이트에 있을 것이라 생각하는가?***
