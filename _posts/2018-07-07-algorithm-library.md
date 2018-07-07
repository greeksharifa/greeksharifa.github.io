---
layout: post
title: greeksharifa's Library
author: YouWon
categories: Algorithm
tags: [C++]
---

## re_define.h

[코드](https://github.com/greeksharifa/ps_code/blob/master/library/re_define.h)

필자가 만든 라이브러리...라고 하기는 좀 그렇고, 그냥 재정의한 것들을 모아 놓은 헤더 파일이다.  
필자의 코드에서 처음 보는 토큰들이 좀 있을 텐데, 잘 모르겠다면 를 참조하면 된다.

대표적으로 다음과 같은 것들이 있다.

re_defined | original
-------- | --------
ll | long long
all(A) | A.begin(), A.end()
pi | pair\<int,int\>
mp(x,y) | make_pair(x,y)
vi | vector\<int\>
vvi | vector\<vector\<int\> \>


## bit_library.h

[코드](https://github.com/greeksharifa/ps_code/blob/master/library/bit_library.h)

비트 관련 사용자 정의 함수를 모아 놓은 헤더 파일이다.  
bit 연산을 안다면 코드를 보고 이해할 수 있으므로 따로 설명하지는 않겠다.

## conversion_library.h

[코드](https://github.com/greeksharifa/ps_code/blob/master/library/conversion_library.h)

어떤 데이터 타입 변수를 다른 데이터 타입 변수로 바꾸는 함수들을 모아 놓았다.  
예를 들어 string_to_vi 함수는 "1236"과 같은 string을 vi(vector\<int\>)로 변환한다.
