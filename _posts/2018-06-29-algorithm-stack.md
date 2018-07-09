---
layout: post
title: Stack(스택)
author: YouWon
categories: [Algorithm & Data Structure]
tags: [Stack]
---

## 참조

분류 | URL
-------- | --------
문제 | [스택](https://www.acmicpc.net/problem/10828), [괄호](https://www.acmicpc.net/problem/9012)
응용 문제 | [스포일러 문제](https://www.acmicpc.net/problem/6549)
이 글에서 설명하는 라이브러리 | [std::stack](http://www.cplusplus.com/reference/stack/stack/)


--- 

## 개요

### 시간복잡도: $ O(M) $
### 공간복잡도: $ O(N) $
- N은 원소의 수, M은 연산의 수이다.
- 스택 자체는 알고리즘이 아닌 자료구조이지만, 스택을 쓰는 경우 복잡도는 위와 같이 나온다.

이 글에서는 스택(stack)이라고 하는 자료구조와, 이를 활용한 문제들을 살펴볼 것이다.

스택은 간단하면서도 매우 유용한 자료구조이다. 여러분의 실생활에서도 자주 볼 수 있는 개념이다.  
그리고 알고리즘으로 적용하기에도 쉽다. 그럼 스택은 무엇인가?

---

## Stack(스택)

스택([stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)))의 사전적 의미는 더미(무더기)이다.

스택은 다음 그림으로 설명된다.

![Stack](https://upload.wikimedia.org/wikipedia/commons/b/b4/Lifo_stack.png)

책상 위에 책을 몇 권 쌓는 것과 같다. 1번 책을 쌓고, 2번 책을 쌓고, ... , 6번 책까지 쌓았다고 하자.  
그리고 책을 한 권을 집는다고 치자. 그러면 여러분은 (정상적이라면) 몇 번 책을 집겠는가?  
사서 고생하는 사람이 아니라면 굳이 아래쪽 책을 힘들게 빼진 않을 것이다. 즉, 6번 책을 뺄 것이다. 위의 그림과 갈다.  
첫 번째 pop(책을 빼는 것) 연산은 가장 마지막에 들어온 6번부터 행해진다.

이것을 Last-In-First-Out이라 부른다. 즉 가장 나중에 들어온 것이 제일 먼저 나간다는 의미이다.  
이것이 스택의 전부이다. 그리고 C++의 STL에는 이것이 친절히 구현되어 있다. 다음에서 살펴보자.


## std::stack 사용법

### Include

우선 include를 해야 한다. 특별히 큰 프로젝트에서 쓰는 것이 아니라면, `std::`를 매번 쓰기 귀찮으니 namespace도 써 주자.

```cpp
#include <stack>

using namespace std;
```

### 선언

stack은 generic으로 구현된 template이다. 즉, stack에 들어갈 데이터 타입을 정해야 한다.  
보통 int나 char 등을 사용하게 될 것이다. 물론 기본형 뿐만 아니라 사용자 정의 타입도 가능하다.

```cpp
stack<int> st; // 현재 비어 있다.
// stack<char> st_c;
// stack<dot_2d> st_person; // dot_2d는 알아서 정의하시길...
```

### push(e)

스택에 무언가를 집어넣는(push) 연산이다. e는 집어넣을 원소이다.

```cpp
st.push(10);  // 10
st.push(20);  // 10 20
st.push(30);  // 10 20 30
st.push(777); // 10 20 30 777
``` 

### size()

스택의 현재 size를 반환한다. 몇 개나 들어 있는지 알고 싶을 때 쓰면 된다.

```cpp
printf("size: %d\n", st.size());  // size: 4
st.push(999);
printf("size: %d\n", st.size());  // size: 5
```

### top()

스택의 맨 위 원소를 반환한다. 스택에서는 맨 위의 것만 빼낼 수 있다. 다른 원소에는 접근이 불가능하다.  
물론 스택을 직접 구현한다면 접근 가능하게 할 수도 있지만, 스택을 쓰는 데 그렇게 할 이유가..?

```cpp
int e = st.top();
printf("top: %d\n", e);   // top: 999
```

### pop()

스택의 맨 위 원소를 제거한다. 책을 하나 가져갔다고 생각하면 된다.  
반환값이 `void`이므로 리턴값으로 top 값을 알아낼 수는 없다.  
또, 비어 있는데 pop을 수행하려고 하면 런타임 에러를 발생시킨다.

```cpp
st.pop(); // 999가 제거됨
for (int i = 0; i < 3; i++)
    st.pop(); 
// 777, 30, 20이 차례로 제거됨
```

### empty()

스택이 비었는지를 검사한다. `size() == 0` 구문으로 체크할 수도 있지만, 이쪽이 더 직관적이다. 그리고 생각보다 자주 쓰게 된다.

```cpp
if (st.empty())
    printf("stack is empty, 1\n");
st.pop();   // 10이 제거됨
if (st.empty())
    printf("stack is empty, 2\n");

// stack is empty, 2
```

### emplace(e)

STL에서 emplace는 생성자를 호출하면서 `push`(혹은 `push_back`)하는 것과 동일하다.  
이 기능은 stack에 기본형 말고 `dot_2d`와 같은 사용자 정의 함수나 생성자 호출이 필요한 데이터 타입을 넣었을 때 필요하다.  
int와 같은 기본형을 넣을 때는 별 차이가 없다.

```cpp
st.emplace(-3);
printf("top: %d\n", st.top());  // top: -3
```

### swap(another_stack)

같은 데이터 타입을 담고 있는 다른 스택과 원소 전체를 swap한다.

```cpp
stack<int> another_st;
st.swap(another_st);
if (st.empty())
    printf("stack is empty, 3\n");
// stack is empty, 3
```

스택은 사실상 이게 전부이다. 그리고 생각보다 많은 문제를 풀 수 있다.  
물론 대부분은 너무 뻔히 풀이가 스택이라는 것이 보이지만, 안 그런 것도 있다(스포일러 문제 참조)  



## 문제 풀이

### BOJ 10828(스택)

문제: [스택](https://www.acmicpc.net/problem/10828)

풀이: [BOJ 10828(스택) 문제 풀이](https://greeksharifa.github.io/ps/2018/07/08/PS-10828/)


### BOJ 09012(괄호)

문제: [괄호](https://www.acmicpc.net/problem/9012)

풀이: [BOJ 09012(괄호) 문제 풀이](https://greeksharifa.github.io/ps/2018/07/08/PS-09012/)

### 스포일러 문제

문제: [스포일러 문제](https://www.acmicpc.net/problem/6549)

풀이: [스포일러 풀이            ](https://greeksharifa.github.io/ps/2018/07/07/PS-06549/)
