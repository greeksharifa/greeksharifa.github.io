---
layout: post
title: PyCharm 사용법 2 - 단축키
author: YouWon
categories: References
tags: [PyCharm, usage]
---

---

PyCharm(파이참)은 Jetbrains 사에서 제작 및 배포하는 **유료**/무료 프로그램이다.  
Professional 버전은 돈을 주고 구입하거나, 학생이라면 [학생 인증](https://www.jetbrains.com/student/)을 하고 무료로 사용할 수 있다.

글이 길기 때문에 사용법을 검색하고 싶다면 `Ctrl + F` 키를 누른 다음 검색해 보자.

기본적인 사용법 및 고급 기능은 [PyCharm 사용법(파이참 설치 및 사용법)](https://greeksharifa.github.io/references/2019/02/07/PyCharm-usage/)에서 확인하자.

*알아둘 점*

- 단축키를 누를 때는 반드시 적힌 순서대로 눌러야 한다.  
- 또한 일반 1~0 숫자키 및 사칙연산 키와 numpad에 있는 키는 다르게 취급한다. 예를 들면 `-`와 `numpad-`로 각기 다르게 표시한다.
- `[1~5]`와 같이 대괄호로 표시된 것은 이 중 하나를 눌러도 된다는 뜻으로 예시는 숫자 1 또는 2~5를 누를 때 효과가 조금씩 다르다.
- `,`로 표시된 경우는 `,` 전까지의 단축키를 누른 다음 `,` 이후 키를 따로 누를 수 있다. 예를 들어 `Ctrl + NumPad*, [1~5]`의 경우 `Ctrl`를 누른 채로 `NumPad*`키를 누른 다음, 손을 뗀 다음에 숫자 `1`을 눌러도 된다. 그리고 이 경우 `NumPad*`까지 누르고 PyCharm 제일 왼쪽 아래를 보면 다음 키를 누를 수 있는 옵션이 표시되어 있다.

*2023.04.22 updated*

---

## 코드 접기 및 펼치기(expand and collapse)

참고: https://www.jetbrains.com/help/rider/Code_Folding.html#folding_menu

| 설명 | 단축키 |
| -------- | -------- |
코드 블록 펼치기 / 접기 | line number 옆에 있는 `+`나 `-`버튼
선택한 부분을 펼치기 | Ctrl + +
선택한 부분을 접기 | Ctrl + -
선택한 부분을 재귀적으로 펼치기 | Ctrl + Alt + NumPad+
선택한 부분을 재귀적으로 접기 | Ctrl + Alt + NumPad-
파일의 모든 코드 블록 펼치기 | Ctrl + Shift + NumPad+
파일의 모든 코드 블록 접기 | Ctrl + Shift + NumPad-
선택한 부분을 level 1~5 까지 코드 펼치기 | Ctrl + NumPad*, [1~5]
파일의 모든 코드 블록을 level 1~5 까지 코드 펼치기 | Ctrl + Shift + NumPad*, [1~5]



<center><img src="/public/img/2023-04-22-PyCharm-shortcut/01.png" width="70%"></center>


---

## References

[공식 홈페이지](https://www.jetbrains.com/pycharm/)에서 더 자세한 사용법을 찾아볼 수 있다.
