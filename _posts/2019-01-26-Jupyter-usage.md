---
layout: post
title: Jupyter Notebook(Python) 사용법
author: YouWon
categories: References
tags: [usage]
---

## 설치 및 실행

두 가지 

### 실행 옵션

명령 옵션의 도움말을 표시한다.
```
jupyter notebook --help
```

실행 속도 상승을 위해 MathJax를 무효화할 수 있다. MathJax는 수식 입력을 위해 추가되는 JavaScript 라이브러리이다.
```
jupyter notebook --no-mathjax
```

웹 브라우저를 지정하거나 실행시키지 않을 수 있다. 포트 번호 지정도 가능하다.
```
jupyter notebook --browser="safari"
jupyter notebook --no-browser
jupyter notebook --port=8889
```

노트북 실행 시 실행 디렉토리를 지정할 수 있다.
```
jupyter notebook --notebook-dir=/Users/pydata/projects/jupyter/notebooks
```

### 설정 파일 수정

Jupyter Notebook의 기본 설정을 변경하고 싶으면 다음 명령을 입력한다.
```
jupyter notebook --generate-config
```
그러면 Jupyter가 실행되는 대신 설정 파일이 열린다.  
설정 파일에서 필요한 옵션을 변경하여 사용하면 된다. 기본적으로 사용하지 않는 옵션은 모두 주석 처리되어 있다. 

설정 파일을 재지정하고 싶으면 다음과 같이 입력한다.
```
jupyter notebook --config=custom_config.py
```


---

## 새 파일 생성


---

## Jupyter의 기본 사용법

### 편집 / 명령 모드

편집 모드에서는 셀의 내용을 편집할 수 있고(셀의 테두리가 초록색), 명령 모드는 편집중이 아닌 상태 또는 셀 자체에 조작을 가하는 상태(셀의 테두리가 파란색)이다.  
명령 모드에서 편집 모드로 들어가려면 `Enter`키를, 반대로는 `Esc` 키를 누르면 된다.

### 셀의 타입

Code 타입, Markdown 타입이 있다.  
Code 타입은 일반 코드를 실행할 수 있는 셀이다. 기본적으로 셀을 생성하면 Code 타입으로 생성된다.  
Markdown 타입은 [Markdown](https://greeksharifa.github.io/references/2018/06/29/markdown-usage/)으로 셀의 내용을 작성할 수 있다. 코드로 실행되지는 않으며, 수식을 작성할 수 있다. 수식은 MathJax에 의해 지원된다. 수식 작성 방법은 [여기](https://greeksharifa.github.io/references/2018/06/29/equation-usage/)를 참고한다.

### 셀 실행

실행하고 싶은 셀의 아무 곳에나 커서를 위치시킨 후 `Shift + Enter` 키를 누른다.  
실행하면 셀 아래쪽에는 실행 결과가 표시되고, 셀 옆의 'In [ ]'과 'Out [ ]'에 몇 번째로 실행시켰는지를 나타내는 숫자가 표시된다.

### 자동완성

웬만한 IDE에는 다 있는 자동완성 기능이다. 변수나 함수 등을 일부만 입력하고 `Tab` 키를 누르면 된다.

---

## 단축키

단축키 정보는 [Help] - [Keyboard Shortcuts] 또는 명령 모드에서 `H`를 눌러서 표시할 수 있다.

공용 단축키 | 설명
-------- | --------
Shift + Enter | 액티브 셀을 실행하고 아래 셀을 선택한다.
Ctrl + Enter | 액티브 셀을 실행한다.
Alt + Enter | 액티브 셀을 실행하고 아래에 셀을 하나 생성한다.

편집 모드 단축키 | 설명
-------- | --------
Ctrl + Z | Undo 명령이다.
Ctrl + Shift + Z | Rdeo 명령이다.
Tab | 자동완성 또는 Indent를 추가한다.
Shift + Tab | 툴팁 또는 변수의 상태를 표시한다.

명령 모드 단축키 | 설명
-------- | --------
↑, ↓ | 셀 선택
A | 액티브 코드 셀의 위(Above)에 셀을 하나 생성한다.
B | 액티브 코드 셀의 위(Below)에 셀을 하나 생성한다.
Ctrl + S | Notebook 파일을 저장한다.
Shift + L | 줄 번호 표시를 토글한다.
D, D | (D 두번 연속으로 타이핑)액티브 코드 셀을 삭제한다.
Z | 삭제한 셀을 하나 복원한다.
Y | 액티브 코드 셀을 Code 타입(코드를 기술하는 타입)으로 한다. 
M | 액티브 코드 셀을 Markdown 타입으로 한다. 
O, O | 커널을 재시작한다.
P | 명령 팔레트를 연다.
H | 단축키 목록을 표시한다. `Enter` 키로 숨긴다.

---

## Jupyter의 기능

### DocString의 표시

선언한 변수 뒤에 `?`를 붙여서 셀을 실행하는 것으로 해당 변수의 상태를 확인할 수 있다.

약간 다른 방법으로 변수를 타이핑한 후 `Shift + Tab`을 누르면 툴팁이 표시된다.  
툴팁에는 DocString의 일부 내용이 표시된다.

### 이미지 첨부하기

Drag & Drop으로 첨부할 수 있다.

### shell(명령 프롬프트)의 이용

명령창에서 쓰는 명령을 그대로 쓰되, 맨 앞에 `!`를 입력하여 사용 가능하다.

```
!cd Documents
```

### 매직 명령어 이용

맨 앞에 `%`를 붙이고 특정 명령을 수행할 수 있다.

매직 명령어 | 설명
-------- | --------
%pwd | 현재 디렉토리 경로 출력
%time `코드` | `코드`의 실행 시간을 측정하여 표시
%timeit `코드` | `코드`를 여러 번 실행한 결과를 요약하여 표시
%history -l 3 | 최근 3개의 코드 실행 이력 취득
%ls | 윈도우의 dir, Linux의 ls 명령과 같음
%autosave `n` | 자동저장 주기를 설정한다. 초 단위이며, 0이면 무효로 한다.
%matplotlib | 그래프를 그리는 코드 위에 따로 설정한다. `%matplotlib inline`으로 설정하면 코드 셀의 바로 아래에, `%matplotlib tk`로 설정하면 별도 창에 그래프가 출력된다. `%matplotlib notebook`으로 하면 코드 셀 바로 아래에 동적으로 그래프를 조작할 수 있는 그래프가 생성된다.

```python
# 코드 실행 시간 측정
%time sum(range(10000))
# 결과:
# CPU times: user 225 us, sys: 0 ns, total: 225 us
# Wall time: 228 us
# 499950000

# 1000회 반복, 3회 실행
%timeit sum(range(10000))
# 결과: 
# 1000 loops, best of 3: 238 us for loop

# 옵션 지정하기
%timeit -n 2000 -r 5 sum(range(10000))

# 셀 전체의 시간 측정
%%timeit -n 1000 -r 3
s = 0
for i in range(10000):
    s += i
```

---

## 둥

Kernel Interrupt

---

### 코드 입력

```python
def p_print(c):
    print('hi %s!' % c)

p_print('Gorio')
```

### 수식 입력

참조: [stackexchange.com](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

표로 정리한 글: [YW & YY's blog](https://greeksharifa.github.io/references/2018/06/29/equation-usage/)

수식 입력은

inline style: `$ a^2 + b^2 = c^2 $`

$ a^2 + b^2 = c^2 $

display style: `$$ a^2 + b^2 = c^2 $$`

$$ a^2 + b^2 = c^2 $$

로 한다.

---

### 이미지 추가

반드시 /로 시작해야 함.

예시:

<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)

---

### Markdown

다음 글을 참조한다. <https://greeksharifa.github.io/references/2018/06/29/markdown-usage/>

---

### References

다음의 글에 정리한다. <https://greeksharifa.github.io/references/2018/06/29/references/>
