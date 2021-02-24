---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

[파이썬 정규표현식(re) 사용법 - 01. Basic](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)  
**[파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)**  
[파이썬 정규표현식(re) 사용법 - 03. OR, 반복](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/)  
[파이썬 정규표현식(re) 사용법 - 04. 그룹, 캡처](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/)  
[파이썬 정규표현식(re) 사용법 - 05. 주석, 치환, 분리](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)  
[파이썬 정규표현식(re) 사용법 - 06. 치환 함수, 양방탐색, 조건문](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/05/regex-usage-06-advanced/)  
[파이썬 정규표현식(re) 사용법 - 07. 예제(숫자)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-07-example/)  
[파이썬 정규표현식(re) 사용법 - 08. 예제(단어, 행)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-08-example/)  
[파이썬 정규표현식(re) 사용법 - 09. 기타 기능](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/24/regex-usage-09-other-functions/)  

---

이 글에서는 정규표현식 기초와 python library인 `re` 패키지 사용법에 대해서 설명한다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

파이썬 버전은 3.6을 기준으로 하나, 3.x 버전이면 (아마) 동일하게 쓸 수 있다.  
2.7 버전은 한글을 포함한 비 알파벳 문자 처리가 다르다.

---

## 특수문자


### 메타문자

메타문자에 대해서는 [이전 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/#%EB%A9%94%ED%83%80%EB%AC%B8%EC%9E%90)에서 설명했다.

### 비인쇄 문자

벨`\a`, 이스케이프`\e`, 폼 피드`\f`, 라인 피드(개행문자)`\n`, 캐리지 리턴`\r`, 가로 탭`\t`, 세로 탭`\v`는 다음 두 가지 방식으로 쓸 수 있다.
> \a \e \f \n \r \t \v  
> \x07 \x1B \f \n \r \t \v

정규표현식을 쓰면서 다른 것들은 거의 볼 일이 없을 것이지만, `\t`와 `\n`은 알아두는 것이 좋다.

```python
matchObj = re.findall('\t ', 'a\tb\tc\t \t d')
print(matchObj)
```
결과
```
['\t ', '\t ']
```

탭 문자와 공백 문자가 붙어 있는 것은 2개임을 확인할 수 있다.

### 이스케이프 `\` 

이스케이프 문자 `\`는 메타문자를 일반 리터럴 문자로 취급하게끔 해 준다.  
예를 들어 여는 괄호 `[`는 메타 문자지만, `\[`와 같이 처리하면 리터럴 문자인 일반 대괄호 문자 '['와 매칭될 수 있게 된다.

하지만, 일반 영수 문자(알파벳 또는 숫자)를 이스케이프 처리하면 에러가 나거나 혹은 전혀 다른 의미의 정규식 토큰이 생성된다.  
예를 들어 파이썬에서 `\1`의 경우에는 캡처한 문자열 중 첫번째를 재사용한다는 의미(나중에 자세히 설명할 것이다)가 되어 버린다. 따라서 `\`를 남용하면 안 된다.


---

<script data-ad-client="ca-pub-9951774327887666" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>


## [ ] 대괄호:여러 문자 중 하나와 일치

대괄호 `[`와 `]` 사이에 원하는 문자를 여러 개 넣으면, 문자열이 넣은 문자 중 하나와 일치하면 매칭이 이루어진다. 즉 OR 개념이라고 할 수 있다.  
여기서 중요한 것은 `[ ]` 안에 얼마나 많은 문자 종류가 있는지에 상관없이 딱 한 문자와 일치된다는 것이다.

예를 들어 정규식 표현이 `[abc]`이고 문자열이 'a'이면 **re.match**는 매칭되었다고 할 것이다.  
문자열이 'b'이거나 'c'이어도 매칭이 된다. 다만 문자열이 'd'이거나 '가나다' 같은 것이면 매칭이 되지 않는다.

```python
matchObj = re.fullmatch("You[;']re studying re module[.,]", \
                        'You;re studying re module,')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 26), match='You;re studying re module,'>
```

사용자의 오타를 잡기에 괜찮은 기능이다.

대괄호 `[ ]`에는 다른 기능이 더 있다. 이전 글에서 [semi-메타문자](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/#semi-%EB%A9%94%ED%83%80%EB%AC%B8%EC%9E%90)를 설명했었는데, 문자 `-`는 대괄호 안에서는 메타문자 역할을 한다.

하이픈 `-`는 범위를 형성한다. 예를 들어 `[a-z]`는 알파벳 소문자 중 하나이기만 하면 매칭이 된다. 또 `[A-Z]`, `[0-9]`는 각각 알파벳 대문자와 숫자 하나에 매칭된다.  
물론 위의 경우뿐만 아니라 넓은 범위도 가능하다. `[가-힣]`의 경우는 한글 한 글자에 일치된다.  
`[A-z]`는 영문 대소문자와 몇 개의 특수문자를 포함한다. 하지만 여러분이 잘 모르는 문자까지 포함될 수 있으므로 영문자는 `[A-Za-z]`와 같이 쓰기를 권한다.

참고로, 대괄호 안에서는 메타문자 역할을 하는 것은 오직 `\`, `^`, `-`, `]` 4개뿐이다. 즉, 이전에 메타문자라고 설명했었던 `.`, `*`, `+` 등은 대괄호 안에서는 그냥 문자 '.', '*', '+' 하나에 매칭된다.  
그러나 헷갈릴 소지가 다분하기 때문에 원래 메타문자인 문자들은 그냥 대괄호 안에서도 `\` 이스케이프 처리하는 것이 편할 것이다.  
물론 IDE가 좋다면 redundant escape character라는 경고를 띄워 줄지도 모른다.

캐릿(caret)`^` 문자가 여는 대괄호 바로 뒤에 있으면 문자가 반전된다. 바로 예시를 보도록 하자.

```python
matchObj = re.search('Why [a-z]o serious\?', 'Why so serious?')
print(matchObj)
matchObj = re.search('Why [^0-9]o serious\?', 'Why so serious?')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 15), match='Why so serious?'>
<_sre.SRE_Match object; span=(0, 15), match='Why so serious?'>
```

`[a-z]`는 영문 소문자 하나('s')와 일치되므로 매칭 결과가 반환되었다.  
`[^0-9]`는 숫자를 제외한 문자 하나에 일치되므로, 's'는 숫자가 아니기에 매칭이 되었다.

`[z-a]`와 같이 거꾸로 쓰는 것은 불가능하다.

대괄호 안의 `-`는 또 다른 기능이 있다. 바로 진짜 빼기(마이너스), 즉 차집합 연산이다.  
대괄호 한 쌍을 집합으로 보면 차집합이란 말이 이해가 될 것이다. `[a-z-[g-z]]`의 경우 a-f와 같은 의미이다.  
또 &&를 안에 쓰면 C언어 문법의 and 기능처럼 교집합을 의미한다고 한다.  
하지만 필자가 글을 쓰는 시점에서 이 문법이 유효한지는 확인되지 않았다. 파이썬 버전에 따라 다를 수도 있고, 지원하지 않는 기능일 수도 있다.

---


## . 마침표: 모든 문자와 일치

개행문자를 제외한 모든 문자와 일치하는 정규표현식은 마침표 `.` 이다. 정말로 모든 문자와 일치되기 때문에 별다른 설명은 필요 없을 것 같다.

```python
matchObj = re.findall('r..n[.]', 'ryan. ruin rain round. reign')
print(matchObj)
```
결과
```
['ryan.']
```

대괄호 `[ ]` 안에서는 `.`가 메타문자로 동작하지 않는다고 하였다. 따라서 일치되는 문자열은 'ryan' 하나뿐이다.

### 마침표는 개행 문자와 일치 옵션

파이썬 re 패키지의 많은 함수들은 다음과 같은 인자들을 받는다고 [이전 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/#rematchpattern-string-flags)에서 설명했었다.

> re.match(pattern, string, flags)

여기서 flags는 다음과 같은 종류들이 있다.

syntax  |    long syntax    |   inline flag | meaning
------- | ----------------- | :-----------: | --------
re.I    |	re.IGNORECASE   |   (?i)        |	대소문자 구분 없이 일치
re.M    |	re.MULTILINE    |   (?m)        |	^와 $는 개행문자 위치에서 일치
re.S    |	re.DOTALL       |	(?s)        |   마침표는 개행문자와 일치
re.A    |   re.ASCII        |   (?a)        |   {\w, \W, \b, \B}는 ascii에만 일치
re.U    |	re.UNICODE      |	(?u)        |   {\w, \W, \b, \B}는 Unicode에 일치
re.L    |	re.LOCALE       |	(?L)        |   {\w, \W, \b, \B}는 locale dependent
re.X    |	re.VERBOSE      |	(?x)        |   정규표현식에 주석을 달 수 있음

우선 다른 것들은 나중에 살펴보고, 마침표 옵션만을 보자.

syntax  |    long syntax    |   meaning
------- | ----------------- | ----------
re.S    |	re.DOTALL       |	마침표는 개행문자와 일치

```python
print(re.findall('a..', 'abc a  a\na'))
print(re.findall('a..', 'abc a  a\na', re.S))
print(re.findall('a..', 'abc a  a\na', re.DOTALL))
```
결과
```
['abc', 'a  ']
['abc', 'a  ', 'a\na']
['abc', 'a  ', 'a\na']
```

개행 문자도 마침표에 일치되는지를 설정할 수 있음을 확인하였다.  
문자열을 행 단위로 처리하거나 아니면 전체 문자열을 대상으로 처리할 수 있다는 것에 이 옵션의 존재 의의가 있다.

#### 모드 변경자

아니면 다른 방법도 있다. 정규표현식 내에서 사용할 수도 있다.  
문자열 앞에 `(?s)` 토큰을 넣으면 된다.

```python
print(re.findall('(?s)a..', 'abc a  a\na'))
```
결과
```
['abc', 'a  ', 'a\na']
```

모드 변경자는 여러 개를 중첩하여 사용할 수도 있다.  
또한 일부분에만 사용하고 싶으면 `(?s<regex>)`처럼 모드 변경자의 소괄호 안에 집어넣으면 된다.
```python
print(re.findall('(?is)a..', 'Abc'))
print(re.findall('(?is:a..) and abc is good',
'''
Abc and abc is good.
abc and Abc is good. 
'''))
```
결과
```
['Abc']
['Abc and abc is good']
```
두 번째 **findall**에서 문장을 한 개만 찾은 것을 유의하라.

---


## 문자 집합: \w \W, \d \D, \s \S, \b \B

### \w, \W: 단어 문자, 비 단어 문자

`\w`는 단어 문자 1개와 일치된다. 단어 문자는 영문 대소문자, 숫자 0-9, 언더바 '_' 를 포함한다.  
한글 등 알파벳 이외의 단어는 파이썬 버전에 따라 다른데, Unicode를 기본으로 사용하는 파이썬 3이라면 아마 `\w`의 범위에 한글도 포함될 것이다. 여러분이 스스로 확인해 봐야 할 것이다.


`\W`는 단어 문자 이외의 문자 1개에 일치된다. 즉 공백 문자, 특수 문자 등에 일치된다고 보면 된다.  
`\w`와 정확히 반대의 역할을 한다.

```python
matchObj = re.search('\w\w\w', 'a_가')
print(matchObj)
matchObj = re.findall('\w\W\w', 'a (9_a a')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 3), match='a_가'>
['a a']
```
첫 번째 출력 결과의 경우 단어 3개를 나타내는 정규표현식에 'a_가'가 매칭되었다.
두 번째 출력 결과는 잘 보면  
1) 단어 문자(a)  
2) 비 단어 문자( )  
3) 단어 문자(a)  
순으로 되어 있는데, 그런 결과는 'a a' 하나뿐이다.

### \d, \D: 숫자 문자, 비 숫자 문자

`\d`는 숫자 문자 1개에 일치된다. 마찬가지로 `\D`는 비 숫자 문자 1개에 일치된다.

```python
matchObj = re.search('\d\d', '12abc34')
print(matchObj)
matchObj = re.findall('\d\d\D\D', '11aa11c1')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 2), match='12'>
['11aa']
```
첫 번째 출력 결과는 매칭되는 문자열은 두 군데로 '12'와 '34'이다. 하지만 **re.search**는 제일 처음 하나만 찾아내기 때문에 하나만 반환하였다.
두 번째 출력 결과는 숫자 2개에 비 숫자 문자 2개가 붙어 있는 문자열 '11aa'를 잘 찾아 주었다.

### \s, \S: 공백 문자, 비 공백 문자

`\s`는 공백 문자(빈칸 ' ', 탭 '\t', 개행 '\n') 1개에 일치된다. 마찬가지로 `\S`는 `\s`의 반대 역할이다. 즉, 공백 문자 이외의 모든 문자 1개에 일치된다.

```python
matchObj = re.search(
    'Oh\smy\sgod\s\S',
    '''Oh my\tgod
!''')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 11), match='Oh my\tgod\n!'>
```


### \b, \B: 단어 경계, 비 단어 경계

단어 경계 `\b` 는, 문자 하나와 일치되는 것이 아니라 정말로 단어 경계와 일치된다. 단어 문자와 비 단어 문자 사이와 매칭된다고 보면 된다.

비 단어 경계 `\B` 는 마찬가지로 반대의 역할을 수행한다. 즉, 단어 문자와 단어 문자 사이 혹은 비 단어 문자와 비 단어 문자 사이와 일치된다.

다른 말로는, `\b`는 `\w`에 일치되는 한 문자와 `\W`에 일치되는 한 문자 사이에서 일치되고, `\B`는 `\w`에 일치되는 두 문자 사이 또는 `\W`에 일치되는 두 문자 사이에서 일치된다.

한 가지 주의할 점으로는 `\b`나 `\B`를 사용하기 위해서는 정규표현식 앞에 `r` prefix를 붙여줘야 한다는 것이다.  
예시를 보자.

```python
matchObj = re.findall(r'\w\b\W\B', 'ab  c d  == = e= =f')
print(matchObj)
```
결과
```
['b ', 'd ', 'e=']
```
위의 예시는  
1) 단어 문자  
2) 단어 경계  
3) 비 단어 문자  
4) 비 단어 경계  

순으로 되어 있는 문자열을 찾는다. 위의 조건을 만족시키려면 단어 문자 + 비 단어 문자 + 비 단어 문자 조합을 찾아야 한다. 그리고 실제로 매칭되는 문자열은 단어 문자 + 비 단어 문자이다.  
(주: 여기서 2) 단어 경계는 쓸모가 없다. 이유는 여러분이 알아서 생각하면 된다.)

#### 응용 문제

문제 1: 'line'과는 일치하지만, 'outline'나 'linear' 등과는 일치하지 않는 정규표현식을 작성하라. 즉, 정확히 'line' 단어와만 일치해야 한다.
<details>
    <summary>문제 1 정답보기</summary>
    <p>\bline\b</p>
</details>

<br>

문제 2: 'stacatto'에는 일치하지만, 'cat'이나 'catch', 'copycat' 등과는 일치하지 않는 정규표현식을 작성하라.
<details>
    <summary>문제 2 정답보기</summary>
    <p>\Bcat\B</p>
</details>

<br>

`\b`는 단어 경계로, 다음에 일치된다.

1. 첫 문자가 단어 문자인 경우, 첫 문자 앞에서
2. 인접한 두 문자 중 하나만 단어 문자인 경우, 그 사이에서
3. 끝 문자가 단어 문자인 경우, 끝 문자 뒤에서

즉 문자열의 맨 앞과 맨 끝은 비 단어인 것으로 처리된다.

`\B`는 비 단어 경계로, 다음에 일치된다.

1. 첫 문자가 비 단어 문자인 경우, 첫 문자 앞에서
2. 두 단어 문자 사이 또는 두 비 단어 문자 사이에서
3. 끝 문자가 비 단어 문자인 경우, 끝 문자 뒤에서
4. 빈 문자열에서

(헷갈리는) 예시를 보자.

```python
print(re.findall(r'\b', 'a'))
print(re.findall(r'\B', 'a'))

print(re.findall(r'\b', 'a aa'))
print(re.findall(r'\B', 'a aa'))
```
결과
```
['', '']
[]
['', '', '', '']
['']
```

각각 어디에서 일치된 것인지 이해해 보기 바란다.

### 옵션: r prefix

원래 r prefix란 이스케이프 문자 `\`를 이스케이프 처리 문자가 아닌 일반 리터럴 문자로 인식하게끔 하는 역할을 한다. [영문 설명](https://stackoverflow.com/questions/2241600/python-regex-r-prefix)을 가져오면 아래와 같다.

> When an "r" or "R" prefix is present, a character following a backslash is included in the string without change, and all backslashes are left in the string. For example, the string literal r"\n" consists of two characters: a backslash and a lowercase "n". String quotes can be escaped with a backslash, but the backslash remains in the string; for example, r"\"" is a valid string literal consisting of two characters: a backslash and a double quote; r"\" is not a valid string literal (even a raw string cannot end in an odd number of backslashes). Specifically, a raw string cannot end in a single backslash (since the backslash would escape the following quote character). Note also that a single backslash followed by a newline is interpreted as those two characters as part of the string, not as a line continuation.

해석하면, 
> "r"이나 "R" 접두사가 있으면, \ 뒤에 있는 문자는 문자열에 변화 없이 그대로 남아 있게 되고, 모든 \ 또한 문자열에 남아 있게 된다. 예를 들어, 리터럴 문자열 r"\n"은 \와 소문자 n 2개의 문자로 구성된다. 따옴표 문자열 역시 \가 있으면 이스케이프 처리될 수 있지만, \는 여전히 문자열에 남아 있게 된다. 예를 들어 r"\\""의 경우 \와 " 두 개로 구성된 유효한 문자열이다. r"\\"는 유효하지 않다(raw string은 홀수 개의 \로 끝날 수 없다). 특별히, raw string은 한 개의 \로 끝날 수 없다(\는 다음에 오는, 즉 문자열의 끝을 알리는 따옴표를 이스케이프 처리하므로). newline이 다음에 오는 한 개의 \는 문자열의 일부로서 두 개의 문자로 취급되지, 개행으로 처리되지 않는다. 

예시를 보자.

```
>>> r'\'
SyntaxError: EOL while scanning string literal
>>> r'\''
"\\'"
>>> '\'
SyntaxError: EOL while scanning string literal
>>> '\''
"'"
>>> 
>>> r'\\'
'\\\\'
>>> '\\'
'\\'
>>> print r'\\'
\\
>>> print r'\'
SyntaxError: EOL while scanning string literal
>>> print '\\'
\
```

### Unicode/Locale dependent 옵션

파이썬3은 기본적으로 한글도 "단어 문자"에 포함되기 때문에 쓸 일이 있을지는 모르지만, 이 옵션들도 소개해 본다.

syntax  |    long syntax    |   inline flag | meaning
------- | ----------------- | :-----------: | --------
re.A    |   re.ASCII        |   (?a)        |   {\w, \W, \b, \B}는 ascii에만 일치
re.U    |	re.UNICODE      |	(?u)        |   {\w, \W, \b, \B}는 Unicode에 일치
re.L    |	re.LOCALE       |	(?L)        |   {\w, \W, \b, \B}는 locale dependent

파이썬3은 기본적으로 Unicode를 기준으로 처리되기 때문에 `re.U`는 쓸모가 없다. 그러나 호환성을 위해 아직까지는 살아 있는 옵션이다.  
아스키에만 일치하는 옵션을 쓰고 싶으면 `re.ASCII` 옵션을 사용하면 된다.  

조금 더 자세한 설명은 [여기](https://docs.python.org/3/library/re.html#module-contents)를 참조하라.

다른 flags 사용법과 똑같으므로 생략하도록 하겠다.

---


## `^`, \$, \\A, \\Z: 문자열 전체 또는 행의 시작이나 끝의 대상을 대조

`\A`는 문자열 시작을, `\Z`는 문자열 끝과 일치된다.

이들은 일명 앵커라고 부르는데, 문자와 일치되는 것이 아니라 정규식 패턴을 특정 위치에 고정시켜서 그 위치에 일치시키는 역할을 한다.

`^`와 `$`는 기본적으로 행 시작과 행 끝에 일치된다.

여기서 행은 문자열의 시작과 개행문자 사이, 개행문자와 개행문자 사이, 개행문자와 문자열의 끝 사이 부분이다. 문자열에 개행문자가 없으면 전체 문자열이 한 개의 행이 된다.

`^`와 `$`는 일반적으로 `\A`와 `\Z` 앵커와 효과가 같다. 다른 경우는 옵션을 설정하는 경우인데, re.MULTILINE 옵션을 설정하면 `^`와 `$`는 문자열 전체의 시작/끝이 아닌 행의 시작/끝에서 일치된다.

```python
print(re.findall('\Aryan\d\Z', 'ryan1'))
print(re.findall('^ryan\d$', 'ryan1'))

print(re.findall('\A ryan\d\s\Z', ' ryan1 \n ryan2 \n rain1 \n ryan3 '))
print(re.findall('^ ryan\d\s$', ' ryan1 \n ryan2 \n rain1 \n ryan3 '))
print(re.findall('^ ryan\d\s$', ' ryan1 \n ryan2 \n rain1 \n ryan3 ', re.M))
print(re.findall('^ ryan\d\s$', ' ryan1 \n ryan2 \n rain1 \n ryan3 ', re.MULTILINE))
```
결과
```
['ryan1']
['ryan1']
[]
[]
[' ryan1 ', ' ryan2 ', ' ryan3 ']
[' ryan1 ', ' ryan2 ', ' ryan3 ']
```

Java, .NET 등에서는 `\z` 옵션이 있지만, 파이썬에는 bad escape 에러를 보게 되므로 사용하지 말자.

응용으로, 빈 문자열 혹은 빈 행을 검사할 수 있다.

```python
print(re.fullmatch('\A\Z', ''))
print(re.fullmatch('\A\Z', '\n'))
print(re.fullmatch('^$', ''))
print(re.fullmatch('^$', '\n'))
print(re.findall('^$', '\n', re.M))
```
결과
```
<_sre.SRE_Match object; span=(0, 0), match=''>
None
<_sre.SRE_Match object; span=(0, 0), match=''>
None
['', '']
```

`^`, `$`도 마침표 `.`처럼 옵션을 인라인으로 설정할 수 있다.

```python
print(re.findall('(?m)^$', '\n'))
```
결과
```
['', '']
```

참고로, 옵션을 여러 개 쓰려면 `|`로 OR 연산을 시켜주면 된다.

```python
print(re.findall('^ ryan\d\s$', ' ryan1 \n Ryan2 \n rain1 \n RYAN3 ', re.M | re.IGNORECASE))
```
결과
```
[' ryan1 ', ' Ryan2 ', ' RYAN3 ']
``` 

위의 예시처럼 full-name과 약자를 같이 써도 되지만, 가독성을 생각한다면 굳이 그렇게 할 이유는 없다.


## 유니코드 번호

한 글자 일치와 사용법은 같다.

```python
print(re.findall('\u18ff\d', '0᣿1頶᣿2䅄ሲ᣿3456'))
```
결과
```
['\u18ff1', '\u18ff2', '\u18ff3']
```

참고로 '\u18ff'는 '᣿'이다.

---


[다음 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/)에서는 다자택일(OR), 반복 등을 다루도록 하겠다.
