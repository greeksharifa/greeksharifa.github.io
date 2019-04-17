---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 08. 예제(단어, 행)
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

[파이썬 정규표현식(re) 사용법 - 01. Basic](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)  
[파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)  
[파이썬 정규표현식(re) 사용법 - 03. OR, 반복](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/)  
[파이썬 정규표현식(re) 사용법 - 04. 그룹, 캡처](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/)  
[파이썬 정규표현식(re) 사용법 - 05. 주석, 치환, 분리](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)  
[파이썬 정규표현식(re) 사용법 - 06. 치환 함수, 양방탐색, 조건문](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/05/regex-usage-06-advanced/)  
[파이썬 정규표현식(re) 사용법 - 07. 예제(숫자)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-07-example/)  
**[파이썬 정규표현식(re) 사용법 - 08. 예제(단어, 행)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-08-example/)**  
[파이썬 정규표현식(re) 사용법 - 09. 기타 기능](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/24/regex-usage-09-other-functions/)  

---

이 글에서는 정규표현식으로 처리할 수 있는 예제를 설명한다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

파이썬 버전은 3.6을 기준으로 하나, 3.x 버전이면 (아마) 동일하게 쓸 수 있다.  
2.7 버전은 한글을 포함한 비 알파벳 문자 처리가 다르다.

---

## 단어

### 기초

문제 1: colour, Color, color 등 모든 버전의 color에 일치되는 정규식을 작성하라.
<details>
    <summary>문제 1 정답보기</summary>
    <p>r'\b(?i)colou?r\b'</p>
</details>

<br>

`(?i)` 모드 변경자 대신에 **re.IGNORECASE** 옵션을 주어도 상관없다.

다음 문제는 이전에 본 적이 있는 문제이다.

문제 2: cat에는 일치되지 않고, cat을 포함하면서 그보다 긴 단어, staccato, cats, tomcat 등에 일치되는 정규식을 작성하라.
<details>
    <summary>문제 2 정답보기</summary>
    <p>r'\Bcat\B'</p>
</details>

<br>

---

### 특정 단어 배제

문제 3: 이번에는 특정 단어를 제외시켜 보자. reg를 제외한 모든 단어에 일치되도록 하는 정규식을 작성하라.
<details>
    <summary>문제 3 정답보기</summary>
    <p>r'\b(?!reg\b)\w+'</p>
</details>

<br>

생각하기 조금 어려웠을 것 같다.

부정 전방탐색이 빛을 발하는 순간이다. 부정 전방탐색을 씀으로써 앞에 단어 경계가 있고(`\b`), 'reg'로 이루어져 있으며(`reg`), 바로 뒤에 `\b`이 나와 끝나버리는, 즉 'reg'에 정확히 일치되는 순간이 되면 위 정규식은 일치에 실패한다.  
따라서 특정 단어 'reg'를 배제할 수 있게 된다. 부정 전방탐색의 일치에 실패하면, 그 다음 부분인 `\w+`를 갖고 일치를 시도한다. 단어면 성공할 것이다.

문제 4: 이번엔 특정 단어뿐 아니라 그 단어를 포함하는 단어도 제외시켜 보자. reg뿐만 아니라 regex, register, dreg 등은 모두 제외되어야 한다.
<details>
    <summary>문제 4 정답보기</summary>
    <p>r'\b(?:(?!cat)\w)+\b'</p>
</details>

<br>

문제 3번을 생각해냈으면 이 문제도 풀 수 있을 것이라고 생각한다.

주어진 조건은 프로그래밍이 가능한 다음 세 문장으로 압축할 수 있다.
1. 단어 중간의 어떤 부분에서든 그 앞에 'reg'가 와서는 안 된다. `(?!cat)\w`
2. 1번을 만족했으면, 단어 문자가 1개 이상은 있어야 한다. `( ... \w)+`
3. 물론 양끝에 단어 경계는 있어야 한다. `\b ... \b`

위 세 조건을 적용시키면 'reg'가 단어에 포함된다면 일치되지 않는다는 사실을 알 수 있다.

문제 5: 바로 뒤에 숫자가 따라오는 단어를 무시하는, 다른 모든 단어를 검색하는 정규식을 작성하라. 단어와 숫자 사이에는 비 단어 문자가 온다고 가정하자.
<details>
    <summary>문제 5 정답보기</summary>
    <p>r'\b\w+\b(?!\W+\d+\b'</p>
</details>

<br>

이번에도 부정 전방탐색을 사용하면 된다. 무엇을 제외하고 싶다면 부정 양방탐색을, 다른 조건을 더 추가해야 한다면 긍정 양방탐색을 사용한다는 것을 기억하자.

---

## 행

### 빈 행 제거

문제 6: 빈 행을 제거하자. 예컨대 '\n\n' 을 '\n' 하나로 치환하여 불필요한 행을 제거하는 식이다. 운영체제에 독립적으로 작성되어야 한다.
<details>
    <summary>문제 6 정답보기</summary>
    <p>re.sub(r'\n+', r'\n', string)</p>
</details>

<br>

개행 문자 말고 모든 공백 문자를 하나로 치환하려면 다음을 쓰면 된다.

```python
re.sub('\s+', ' ', string)
```
가로 공백 문자만 치환하려면 `\s` 대신 `[ \t]`를 사용하라.

### 중복 행 제거

문제 7: 중복 행을 제거하자. 똑같은 내용의 행이 두 번 이상 반복되는 행을 하나만 남겨두고 모두 제거하자.
<details>
    <summary>문제 7 정답보기</summary>
    <p>re.sub(r'^(.*)(?:(?:\r?\n|\r)\1)+$', r'\1', re.MULTILINE)</p>
</details>

<br>

복잡해 보이지만 크게 어렵진 않다. 중복 행이란 다음과 같은 구조일 것이다.

1. 행 내용
2. 행 구분자
3. 행 내용
4. 행 구분자
5. 행 내용
   ...

행 내용은 `(.*)`으로 간단히 해결된다. 나중에 똑같은 것을 일치시켜야 하므로 캡처 그룹을 사용한다.  
그리고 행 구분자를 사용하는데, 운영체제에 상관없이 사용하기 위해 조금 복잡하게(`\r?\n|\r`)이 들어갔지만, 자신이 무슨 운영체제에서 프로그래밍하고 있으며 어떤 운영체제에서 작성된 파일을 갖고 있는지 안다면 하나만 써도 무방하다.  
그 다음은 행 내용을 재사용하기 위한 `\1`, 1회 이상 반복을 위한 `+`, 그리고 한 행만 남겨두고 제거하기 위한 `r'\1'`을 사용하면 해결된다.

주의할 점으로는 **re.MULTILINE** 옵션을 주어야 하며, **re.DOTALL**은 절대로 주어서는 안 되는 옵션이라는 점이다.

---

조금 더 복잡한 예제를 정리해 두면 좋겠지만, 그때그때 맞게 쓰는 것이 더 나을 것 같아서 굳이 따로 정리할 필요는 없을 것 같다.

[다음 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/24/regex-usage-09-other-functions/)에서는 **re** 패키지에 포함된 다른 함수들을 다루도록 하겠다.
