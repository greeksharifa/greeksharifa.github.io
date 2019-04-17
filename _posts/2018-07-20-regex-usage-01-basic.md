---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 01. Basic
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

**[파이썬 정규표현식(re) 사용법 - 01. Basic](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)**  
[파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)  
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

## 정규표현식의 기초

### 일대일 매칭되는 문자

정규표현식 안에서, 바로 다음 절에서 설명하는 메타문자를 제외한 모든 문자 하나는 일반 문자열 하나와 매칭된다. 예를 들어, `a`는 a와 매칭되고, `가`는 '가'와 매칭되는 식이다.  
당연히 `a`가 'b' 또는 '가'와 매칭되지는 않는다.

### 메타문자

어떤 프로그래밍 언어의 정규표현식이든 메타문자라는 것이 존재한다.  
이는 특수한 기능을 하는 문자로, `import` 등 파이썬의 예약어와 비슷한 역할을 맡는 문자라고 생각하면 된다.

파이썬 re 모듈의 메타문자는 총 12개로 다음과 같은 것들이 있다.

` $()*+.?[\^{| `

이들 메타문자는 각각의 문자 하나에 매칭되지 않는다.  
예를 들어 일반 문자인 `a`는 문자 'a'에 매칭하지만, 여는 소괄호 `(`는 문자 '('와 매칭하지 않는다. 
> 그럼 찾고자 하는 문자열에 소괄호가 있으면 어떻게 하나?

위의 문자들의 앞에 백슬래시 `\`를 붙여 주면 일반 문자처럼 한 글자에 매칭된다. 
예를 들어 `\(`는 문자 '('와 매칭된다.

이들의 사용법은 차차 알아보도록 하자.

### semi-메타문자

사실 이건 필자가 붙인 이름이지만... 이들 문자는 평소에는 메타문자가 아니지만, 특수한 상황에서는 메타문자 역할을 하는 문자들이다.  
`]`, `-`, `)` 가 있다.

닫는 괄호야 당연히 여는 괄호에 대응된다는 것은 알 수 있을 것이다. `-`는 이후에 설명한다.

---

## re 패키지 기본 method

### import

물론 `py` 파일에서는 `import re`를 해주어야 쓸 수 있다.

### re.match(pattern, string, flags)

![01_match](/public/img/정규표현식(re)/2018-07-20-regex-usage-01-basic/01_match.PNG)

**re.match** 함수는 "문자열의 처음"부터 시작하여 패턴이 일치되는 것이 있는지를 확인한다.  
다음과 같이 사용한다.

```python
matchObj = re.match('a', 'a')
print(matchObj)

print(re.match('a', 'aba'))
print(re.match('a', 'bbb'))
print(re.match('a', 'baa'))
# 사실 match의 결과를 바로 print하지는 않는다. 결과를 활용하는 방법은 나중에 설명할 matchObj.group 함수를 쓰는 것이다.
```

결과
```
<_sre.SRE_Match object; span=(0, 1), match='a'>
<_sre.SRE_Match object; span=(0, 1), match='a'>
None
None
```

**re.match** 함수는 문자열의 처음부터 시작하여 패턴이 일치되는 것이 있는지를 확인한다.  
위의 예시에서 첫번째는 패턴과 문자열이 동일하므로 매치되는 것을 확인할 수 있다.  
두번째 예시는 문자열이 'a'로 시작하기 때문에 매치가 된다.  
나머지 두 개는 'a'로 시작하지 않아 패턴 `a`와 매치되지 않는다. 매치되지 않을 때 **re.match** 함수는 None을 반환한다. 

매치가 되었을 때는 match Object를 반환한다. 위의 결과에서 `_sre.SRE_Match object`를 확인할 수 있다.

**re.match** 함수는 인자로 1)pattern 2)string 3)flags를 받는다. 3번은 필수 인자는 아닌데, 어떤 옵션이 있는지는 뒤에서 설명한다.  
각 인자는 각각 1)패턴 2)패턴을 찾을 문자열 3)옵션을 의미한다.

### re.search(pattern, string, flags)

![02_search](/public/img/정규표현식(re)/2018-07-20-regex-usage-01-basic/02_search.PNG)

**re.search** 함수는 **re.match**와 비슷하지만, 반드시 문자열의 처음부터 일치해야 하는 것은 아니다.

다음 예시를 보자.

```python
matchObj = re.search('a', 'a')
print(matchObj)

print(re.search('a', 'aba'))
print(re.search('a', 'bbb'))
print(re.search('a', 'baa'))
```

결과
```
<_sre.SRE_Match object; span=(0, 1), match='a'>
<_sre.SRE_Match object; span=(0, 1), match='a'>
None
<_sre.SRE_Match object; span=(1, 2), match='a'>
```

네 번째 결과가 달라졌음을 볼 수 있다. **re.search** 함수는 문자열의 처음뿐 아니라 중간부터라도 패턴과 일치되는 부분이 있는지를 찾는다.  
따라서 네 번째 문자열 'baa'의 경우 1번째 index(두 번째 문자) 'a'와 매치된다.

위의 결과에서 `span=(0, 1)` 를 확인할 수 있다. 위의 두 결과는 `span=(0, 1)`인데,  
이는 0번째 문자부터 1번째 문자 전까지(즉, 0번째 문자 하나인 'a')가 패턴과 매치되었음을 뜻한다.  
`span=(1, 2)`의 경우 1번째 문자('baa' 의 첫 번째 'a'이다)가 패턴과 매치되었음을 볼 수 있다.

### re.findall(pattern, string, flags)

![03_findall](/public/img/정규표현식(re)/2018-07-20-regex-usage-01-basic/03_findall.PNG)

이름에서 알 수 있듯이 **re.findall** 함수는 문자열 중 패턴과 일치되는 모든 부분을 찾는다.

```python
matchObj = re.findall('a', 'a')
print(matchObj)

print(re.findall('a', 'aba'))
print(re.findall('a', 'bbb'))
print(re.findall('a', 'baa'))
print(re.findall('aaa', 'aaaa'))
```

결과
```
['a']
['a', 'a']
[]
['a', 'a']
['aaa']
```

각 예시에서, 문자열의 a의 개수를 세어 보면 잘 맞는다는 것을 확인할 수 있다.

함수 설명을 잘 보면, "non-overlapping" 이라고 되어 있다. 즉 반환된 리스트는 서로 겹치지 않는다는 뜻이다.  마지막 예시가 이를 말해주고 있다. 겹치는 것을 포함한다면 두 개가 반환되어야 했다.

### re.finditer(pattern, string, flags)

![04_finditer](/public/img/정규표현식(re)/2018-07-20-regex-usage-01-basic/04_finditer.PNG)

**re.findall**과 비슷하지만, 일치된 문자열의 리스트 대신 matchObj 리스트를 반환한다.

```python
matchObj_iter = re.finditer('a', 'baa')
print(matchObj_iter)

for matchObj in matchObj_iter:
    print(matchObj)
```

결과
```
<callable_iterator object at 0x000002795899C550>
<_sre.SRE_Match object; span=(1, 2), match='a'>
<_sre.SRE_Match object; span=(2, 3), match='a'>
```

iterator 객체 안에 matchObj가 여러 개 들어 있음을 확인할 수 있다.

### re.fullmatch(pattern, string, flags)

![05_fullmatch](/public/img/정규표현식(re)/2018-07-20-regex-usage-01-basic/05_fullmatch.PNG)

**re.fullmatch**는 패턴과 문자열이 남는 부분 없이 완벽하게 일치하는지를 검사한다.

```python
matchObj = re.fullmatch('a', 'a')
print(matchObj)

print(re.fullmatch('a', 'aba'))
print(re.fullmatch('a', 'bbb'))
print(re.fullmatch('a', 'baa'))
print(re.fullmatch('aaa', 'aaaa'))
```

결과
```
<_sre.SRE_Match object; span=(0, 1), match='a'>
None
None
None
None
```

맨 위의 예시만 문자열이 남는 부분 없이 정확하게 일치하므로 매칭 결과를 반환했다. 나머지 예시는 문자열이 뒤에 남기 때문에 매치되는 결과 없이 None을 반환했다.


### match Object의 메서드들

match Object를 그대로 출력해서 쓰고 싶은 사람은 별로 없을 것이다. **re.match** 등의 결과로 얻은 matchObj를 활용하는 방법을 정리하면 다음과 같다.

Method | Descrption
------ | ----------
group() | 일치된 문자열을 반환한다.
start() | 일치된 문자열의 시작 위치를 반환한다.
end()   | 일치된 문자열의 끝 위치를 반환한다.
span()  | 일치된 문자열의 (시작 위치, 끝 위치) 튜플을 반환한다.

```python
matchObj = re.search('match', "'matchObj' is a good name, but 'm' is convenient.")
print(matchObj)

print(matchObj.group())
print(matchObj.start())
print(matchObj.end())
print(matchObj.span())
```
결과
```
<_sre.SRE_Match object; span=(1, 6), match='match'>
match
1
6
(1, 6)
```

잘 세어보면 'match'가 1번째 문자부터 6번째 문자 직전까지임을 알 수 있다. 인덱스는 0부터이다.


---

[다음 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)에서는 정규표현식의 기초를 더 살펴보도록 한다.
