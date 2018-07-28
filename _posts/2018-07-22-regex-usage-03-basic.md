---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 03. OR, 반복
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

이 글에서는 정규표현식 기초와 python library인 `re` 패키지 사용법에 대해서 설명한다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

---

## 정규표현식의 기초: OR, 반복

### `|` : 다자택일

단어 'one', 'two', 'three' 중 하나에 대응하고 싶다면 `|`를 쓰면 된다(백슬래시 또는 원화로 되어 있는 `\` 키의 shift 버전이다).

```python
matchObj = re.findall('one|two|three', 'one four two three zero')
print(matchObj)
```
결과
```
['one', 'two', 'three']
```

작동 과정을 살펴보자.

1. 맨 앞에서 바로 'one'이 일치된다.
2. 공백 한 개는 `o`, `t` 어느 것에도 일치되지 않으므로 건너뛴다. 'f'도 마찬가지이다.
3. 'four' 의 'o'에 도달했다. `o`는 일치되기 때문에, 'u'에 `n`을 일치시켜본다. 물론 아니다.
4. 계속 넘어가서 'two'의 't'에 도달했다. 't'는 `t`에 일치된다.
5. 'w'에서는 `w`와 `h` 중 일치되는 것을 찾는다. 현재 `tw`까지 일치되었다.
6. 'o'까지 일치되어 'two`를 찾았다. 
7. 이와 비슷한 과정을 반복하여 'three'까지 찾고 종료한다.

일반적으로 `|`로 나열한 단어들의 순서가 중요하지는 않다. 하지만 중요한 순간이 있다.  
다음 예시를 보자.

```python
matchObj = re.findall('one|oneself|onerous', 'oneself is the one thing.')
print(matchObj)
```
결과
```
['one', 'one']
```

'oneself'가 있음에도 `oneself`에 일치되지 않았다. 그 이유는 이미 'one'을 찾아버렸고, 정규식은 overlapping된 부분을 또 찾지 않기 때문에, 'one'을 찾고 나서 남은 문자열은 'self is the one thing.'이다. 따라서 남은 문자열에서는 더 이상 `oneself`를 찾을 수 없는 것이다.

이 문제의 해결 방법은 두 가지다.

1. 당연하게도 더 긴 `oneself`를 `one` 앞에다 두면 해결된다.
2. 아니면 단어 경계를 활용한다. `\bone\b|\boneself\b`로 쓰면 된다.


### * : 0회 이상 반복

어떤 문자나 기호 뒤에 *(asterisk)를 붙이면 그 문자가 일치되는 만큼 일치된다. 예를 들어 `a*`의 경우 'a'나 'aaa' 혹은 ''(빈 문자열)과도 일치된다.

예시를 보자. 
```python
print(re.match('a*', ''))
print(re.match('a*', 'a'))
print(re.search('a*', 'aaaa'))
print(re.fullmatch('a*', 'aaaaaa'))
print(re.findall('a*', 'aaabaaa aa  '))

matchObj = re.search('<p>.*</p>', '<p> Lorem ipsum... is boring. </p>')
print(matchObj)
```
결과
```
<_sre.SRE_Match object; span=(0, 0), match=''>
<_sre.SRE_Match object; span=(0, 1), match='a'>
<_sre.SRE_Match object; span=(0, 4), match='aaaa'>
<_sre.SRE_Match object; span=(0, 6), match='aaaaaa'>
['aaa', '', 'aaa', '', 'aa', '', '', '']
<_sre.SRE_Match object; span=(0, 34), match='<p> Lorem ipsum... is boring. </p>'>
```

여섯 번째 결과의 경우, 파이썬 버전에 따라 **None**이 반환될 수도 있다.

그런데 한 가지 이상한 결과가 보인다. 다섯 번째 실행문이다.

```python
print(re.findall('a*', 'aaabaaa aa  '))
# ['aaa', '', 'aaa', '', 'aa', '', '', '']
```

빈 문자열이 이상하리만큼 많이 매칭되었다. 굉장히 비직관적인 결과이지만, 빈 문자열에도 일치된다는 것을 생각했을 때 아예 틀린 것은 분명히 아니다.  
매칭되는 빈 문자열들은 a가 아닌 다른 문자들과의 경계에서 발생한다고 생각하면 될 듯하다. 하지만, 아마 대부분 이것은 원하는 결과가 아닐 것이기 때문에, 'a' 덩어리를 찾고 싶다면 다음 메타문자를 보자.

### + : 1회 이상 반복

`*`과 비슷하지만 무조건 한 번이라도 등장해야 한다. 위와 거의 같은 예시를 보자.

```python
print(re.match('a+', ''))
print(re.match('a+', 'a'))
print(re.search('a+', 'aaaa'))
print(re.fullmatch('a+', 'aaaaaa'))
print(re.findall('a+', 'aaabaaa aa  '))

matchObj = re.search('<p>.+</p>', '<p> Lorem ipsum... is boring. </p>')
print(matchObj)
```
결과
```
None
<_sre.SRE_Match object; span=(0, 1), match='a'>
<_sre.SRE_Match object; span=(0, 4), match='aaaa'>
<_sre.SRE_Match object; span=(0, 6), match='aaaaaa'>
['aaa', 'aaa', 'aa']
<_sre.SRE_Match object; span=(0, 34), match='<p> Lorem ipsum... is boring. </p>'>
```

아마 이것이 여러분이 원하는 'a' 덩어리를 찾은 결과일 것이다.  
빈 문자열이 일치되지 않은 것을 기억하자.

### {n, m} : 지정 횟수만큼 반복

중괄호는 지정한 횟수만큼 정규식을 반복시키는 것이다. 이 쓰임으로 중괄호를 쓸 때 쓰는 방법은 세 가지가 있다.

1. {n} : 정확히 n회만큼 반복
2. {n, m} : n회 이상 m회 이하 반복
3. {n, } : n회 이상 반복. 무한히 일치될 수 있다.

물론 n은 자연수, m은 n보다 큰 정수이다.
그리 어렵지 않으므로 바로 예시를 보자.

```python
print(re.search('a{3}', 'aaaaa'))
print(re.findall('a{3}', 'aaaaaaaa'))
print(re.findall('a{2,4}', 'a aa aaa aaaa aaaaa'))
print(re.findall('a{2,}', 'a aa aaa aaaa aaaaa'))
```
결과
```
<_sre.SRE_Match object; span=(0, 3), match='aaa'>
['aaa', 'aaa']
['aa', 'aaa', 'aaaa', 'aaaa']
['aa', 'aaa', 'aaaa', 'aaaaa']
```

예상과는 조금 다른 결과일지도 모르겠다. 오직 'aaa'만을 찾고 싶을 때 `a{3}`처럼 쓰면 'aaaaa'의 일부분인 'aaa'에도 일치될 수 있다. 따라서 정확히 'aaa'만을 찾으려면 `\baaa\b`처럼 단어 경계를 활용하는 쪽이 좋다.

참고로 `{0, }`은 `*`과 같고, `{1,}`은 `+`와 같다. 

![01](\public\img\정규표현식(re)\2018-07-20-regex-usage-03-basic\01.{0,1,}.PNG)


### ? : 0회 또는 1회 반복

이 메타문자도 어렵지는 않을 것이라 생각된다. `?`는 `{0,1}`과 같다. 

```python
print(re.findall('ab?a', 'aa aba aaaa'))
```
결과
```
['aa', 'aba', 'aa', 'aa']
```

정규표현식은 항상 최대로 일치시키려 한다는 것을 기억하자.


### 응용 문제

문제 1: 1~8자리 10진수에 일치하는 정규표현식을 작성하라.
<details>
    <summary>문제 1 정답보기</summary>
    <p>r'\b\d{1,8}\b'</p>
</details>

<br>

문제 2: 4자리 또는 8자리 16진수에 일치하는 정규표현식을 작성하라. 16진수는 0~9, a~f를 사용한다. 예시는 abcd1992, 7fffffff, 2dfa9a00이다.
윈도우 오류에서 '0xC1900101' 비슷한 에러를 많이 봤을 것이다.
<details>
    <summary>문제 2 정답보기</summary>
    <p>r'\b[0-9a-f]{4}\b|\b[0-9a-f]{8}\b'</p>
</details>

<br>


문제 3: 1.2이나 3.72e3, 1.002e-12 같은 수를 부동소수점 수 또는 과학적 표기법으로 표기한 수라고 한다. 이와 같은 수에 일치하는 정규표현식을 작성하라.
<details>
    <summary>문제 3 정답보기</summary>
    <p>r'\b\d*\.\d+(e\d+)?'</p>
</details>

<br>

파이썬 버전 3.6 기준으로, `\b`를 쓰려면 **r prefix**를 붙여 주어야 한다고 했었다.

문제 3의 정답에 아직 설명하지 않은 소괄호 `( )`가 있다. 이는 다음 글에서 설명한다.