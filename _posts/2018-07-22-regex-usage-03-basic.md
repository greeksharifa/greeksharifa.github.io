---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 03. Basic
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

이 글에서는 정규표현식 기초와 python library인 `re` 패키지 사용법에 대해서 설명한다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

---

## 정규표현식의 기초: OR, 반복

### | : 다자택일

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

1. 당연하게도 더 긴 `oneself`를 `one` 앞에다 둬 버리면 해결된다.
2. 아니면 단어 경계를 활용한다. `\bone\b|\boneself\b`로 쓰면 된다.


### * : 0회 이상 반복

