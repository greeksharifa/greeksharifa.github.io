---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 04. 그룹, 캡처
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

[파이썬 정규표현식(re) 사용법 - 01. Basic](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)  
[파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)  
[파이썬 정규표현식(re) 사용법 - 03. OR, 반복](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/)  
**[파이썬 정규표현식(re) 사용법 - 04. 그룹, 캡처](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/)**  
[파이썬 정규표현식(re) 사용법 - 05. 주석, 치환, 분리](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)  
[파이썬 정규표현식(re) 사용법 - 06. 치환 함수, 양방탐색, 조건문](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/05/regex-usage-06-advanced/)  
[파이썬 정규표현식(re) 사용법 - 07. 예제(숫자)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-07-example/)  
[파이썬 정규표현식(re) 사용법 - 08. 예제(단어, 행)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-08-example/)  
[파이썬 정규표현식(re) 사용법 - 09. 기타 기능](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/24/regex-usage-09-other-functions/)  

---

이 글에서는 정규표현식 중급 기술과 python library인 `re` 패키지 사용법에 대해서 설명한다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

파이썬 버전은 3.6을 기준으로 하나, 3.x 버전이면 (아마) 동일하게 쓸 수 있다.  
2.7 버전은 한글을 포함한 비 알파벳 문자 처리가 다르다.

---

## 정규표현식의 중급: 그룹, 캡처 = ( )

소괄호 `( )`에는 중요한 기능이 두 가지 있다. 그룹화와 캡처인데, 정규식의 여러 문자를 그룹으로 묶어주는 것과 정규식의 일부분에 해당하는 문자열에만 관심이 있을 때 그 부분을 따로 빼서 캡처하는 기능이다.  
여담으로 그룹화는 기초 과정이지만 캡처와 더불어 중급 과정에 넣었다.

### 그룹화

그룹화는 말 그대로 그룹으로 묶어주는 것이다. 지금까지의 글에서는 정규식 메타문자들의 효력은 대개 한 문자에만 적용이 되었다.

```python
print(re.findall('12+', '12 1212 1222'))
```
결과
```
['12', '12', '12', '1222']
```
'1212'와 같은 문자열을 찾고 싶었는데, '12' 혹은 '1222'만 찾아진다. 즉 메타문자 `+`는 `2`에만 적용이 된 것이다. 이를 `12` 모두에 적용시키려면 소괄호 `( )`로 그룹화시켜주면 된다.

```python
print(re.match('(12)+', '1212'))
print(re.search('(12)+', '1212'))
print(re.findall('(12)+', '1212'))
print(re.fullmatch('(12)+', '1212'))
```
결과
```
<_sre.SRE_Match object; span=(0, 4), match='1212'>
<_sre.SRE_Match object; span=(0, 4), match='1212'>
['12']
<_sre.SRE_Match object; span=(0, 4), match='1212'>
```
정규식은 항상 최대로 일치시키는 쪽으로 문자열은 탐색하기 때문에, '12'가 아닌 '1212'를 잘 찾았다. 그런데 한 가지 이상한 결과는 **re.findall** 결과이다.

다른 예시를 한번 보자.
```python
print(re.findall('A(12)+B', 'A12B'))
print(re.findall('A(12)+B', 'A1212B'))
print(re.findall('A(12)+B', 'A121212B'))
print(re.findall('A(12)+B', 'A12121212B'))
```
결과
```
['12']
['12']
['12']
['12']
```
'A'와 'B'를 통해 문자열 전체가 정규식과 일치된 것을 확인할 수 있으나, '12'가 몇 개인지에 관계없이 딱 '12'만 일치되어 결과로 반환되었다. 이는 괄호가 가진 다른 기능인 캡처 때문이다.



### 캡처

캡처란 원하는 부분만을 추출하고 싶을 때 사용하는 것이다. 예를 들어 'yyyy-mm-dd'와 같이 날짜를 나타내는 문자열 중 월, 일을 각각 따로 빼서 쓰고 싶다고 하자.  
그러면 따로 빼고 싶은 부분인 'mm'과 'dd' 부분에만 소괄호의 캡처 기능을 사용하면 된다.

```python
print(re.findall('\d{4}-(\d\d)-(\d\d)', '2028-07-28'))
print(re.findall('\d{4}-(\d\d)-(\d\d)', '1999/05/21 2018-07-28 2018-06-31 2019.01.01'))
```
결과
```
[('07', '28')]
[('07', '28'), ('06', '31')]
```
월과 일에 해당하는 부분만 따로 빠졌음을 알 수 있다. 그리고 날짜 형식이 맞지 않는 경우에는 아예 캡처되지 않았음을 확인할 수 있다.

여기서 한 가지 문제점은, 6월 31일은 존재하지 않는 날짜란 점이다. 위의 정규식은 숫자로만 처리를 했기 때문에 '9999-99-99'도 일치된다는 문제가 있다. 이러한 문제를 해결하는 방법은 함수를 정규식에 쓰는 것인데, 이 방법에 대해서는 [나중](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)에 알아보도록 한다.



### matchObj.groups()

여러분은 [첫 번째 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)에서 다음 예시를 보았을 것이다.

```python
matchObj = re.search('match', "'matchObj' is a good name, but 'm' is convenient.")
print(matchObj)

print(matchObj.group())
print(matchObj.start())
print(matchObj.end())
print(matchObj.span())
# matchObj를 오랜만에 가져와 보았다. 캡처를 잘 쓰기 위해서는 matchObj가 필요하다.
```
결과
```
<_sre.SRE_Match object; span=(1, 6), match='match'>
match
1
6
(1, 6)
```

이제 정규식을 캡처를 포함한 식으로 바꿔보자.

```python
matchObj = re.search('match', "'matchObj' is a good name, but 'm' is convenient.")
print(matchObj)

print(matchObj.group())
print(matchObj.groups())

print('# ---------------------------------------------------------------- #')

m = re.search('\d{4}-(\d?\d)-(\d?\d)', '1868-12-10')
print(m)

print(m.group())
print(m.groups())
```
결과
```
<_sre.SRE_Match object; span=(1, 6), match='match'>
match
()
# ---------------------------------------------------------------- #
<_sre.SRE_Match object; span=(0, 10), match='1868-12-10'>
1868-12-10
('12', '10')
```
matchObj의 **group** 메서드는 정규식 전체의 일치부를 찾는다. 반면에 **groups** 메서드는 명시적으로 캡처(`( )`로 감싼 부분)한 부분을 반환한다.  

위의 matchObj는 캡처 구문이 없기 때문에 **groups** 결과가 빈 튜플이 되는 것이다.  
반면 m의 경우 월과 일에 해당하는 부분을 반환하였다.

**group**과 **groups**의 사용법을 좀 더 보도록 하자.

![01](\public\img\정규표현식(re)\2018-07-20-regex-usage-04-intermediate\01_group.PNG)

<img src="\public\img\정규표현식(re)\2018-07-20-regex-usage-04-intermediate\02_groups.PNG" width="75%">

```python
m = re.search('\d{4}-(\d?\d)-(\d?\d)', '1868-12-10')
print('m:', m)

print('m.group():', m.group())

for i in range(0, 3):
    print('m.group({}): {}'.format(i, m.group(i)))

print('m.groups():', m.groups())
```
결과
```
m: <_sre.SRE_Match object; span=(0, 10), match='1868-12-10'>
m.group(): 1868-12-10
m.group(0): 1868-12-10
m.group(1): 12
m.group(2): 10
m.groups(): ('12', '10')
```
결과를 보면 대략 사용법을 알 수 있을 것이다.

1. group(i)는 i번째 소괄호에 명시적으로 캡처된 부분만을 반환한다.
2. group(0)은 전체 일치부를 반환하며, group()과 효과가 같다.
3. groups()는 명시적으로 캡처된 모든 부분 문자열을 반환한다.

i번째 캡처된 부분은, i번째 여는 괄호와 대응된다고 생각하면 된다. 캡처를 중첩해서 사용하는 경우`((12)+)`, 첫 번째 캡처는 바깥쪽 소괄호이다.

주의할 점은 group(0)이 0번째 캡처를 의미하는 것이 아니라 전체 일치부를 반환한다는 것이다.

---

### 비 캡처 그룹

그룹화를 위해 소괄호를 반드시 써야 하는데, 굳이 캡처하고 싶지는 않을 때가 있다. 예를 들어 다음과 같이 쓴다고 하자.
```python
matchObj = re.search('((ab)+), ((123)+) is repetitive\.', 'Hmm... ababab, 123123 is repetitive.')
print(matchObj.group())
print(matchObj.group(1))
print(matchObj.group(2)) # don't want
print(matchObj.group(3)) 
print(matchObj.group(4)) # don't want
```
결과
```
ababab, 123123 is repetitive.
ababab
ab
123123
123
```
캡처 기능을 사용할 때 위의 'ababab', '123123'을 얻고 싶을 뿐 'ab'나 '123'을 얻고 싶지는 않을 때가 있다. 그러나 소괄호는 기본적으로 캡처 기능을 갖고 있기 때문에 group(2)에는 '123123' 대신 'ab'가 들어가 있다.  
이는 원하는 결과가 아닐 때가 많다. 그래서 정규표현식은 비 캡처 기능을 지원한다. 

비 캡처 그룹은 `(?:<regex>)`와 같이 사용한다. 위의 예시를 다시 써 보자.

```python
matchObj = re.search('((?:ab)+), ((?:123)+) is repetitive\.', 'Hmm... ababab, 123123 is repetitive.')
print(matchObj.group())
print(matchObj.group(1))
print(matchObj.group(2))
```
결과
```
ababab, 123123 is repetitive.
ababab
123123
```
예상대로 동작하였다.

비 캡처 그룹의 장점은 캡처 그룹의 번호를 이상하게 만들지 않게 할 수 있다는 것과, 쓸데없는 캡처 그룹을 **group**의 반환값에 집어넣지 않게 되므로 성능상의 이점이 있다.  
그러나 성능 향상은 보통 상황이라면 체감하기 어려울 정도이긴 하다.

참고로 [모드 변경자]()나 비 캡처 그룹처럼 여는 소괄호 뒤에 `?`가 있으면, [0회 또는 1회 반복](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/#--0%ED%9A%8C-%EB%98%90%EB%8A%94-1%ED%9A%8C-%EB%B0%98%EB%B3%B5)이나 기타 다른 의미가 아닌 특별한 기능을 하는 토큰이 된다. 앞으로 이러한 토큰들을 여럿 볼 수 있을 것이다.

#### 모드 변경자가 있는 그룹

[여기]()에서 (?s)와 같은 모드 변경자를 본 적이 있을 것이다. 

이러한 모드 변경자는 소괄호를 쓰긴 하지만 캡처 그룹으로 작동하지 않는다.
```python
matchObj = re.search('case sensitive(?i) irrelevant', 'case sensitive IrreLEVant')
print(matchObj.group(0))
print(matchObj.group(1))
```
결과
```
case sensitive IrreLEVant
Traceback (most recent call last):
  File "<input>", line 3, in <module>
IndexError: no such group
```

---

### \\ (숫자): 앞서 일치된 문자열을 다시 비교

앞뒤가 똑같은 세 글자 단어를 찾는다고 해보자. 이를 위해서는 조금 전 살펴본 캡처가 꼭 필요하다.

i번째 캡처된 문자열은 **group(i)** 메서드를 통해 접근할 수 있다고 하였다. 그런데 그건 matchObj을 얻은 후의 얘기고, 정규식 내에서는 다른 방법을 쓴다. 바로 `\(숫자)`이다. 예를 들면 `\1`, `\2`, ...이다.  
이를 재참조부라 한다.

아마 그럴 리는 없겠지만 재참조부가 10개 이상인 경우 그냥 두 자리 수를 쓰면 된다. `\10`, `\11`, ...

`\b`와 마찬가지로 `\1`과 같은 문법을 쓸 때에는 앞에 **r prefix**를 붙여 주어야 한다.

우선 예시를 보자. 단어 경계는 정규식이 더 복잡해 보이므로 일부러 넣지 않았다. 분리된 단어만을 보고 싶다면, `\b`를 넣으면 된다.

```python
print(re.search(r'(\w)\w\1', '토마토 ABC aba xyxy ').group())
print(re.findall(r'(\w)\w\1', '토마토 ABC aba xyxy '))
```
결과
```
토마토
['토', 'a', 'x']
```

첫 번째 결과는 원하는 결과이다. 그러나 **search**는 하나밖에 찾지 못하므로 완벽한 답은 아니다.  
두 번째 결과는 원하는 결과가 아닐 것이다. 이는 `( )`가 들어가면 앞에서 말했듯 캡처 그룹만을 반환하기 때문이다.

전체를 참조하려면 여러 방법이 있지만, 세 가지를 소개한다.

첫 번째는 **search**로 하나를 찾은 다음 남은 문자열로 다시 **search**를 하는 것이다. 그러나 이는 괜한 코딩량이 늘어난다.

두 번째는 캡처를 하나 더 만드는 것이다.

```python
match_list = re.findall(r'((\w)\w\2)', '토마토 ABC aba xyxy ')

for match in match_list:
    print(match[0])
```
결과
```
토마토
aba
xyx
```
재참조부가 `\1`이 아니라 `\2`인 이유는, 여는 소괄호(opening parenthesis)의 순서를 잘 살펴보라. 바깥쪽 소괄호인, 전체를 감싸는 소괄호가 첫 번째 캡처 부분이다. 따라서 안쪽 `(\w)`가 `\2`에 대응된다.

그러나 이 방법은 나쁘지 않지만, **findall**로 찾기 때문에 위치를 찾아주지는 않는다는 단점이 있다.  
일치부의 시작/끝 위치까지 알고 싶을 때에는 **finditer**을 이용한다.

```python
matchObj_iter = re.finditer(r'((\w)\w\2)', '토마토 ABC aba xyxy ')

for matchObj in matchObj_iter:
    print('string: {}, \t start/end position={}, \t 반복 부분: {}'.
          format(matchObj.group(), matchObj.span(), matchObj.group(2)))
```
결과
```
string: 토마토, 	 start/end position=(0, 3), 	 반복 부분: 토
string: aba, 	 start/end position=(8, 11), 	 반복 부분: a
string: xyx, 	 start/end position=(12, 15), 	 반복 부분: x
```

참고로, 이러한 `\1`, `\2`, ... 들은 비 명명 그룹이라고도 한다. 그 이유는, 바로 다음에 설명할 명명 그룹 때문이다.

### 명명 그룹

`\1`, `\2`, ...는 간편하긴 하지만, 그다지 눈에 잘 들어오지는 않는다. 코딩할 때 변수명을 'a', 'b' 같은 것으로 지어 놓으면 남이 알아보기 힘든 것과 갈다.

많은 프로그래밍 언어의 정규표현식은 명명 그룹 기능을 지원한다.  
언어마다 쓰는 방법이 다르지만, 파이썬 기준으로는 `(?P<name>regex)` 형식으로 쓴다.

앞 절의 내용을 이해했으면 어려운 내용이 아니다. 

예시를 하나 보자.  
'2018-07-28 2018.07.28'처럼, 형식만 다른 똑같은 날짜가 있는지를 확인하는 상황을 생각하자.

```python
matchObj = re.match(
    r'(?P<year>\d{4})-(?P<month>\d\d)-(?P<day>\d\d) (?P=year)\.(?P=month)\.(?P=day)',
    '2018-07-28 2018.07.28')

print(matchObj.group())
print(matchObj.groups())
print(matchObj.group(1))
```
결과
```
2018-07-28 2018.07.28
('2018', '07', '28')
2018
```

명명 그룹의 재참조는 `(?P=name)` 형식으로 쓰면 된다. 

사실 명명 그룹과 비 명명 그룹을 섞어 쓸 수는 있다.

```python
matchObj = re.match(
    r'(?P<year>\d{4})-(?P<month>\d\d)-(?P<day>\d\d) (?P=year)\.\2\.\3',
    '2018-07-28 2018.07.28')

print(matchObj.group())
```
결과
```
2018-07-28 2018.07.28
```
하지만 기껏 가독성 높이려고 명명 그룹을 썼는데 저렇게 쓰면 가독성이 더 나빠진다. 지양하도록 하자.

한 가지 주의할 점은 `name` 부분은 `\w`에 일치되는 문자들로만 구성해야 한다. 그렇지 않으면 'invalid group name'이라는 메시지를 볼 수 있을 것이다.


---

### 반복 부분의 캡처

[이 글의 앞부분](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/#%EA%B7%B8%EB%A3%B9%ED%99%94)에서 `12`를 반복시키려고 `(12)+` 정규식을 썼는데 원치 않는 결과가 나온 것을 보았을 것이다.

```python
print(re.findall('A(12)+B', 'A121212B'))
```
결과
```
['12']
```

위의 예시처럼 문자가 한 종류(12)로 정해져 있으면 그냥 전체에다 캡처 그룹을 하나 더 만드는 것으로 해결 가능하지만, 정해진 것이 아닌 문자 집합 같은 것이라면 꽤 어려워진다.

```python
print(re.findall(r'\b(\d\d)+\b', '1, 25, 301, 4000, 55555'))
```
결과
```
['25', '00']
```

위의 예시는 길이가 짝수인 정수를 찾고 싶은 것이다.  
그러나 '4000' 대신 '00'을 찾고 싶은 사람은 별로 없을 것 같다.

이를 캡처 그룹으로 한번에 묶어내는 우아한 방법은 없지만, 다른 괜찮은 해결 방법은 있다.

```python
matchObj_iter = re.finditer(r'\b(\d\d)+\b', '1, 25, 301, 4000, 55555')

for matchObj in matchObj_iter:
    print(matchObj.group())
```
결과
```
25
4000
```

stackoverflow에서 찾은 답변 중에는 패턴을 expand하거나 일치하는 부분만 잘라낸 다음 추가 처리를 하라는 답변이 있었는데, 그런 것보다는 위의 방법이 더 깔끔한 것 같다. 

---

[다음 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)에서는 주석, 치환, 컴파일 등을 살펴보도록 한다.