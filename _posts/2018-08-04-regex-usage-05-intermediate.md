---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 05. 주석, 치환, 분리
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

---

[파이썬 정규표현식(re) 사용법 - 01. Basic](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)  
[파이썬 정규표현식(re) 사용법 - 02. 문자, 경계, flags](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)  
[파이썬 정규표현식(re) 사용법 - 03. OR, 반복](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/22/regex-usage-03-basic/)  
[파이썬 정규표현식(re) 사용법 - 04. 그룹, 캡처](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/)  
**[파이썬 정규표현식(re) 사용법 - 05. 주석, 치환, 분리](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/)**  
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

## 정규표현식 중급: 주석 추가

복잡한 정규식은 써 놓고 나중에 보면 한 줄밖에 안 되는 주제에 다른 스파게비 코드만큼이나 읽기 힘들다. 위에 주석으로 한 줄 이 정규식의 의미를 써놓는 게 부족한 경우가 있을지도 모른다. 그래서 정규식 내부에 주석을 넣는 기능이 있다.

주석은 특별히 어려운 부분이 아니다. 단지 [옵션](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/#%EB%A7%88%EC%B9%A8%ED%91%9C%EB%8A%94-%EA%B0%9C%ED%96%89-%EB%AC%B8%EC%9E%90%EC%99%80-%EC%9D%BC%EC%B9%98-%EC%98%B5%EC%85%98) 중 하나인 `re.VERBOSE` 옵션을 사용하기만 하면 된다. 약자는 `re.X`이다.

위 옵션을 추가하면 정규식 내의 공백들이 무시된다.

```python
print(re.search(r'''
010-# 핸드폰 앞자리 
\d{4}-# 중간자리
\d{4}# 뒷자리''',
                '010-1234-5678',
                re.VERBOSE).group())
```
결과
```
010-1234-5678
```

인라인으로 쓰기 위해서는 모드 변경자 `(?x)`를 쓰면 된다. 

위의 예시처럼 `re.VERBOSE`를 쓸 때는 삼중따옴표를 쓰는 것이 가장 좋다.  
또 일반 공백 문자는 무시되기 때문에, 정규식 내에 공백문자를 넣고 싶으면 `\ `(공백문자를 이스케이프 처리)를 사용하거나 `[ ]`처럼 대괄호 내에 공백문자를 집어넣으면 된다.  
탭 문자는 `\t`로 동일하고, 개행 문자는 `\n`은 무시되기 때문에(예시에선 개행을 했음에도 문자열이 이를 무시하고 일치되었다) `'\\n'` 또는 r`'\n'`으로 하는 것이 라인피드와 일치된다.


---

## 정규표현식 중급: 치환

사실 치환도 어려운 개념은 아니지만, 활용하는 방법은 꽤 많기 때문에 중급으로 분류하였다.

치환은 말 그대로 정규식에 일치하는 부분 문자열을 원하는 문자열로 치환하는 것이다.  
파이썬 문자열은 기본적으로 `replace` 메서드를 갖고 있기 때문에 일반적인 문자열은 그냥 치환이 가능하다.

```python
origin_str = 'Ryan keep a straight face.'
edited_str = origin_str.replace('keep', 'kept')
print(edited_str)
```
결과
```
Ryan kept a straight face.
```

그러나 이 `replace`는 정규식 패턴에 대응하는 문자열을 찾아주지는 못한다. 그래서 **re.sub** 메서드가 필요하다.

### re.sub(pattern, repl, string, count, flags)

![01](\public\img\정규표현식(re)\2018-08-05-regex-usage-05-intermediate\01_sub.PNG)

간단한 사용법은 다음과 같다.

```python
print(re.sub('\d{4}', 'XXXX', '010-1234-5678'))
```
결과
```
010-XXXX-XXXX
```

인자만 보아도 대략 감이 올 것이다. pattern, string, flags는 우리가 지금까지 써 왔던 것들이다.

나머지 두 개도 어려운 부분은 아니다. **re.sub**은 패턴에 일치되는 문자열은 다른 문자열로 바꿔주는 것이므로, repl은 당연히 그 '다른 문자열'에 해당하는 부분이다.

count 인자는, 최대 몆 개까지 치환할 것인가를 지정한다. 만약 일치되는 문자열이 3인데 count=2로 지정되어 있으면 마지막 세 번째 문자열은 치환되지 않는다.  
물론 일치되는 문자열이 count보다 적으면 그냥 전부 다 치환된다.

<script data-ad-client="ca-pub-9951774327887666" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

```python
print(re.sub(pattern='Gorio', repl='Ryan', count=2, \
             string='Gorio, Gorio, Gorio keep a straight face.'))
```
결과
```
Ryan, Ryan, Gorio keep a straight face.
```

인자가 많으므로 이번엔 파이썬의 특징인 인자 명시적 지정을 사용해 보았다.

결과에서 이해되지 않는 부분은 딱히 없을 것이다.  
참고로 **re.sub**은 일치된 위치를 따로 반환해 주지 않는다.


### re.subn(pattern, repl, string, count, flags)

![02](\public\img\정규표현식(re)\2018-08-05-regex-usage-05-intermediate\02_subn.PNG)

**re.subn**은 **re.sub**과 매우 유사하지만, 리턴하는 값이 치환된 문자열과 더불어 치환된 개수의 튜플이라는 것이 다른 점이다.

```python
print(re.subn(pattern='Gorio', repl='Ryan', count=2, \
              string='Gorio, Gorio, Gorio keep a straight face.'))
```
결과
```
('Ryan, Ryan, Gorio keep a straight face.', 2)
```

문자열이 두 개 치환되었으므로 2를 같이 리턴한다.

### 정규식 일치부를 문자열에서 제거

왜 파이썬에서는 제거 메서드를 따로 만들지 않았는지는 잘 모르지만, **re.sub**으로 간단히 구현 가능하다.

```python
print(re.sub('Tube', '', 'Tube Ryan'))
```
결과
```
 Ryan
```

문자열을 제거할 때 공백 문자 하나가 남는 것은 흔히 하는 실수이다. 결과를 보면 'Ryan' 앞에 공백 문자가 하나 있는데, 이를 실제 상황에서 본다면 은근히 신경 쓰일 것이다. 제거할 문자열 앞이나 뒤에 공백 문자를 하나 넣어서 같이 제거하도록 하자.


### 치환 텍스트에 정규식 일치부 삽입

이번에는 문자열 치환 시 그냥 literal text가 아닌 일치된 부분을 재사용하는 방법을 알아보겠다. [재참조부](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/#-%EC%88%AB%EC%9E%90-%EC%95%9E%EC%84%9C-%EC%9D%BC%EC%B9%98%EB%90%9C-%EB%AC%B8%EC%9E%90%EC%97%B4%EC%9D%84-%EB%8B%A4%EC%8B%9C-%EB%B9%84%EA%B5%90)와 약간 비슷한 개념이다.

URL을 markdown link로 변환해 보겠다. Markdown link는 

> `[이름](URL)`

이렇게 구성된다.

파이썬에서 전체 일치부를 치환 텍스트에 삽입하려면 `\g<0>`이라는 토큰을 사용해야 한다.

```python
print(re.sub('https?://\S+',
             '[링크](\g<0>)',
             'http://www.google.com and https://greeksharifa.github.io'))
```
결과
```
[링크](http://www.google.com) and [링크](https://greeksharifa.github.io)
```


### 치환 텍스트에 정규식 부분 일치부 삽입

여러분은 [여기](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/#-%EC%88%AB%EC%9E%90-%EC%95%9E%EC%84%9C-%EC%9D%BC%EC%B9%98%EB%90%9C-%EB%AC%B8%EC%9E%90%EC%97%B4%EC%9D%84-%EB%8B%A4%EC%8B%9C-%EB%B9%84%EA%B5%90)에서 재참조부 일부를 사용하는 방법을 배웠다. 그러면 치환 텍스트에도 이처럼 일부분을 삽입하는 방법이 있을 것이다.

```python
print(re.sub('(\d{4})-(\d{2})-(\d{2})', 
             r'\1.\2.\3',
             '1900-01-01'))
```
결과
```
1900.01.01
```

yyyy-mm-dd 형식이 yyyy.mm.dd 형식으로 바뀌었다.  
`\1`과 같은 문법을 쓸 때에는 **r prefix**를 붙여야 한다는 것을 기억하고 있을 것이다.

만약 위와 같은 상황에서 `\4`를 사용한다면, 아무것도 캡처된 것이 없으므로 에러 메시지를 볼 수 있을 것이다.

#### 명명 그룹을 사용한 경우

조금 위에서 `\g<0>`을 사용했었다. 그러면 명명 그룹의 경우 `\g<name>`을 사용한다는 것을 눈치챌 수 있을 것이다. 
비 명명 그룹은 `\g<1>`, `\g<2>`, ... 을 사용한다.

```python
print(re.sub('(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
             '\g<year>.\g<month>.\g<day>',
             '1900-01-01'))
```
결과
```
1900.01.01
```

물론 명명 그룹과 비 명명 그룹을 혼용하는 것은 여러 번 말했듯이 좋은 생각이 아니다.

---

## 정규표현식 중급: split

**re.sub** 말고도 유용한 함수는 **re.split**이다. 이 메서드는 파이썬 문자열의 기본 메서드인 split과 매우 유사하나, 정규식을 처리할 수 있다.

이 역시 어려운 함수는 아니기 때문에, 예시 하나를 바로 보도록 하겠다.  
html 태그 내에서 태그를 제외한 부분으로 split하는 예제이다.

```python
print(re.split('<[^<>]*>',
               '<html> Wow <head> header </head> <body> Hey </body> </html>'))
```
결과
```
['', ' Wow ', ' header ', ' ', ' Hey ', ' ', '']
```

물론 이렇게만 하면 빈 문자열 등도 많이 나온다. 이는 정규식으로 따로 처리하거나, 다음과 같이 쓰면 된다.
```python
result = re.split('<[^<>]*>',
                  '<html> Wow <head> header </head> <body> Hey </body> </html>')

result = list(map(lambda x: x.strip(), result))
result = list(filter(lambda x: x != '', result))
print(result)
```
결과
```
['Wow', 'header', 'Hey']
```
정규식이 깔끔하긴 하지만, 한 번에 모든 것을 처리하려고 하면 힘들 수 있다. 파이썬 기본 기능도 잘 활용하자.


---

## 정규표현식 초급: re.compile

여기서는 **re.compile** 메서드를 알아볼 것이다.

여러분은 지금까지 `import re`를 사용하여 `re`로부터 직접 메서드를 호출해왔다. **re.match**, **re.sub** 등이 그 예시이다.

이 방식은 한두번 쓰고 말기에는 괜찮지만, 같은 정규식 패턴을 반복문 내에서 반복적으로 사용해야 할 경우 성능상의 부담이 있다.  
이는 정규식은 컴파일이란 과정을 거치기 때문인데, `re` 모듈로부터 직접 갖다 쓰면 매번 컴파일이란 비싼 계산을 해야 하기 때문에 성능이 떨어진다.

**re.compile**은 컴파일을 미리 해 두고 이를 저장할 수 있다. 예시를 보자.

여러분은 지금까지 이렇게 해 왔다.

```python
print(re.search(r'\b\d+\b', 'Ryan 1 Tube 2 345'))
```

이를 한 번 정도 쓰는 것이 아닌, 반복문 내에서 여러 번 쓴다면 이렇게 쓰는 것이 좋다.

```python
with open('ryan.txt', 'r', encoding='utf-8') as f_in:
    reObj = re.compile(r'\b\d+\b')
    for line in f_in:
        matchObj = reObj.search(line)
        print(matchObj.group())
```

미리 컴파일해 두면 성능상의 이점이 있다.

사용법이 조금 달라진 것이 눈에 띌 것이다.

여러분은 지금까지,
1. `re` 모듈로부터 직접 match, search 등 메서드를 써 왔다. 
    - 인자는 기본적으로 1) 정규식 패턴과 2) 찾을 문자열이 있었다.

컴파일을 미리 하는 버전은,
1. re.compile 메서드로부터 reObj를 만든다. 
    - 인자는 기본적으로 1) 정규식 패턴 하나이다.
2. reObj.match 혹은 search 등으로 문자열을 찾는다. 
    - 인자는 정규식 패턴은 이미 저장되어 있으므로 search 메서드에는 1) 찾을 문자열 하나만 주면 된다.

reObj가 무슨 정규식 패턴을 가졌는지 보려면 다음을 수행해 보라.
```python
print(re.compile('\d+'))
```
결과
```
re.compile('\\d+')
```

'\'를 구분하기 위해는 '\' 도 `\`에 의해 이스케이프 처리되어야 하므로 '\'가 두 개 있다는 점을 주의하라.

---

[다음 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/05/regex-usage-06-advanced/)에서는 정규표현식 고급 기술을 다루도록 하겠다.
