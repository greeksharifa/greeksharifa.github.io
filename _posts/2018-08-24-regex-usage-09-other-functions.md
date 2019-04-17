---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 09. 기타 기능
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
[파이썬 정규표현식(re) 사용법 - 08. 예제(단어, 행)](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/06/regex-usage-08-example/)  
**[파이썬 정규표현식(re) 사용법 - 09. 기타 기능](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/24/regex-usage-09-other-functions/)**  

---

이 글에서는 **re** 패키지에 포함된, 지금까지의 글에서 다루지 않았던 함수와 속성 등을 다루도록 하겠다.

본 글에서 정규표현식은 `regex`와 같이, 일반 문자열은 'regex'와 같이 표시하도록 한다.

파이썬 버전은 3.6을 기준으로 하나, 3.x 버전이면 (아마) 동일하게 쓸 수 있다.  
2.7 버전은 한글을 포함한 비 알파벳 문자 처리가 다르다.

---

## 함수

### re.escape(string)

**re.escape** 함수는 문자열을 입력받으면 특수문자들을 이스케이프 처리시켜 준다.

```python
pattern = r'((\d)\2{4,})'
print(re.escape(pattern))
```
결과
```
\(\(\\d\)\\2\{4\,\}\)
```

### re.purge()

사실 설명하지 않은 것이 있는데, **re** 패키지는 **re.compile**로 만들어 놓은 객체들을 cache에 저장해 둔다. 최대 100개까지라고 알려져 있으며, 그 수를 넘어갈 경우 초기화된다고 한다.  
물론 여러분은 아마 한 프로그램 내에서 100개 이상의 다른 정규식을 쓸 일은 없으니 크게 신경 쓸 필요는 없다.

**re.purge** 함수는 이 cache를 초기화하는 함수이다.

```python
re.purge()
```
결과

결과는 아무것도 출력되지 않는다.

---

## 속성

### re.RegexFlag

[이전 글](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/#%EB%A7%88%EC%B9%A8%ED%91%9C%EB%8A%94-%EA%B0%9C%ED%96%89-%EB%AC%B8%EC%9E%90%EC%99%80-%EC%9D%BC%EC%B9%98-%EC%98%B5%EC%85%98)에서 flags를 설명했었는데, 이 flag들이 어떤 것이 있는지 알려주는 객체가 re 안에 내장되어 있다.

```python
for flag in re.RegexFlag:
    print(flag)
```
결과
```
RegexFlag.ASCII
RegexFlag.IGNORECASE
RegexFlag.LOCALE
RegexFlag.UNICODE
RegexFlag.MULTILINE
RegexFlag.DOTALL
RegexFlag.VERBOSE
RegexFlag.TEMPLATE
RegexFlag.DEBUG
```

---

### re.TEMPLATE

아마 쓸 일이 없을 듯하므로 설명은 생략한다. (?)

다만 이런 것이 있다는 것만 소개한다.

---


### re.DEBUG

reObj를 출력하면 컴파일한 정규식을 그대로 출력하던 것을 기억할 것이다. **re.debug**는 일종의 디버깅 모드로서, 정규식의 대략적인 구조를 알 수 있다.  
말 그대로 디버깅용으로 쓰면 될 듯하다.

```python
r = re.compile('\d{3,6}', re.DEBUG)
print(r)
print(r.findall('AS 123123 ars'))
```
결과
```
MAX_REPEAT 3 6
  IN
    CATEGORY CATEGORY_DIGIT
re.compile('\\d{3,6}', re.DEBUG)
['123123']
```

reObj의 사용법은 기본 compile된 객체와 완전히 같다.

---

### re.error

**re.error**는 compile 함수에 전달된 문자열이 유효하지 않은 정규식일 때 발생하는 에러 타입이다. try-except 구문으로 처리하면 된다. 자세한 사용법은 아래 예시로만 보여도 충분할 듯 하다.  

참고로 아래 코드의 phi는 원주율을 소수점 1만 자리까지 저장한 문자열이다.

```python
regex_list = [
    r'((\d)\2{4,})',
    r'((\d)\1{4,})'
]

for regex in regex_list:
    try:
        reObj = re.compile(regex)
        print(list(map(lambda x: x[0], reObj.findall(phi))))
    except re.error:
        print("<Invalid regular expression %s>" % regex)
    finally:
        print('done')
```
결과
```
['999999']
done
<Invalid regular expression ((\d)\1{4,})>
done
```

무엇이 유효하지 않은지는 *연습문제로 남겨두도록 하겠다.*

조금 더 자세한 사용법은 [여기](https://docs.python.org/3/library/re.html#re.error)를 참조한다.

---

이것으로 정규표현식에 대한 글을 마치도록 한다.  
조금 더 복잡한 예제를 정리해 두면 좋겠지만, 그때그때 맞게 쓰는 것이 더 나을 것 같아서 굳이 따로 정리할 필요는 없을 것 같다.
