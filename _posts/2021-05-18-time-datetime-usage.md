---
layout: post
title: Python time, datetime 사용법(Python 시간 다루기)
author: YouWon
categories: References
tags: [time, datetime, usage]
---

이 글에서는 Python 라이브러리인 time과 datetime에 대해 알아본다. 가끔 쓰는데 막상 쓰려면 언제 봐도 헷갈리는 라이브러리 중 하나인 듯 하다.

---

## Import

```python
import time
import datetime
```

---

## time


### 현재 시각: time.time()

1970년 1월 1일 0시 0분 0초 이후 경과한 시간을 초 단위로 반환한다. 2021년 기준 대략 16억의 값을 가진다.

```python
print(time.time())
```
```
# result
1620055042.444191
```

### time 객체: time.localtime(secs)

float 형식의 seconds를 입력으로 주면 지역 시간대에 맞는 time 객체로 변환할 수 있다. 입력을 안 주면 현재 시간으로 계산한다.

반환된 값에서 `.tm_year` 등의 값을 그대로 가져올 수 있다. 연, 월, ..., 초, 요일(`tm_wday`, 일요일=0, 토요일=6), 몇 번째 날짜(`tm_yday`, 1월 1일=0, 1월 2일=1), 일광 절약 시간(`tm_isdst`, 미적용=0, 적용=양수, 정보없음=음수)



```python
print(time.localtime())
print(time.localtime(secs=time.time()))
print(time.localtime(time.time()).tm_year)
```
```
# result
time.struct_time(tm_year=2021, tm_mon=5, tm_mday=4, tm_hour=0, tm_min=19, 
                 tm_sec=51, tm_wday=1, tm_yday=124, tm_isdst=0)
time.struct_time(tm_year=2021, tm_mon=5, tm_mday=4, tm_hour=0, tm_min=19, 
                 tm_sec=51, tm_wday=1, tm_yday=124, tm_isdst=0)
2021
```

### 출력 포맷: time.strftime(format, time object)

datetime에도 비슷한 메서드가 있는데, string format time 정도라고 생각하면 된다.

time 객체가 주어지면 지정한 format으로 출력해준다.

```python
now = time.localtime()
print(time.strftime('%Y%m%d', now))
print(time.strftime('%c', now))
print(time.strftime('%x', now))
print(time.strftime('%X', now))
print(time.strftime('%H%M%S', now))
```
```
# result
20210504
Tue May  4 00:41:07 2021
05/04/21
00:41:07
004107
```

### 출력 포맷 종류

출력 포맷은 다음과 같은 것이 있다. 

Format | Description | Example
-------- | -------- | --------
%c | 날짜, 요일, 시간을 출력, 현재 시간대 기준 | Tue May  4 00:33:26 2021
%x | 날짜를 출력, 현재 시간대 기준 | 05/04/21
%X | 시간을 출력, 현재 시간대 기준 | 00:33:26 
%a | 요일 줄임말 | Sun, Mon, ... Sat
%A | 요일 | Sunday, Monday, ..., Saturday
%w | 요일을 숫자로 표시, 월~일 | 0, 1, ..., 6
%d | 일 | 01, 02, ..., 31
%b | 월 줄임말 | Jan, Feb, ..., Dec
%B | 월 | January, February, …, December
%m | 숫자 월 | 01, 02, ..., 12
%y | 두 자릿수 연도 | 01, 02, ..., 99
%Y | 네 자릿수 연도 | 0001, 0002, ..., 2017, 2018, 9999
%H | 시(24hour) | 00, 01, ..., 23
%I | 시(12hour) | 01, 02, ..., 12
%p | AM, PM | AM, PM
%M | 분 | 00, 01, ..., 59
%S | 초 | 00, 01, ..., 59
%Z | 시간대 | 대한민국 표준시
%j | 1월 1일부터 경과한 일수 | 001, 002, ..., 366
%U | 1년중 주차(월요일이 한 주의 시작) | 00, 01, ..., 53
%W | 1년중 주차(월요일이 한 주의 시작) | 00, 01, ..., 53

---

## datetime

## 현재 시각: datetime.datetime.today()

현재 시각 정보를 포함하는 datetime 객체를 반환한다. 연도부터 마이크로초까지 나온다.  
물론 각 원소는 `.year`와 같이 접근할 수 있다.

```python
print(datetime.datetime.today())
print(datetime.datetime.today().year)
```
```
# result
datetime.datetime(2021, 5, 4, 0, 44, 5, 707495)
2021
```

### 원하는 시각으로 datetime 객체 생성하기

메서드는 다음과 같이 생겼다. 연, 월, 일 등을 지정하여 datetime 객체를 생성할 수 있다.

`datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0)`

```python
print(datetime.datetime(2021, 12, 25))
```
```
# result
2021-12-25 00:00:00
```

### 문자열로 datetime 객체 생성하기

문자열과 그 문자열이 어떻게 생겼는지를 지정하는 format을 같이 주면 datetime 객체가 생성된다.

```python
d = datetime.datetime.strptime('20211225', '%Y%m%d')
print(type(d))
print(d)
```
```
# result
<class 'datetime.datetime'>
2021-12-25 00:00:00
```


(지속 업데이트 예정)

---


```python

```
```
# result

```





---

# References

- [원문](https://docs.python.org/3/library/argparse.html)
- [번역본](https://docs.python.org/ko/3.7/library/argparse.html)