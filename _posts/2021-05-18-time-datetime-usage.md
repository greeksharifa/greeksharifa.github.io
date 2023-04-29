---
layout: post
title: Python time, datetime 사용법(Python 시간 다루기)
author: YouWon
categories: References
tags: [time, datetime, usage]
---

이 글에서는 Python 라이브러리인 time과 datetime에 대해 알아본다. 가끔 쓰는데 막상 쓰려면 언제 봐도 헷갈리는 라이브러리 중 하나인 듯 하다.

추가로 numpy와 pandas에서 사용하는 `datetime64`에 대해서도 정리한다.

---

## Import

```python
import time
import datetime
```

---


공통적으로, `strptime()`은 str을 datetime 객체로, `strftime()`은 datetime을 str로 바꿔준다.

## time


### 현재 시각: time.time()

1970년 1월 1일 0시 0분 0초 이후 경과한 시간을 초 단위로 반환한다. 2021년 기준 대략 16억의 값을 가진다.

```python
print(time.time())
# 1620055042.444191
```


### time 객체: time.localtime(secs)

float 형식의 seconds를 입력으로 주면 지역 시간대에 맞는 time 객체로 변환할 수 있다. 입력을 안 주면 현재 시간으로 계산한다.

반환된 값에서 `.tm_year` 등의 값을 그대로 가져올 수 있다. 연, 월, ..., 초, 요일(`tm_wday`, 일요일=0, 토요일=6), 몇 번째 날짜(`tm_yday`, 1월 1일=0, 1월 2일=1), 일광 절약 시간(`tm_isdst`, 미적용=0, 적용=양수, 정보없음=음수)



```python
print(time.localtime())
print(time.localtime(secs=time.time()))
print(time.localtime(time.time()).tm_year)

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

`datetime.time`은 시간 기능(시, 분, 초, 마이크로초)을, `datetime.date`는 날짜 기능을(연, 월, 일), `datetime.datetime`은 날짜+시간을, `datetime.timedelta`는 시간의 차이를 구한다.

아래에서 소개하는 함수들은 `datetime.date, datetime.time, datetime.datetime` 모두에서 사용 가능하다.

## 현재 시각: datetime.datetime.today()

현재 시각 정보를 포함하는 datetime 객체를 반환한다. 연도부터 마이크로초까지 나온다.  
물론 각 원소는 `.year`와 같이 접근할 수 있다.

```python
print(datetime.datetime.today())
print(datetime.datetime.today().year)

# result
datetime.datetime(2021, 5, 4, 0, 44, 5, 707495)
2021
```

`datetime.date.today()`로는 현재 날짜를 구할 수 있다.
```python
datetime.date.today() 

# result
datetime.date(2023, 4, 29)
```

### 원하는 시각으로 datetime 객체 생성하기

메서드는 다음과 같이 생겼다. 연, 월, 일 등을 지정하여 datetime 객체를 생성할 수 있다.

`datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0)`

```python
print(datetime.datetime(2021, 12, 25))
print(datetime.date(2021,12,25))

# result
2021-12-25 00:00:00
2021-12-25
```

### 문자열로 datetime 객체 생성하기 - strptime

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


### datetime을 문자열로 바꾸기 - strftime

```python
today = datetime.datetime.today()
print(today.strftime('%Y-%m-%d %H'))

# result
2023-04-29 21
```


---

## numpy datetime64

- 날짜 및 시간과 관련된 ISO 8601 국제표준 방식으로 str type의 값을 전달해 생성하거나
- 유닉스 시각(UTC 1970.1.1 자정)부터 경과 시간을 초 단위로 환산해 나타낼 수 있다.
- np의 datetime64는 마이크로초($10^{-6}$)가 아닌 아토초($10^{-18}$)까지 저장한다.
- 단위 코드는 Y, M, W, M, h, m, s, ms, us, ns, ps, fs이다.

### str로 생성하기

```python
import numpy as np
np.datetime64('2023-04-29')

# result
numpy.datetime64('2023-04-29')
```


### 경과한 시간으로 생성하기

아래는 각각 나노초(ns), 일(Day), 초(second)으로 생성한 결과이다.

```python
np.datetime64(1000, 'ns')
np.datetime64(100000, 'D')
np.datetime64(123456789, 's')

# result
numpy.datetime64('1970-01-01T00:00:00.000001000')
numpy.datetime64('2243-10-17')
numpy.datetime64('1973-11-29T21:33:09')
```

### 일련의 날짜 객체 생성하기

`np.arange`로 범위를 줄 때는 `[시작, 끝)` 범위로 주고, **D**ay, **M**onth, **Y**ear 간격으로 생성할 수 있다.

```python
np.array(['2021-01-01', '2022-01-01', '2023-01-01'], dtype='datetime64')
# np.arange('2010-01-01', '2010-01-03', dtype='datetime64[H]') # TypeError
np.arange('2010-01', '2010-02', dtype='datetime64[D]')
np.arange('2010-01', '2011-09', dtype='datetime64[M]')
np.arange('2010-01', '2012-09', dtype='datetime64[Y]')

# result
array(['2021-01-01', '2022-01-01', '2023-01-01'], dtype='datetime64[D]')

array(['2010-01-01', '2010-01-02', '2010-01-03', '2010-01-04',
       '2010-01-05', '2010-01-06', '2010-01-07', '2010-01-08',
       '2010-01-09', '2010-01-10', '2010-01-11', '2010-01-12',
       '2010-01-13', '2010-01-14', '2010-01-15', '2010-01-16',
       '2010-01-17', '2010-01-18', '2010-01-19', '2010-01-20',
       '2010-01-21', '2010-01-22', '2010-01-23', '2010-01-24',
       '2010-01-25', '2010-01-26', '2010-01-27', '2010-01-28',
       '2010-01-29', '2010-01-30', '2010-01-31'], dtype='datetime64[D]')

array(['2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06',
       '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12',
       '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06',
       '2011-07', '2011-08'], dtype='datetime64[M]')

array(['2010', '2011'], dtype='datetime64[Y]')
```

### datetime64 날짜 간격 계산

둘 중 최소 날짜 단위로 계산해준다.

```python
np.datetime64('2023-04-29') - np.datetime64('2021-05-18')
np.datetime64('2023-04') - np.datetime64('2021')

# result
numpy.timedelta64(711,'D')
numpy.timedelta64(27,'M')
```

---

## Pandas

개념 | Scalar class | Array class | data type | 생성 방법
-------- | -------- | -------- | -------- | --------
Date times | Timestamp | DatetimeIndex | datetime64[ns] 또는 datetime64[ns, tz] | to_datetime 또는 date_range
Time deltas | Timedelta | TimedeltaIndex | timedelta64[ns] | to_timedelta 또는 timedelta_range
Time spans | Period | PeriodIndex | period[freq] | Period 또는 period_range
Date offsets | DateOffset | None | None | DateOffset


### Timestamp

- Datetimes는 python 표준 라이브러리인 datetime이다. 특정 시점 하나를 지칭한다. (Timestamp)
- 두 개 이상의 배열을 다룰 때는 DatetimeIndex를 사용한다.
- `date_range()` 함수(또는 `~Index`)는 연 또는 월 단위로 인자를 주면 `[시작 연/월의 시작일, 끝 연/월의 시작일]`범위로 생성해 준다.


```python
import pandas as pd
pd.Timestamp(2323.345689, unit='D')
pd.Timestamp('2023-04-29')

pd.to_datetime('2023-04-29 21')
pd.to_datetime(['2023-04-29', '2023-04-30'])

pd.date_range('2020-02-01', '2020-02-07')
pd.date_range('2020-01', '2020-09')
# pd.date_range('2020', '2023')

# result
Timestamp('1976-05-12 08:17:47.529600017')
Timestamp('2023-04-29 00:00:00')

Timestamp('2023-04-29 21:00:00')
DatetimeIndex(['2023-04-29', '2023-04-30'], dtype='datetime64[ns]', freq=None)

DatetimeIndex(['2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04',
               '2020-02-05', '2020-02-06', '2020-02-07'],
              dtype='datetime64[ns]', freq='D')
DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
               '2020-01-09', '2020-01-10',
               ...
               '2020-08-23', '2020-08-24', '2020-08-25', '2020-08-26',
               '2020-08-27', '2020-08-28', '2020-08-29', '2020-08-30',
               '2020-08-31', '2020-09-01'],
              dtype='datetime64[ns]', length=245, freq='D')
```

### Time spans

- Time spans는 특정 시점이 아닌 기간을 의미한다. 하루 데이터면 0시 0분 0초부터 23시 59분 59초까지이다. 하나면 Period, 두 개 이상이면 PeriodIndex를 사용한다.

```python
pd.Period('2019-01')
pd.Period('2019-04', freq='D')
pd.period_range('2020-01', '2020-02', freq='D')

# result
Period('2019-01', 'M')
Period('2019-04-01', 'D')
PeriodIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
             '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
             '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12',
             '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16',
             '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20',
             '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',
             '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28',
             '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-01'],
            dtype='period[D]')
```


Date times와 Time spans의 차이는 아래로 확인할 수 있다.

```python
spans = pd.Period('2023-04-29')
stamp = pd.Timestamp('2023-04-29 12:34')
print(spans.start_time < stamp < spans.end_time)

# result
True
```

`date_range(), period_range()` 함수에 freq 인자를 다양하게 넣어줄 수 있다.

```python
pd.date_range('2019-04', '2019-05', freq='B')
pd.date_range('2019-04', '2019-05', freq='W')
pd.date_range('2019-04', '2019-05', freq='W-MON')

# result
DatetimeIndex(['2019-04-01', '2019-04-02', '2019-04-03', '2019-04-04',
               '2019-04-05', '2019-04-08', '2019-04-09', '2019-04-10',
               '2019-04-11', '2019-04-12', '2019-04-15', '2019-04-16',
               '2019-04-17', '2019-04-18', '2019-04-19', '2019-04-22',
               '2019-04-23', '2019-04-24', '2019-04-25', '2019-04-26',
               '2019-04-29', '2019-04-30', '2019-05-01'],
              dtype='datetime64[ns]', freq='B')

DatetimeIndex(['2019-04-07', '2019-04-14', '2019-04-21', '2019-04-28'], dtype='datetime64[ns]', freq='W-SUN')

DatetimeIndex(['2019-04-01', '2019-04-08', '2019-04-15', '2019-04-22',
               '2019-04-29'],
              dtype='datetime64[ns]', freq='W-MON')
```

주기는 아래와 같은 것들이 있다.

Frequency | Description
-------- | -------- 
None | 일반적인 간격, 달력상 1일 간격
B | 영업일(평일), Business Days
W | 주말
M | 각 월의 마지막 날
MS | 각 월의 첫날
BM | 주말이 아닌 평일 중 각 월의 마지막 날
BMS | 주말이 아닌 평일 중 각 월의 첫날
W-MON | 주(월요일)



---



```python


# result

```



---

# References

- [원문](https://docs.python.org/3/library/argparse.html)
- [번역본](https://docs.python.org/ko/3.7/library/argparse.html)