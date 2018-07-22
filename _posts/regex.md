---
layout: post
title: 파이썬 정규표현식(re) 사용법 - 02. Basic
author: YouWon
categories: [정규표현식(re)]
tags: [Regex, re]
---

<(?i)regex> 대소문자 구분 없음
(?-i) 위의 설정 없애기(토글)
(?s) : 마침표는 개행문자와 일치
^와 $는 개행문자 위치에서 일치 옵션 해제: ^와 $는 행 시작/끝

개행문자를 포함한 모든 문자
. + 마침표는 개행문자와 일치

여기서 flags는 다음과 같은 종류들이 있다.

syntax  |    long syntax    |   meaning
------- | ----------------- | ----------
re.I    |	re.IGNORECASE   |	대소문자 구분 없이 일치
re.M    |	re.MULTILINE    |	^와 $는 개행문자 위치에서 일치
re.S    |	re.DOTALL       |	마침표는 개행문자와 일치
re.U    |	re.UNICODE      |	{\w, \W, \b, \B}는 Unicode dependent
re.L    |	re.LOCALE       |	{\w, \W, \b, \B}는 locale dependent
re.X    |	re.VERBOSE      |	정규표현식에 주석을 달 수 있음





