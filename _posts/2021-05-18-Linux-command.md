---
layout: post
title: Linux(Ubuntu) terminal 명령어 정리
author: YouWon
categories: References
tags: [Linux, Ubuntu, usage]
---

이 글에서는 ubuntu terminal에서 사용하는 명령어를 정리한다.

`Ctrl + F`로 원하는 명령어를 검색하면 좋다.

---

# 파일 탐색 관련

## ls 기본 옵션

### 상세히 출력하기 

```bash
ls -l
```

### 출력 개수 제한

```bash
# 상위 4줄만 표시
ls | head -4
```

## 개수 세기

### 현재 위치에 있는 디렉토리 개수

하위 디렉토리는 체크하지 않으며, 현재 위치의 파일은  개수에 포함되지 않는다.

```bash
ls -l | grep ^d | wc -l
```

### 현재 위치에 있는 파일 개수

하위 디렉토리는 체크하지 않으며, 현재 위치의 디렉토리는 개수에 포함되지 않는다.

```bash
ls -l | grep ^- | wc -l
```

### 현재 디렉토리에 포함되는 전체 파일 개수

하위 디렉토리 내의 파일을 포함하며, 디렉토리는 개수에 포함되지 않는다.

```bash
find . -type f | wc -l
```

## Find 사용

find는 파일을 검색하는 명령어이다. `ls`와 `grep`을 조합하는 방법보다 효율적인 명령어라고 한다.

`fine --help` 명령어를 터미널에 입력하면 도움말을 볼 수 있다.


```bash
Usage: find [-H] [-L] [-P] [-Olevel] [-D help|tree|search|stat|rates|opt|exec] [path...] [expression]
```

기본 옵션은 `-print`이므로 그냥 `find`만 입력하면 현재+하위 디렉토리의 모든 파일을 보여준다. `find .`와 같다.

```
.
./cuda_11.1.1_455.32.00_linux.run
./cuda
./cuda/include
./cuda/include/cudnn_cnn_train.h
...
```

### 파일 타입별로 검색

```bash
find . -type -d
# -type -d는 디렉토리 타입만
# -type -f는 파일 타입만

# 결과 예시
.
./cuda
./cuda/include
./cuda/lib64
```

- [참고](https://gracefulprograming.tistory.com/84)

### 필터링 및 대소문자 (미)구분

`-name <표현식>`은 표현식 형식에 맞는 파일만 검색하는 옵션이다.(대소문자 구분)  
`-iname <표현식>`은 대소문자를 구분하지 않는다.

```bash
find . -type f -iname "*.txt"
find . -type d -name "*lib*"
```

### 파일 수정 시간 필터링

- -mmin : 수정시간 기준(분), modify + minute
- -mtime: 수정시간 기준(일)
- -cmin : 생성시간 기준(분), create + minute
- -ctime: 생성시간 기준(일)

```bash
# 수정한 지 60분 이상 된 파일을 검색한다.
find . -mmin +60 -type f
```

### 파일 크기 필터링

```bash
# 100byte 이하 파일만 검색
find . -size -100c -type f
# 128kb 이상 파일만 검색
find . -size +128k -type f
```

### 현재+하위 디렉토리에서 특정 확장자 파일 모두 제거

```bash
# 아래는 확장자가 tsv인 파일을 찾아 제거한다.
find . -type f -name "*.tsv" -exec rm {} \;
```

## 파일 용량

### 전체 디렉토리 총 용량 표시

`du` 명령어는 현재 디렉토리(혹은 지정한 디렉토리) 및 그 하위 디렉토리의 총 용량을 전부 표시한다.

```bash
du # du .와 같다.
du /etc
```

### 특정 디렉토리 총 용량 표시

`-s` 옵션을 붙인다. kbyte 단위로 표시되며, MB, GB 등의 단위로 편하게 보려면 `-h` 옵션을 추가한다.

```bash
du -s /etc
du -sh /etc
du -sh /etc/*
```

### 특정 디렉토리 및 하위 n단계 디렉토리 총 용량 표시

`-d <number>` 옵션을 붙인다. 아래 예시는 `/etc` 및 바로 아래 포함된 디렉토리만의 총 용량을 표시한다.

```bash
du -d 1 /etc
```

### 디렉토리 및 파일의 용량 표시

`du` 명령어에서 `-a` 옵션을 붙이면 하위 파일의 용량까지 같이 표시할 수 있다.

```bash
du -a /etc
```

### 디스크 사용량

`df` 명령어를 사용한다. 위와 마찬가지로 편한 단위로 보려면 `-h` 옵션을 쓴다.

```bash
df
df -h
```


---

# 압축 관련

## zip, unzip

### 여러 파일 압축 한 번에 풀기

`unzip '<zipfiles>' [-d <dir>]`의 형식이다.

따옴표를 쓰지 않으면 `filename not matched` 에러가 뜬다.

```bash
unzip '*.zip' -d data/
```


---


# Process 관련

## Defunct Process

### Defunct Process(좀비 프로세스) 찾기

```bash
ps -ef | grep defunct | grep -v grep
```

### Defunct Process(좀비 프로세스) 개수 출력

```bash
ps -ef | grep defunct | grep -v grep | wc -l
```

### defunct(zombie) process(좀비 프로세스) 죽이기

```bash
ps -ef | grep defunct | awk '{print $3}' | xargs kill -9
```


## Kill process

```bash
sudo kill -9 $PID
```

---


# Nvidia-smi

## 0.5초마다 업데이트

```bash
watch -d -n 0.5 nvidia-smi
```

---

# PATH 확인

## cuda 위치 확인

보통 `/usr/local/cuda`에 있다.

```bash
locate cuda | grep /cuda$
# or
find / -type d -name cuda 2>/dev/null
```

---

# 기타

## 오류 해결 

### `\r command not found`

파일에 '\r'이 포함되어 코드가 제대로 작동하지 않는 오류(보통 운영체제의 차이로 인해 발생). 다음 코드를 수행하면 해결된다.

```bash
sed -i 's/\r$//' <filename>
```



(지속 업데이트 예정)

<!--
<center><img src="/public/img/2020-12-16-Latex-usage/0.png" width="80%" alt="KnowIT VQA"></center>
-->