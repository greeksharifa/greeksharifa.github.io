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

---

# 압축 관련

## zip, unzip

### 여러 파일 압축 한 번에 풀기

`unzip <zipfiles> [-d <dir>]`의 형식이다.

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



# PATH 확인

## cuda 위치 확인

보통 `/usr/local/cuda`에 있다.

```bash
locate cuda | grep /cuda$
# or
find / -type d -name cuda 2>/dev/null
```


(지속 업데이트 예정)

<!--
<center><img src="/public/img/2020-12-16-Latex-usage/0.png" width="80%" alt="KnowIT VQA"></center>
-->