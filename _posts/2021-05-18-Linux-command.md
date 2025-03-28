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

심심하면 입력하는 기본 명령

```bash
ls
```

### 상세히 출력하기 

```bash
# -l 옵션은 파일 권한, 생성/수정한 user, 파일 크기, 수정 시각 등의 자세한 옵션을 표시한다.
# -h 옵션은 -l과 같이 쓰면 파일 크기를 알아보기 쉽게 표시해준다.
# -a 옵션은 숨겨진 파일 혹은 링크까지 같이 표시해준다.
ls -ahl
```

### 출력 개수 제한

```bash
# 상위 4줄만 표시
ls | head -4
```

### 절대 경로 출력
```bash
# 현재 위치에서 절대 경로 출력
pwd

# Use this for dirs (the / after ** is needed in bash to limit it to directories):
ls -d -1 "$PWD/"**/

# this for files and directories directly under the current directory, whose names contain a .:
ls -d -1 "$PWD/"*.*

# this for everything:
ls -d -1 "$PWD/"**/*

# Taken from here http://www.zsh.org/mla/users/2002/msg00033.html
# In bash, ** is recursive if you enable shopt -s globstar.
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

# Symbolic link 관련

hard link과 symbolic link 두 종류가 존재하는데, 둘 다 바로가기 같은 개념이지만 차이가 조금 있다.

- `ln` 뒤에 `-s` 옵션을 주면 symbolic link로 생성된다.
- 원본 파일과 hard link 파일은 디스크에서 완전히 동일한 파일이다.
- symbolic link 파일은 원본과는 다른 파일이고 링크만 가지고 있다.


## link 생성

```bash
# 원본 경로는 파일일 수도, directory일 수도 있다.
ln -s <원본 경로> <link 경로>
ln -s ~/data/Charades/Charades_v1_480/  dataset/ag/videos
# hard link는 -s 옵션을 붙이지 않는다.
ln <원본 경로> <link 경로>
```

## 링크 삭제

```bash
rm -f <link 경로>
rm -f dataset/ag/videos
# 경로 마지막에 /를 붙이면 삭제가 되지 않는다.
```

폴더에 대한 삭제나 파일에 대한 삭제 모두 `rm -f`로 삭제할 수 있다. 원본은 삭제되지 않는다.  
단 `rm -rf`로 삭제하면 원본이 삭제된다.

참고로, symbolic link의 원본을 삭제하면 link는 존재하지만 실제로 파일에 접근할 수는 없다.  
`ls` 등의 명령어로 확인하면 빨간색으로 표시된다.

hard link의 경우에는 원본을 삭제해도 (원본의 복사본) 파일에 접근 가능하다.


---

# 압축 관련

## zip, unzip

### 압축하기

```bash
zip <압축파일명.zip> [-r] <압축할 파일or디렉토리 path 1> [ <압축할 파일or디렉토리 path 2> ...]
zip gorio.zip gorio1.txt gorio2.txt
zip gorio.zip -r gorio_dir/
zip gorio.zip -r ./Downloads/*
# ...
```

### 압축풀기

```bash
unzip <filename>
# 해제된 파일들이 저장될 디렉토리를 지정하고 싶으면 -d 옵션을 사용한다.(d=directory)
unzip <filename> -d <path>
```

### 여러 압축파일 한 번에 풀기

`unzip '<zipfiles>' [-d <dir>]`의 형식이다.

따옴표를 쓰지 않으면 `filename not matched` 에러가 뜬다.

```bash
# 여러 압축 파일을 지정해도 된다.
unzip a.zip b.zip c.zip
# 전부를 지정할 수도 있다. 이때는 따옴표를 꼭 써 줘야 한다.
unzip '*.zip' -d data/
```


## tar, tar.gz

용량을 줄이는 압축 방법은 `tar.gz` 형식으로 압축하는 것이고, `tar` 형식은 단지 하나의 파일로 합치는 archiving 방식이다.

`tar.gz`로 처리하려면 옵션에 `z`를 더 붙이면 된다.

옵션 설명은 아래와 같다.
- -c : 파일을 tar로 묶음
- -x : tar 압축을 풀때 사용함
- -v : 묶거나 파일을 풀때 과정을 화면에 출력
- -f : 파일이름을 지정
- -z : gzip으로 압축하거나 해제
- -C : 경로를 지정
- -p : 파일 권한을 저장

### 압축하기: tar -cvf, tar -zcvf

```bash
# 디렉토리를 tar로 압축하기
tar -cvf [압축파일명.tar] [압축하기위한 디렉토리]
# 파일들을 tar로 압축하기
tar -cvf [압축파일명.tar] [파일1] [파일2] [...]

# 디렉토리를 tar.gz로 압축하기
tar -zcvf [압축파일명.tar.gz] [압축하기위한 디렉토리]
# 파일 tar.gz 압축하기
tar -zcvf [압축파일명.tar.gz] [파일1] [파일2] [...]
```

### 압축풀기: tar -xvf, tar -zxvf

`-c` 옵션을 `-x`로 바꿔주기만 하면 된다.

```bash
# tar 파일 압축풀기
tar -xvf [압축파일명.tar]

# tar.gz 파일 압축풀기
tar -zxvf [압축파일명.tar.gz] 
```


### 여러 압축파일 한 번에 풀기

`tar` 또는 `tar.gz` 확장자를 갖는 파일들을 찾고 위의 압축풀기 명령을 각각에 대해 수행하는 코드이다.

```bash
# tar 형식
find . -name '*.tar' -exec tar -xvf {} \;
# tar.gz 형식
find . -name '*.tar.gz' -exec tar zxvf {} \;
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

### \r command not found

파일에 '\r'이 포함되어 코드가 제대로 작동하지 않는 오류(보통 운영체제의 차이로 인해 발생). 다음 코드를 수행하면 해결된다.

```bash
sed -i 's/\r$//' <filename>
```



(지속 업데이트 예정)

<!--
<center><img src="/public/img/2020-12-16-Latex-usage/0.png" width="80%" alt="KnowIT VQA"></center>
-->