---
layout: post
title: Python glob, os, platform, shutil 사용법(Python os 다루기)
author: YouWon
categories: References
tags: [os, platform, usage]
---

이 글에서는 Python 라이브러리인 os와 platform에 대해 알아본다. 가끔 쓰는데 막상 쓰려면 언제 봐도 헷갈리는 라이브러리 중 하나인 듯 하다.

---

## Import

```python
import glob
import os
import platform
```

--

## glob

특정 경로에 존재하는 파일 or 디렉토리의 리스트를 불러온다.

현재 디렉토리의 구조가 다음과 같다고 하자.

```
current
├── .gitignore
├── readme.md
├── data
|   ├── 1.jpg
|   ├── 2.jpg
|   ├── 3.png
|   ├── 4.png
├── model
|   ├── faster_rcnn_r50_1.pth
|   ├── faster_rcnn_r50_2.pth
```

아주 중요한 문자가 있다.

- `*`: 임의의 문자열과 매치된다.
- `/`: 특정 디렉토리를 경로로 지정할 때 사용한다.
- `**`: 재귀적으로 디렉토리 전체를 탐색할 때 사용한다.
- `?`: 하나의 임의의 문자와 매치된다.
- `[]`: 대괄호 내의 문자들 중 하나와 매치된다.

자세한 설명은 [정규표현식](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/21/regex-usage-02-basic/)을 살펴보면 된다.

사용법은 아래 예시를 보면 쉽게 알 수 있다.

### glob.glob

```python
print(glob.glob('data/*.jpg')
# ['data/1.jpg', 'data/2.jpg']
print(glob.glob('./data/*.jpg')
# ['./data/1.jpg', './data/2.jpg']
```

`glob.glob('./[0-9].*')`처럼 사용할 수도 있다. 파일 이름이 숫자 한개인 파일들과 매치된다.

glob으로 가져오는 경로는 사용자가 지정한 디렉토리 경로 전체도 같이 가져오므로 "파일명"만 갖고 싶을 때에는 `.split()` 함수 등으로 쪼개서 가져와야 한다.  
윈도우에서는 `/` 대신 `\`나 `\\`로 표시될 수 있다. 


### glob.iglob

`glob.glob` 함수와 거의 동일한데 리스트가 아닌 반복가능한(iterable) 형태로만 나온다. 파일이 매우 많다면 이쪽을 쓰는 것이 메모리 등에서 좋다.

```python
for filename in glob.glob('data/*.png'):
    print(filename)
    
# data/3.png
# data/4.png
```

### glob.glob('**/*.jpg', recursive=True)

glob은 기본적으로 특정 디렉토리 하나의 파일만을 탐색한다. 그래서 재귀적으로 디렉토리 내부에 존재하는 모든 디렉토리들을 포함해서 탐색하고 싶으면 `**` 문자와 recursive 옵션을 써야 한다.

```python
for filename in glob.glob('**/*', recursive=True):
    print(filename)

# data
# model
# readme.md
# data/1.jpg
# data/2.jpg
# data/3.png
# data/4.png
# model/faster_rcnn_r50_1.pth
# model/faster_rcnn_r50_2.pth
```

특정 확장자만 모든 디렉토리 내에서 찾고 싶다면 `glob.glob('**/*.png', recursive=True)`와 같이 쓰면 된다.


### glob.escape

모든 특수 문자(`?`, `*`  `[`)를 이스케이프 처리한다.
glob과 비슷하지만 특수문자가 들어있는 파일명을 처리하고 싶을 때 사용한다.

---

## os

### working directory

#### os.getcwd()

현재 working directory 경로를 반환한다.

```python
print(os.getcwd())

# result
'C:\\Users\\gorio\\Documents\\current'
```

#### os.chdir()

working directory를 변경한다.

```python
os.chdir('./data)'
```


### 파일 탐색

#### os.walk(path)

파일이나 디렉토리의 목록을 iterable 형태로 얼을 수 있다.

```python
# 현재 디렉토리 안에 있는 모든 파일 탐색
for curDir, dirs, files in os.walk('.'):
     for f in files:
        print(os.path.join(curDir, f))


# result
./.gitignore
./readme.md
./data/1.jpg
./data/2.jpg
./data/3.png
./data/4.png
./model/faster_rcnn_r50_1.pth
./model/faster_rcnn_r50_2.pth
```

#### os.listdir()

지정한 디렉토리의 하위 파일 및 디렉토리만 반환한다. 디렉토리 내부의 파일은 탐색하지 않는다.

```python
os.listdir('.')

# result
['.gitignore', 'data', 'model', 'readme.md']
```


### os.path

파일이나 디렉토리의 존재를 확인하거나 경로를 얻거나, 경로와 파일명을 결합하는 등의 함수를 포함한다.

```python
os.path.isfile('readme.md')

# result
True
```

함수명 | 설명 
-------- | --------
basename(path) | path에서 앞의 디렉토리 경로는 다 떼고 파일명만 반환한다.
dirname(path) | path에서 파일명은 떼고 디렉토리 경로만 반환한다.
isdir(path) |  주어진 경로가 디렉토리이면 True를 반환한다.
isfile(path) | 주어진 경로가 파일이면 True를 반환한다.


---

## shutil

### shutil.copy(src, dst, *, follow_symlinks=True)

`src` 경로의 파일을 `dst` 경로로 복사한다. `dst`가 디렉토리 경로이면 해당 디렉토리에 파일을 복사하고, 파일명까지 포함되어 있으면 해당 파일명으로 변경하며 복사한다.

```python
shutil.copy('1.jpg', 'data/')
```

참고로, 

- `follow_symlinks`가 False이고 `src`가 심볼릭 링크이면, `dst`는 심볼릭 링크로 만들어진다. 
- `follow_symlinks`가 True이고 `src`가 심볼릭 링크이면, `dst`는 `src`가 참조하는 파일의 사본으로 만들어진다.


### shutil.copy2(src, dst, *, follow_symlinks=True)

파일 메타 데이터를 보존한다는 것을 빼면 `shutil.copy`와 동일하다.

### shutil.copyfile(src, dst, *, follow_symlinks=True)

`dst`는 디렉토리로 지정할 수 없다. 이외에는 사실상 동일하다.

### shutil.move(src, dst, copy_function=copy2)

파일이나 디렉토리를 (재귀적으로) 이동시킨다.


### shutil.copytree

```
shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
```

특정 경로 하위에 있는 모든 파일와 디렉토리를 원하는 위치에 복사할 수 있다.

```python
from distutils.dir_util import copy_tree
copy_tree("./test1", "./test2")
```


### shutil.rmtree(path, ignore_errors=False, onerror=None)

전체 디렉토리 tree를 삭제한다.


실질적으로 잘 안쓰지만 이런 함수들도 있다.

- shutil.copymode: `src`의 권한 bit를 `dst`로 복사한다. linux에서 w,r 등의 옵션을 가리킨다.
- shutil.copystat: 권한 bit, 마지막 액세스 시간, 마지막 수정 시간 및 flag를 복사한다.
- shutil.disk_usage(path): 지정한 경로의 디스크 사용량 통계를 `(total, used, free)` attribute를 갖는 named tuple로 반환한다.

---

## platform

시스템 정보를 확인하는 모듈이다.

기본적으로 값을 판별할 수 없으면 빈 문자열을 반환한다.


### platform.system()

현재 구동되고 있는 OS(Operating System)의 종류를 알려준다.

```python
platform.system()

# result
'Windows'
```

### 기타 플랫폼 관련 함수


**platform.node()**

네트워크 이름 또는 사용자명을 반환한다.

```python
platform.node()

# result
'gorio'
```

**platform.release()**

릴리즈 번호를 반환한다.

```python
platform.release()

# result
'10'
```

**platform.machine()**

기계 유형을 반환한다. 어디선가 본 듯한 amd64, i386 등이 반환된다.

```python
platform.machine()

# result
'AMD64'
```


### python 버전 관련

예시만 보면 바로 사용법을 알 수 있다.

```python
>>> platform.python_implementation()
'CPython'
>>> platform.python_version()
'3.9.10'
>>> platform.python_version_tuple()
('3', '9', '10')

>>> platform.python_build()
('tags/v3.9.10:f2f3f53', 'Jan 17 2022 15:14:21')
>>> platform.python_compiler()
'MSC v.1929 64 bit (AMD64)'
>>> platform.python_branch()
'tags/v3.9.10'
```

---



# References

- [platform](https://docs.python.org/ko/3/library/platform.html)
