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

### glob.glob


### glob.iglob


### glob.glob('**/*.jpg', recursive=True)

---

## os


### os.getcwd()

---

## shutil

shutil.copy

shutil.copy2

shutil.copyfile

shutil.copytree

```python
from distutils.dir_util import copy_tree
copy_tree("./test1", "./test2")
```


---

## platform


### platform.system()

현재 구동되고 있는 OS(Operating System)의 종류를 알려준다.

```python
platform.system()
```
```
# result
'Windows'
```



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