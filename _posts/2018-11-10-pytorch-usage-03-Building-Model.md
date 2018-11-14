---
layout: post
title: PyTorch 사용법 - 03. Building Model
author: YouWon
categories: PyTorch
tags: [PyTorch]
---

이 글에서는 PyTorch 모델을 만드는 방법에 대해서 알아본다.

사용되는 torch 함수들의 사용법은 [여기](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)에서 확인할 수 있다.

---

## 프로젝트 구조

- 03_Building_Model/
    - main.py
    - models.py
    - data/
        - 03_.csv

1. 

---

## Import

```python
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
```

다음 파일을 다운로드하여 data/ 디렉토리에 넣는다.

[02_Linear_Regression_Model_Data.csv](https://github.com/greeksharifa/Tutorial.code/blob/master/Python/PyTorch_Usage/data/02_Linear_Regression_Model_Data.csv)

---

## Load preprocessed Data

### 데이터 준비


---

## Load Model

---

## Set Loss function(creterion) and Optimizer

---

## Train Model


---

## Display output (and graph) and save results

---

전체 코드는 [여기]()에서 살펴볼 수 있다.
