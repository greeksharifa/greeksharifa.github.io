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
    - 03_make_data_file.py
    - main.py
    - models.py
    - data/
        - 03_Building_Model_01_Parabolic_Data.csv
    - output/
        - figures/
        - weights/

1. `03_make_data_file.py`는 단지 dataset 파일을 만드는 코드이다. 실제 프로젝트에서는 데이터를 임의로 생성할 수 없으므로 원래는 존재해서는 안 되는 파일이다. 대신 preprocess를 위한 code는 포함될 수 있다.
2. `models.py`는 여러 모델들을 포함한다. 이번 예제의 `main.py`에서는 `models.py`의 모델들을 하나씩 꺼내와서 학습을 시킬 것이지만, 실제 프로젝트에서는 이것과 똑같이 진행되지는 않는다.
3. data/ 디렉토리의 각 파일은 데이터셋 하나씩을 포함하며, `models.py`의 각 모델은 한 개의 파일에 대응된다.
4. output/ 디렉토리는 이름에서 알 수 있듯이 결과를 저장하는 디렉토리이다. output/는 그래프를 저장하는 figures/와 학습된 모델의 weights를 저장하는 weights/ 디렉토리로 구성되어 있다.

---

## Import

```python
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
```

data/ 디렉토리는 다음 폴더의 내용을 복사하면 된다.

[둥]()

---

## Load preprocessed Data

### 데이터 준비

[이전 글](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)에서 본 것과 거의 유사하다. 그래프를 그릴 때는 scale에 주의하도록 한다.



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
