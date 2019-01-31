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

## Pytorch Model

**Layer** : Model 또는 Module을 구성하는 한 개의 층, Convolutional Layer, Linear Layer 등이 있다.  
**Module** : 1개 이상의 Layer가 모여서 구성된 것. Module이 모여 새로운 Module을 만들 수도 있다.  
**Model** : 여러분이 최종적으로 원하는 것.  

예를 들어 **nn.Linear**는 한 개의 layer이기도 하며, **nn.Linear** 한 개만으로도 module을 구성할 수 있다. 단순 Linear Model이 필요하다면, `model = nn.Linear(1, 1, True)`처럼 사용해도 무방하다.


PyTorch의 모든 모델은 기본적으로 다음 구조를 갖는다. PyTorch 내장 모델뿐 아니라 사용자 정의 모델도 반드시 이 정의를 따라야 한다.

```python
import torch.nn as nn
import torch.nn.functional as F

class Model_Name(nn.Module):
    def __init__(self):
        super(Model_Name, self).__init__()
        self.module1 = ...
        self.module2 = ...
        """
        ex)
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        """

    def forward(self, x):
        x = some_function1(x)
        x = some_function2(x)
        """
        ex)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        """
        return x
```

PyTorch 모델로 쓰기 위해서는 다음 조건을 따라야 한다. 내장된 모델들(**nn.Linear** 등)은 당연히 이 조건들을 만족한다.

1. **torch.nn.Module**을 상속해야 한다.
2. `__init__()`과 `forward()`를 override해야 한다.
    - 사용자 정의 모델의 경우 init과 forward의 인자는 자유롭게 바꿀 수 있다. 이름이 x일 필요도 없으며, 인자의 개수 또한 달라질 수 있다.

이 두 가지 조건은 PyTorch의 기능들을 이용하기 위해 필수적이다. 

따르지 않는다고 해서 에러를 내뱉진 않지만, 다음 규칙들은 따르는 것이 좋다:

1. `__init__()`에서는 모델에서 사용될 module을 정의한다. module만 정의할 수도, activation function 등을 전부 정의할 수도 있다. 대개 module만 정의하는 편이다.
    - 아래에서 설명하겠지만 module은 **nn.Linear**, **nn.Conv2d** 등을 포함한다.
    - activation function은 **nn.functional.relu**, **nn.functional.sigmoid** 등을 포함한다.
2. `forward()`에서는 모델에서 행해져야 하는 계산을 정의한다(대개 train할 때). 모델에서 forward 계산과 backward gradient 계산이 있는데, 그 중 forward 부분을 정의한다. input을 네트워크에 통과시켜 어떤 output이 나오는지를 정의한다고 보면 된다.
    - `__init__()`에서 정의한 module들을 그대로 갖다 쓴다.
    - 위의 예시에서는 `__init__()`에서 정의한 `self.conv1`과 `self.conv2`를 가져다 썼고, activation은 미리 정의한 것을 쓰지 않고 즉석에서 불러와 사용했다.

### nn.Module

[여기](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/#import)를 참고한다. 요약하면 **nn.Module**은 모든 PyTorch 모델의 base class이다.

---

## Pytorch Layer의 종류

1. Linear layers
    - nn.Linear
    - nn.Bilinear
2. Convolution layers
    - nn.Conv1d, nn.Conv2d, nn.Conv3d
    - nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
    - nn.Unfold, nn.Fold
3. Pooling layers
    - nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
    - nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d
    - nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d
    - nn.FractionalMaxPool2d
    - nn.LPPool1d, nn.LPPool2d
    - nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d
    - nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d
4. Padding layers
    - nn.ReflectionPad1d, nn.ReflectionPad2d
    - nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d
    - nn.ZeroPad2d
    - nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d
5. Normalization layers
    - nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
    - nn.GroupNorm
    - nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
    - nn.LayerNorm
    - nn.LocalResponseNorm
6. Recurrent layers
    - nn.RNN, nn.RNNCell
    - nn.LSTM, nn.LSTMCell
    - nn.GRU, nn.GRUCell
7. Dropout layers
    - nn.Dropout, nn.Dropout2d, nn.Dropout3d
    - nn.AlphaDropout
8. Sparse layers
    - nn.Embedding
    - nn.EmbeddingBag

---

## PyTorch Activation function의 종류

1. Non-linear activations
    - nn.ELU, nn.SELU
    - nn.Hardshrink, nn.Hardtanh
    - nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU
    - nn.Sigmoid, nn.LogSigmoid
    - nn.Softplus, nn.Softshrink, nn.Softsign
    - nn.Tanh, nn.Tanhshrink
    - nn.Threshold
2. Non-linear activations (other)
    - nn.Softmin
    - nn.Softmax, nn.Softmax2d, nn.LogSoftmax
    - nn.AdaptiveLogSoftmaxWithLoss

---
 
Module 설계 시 자주 쓰는 것으로 **nn.Sequential**이 있다.

## nn.Sequential

이름에서 알 수 있듯 여러 module들을 연속적으로 연결하는 모델이다. 

```python
# Example of using Sequential
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
"""
이 경우 model(x)는 nn.ReLU(nn.Conv2d(20,64,5)(nn.ReLU(nn.Conv2d(1,20,5)(x))))와 같음.
"""

# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

조금 다르지만 비슷한 역할을 할 수 있는 것으로는 nn.ModuleList, nn.ModuleDict가 있다.

---

## How to Build the Model

크게 네 가지 정도의 방법이 있다.

### Simple method

```python
model = nn.Linear(in_features=1, out_features=1, bias=True)
```

[이전 글](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)에서 썼던 방식이다.



## Import

```python
import pandas as pd

import torch
from torch import nn

from models import *

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
