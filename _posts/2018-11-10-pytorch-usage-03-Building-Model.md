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

# Import

```python
# basic
import pandas as pd

# pytorch
import torch
from torch import nn
from torch.nn import functional as F

# visualization
import matplotlib.pyplot as plt
```

---

# Load Data

전처리하는 과정을 설명할 수는 없다. 데이터가 어떻게 생겼는지는 직접 봐야 알 수 있다.  
다만 한 번 쓰고 말 것이 아니라면, 데이터가 추가되거나 변경점이 있더라도 전처리 코드의 대대적인 수정이 발생하도록 짜는 것은 금물이다.

## 



---

# Define and Load Model

## Pytorch Model

**Layer** : Model 또는 Module을 구성하는 한 개의 층, Convolutional Layer, Linear Layer 등이 있다.  
**Module** : 1개 이상의 Layer가 모여서 구성된 것. Module이 모여 새로운 Module을 만들 수도 있다.  
**Model** : 여러분이 최종적으로 원하는 것. 한 개의 Module일 수도 있다. 

예를 들어 **nn.Linear**는 한 개의 layer이기도 하며, 이것 하나만으로도 module이나 Model을 구성할 수 있다. 단순 Linear Model이 필요하다면, `model = nn.Linear(1, 1, True)`처럼 사용해도 무방하다.

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

1. `__init__()`에서는 모델에서 사용될 module을 정의한다. module만 정의할 수도, activation function 등을 전부 정의할 수도 있다. 
    - 아래에서 설명하겠지만 module은 **nn.Linear**, **nn.Conv2d** 등을 포함한다.
    - activation function은 **nn.functional.relu**, **nn.functional.sigmoid** 등을 포함한다.
2. `forward()`에서는 모델에서 행해져야 하는 계산을 정의한다(대개 train할 때). 모델에서 forward 계산과 backward gradient 계산이 있는데, 그 중 forward 부분을 정의한다. input을 네트워크에 통과시켜 어떤 output이 나오는지를 정의한다고 보면 된다.
    - `__init__()`에서 정의한 module들을 그대로 갖다 쓴다.
    - 위의 예시에서는 `__init__()`에서 정의한 `self.conv1`과 `self.conv2`를 가져다 썼고, activation은 미리 정의한 것을 쓰지 않고 즉석에서 불러와 사용했다.
    - backward 계산은 PyTorch가 알아서 해 준다. `backward()` 함수를 호출하기만 한다면.

### nn.Module

[여기](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/#import)를 참고한다. 요약하면 **nn.Module**은 모든 PyTorch 모델의 base class이다.

---

## [Pytorch Layer의 종류](https://pytorch.org/docs/stable/nn.html#module)

참고만 하도록 한다. 좀 많다.

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

## [PyTorch Activation function](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)의 종류

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
 
## [Containers](https://pytorch.org/docs/stable/nn.html#containers)

여러 layer들을 하나로 묶는 데 쓰인다.  
종류는 다음과 같은 것들이 있는데, Module 설계 시 자주 쓰는 것으로 **nn.Sequential**이 있다.
- nn.Module
- nn.Sequential
- nn.ModuleList
- nn.ModuleDict
- nn.ParameterList
- nn.ParameterDict

### [nn.Sequential](https://pytorch.org/docs/stable/nn.html#sequential)

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

## 모델 구성 방법

크게 5가지 정도의 방법이 있다.

### 단순한 방법

```python
model = nn.Linear(in_features=1, out_features=1, bias=True)
```

[이전 글](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/#define-and-load-model)에서 썼던 방식이다. *매우* 단순한 모델을 만들 때는 굳이 nn.Module을 상속하는 클래스를 만들 필요 없이 바로 사용 가능하며, 단순하다는 장점이 있다.

### nn.Sequential을 사용하는 방법

```python
sequential_model = nn.Sequential(
    nn.Linear(in_features=1, out_features=20, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=1, bias=True),
)
```

여러 [Layer](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-Building-Model/#pytorch-layer%EC%9D%98-%EC%A2%85%EB%A5%98)와 [Activation function](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-Building-Model/#pytorch-activation-function%EC%9D%98-%EC%A2%85%EB%A5%98)들을 조합하여 하나의 sequential model을 만들 수 있다. 역시 상대적으로 복잡하지 않은 모델 중 모델의 구조가 sequential한 모델에만 사용할 수 있다.

### 함수로 정의하는 방법

```python
def TwoLayerNet(in_features=1, hidden_features=20, out_features=1):
    hidden = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
    activation = nn.ReLU()
    output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
    
    net = nn.Sequential(hidden, activation, output)
    
    return net

model = TwoLayerNet(1, 20, 1)
```

바로 위의 모델과 완전히 동일한 모델이다. 함수로 선언할 경우 변수에 저장해 놓은 layer들을 재사용하거나, skip-connection을 구현할 수도 있다. 하지만 그 정도로 복잡한 모델은 아래 방법을 쓰는 것이 낫다.

### nn.Module을 상속한 클래스를 정의하는 방법

가장 정석이 되는 방법이다. 또한, 복잡한 모델을 구현하는 데 적합하다.

```python
from torch import nn
import torch.nn.functional as F

class TwoLinearLayerNet(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super(TwoLinearLayerNet, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

model = TwoLinearLayerNet(1, 20, 1)
```

역시 동일한 모델을 구현하였다. 여러분의 코딩 스타일에 따라, [ReLU](https://pytorch.org/docs/stable/nn.html#relu) 등의 Activation function을 `forward()`에서 바로 정의해서 쓰거나, `__init__()`에 정의한 후 forward에서 갖다 쓰는 방법을 선택할 수 있다. 후자의 방법은 아래와 같다.  
물론 변수명은 전적으로 여러분의 선택이지만, activation1, relu1 등의 이름을 보통 쓰는 것 같다.

```python
from torch import nn

class TwoLinearLayerNet(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features):
        super(TwoLinearLayerNet, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        
    def forward(self, x):
        x = self.activation1(self.linear1(x))
        return self.linear2(x)

model = TwoLinearLayerNet(1, 20, 1)
```

두 코딩 스타일의 차이점 중 하나는 import하는 것이 다르다(F.relu와 nn.ReLU는 사실 거의 같다). Activation function 부분에서 `torch.nn.functional`은 `torch.nn`의 Module에 거의 포함되는데, `forward()`에서 정의해서 쓰느냐 마느냐에 따라 다르게 선택하면 되는 정도이다.

### cfg(config)를 정의한 후 모델을 build하는 방법

처음 보면 알아보기 까다로운 방법이지만, *매우* 복잡한 모델의 경우 `.cfg` 파일을 따로 만들어 모델의 구조를 정의하는 방법이 존재한다. 많이 쓰이는 방법은 대략 두 가지 정도인 것 같다.

먼저 PyTorch documentation에서 찾을 수 있는 방법이 있다. 예로는 [VGG](https://arxiv.org/abs/1409.1556)를 가져왔다. 코드는 [여기](https://pytorch.org/docs/0.4.0/_modules/torchvision/models/vgg.html)에서 찾을 수 있다.

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(...)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):...

    def _initialize_weights(self):...

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
```

여기서는 `.cfg` 파일이 사용되지는 않았으나, `cfg`라는 변수가 configuration을 담당하고 있다. VGG16 모델을 구성하기 위해 cfg 변수의 해당하는 부분을 읽어 `make_layer` 함수를 통해 모델을 구성한다.

더 복잡한 모델은 아예 따로 `.cfg` 파일을 빼놓는다. [YOLO](https://greeksharifa.github.io/paper_review/2018/10/26/YOLOv2/)의 경우 수백 라인이 넘기도 한다.

`.cfg` 파일은 대략 [다음](https://github.com/marvis/pytorch-yolo2/blob/master/cfg/yolo.cfg)과 같이 생겼다.
```
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=8
...

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2
...
```

이를 파싱하는 [코드](https://github.com/marvis/pytorch-yolo2/blob/master/cfg.py)도 있어야 한다.
```python
def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks
```

이 방법의 경우 대개 depth가 수십~수백에 이르는 아주 거대한 모델을 구성할 때 사용되는 방법이다. 많은 수의 github 코드들이 이런 방식을 사용하고 있는데, 그러면 그 모델은 굉장히 복잡하게 생겼다는 뜻이 된다.

---

# Set Loss function(creterion) and Optimizer

---

# Train Model


---

# Visualize and save results

---
