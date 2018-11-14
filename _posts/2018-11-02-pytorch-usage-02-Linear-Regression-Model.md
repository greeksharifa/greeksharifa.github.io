---
layout: post
title: PyTorch 사용법 - 02. Linear Regression Model
author: YouWon
categories: PyTorch
tags: [PyTorch]
---

이 글에서는 가장 기본 모델인 Linear Regression Model의 Pytorch 프로젝트를 살펴본다.  

사용되는 torch 함수들의 사용법은 [여기](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)에서 확인할 수 있다.

---

## 프로젝트 구조

- 02_Linear_Regression_Model/
    - main.py
    - data/
        - 02_Linear_Regression_Model_Data.csv

1. 일반적으로 데이터는 data/ 디렉토리에 넣는다. 
2. 코드는 git에 두고, data/는 `.gitignore` 파일에 추가하여 데이터는 git에 올리지 않는다. 파일은 다른 서버에 두고 필요할 때 다운로드한다.

물론 이 예제 프로젝트는 너무 간단하여 그냥 data/ 디렉토리 없이 해도 상관없다.

---

## import

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

지금의 경우는 전처리할 필요가 없으므로 그냥 데이터를 불러오기만 하면 된다. 데이터가 어떻게 생겼는지도 확인해 보자.

```python
data = pd.read_csv('data/02_Linear_Regression_Model_Data.csv')
# Avoid copy data, just refer
x = Variable(torch.from_numpy(data['x'].values).unsqueeze(1).float())
y = Variable(torch.from_numpy(data['y'].values).unsqueeze(1).float())

plt.xlim(0, 11);    plt.ylim(0, 8)
plt.title('02_Linear_Regression_Model_Data')
plt.scatter(x, y)

plt.show()
```

![02_Linear_Regression_Model_Data](/public/img/PyTorch/2018-11-02-pytorch-usage-02-Linear-Regression-Model/02_Linear_Regression_Model_Data.png)
 
 
참고: 이 데이터는 다음 코드를 통해 생성되었다.

```python
x = torch.arange(1, 11, dtype=torch.float).unsqueeze(1)
y = x / 2 + 1 + torch.randn(10).unsqueeze(1) / 5

data = torch.cat((x, y), dim=1)
data = pd.DataFrame(data.numpy())

data.to_csv('data/02_Linear_Regression_Model_Data.csv', header=['x', 'y'])
```

---

## Load Model

매우 간단한 모델이므로 코드도 짧다.  
여기서는 편의를 위해 parameter 이름을 명시하도록 한다.

PyTorch에서 Linear 모델은 `torch.nn.Linear` 클래스를 사용한다. 여기서는 x를 단지 y로 mapping하는 일차원 직선($ y = wx + b $)을 찾고 싶은 것이므로, `in_features`와 `out_features`는 모두 1이다.

```python
from torch import nn

model = nn.Linear(in_features=1, out_features=1, bias=True)
print(model)
print(model.weight)
print(model.bias)

"""
Linear(in_features=1, out_features=1, bias=True)
Parameter containing:
tensor([[-0.9360]], requires_grad=True)
Parameter containing:
tensor([0.7960], requires_grad=True)
"""
```

---

## Set Loss function(creterion) and Optimizer

적절한 모델을 선정할 때와 마찬가지로 loss function과 optimizer를 결정하는 것은 학습 속도와 성능을 결정짓는 중요한 부분이다.  
그러나 지금과 같이 간단한 Linear Regression Model에서는 어느 것을 사용해도 학습이 잘 된다. 하지만, 일반적으로 성능이 좋은 `AdamOptimizer`를 사용하도록 하겠다.

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model(x))

"""
tensor([[-0.1399],
        [-1.0759],
        [-2.0119],
        [-2.9478],
        [-3.8838],
        [-4.8197],
        [-5.7557],
        [-6.6917],
        [-7.6276],
        [-8.5636]], grad_fn=<ThAddmmBackward>)
"""
```

참고: 보통 변수명은 criterion 혹은 loss_function 등을 이용한다.

---

## Train Model

Train은 다음과 같이 이루어진다.

1. `prediction`: 모델에 데이터(x)를 집어넣었을 때 예측값(y). 여기서는 $ y = wx + b $의 결과들이다.
2. `loss`: criterion이 MSELoss로 설정되어 있으므로, prediction과 y의 평균제곱오차를 계산한다.
3. `optimizer.zero_grad()`: optimizer의 grad를 0으로 설정한다. PyTorch는 parameter들의 gradient를 계산해줄 때 grad는 계속 누적되도록 되어 있다. 따라서 gradient를 다시 계산할 때에는 0으로 세팅해주어야 한다.
4. `loss.backward()`: gradient 계산을 역전파(backpropagation)한다.
5. `optimizer.step()`: 계산한 gradient를 토대로 parameter를 업데이트한다($ w \leftarrow w - \alpha \Delta w, b \leftarrow b - \alpha \Delta b $)
6. 학습 결과를 중도에 확인하고 싶으면 그래프를 중간에 계속 그려주는 것도 한 방법이다.

```python
for step in range(500):
    prediction = model(x)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        """
        Show your intermediate results
        """
        pass
```

---

## Display output (and graph) and save results

결과를 그래프로 보여주는 부분은 `matplotlib.pyplot`에 대한 내용이므로 여기서는 넘어가도록 하겠다.

```python
def display_results(model, x, y):
    prediction = model(x)
    loss = criterion(prediction, y)
    
    plt.clf()
    plt.xlim(0, 11);    plt.ylim(0, 8)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
    plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.data.item(), model.bias.data.item()))
    plt.show()
    # plt.savefig('02_Linear_Regression_Model_trained.png')

display_results(model, x, y)
```

![02_Linear_Regression_Model_Trained](/public/img/PyTorch/2018-11-02-pytorch-usage-02-Linear-Regression-Model/02_Linear_Regression_Model_trained.png)

모델을 저장하려면 `torch.save` 함수를 이용한다.

```python
torch.save(obj=model, f='02_Linear_Regression_Model.pt')
```

참고: `.pt` 파일로 저장한 PyTorch 모델을 load해서 사용하려면 다음과 같이 한다. 이는 나중에 자세히 다루도록 하겠다.

```python
loaded_model = torch.load(f='02_Linear_Regression_Model.pt')

display_results(loaded_model, x, y)
```

정확히 같은 결과를 볼 수 있을 것이다.

---

전체 코드는 [여기](https://github.com/greeksharifa/Tutorial.code/blob/master/Python/PyTorch_Usage/02_Linear_Regression_Model.py)에서 살펴볼 수 있다.
