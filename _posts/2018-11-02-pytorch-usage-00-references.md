---
layout: post
title: PyTorch 사용법 - 00. References
author: YouWon
categories: PyTorch
tags: [PyTorch, usage]
---

---

**[PyTorch 사용법 - 00. References](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)**  
[PyTorch 사용법 - 01. 소개 및 설치](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-01-introduction/)  
[PyTorch 사용법 - 02. Linear Regression Model](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)  
[PyTorch 사용법 - 03. How to Use PyTorch](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/)  

---

본 글의 일부 예제는 [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)에서 가져왔음을 밝힙니다.

---

## [데이터 타입(dtype)](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=dtype#torch.torch.dtype)

모든 텐서는 기본적으로 dtype을 갖고 있다. 데이터 타입(dtype)이란 데이터가 정수형인지, 실수형인지, 얼마나 큰 범위를 가질 수 있는지 등을 나타낸다.  
종류는 아래 표와 같다.

| Data type | dtype | CPU tensor | GPU tensor
| -------- | -------- | -------- | -------- 
| 32-bit floating point | torch.float32 or torch.float | torch.FloatTensor | torch.cuda.FloatTensor
| 64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor
| 16-bit floating point | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor
| 8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor
| 8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor
| 32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor
| 64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor

사용법은 어렵지 않다. 텐서 생성시 `dtype=torch.float`과 같이 parameter를 지정해 주기만 하면 된다.

---

## Tensor Creation

### [torch.arange](https://pytorch.org/docs/stable/torch.html?highlight=arange#torch.arange)

```python
# torch.arange(start=0, end, step=1, out=None, dtype=None, 
#              layout=torch.strided, device=None, requires_grad=False) → Tensor
```

start 이상 end 미만까지 step 간격으로 dtype 타입인 1차원 텐서를 **생성**한다.  

`out` parameter로 결과 텐서를 저장할 변수(텐서)를 지정할 수 있다.  

```python
>>> torch.arange(start=1, end=9, step=2)
tensor([1, 3, 5, 7])
```

### [torch.linspace](https://pytorch.org/docs/stable/torch.html?highlight=linspace#torch.linspace)

```python
# torch.linspace(start, end, steps=100, out=None, dtype=None, 
#                layout=torch.strided, device=None, requires_grad=False) → Tensor
```

start 이상 end 미만까지 총 steps 개수의 dtype 타입인 1차원 텐서를 **생성**한다.  
**torch.arange**에서 step은 간격을, **torch.linspace**에서 steps는 개수를 의미한다.

```python
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(0, 10, steps=10)
tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  
         5.5556,  6.6667,  7.7778,  8.8889, 10.0000])
```

### [torch.from_numpy](https://pytorch.org/docs/stable/torch.html?highlight=from_numpy#torch.from_numpy)

```python
# torch.from_numpy(ndarray) → Tensor
```

numpy array인 ndarray로부터 텐서를 만든다. 이 함수는 데이터를 **복사가 아닌 참조**를 한다.  
`from_numpy`로 만들어진 텐서는 해당 ndarray와 메모리를 공유하며, 어느 한쪽의 데이터를 변경 시 둘 다 변경된다.

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> print(t)
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> print(a)
array([-1,  2,  3])
```


### [torch.randn](https://pytorch.org/docs/stable/torch.html?highlight=randn#torch.randn)

```python
# torch.randn(*sizes, out=None, dtype=None, 
#             layout=torch.strided, device=None, requires_grad=False) → Tensor
```

N(0, 1) 정규분포를 따르는 sizes 크기의 텐서를 **생성**한다.

```python
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```

---

## Tensor Reshape

### [torch.unsqueeze(Tensor.unsqueeze)](https://pytorch.org/docs/stable/torch.html#torch.unsqueeze)

```python
# torch.unsqueeze(input, dim, out=None) → Tensor
```

`dim` parameter 위치에 길이 1짜리 차원을 추가한 텐서를 만든다. 이 함수는 데이터를 **복사가 아닌 참조**를 한다. 원본 텐서와 메모리를 공유하며, 어느 한쪽의 데이터를 변경 시 둘 다 변경된다.

`dim`은 [ -input.dim() - 1, input.dim() + 1] 범위를 갖는다. 음수 dim은 dim + input.dim() + 1과 같다.  
원본 텐서의 size가 (2, 3, 4)라면, unsqueeze(1) 버전은 (2, 1, 3, 4), unsqueeze(2) 버전은 (2, 3, 1, 4)이다.

```python
>>> x = torch.tensor([1, 2, 3])
>>> x
tensor([1, 2, 3])
>>> y = x.unsqueeze(1)
>>> y
tensor([[1],
        [2],
        [3]])
>>> x.size(), y.size()
(torch.Size([3]), torch.Size([3, 1]))

>>> y[0][0] = -1
>>> y
tensor([[-1],
        [ 2],
        [ 3]])
>>> x
tensor([-1,  2,  3])
```

## Tensor Operation

### [torch.cat](https://pytorch.org/docs/stable/torch.html?highlight=cat#torch.cat)

```python
# torch.cat(seq, dim=0, out=None) → Tensor
```

두 텐서를 이어 붙인다(concatenate). 데이터를 **복사**한다.  
concatenate하는 차원을 제외하고는 size가 같거나 empty여야 한다. 즉 shape=(2, 3, 4)인 텐서는 shape=(2, 1, 4)와는 `dim=1`일 때만 concatenate가 가능하다.

```python
>>> x = torch.arange(0, 6).reshape(2, 3)
>>> y = torch.arange(100, 104).reshape(2, 2)
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> y
tensor([[100, 101],
        [102, 103]])
>>> torch.cat((x, y), dim=1)
tensor([[  0,   1,   2, 100, 101],
        [  3,   4,   5, 102, 103]])
```

### [torch.Tensor.backward](https://pytorch.org/docs/stable/autograd.html?highlight=backward#torch.Tensor.backward)


---

## torch.nn

### [torch.nn.Linear](https://pytorch.org/docs/stable/nn.html?highlight=linear#torch.nn.Linear)

```python
# class torch.nn.Linear(in_features, out_features, bias=True)
```

Linear 모델 클래스를 생성한다.  
`in_features` 길이의 데이터를 Linear Transformation을 통해 `out_features` 길이의 데이터로 변환할 수 있다.

```python
>>> from torch import nn
>>> model = nn.Linear(in_features=3, out_features=2, bias=True)

>>> print(model)
Linear(in_features=3, out_features=2, bias=True)
>>> print(model.weight)
Parameter containing:
tensor([[-0.3469,  0.1542, -0.4830],
        [-0.2903,  0.4949,  0.4592]], requires_grad=True)
>>> print(model.bias)
Parameter containing:
tensor([-0.0965,  0.5427], requires_grad=True)
```

### [torch.nn.MSELoss](https://pytorch.org/docs/stable/nn.html?highlight=mseloss#torch.nn.MSELoss)

---

## torch.optim

### [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam)


### [torch.optim.Optimizer.zero_grad](https://pytorch.org/docs/stable/optim.html?highlight=zero_grad#torch.optim.Optimizer.zero_grad)

### [torch.optim.Optimizer.step](https://pytorch.org/docs/stable/optim.html?highlight=optimizer%20step#torch.optim.Optimizer.step)

---

## Save and Load

### [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)





---

<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)

---
