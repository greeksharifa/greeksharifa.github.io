---
layout: post
title: PyTorch 사용법 - 02. Tensor 생성 함수
author: YouWon
categories: PyTorch
tags: [PyTorch, TensorFlow]
---

## 참조

[Pytorch.org Tutorials](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)
<https://pytorch.org/docs/stable/tensors.html>

## Tensors

텐서는 원래 수학에서 나온 개념인데, PyTorch를 쓰면서 아주 자세한 내용은 알 필요가 없다.  
딱 한가지만 기억하면 된다. 다차원 배열(혹은 matrix)이다. 물론 PyTorch에서 쓰는 텐서는 기능이 더 많지만, 기본 개념은 다차원 배열이라고만 생각해도 충분하다.  
PyTorch의 모든 연산은 기본적으로 텐서에서 이루어진다. 상수값도 경우에 따라서는 0차원 텐서라고 생각할 수도 있다.

---

## Import

우선 PyTorch를 쓰려면 import를 해야 한다. 다음 코드를 파일의 맨 위에 입력한다.

```python
from __future__ import print_function
import torch
```

---

## 다차원 텐서 생성 함수

### torch.empty(size) - 초기화되지 않은 텐서 생성

| 분류 | 설명
| -------- | --------
| input | 자연수 수열(size)
| output | 차원이 수열의 길이이고, 각 차원의 길이는 수열의 각 원소인 텐서
| Initialization | 초기화되지 않은 쓰레기값
| Description | 초기화하지 않으면 생성 시간이 빠르다.

```python
# Code
x = torch.empty(2, 3) # 5 X 3의 크기를 갖는 2차원 텐서 생성
print(x)

# Results
tensor(1.00000e-31 *
       [[-3.1120,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000]])
```

### torch.zeros(size), torch.ones(size) - 0 또는 1로 초기화된 텐서 생성

| 분류 | 설명
| -------- | --------
| input | 자연수 수열(size)
| output | 차원이 수열의 길이이고, 각 차원의 길이는 수열의 각 원소인 텐서
| Initialization | 0 또는 1로 초기화됨

```python
# Code
x = torch.ones(2)
print(x)
x = torch.zeros(2, 3, 4)
print(x)

# Results
tensor([ 1.,  1.])

tensor([[[ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.]],

        [[ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.]]])
```

### torch.rand(size), torch.randn(size) - 난수로 초기화된 텐서 생성

| 분류 | 설명
| -------- | --------
| input | 자연수 수열(size)
| output | 차원이 수열의 길이이고, 각 차원의 길이는 수열의 각 원소인 텐서
| Initialization | rand: [0, 1] 균등분포, randn: 표준정규분포

```python
# Code
x = torch.rand(2, 3)
print(x)
x = torch.randn(2, 3)
print(x)

# Results
tensor([[ 0.5279,  0.7644,  0.5655],
        [ 0.1436,  0.4151,  0.6750]])

tensor([[ 0.5208,  0.9856,  1.3006],
        [-1.3756,  1.1476, -0.3338]])
```

---

## Line(1차원) 텐서 생성


### torch.randperm(n) - 무작위 permutation 순열인 텐서 생성

| 분류 | 설명
| -------- | --------
| input | 자연수(n)
| output | [0, n-1] permutation과 같은 1차원 텐서
| Description | dtype 지정 가능

```python
# Code
x = torch.randperm(5)
print(x)

# Results
tensor([ 0,  1,  4,  3,  2])
```

### torch.arange(start, end, step) - start 이상 end 미만까지 step만큼 건너뛴 텐서 생성

| 분류 | 설명
| -------- | --------
| input | start는 선택(default=0), end는 필수, step은 선택(default=1)
| output | 1차원 텐서
| Description | 범위: [start, end). python의 range와 비슷함

```python
# Code
x = torch.arange(0, 10, step=2)
print(x)
x = torch.arange(2.5, 6.5, 1)
print(x)

# Results
tensor([ 0.,  2.,  4.,  6.,  8.])

tensor([ 2.5000,  3.5000,  4.5000,  5.5000])
```

### torch.linspace(start, end, steps) - start 이상 end 이하인 steps 수 만큼의 텐서 생성

| 분류 | 설명
| -------- | --------
| input | start는 필수, end는 필수(start보다 작아도 됨), steps는 선택(default=100)
| output | 1차원 텐서
| Description | 범위: [start, end]

```python
# Code
x = torch.linspace(2, 3, 5)
print(x)
x = torch.linspace(100, 1)
print(x)

# Results
tensor([ 2.0000,  2.2500,  2.5000,  2.7500,  3.0000])

tensor([ 100.,   99.,   98.,   97.,   96.,   95.,   94.,   93.,   92.,
          91.,   90.,   89.,   88.,   87.,   86.,   85.,   84.,   83.,
          82.,   81.,   80.,   79.,   78.,   77.,   76.,   75.,   74.,
          73.,   72.,   71.,   70.,   69.,   68.,   67.,   66.,   65.,
          64.,   63.,   62.,   61.,   60.,   59.,   58.,   57.,   56.,
          55.,   54.,   53.,   52.,   51.,   50.,   49.,   48.,   47.,
          46.,   45.,   44.,   43.,   42.,   41.,   40.,   39.,   38.,
          37.,   36.,   35.,   34.,   33.,   32.,   31.,   30.,   29.,
          28.,   27.,   26.,   25.,   24.,   23.,   22.,   21.,   20.,
          19.,   18.,   17.,   16.,   15.,   14.,   13.,   12.,   11.,
          10.,    9.,    8.,    7.,    6.,    5.,    4.,    3.,    2.,
           1.])
```


---

## 직접 생성 함수

### torch.Tensor(x or size) - 직접 생성 혹은 torch.empty(size)와 같음

| 분류 | 설명
| -------- | --------
| input | x(다차원 배열) 또는 size
| output | x와 같은 크기의 텐서 혹은 size 크기의 빈 텐서
| Initialization | 넘겨준 배열 혹은 초기화되지 않은 쓰레기값

```python
# Code
x = torch.Tensor([[1,2], [3,4]])
print(x)
x = torch.Tensor(2,2)
print(x)

# Results
tensor([[ 1.,  2.],
        [ 3.,  4.]])

tensor(1.00000e-31 *
       [[-3.1120,  0.0000],
        [ 0.0000,  0.0000]])
```

### torch.*_like() - 입력받은 텐서와 동일한 성격의 텐서 생성

종류는 다음과 같은 것들이 있다.  
torch.empty_like, torch.full_like, torch.ones_like, torch.rand_like,  
torch.randint_like, torch.randn_like, torch.zeros_like 

| 분류 | 설명
| -------- | --------
| input | 텐서
| output | 같은 크기의 텐서
| Initialization | 넘겨준 텐서와 같음. dtype은 새로 지정 가능

```python
# Code
x = torch.randn(3, dtype=torch.float64)
torch.zeros_like(x)
torch.ones_like(x, dtype=torch.int)


# Results
tensor([ 0.,  0.,  0.], dtype=torch.float64)

tensor([ 1,  1,  1], dtype=torch.int32)
```

### torch.Tensor.new_*() - 입력받은 텐서와 동일한 성격의 텐서 생성

종류는 다음과 같은 것들이 있다.  
new_empty(size, dtype=None, device=None, requires_grad=False) → Tensor  
new_full(size, fill_value, dtype=None, device=None, requires_grad=False) → Tensor  
new_ones(size, dtype=None, device=None, requires_grad=False) → Tensor  
new_tensor(data, dtype=None, device=None, requires_grad=False) → Tensor  
new_zeros(size, dtype=None, device=None, requires_grad=False) → Tensor  

```python
# Code
x = torch.randn(3, dtype=torch.float64)
torch.zeros_like(x)
torch.ones_like(x, dtype=torch.int)


# Results
tensor([ 0.,  0.,  0.], dtype=torch.float64)

tensor([ 1,  1,  1], dtype=torch.int32)
```


## NumPy 변환

### torch.from_numpy(x) - NumPy를 torch.Tensor로 변환

| 분류 | 설명
| -------- | --------
| input | x(NumPy data)
| output | x와 같은 크기의 텐서

```python
# Code
x = np.ndarray(shape=(2,2), dtype=int, buffer=np.array([1,2,3,4]))
tensor_x = torch.from_numpy(x)
print(tensor_x)

# Results
tensor([[ 1,  2],
        [ 3,  4]], dtype=torch.int32)
```

### torch.Tensor.numpy() - torch.Tensor를 NunPy로 변환

| 분류 | 설명
| -------- | --------
| input | None
| output | NumPy 객체
| Description | torch.Tensor 객체에서 직접 사용

```python
# Code
numpy_x = tensor_x.numpy()
print(numpy_x)

# Results
[[1 2]
 [3 4]]
```

## 텐서 생성 옵션

### dtype

모든 텐서는 기본적으로 dtype을 갖고 있다. 데이터 타입(dtype)이란 데이터가 정수형인지, 실수형인지, 얼마나 큰 범위를 가질 수 있는지 등을 나타낸다. 종류는 아래 표와 같다.

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

사용법은 어렵지 않다. 텐서 생성시 `dtype=` 옵션을 붙여 주기만 하면 된다.

```python
# Code
x = torch.linspace(2, 10, steps=5, dtype=torch.float)
y = torch.zeros((2, 3), dtype=torch.long) # 단순 수열 대신 tuple을 size로 전달해도 된다.
z = torch.ones([2, 3], dtype=torch.int8)  # 단순 수열 대신 list를 size로 전달해도 된다.
print(x, y, z, sep='\n')

# Results
tensor([  2.,   4.,   6.,   8.,  10.])
tensor([[ 0,  0,  0],
        [ 0,  0,  0]])
tensor([[ 1,  1,  1],
        [ 1,  1,  1]], dtype=torch.int8)
```

## 그 외

### Tensor.cuda() - GPU연산용으로 변환

| 분류 | 설명
| -------- | --------
| input | None
| output | GPU 텐서 객체로 변환
| Description | torch.cuda.is_available()로 GPU가 이용 가능한지 먼저 확인

```python
# Code
x = torch.Tensor(2, 3)
if torch.cuda.is_available():
    x = x.cuda()
else: 
    print('cuda is not availabile')
```

텐서의 attribute와 method는 [다음](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)에서 알아보도록 한다.
