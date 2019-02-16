---
layout: post
title: PyTorch 사용법 - 03. How to Use PyTorch
author: YouWon
categories: PyTorch
tags: [PyTorch]
---

이 글에서는 PyTorch 프로젝트를 만드는 방법에 대해서 알아본다.

사용되는 torch 함수들의 사용법은 [여기](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)에서 확인할 수 있다.

*주의: 이 글은 좀 길다. ㅎ*

---

# Import

```python
# preprocess, set hyperparameter
import argparse
import os

# load data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# train
from torch import nn
from torch.nn import functional as F

# visualization
import matplotlib.pyplot as plt
```

---

# argparse

```
python train.py --epochs 50 --batch-size 64 --save-dir weights
```
Machine Learning을 포함해서, 위와 같은 실행 옵션은 많은 코드에서 볼 수 있었을 것이다. 학습 과정을 포함하여 대부분은 명령창 또는 콘솔에서 `python 파일 옵션들...`으로 실행시키기 때문에, argparse에 대한 이해는 필요하다.

argparse에 대한 내용은 [여기](https://greeksharifa.github.io/references/2019/02/12/argparse-usage/)를 참조하도록 한다.

---

# Load Data

전처리하는 과정을 설명할 수는 없다. 데이터가 어떻게 생겼는지는 직접 봐야 알 수 있다.  
다만 한 번 쓰고 말 것이 아니라면, 데이터가 추가되거나 변경점이 있더라도 전처리 코드의 대대적인 수정이 발생하도록 짜는 것은 본인 손해이다.

## 단순한 방법

```python
data = pd.read_csv('data/02_Linear_Regression_Model_Data.csv')
# Avoid copy data, just refer
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()
```
`pandas`나 `csv` 패키지 등으로 그냥 불러오는 방법이다. 데이터가 복잡하지 않은 형태라면 단순하고 유용하게 쓸 수 있다. 그러나 이 글에서 중요한 부분은 아니다.

## torch.utils.data.DataLoader

참조: [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

Pytorch는 `DataLoader`라고 하는 괜찮은 utility를 제공한다. DataLoader 객체는 학습에 쓰일 데이터를 batch size에 맞춰 잘라서 저장해 놓고, train 함수가 batch 하나를 요구하면 하나씩 꺼내서 준다고 보면 된다.  
실제 DataLoader를 쓸 때는 다음과 같이 쓰기만 하면 된다.
```python
for idx, (data, label) in enumerate(self.data_loader):
    ...
```

DataLoader 안에 데이터가 어떻게 들어있는지 확인하기 위해, MNIST 데이터를 가져와 보자. DataLoader는 `torchvision.datasets` 및 `torchvision.transforms`와 함께 자주 쓰이는데, 각각 Pytorch가 공식적으로 지원하는 [dataset](https://pytorch.org/docs/stable/torchvision/datasets.html), [데이터 transformation 및 augmentation 함수들](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms)(주로 이미지 데이터에 사용)를 포함한다.  
각각의 사용법은 아래 절을 참조한다.

```python
transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
data_loader = DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)

print('type:', type(data_loader))
print(type(type(data_loader)))

first_batch = data_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))

"""
type: <class 'torch.utils.data.dataloader.DataLoader'>
<class 'type'>
name            | type                      | size
first_batch     | <class 'list'>            | 2
first_batch[0]  | <class 'torch.Tensor'>    | torch.Size([64, 1, 28, 28])
first_batch[1]  | <class 'torch.Tensor'>    | torch.Size([64])
"""
```

### Custom DataLoader 만들기

**nn.Module**을 상속하는 Custom Model처럼, Custom DataLoader는 `torch.utils.data.Dataset`를 상속해야 한다. 또한 override해야 하는 것은 다음 두 가지다. `python dunder`를 모른다면 먼저 구글링해보도록 한다.
- `__len__(self)`: dataset의 전체 개수를 알려준다.
- `__getitem__(self, idx)`: parameter로 idx를 넘겨주면 idx번째의 데이터를 반환한다.

위의 두 가지만 기억하면 된다. 전체 데이터 개수와, i번째 데이터를 반환하는 함수만 구현하면 Custom DataLoader가 완성된다.

완전 필수는 아니지만 `__init__()`도 구현하는 것이 좋다.

[Pytorch Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=dataloader)의 예시는 다음과 같다.
```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')
```

## torchvision.datasets

참조: [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html)

Pytorch가 공식적으로 다운로드 및 사용을 지원하는 datasets이다. 2019.02.12 기준 dataset 목록은 다음과 같다.

- MNIST, Fashion-MNIST, KMNIST, EMNIST,  
- COCO, Captions, Detection,  
- LSUN,  
- *ImageFolder*, *DatasetFolder*,  
- Imagenet-12, CIFAR, STL10, SVHN, PhotoTour, SBU, Flickr, VOC, Cityscapes

각각의 dataset마다 필요한 parameter가 조금씩 다르기 때문에, [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)만 간단히 설명하도록 하겠다. 사실 공식 홈페이지를 참조하면 어렵지 않게 사용 가능하다.

![01_MNIST](/public/img/PyTorch/2018-11-10-pytorch-usage-03-Building-Model/01.PNG)

- root: 데이터를 저장할 루트 폴더이다. 보통 `data/`나 `data/mnist/`를 많이 쓰는 것 같지만, 상관없다.
- train: 학습 데이터를 받을지, 테스트 데이터를 받을지를 결정한다.
- download: true로 지정하면 알아서 다운로드해 준다. 이미 다운로드했다면 재실행해도 다시 받지 않는다.
- transform: 지정하면 이미지 데이터에 어떤 변형을 가할지를 transform function의 묶음(Compose)로 전달한다.
- target_transform: 보통 위의 transform까지만 쓰는 것 같다. 쓰고 싶다면 이것도 쓰자.

## torchvision.transforms

참조: [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms)

1. 이미지 변환 함수들을 포함한다. 상대적으로 자주 쓰이는 함수는 다음과 같은 것들이 있다. 더 많은 목록은 홈페이지를 참조하면 된다. 참고로 parameter 중 `transforms`는 변환 함수들의 list 또는 tuple이다.

    - transforms.CenterCrop(size): 이미지의 중앙 부분을 크롭하여 [size, size] 크기로 만든다.
    - transforms.Resize(size, interpolation=2): 이미지를 지정한 크기로 변환한다. 직사각형으로 자를 수 있다.
        - 참고: transforms.Scale는 Resize에 의해 deprecated되었다.
    - transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'): 이미지의 랜덤한 부분을 [size, size] 크기로 잘라낸다. input 이미지가 output 크기보다 작으면 padding을 추가할 수 있다.
    - transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 3/4), interpolation=2): 이미지를 랜덤한 크기 및 비율로 자른다.
        - 참고: transforms.RandomSizedCrop는 RandomResizedCrop에 의해 deprecated되었다.
    - transforms.RandomRotation(degrees, resample=False, expand=False, center=None): 이미지를 랜덤한 각도로 회전시킨다.
    - transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0): brightness, contrast 등을 변화시킨다.

2. 이미지를 torch.Tensor 또는 PILImage로 변환시킬 수 있다. 사용자 정의 변환도 가능하다.
    - transforms.ToPILImage(mode=None): PILImage로 변환시킨다.
    - transforms.ToTensor(): torch.Tensor로 변환시킨다.
    - transforms.Lambda(lambd): 사용자 정의 lambda function을 적용시킨다.

3. torch.Tensor에 적용해야 하는 변환 함수들도 있다.
    - transforms.LinearTransformation(transformation_matrix): tensor로 표현된 이미지에 선형 변환을 시킨다.
    - transforms.Normalize(mean, std, inplace=False): tensor의 데이터 수치(또는 범위)를 정규화한다.

4. brightness나 contrast 등을 바꿀 수도 있다.
    - transforms.functional.adjust_contrast(img, contrast_factor) 등

5. 위의 변환 함수들을 랜덤으로 적용할지 말지 결정할 수도 있다.

    - transforms.RandomChoice(transforms): `transforms` 리스트에 포함된 변환 함수 중 랜덤으로 1개 적용한다.
    - transforms.RandomApply(transforms, p=0.5): `transforms` 리스트에 포함된 변환 함수들을 p의 확률로 적용한다.

6. 위의 모든 변환 함수들을 하나로 조합하는 함수는 다음과 같다. 이 함수를 `dataloader`에 넘기면 이미지 변환 작업이 간단하게 완료된다.

    - transforms.Compose(transforms)
```python
transforms.Compose([
    transforms.CenterCrop(14),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
```

변환 순서는 보통 resize/crop, toTensor, Normalize 순서를 거친다. Normalize는 tensor에만 사용 가능하므로 이 부분은 순서를 지켜야 한다.

---

# Define and Load Model

## Pytorch Model

gradient 계산 방식 등 Pytorch model의 작동 방식은 [Set Loss function(creterion) and Optimizer 절](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/#set-loss-functioncreterion-and-optimizer)을 보면 된다.

Pytorch에서 쓰는 용어는 Module 하나에 가깝지만, 많은 경우 layer나 model 등의 용어도 같이 사용되므로 굳이 구분하여 적어 보았다.

**Layer** : Model 또는 Module을 구성하는 한 개의 층, Convolutional Layer, Linear Layer 등이 있다.  
**Module** : 1개 이상의 Layer가 모여서 구성된 것. Module이 모여 새로운 Module을 만들 수도 있다.  
**Model** : 여러분이 최종적으로 원하는 것. 당연히 한 개의 Module일 수도 있다. 

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

### nn.Module 내장 함수

[nn.Module](https://pytorch.org/docs/stable/nn.html#module)에 내장된 method들은 모델을 추가 구성/설정하거나, train/eval(test) 모드 변경, cpu/gpu 변경, 포함된 module 목록을 얻는 등의 활동에 초점이 맞춰져 있다.

모델을 추가로 구성하려면,
- `add_module(name, module)`: 현재 module에 새로운 module을 추가한다.
- `apply(fn)`: 현재 module의 모든 submodule에 해당 함수(fn)을 적용한다. 주로 model parameter를 초기화할 때 자주 쓴다.

모델이 어떻게 생겼는지 보려면, 
- `children()`, `modules()`: 자식 또는 모델 전체의 모든 module에 대한 iterator를 반환한다.
- `named_buffers(), named_children(), named_modules(), named_parameters()`: 위 함수와 비슷하지만 이름도 같이 반환한다.

모델을 통째로 저장 혹은 불러오려면, 
- `state_dict(destination=None, prefix='', keep_vars=False)`: 모델의 모든 상태(parameter, running averages 등 buffer)를 딕셔너리 형태로 반환한다. 
- `load_state_dict(state_dict, strict=True)`: parameter와 buffer 등 모델의 상태를 현 모델로 복사한다. `strict=True`이면 모든 module의 이름이 *정확히* 같아야 한다.

학습 시에 필요한 함수들을 살펴보면, 
- `cuda(device=None)`: 모든 model parameter를 GPU 버퍼에 옮기는 것으로 GPU를 쓰고 싶다면 이를 활성화해주어야 한다. 
    - GPU를 쓰려면 두 가지에 대해서만 `.cuda()`를 call하면 된다. 그 두 개는 모든 input batch 또는 tensor, 그리고 모델이다.
    - `.cuda()`는 optimizer를 설정하기 전에 실행되어야 한다. 잊어버리지 않으려면 모델을 생성하자마자 쓰는 것이 좋다.
- `eval()`, `train()`: 모델을 train mode 또는 eval(test) mode로 변경한다. Dropout이나 BatchNormalization을 쓰는 모델은 학습시킬 때와 평가할 때 구조/역할이 다르기 때문에 반드시 이를 명시하도록 한다. 
- `parameters(recurse=True)`: module parameter에 대한 iterator를 반환한다. 보통 optimizer에 넘겨줄 때 말고는 쓰지 않는다.
- `zero_grad()`: 모든 model parameter의 gradient를 0으로 설정한다.

사용하는 법은 매우 간단히 나타내었다. Optimizer에 대한 설명은 [여기](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/#set-loss-functioncreterion-and-optimizer)를 참조하면 된다.

```python
import torchvision
from torch import nn

def user_defined_initialize_function(m):
    pass

model = torchvision.models.vgg16(pretrained=True)
# 예시는 예시일 뿐
last_module = nn.Linear(1000, 32, bias=True)
model.add_module('last_module', last_module)
last_module.apply(user_defined_initialize_function)
model.cuda()

# set optimizer. model.parameter를 넘겨준다.
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

# train
model.train()
for idx, (data, label) in dataloader['train']:
    ...

# test
model.eval()
for idx, (data, label) in dataloader['test']:
    ...
```

---

## Pytorch Layer의 종류

참조: [nn.module](https://pytorch.org/docs/stable/nn.html#module)

참고만 하도록 한다. 좀 많다. 쓰고자 하는 것과 이름이 비슷하다 싶으면 홈페이지를 참조해서 쓰면 된다.

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

## Pytorch Activation function의 종류

참조: [Activation functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

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
 
## Containers

참조: [Containers](https://pytorch.org/docs/stable/nn.html#containers)

여러 layer들을 하나로 묶는 데 쓰인다.  
종류는 다음과 같은 것들이 있는데, Module 설계 시 자주 쓰는 것으로 **nn.Sequential**이 있다.
- nn.Module
- nn.Sequential
- nn.ModuleList
- nn.ModuleDict
- nn.ParameterList
- nn.ParameterDict

### nn.Sequential

참조: [nn.Sequential](https://pytorch.org/docs/stable/nn.html#sequential)

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

크게 6가지 정도의 방법이 있다. **nn** 라이브러리를 잘 써서 직접 만들거나, 함수 또는 클래스로 정의, cfg파일 정의 또는 [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)에 미리 정의된 모델을 쓰는 방법이 있다.

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

### cfg(config)를 정의한 후 모델을 생성하는 방법

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

### torchvision.models의 모델을 사용하는 방법

[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)에서는 미리 정의되어 있는 모델들을 사용할 수 있다. 이 모델들은 그 구조뿐 아니라 `pretrained=True` 인자를 넘김으로써 pretrained weights를 가져올 수도 있다. 

2019.02.12 시점에서 사용 가능한 모델 종류는 다음과 같다.
- AlexNet
- VGG-11, VGG-13, VGG-16, VGG-19
- VGG-11, VGG-13, VGG-16, VGG-19 (with batch normalization)
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- SqueezeNet 1.0, SqueezeNet 1.1
- Densenet-121, Densenet-169, Densenet-201, Densenet-161
- Inception v3

모델에 따라 train mode와 eval mode가 정해진 경우가 있으므로 이는 주의해서 사용하도록 한다. 

모든 pretrained model을 쓸 때 이미지 데이터는 [3, W, H] 형식이어야 하고, W, H는 224 이상이어야 한다. 또 아래 코드처럼 정규화된 이미지 데이터로 학습된 것이기 때문에, 이 모델들을 사용할 때에는 데이터셋을 이와 같이 정규화시켜주어야 한다.
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

사용법은 대략 다음과 같다. 사실 이게 거의 끝이고, 나머지는 다른 일반 모델처럼 사용하면 된다.  

```python
import torchvision.models as models

# model load
alexnet = models.alexnet()
vgg16 = models.vgg16()
vgg16_bn = models.vgg16_bn()
resnet18 = models.resnet18()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()

# pretrained model load
resnet18 = models.resnet18(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
...
```

---

# Set Loss function(creterion) and Optimizer

## Pytorch Loss function의 종류

참조: [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)



## Pytorch Optimizer의 종류

참조: [nn.optim](https://pytorch.org/docs/stable/optim.html)

---

# Train Model

https://pytorch.org/docs/stable/cuda.html
https://pytorch.org/docs/stable/tensor_attributes.html#torch-device


---

# Visualize and save results

---

# Q & A

https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
