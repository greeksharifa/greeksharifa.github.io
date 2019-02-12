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

argparse는 python에 기본으로 내장되어 있다.
```python
import argparse
import os
```
`import os`는 output directory를 만드는 등의 역할을 위해 필요하다.

argparse를 쓰려면 기본적으로 다음 코드가 필요하다.
```python
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
# argument는 원하는 만큼 추가한다.
parser.add_argument('--print-number', type=int, help='an integer for printing repeatably')

args = parser.parse_args()

for i in range(args.print_number):
    print('print number {}'.format(i+1))
```
1. 일단 [ArgumentParser](https://docs.python.org/3/library/argparse.html?highlight=argparse#argparse.ArgumentParser)에 원하는 description을 입력하여 parser 객체를 생성한다. description 외에도 usage, default value 등을 지정할 수 있다.
2. 그리고 `add_argument()` method를 통해 원하는 만큼 인자 종류를 추가한다.
3. `parse_args()` method로 명령창에서 주어진 인자를 파싱한다.
4. `args`라는 이름으로 파싱을 성공했다면 `args.parameter` 형태로 주어진 인자 값을 받아 사용할 수 있다.

실행 결과
```
> python argparseTest.py -h
usage: argparseTest.py [-h] [--print-number PRINT_NUMBER]

Argparse Tutorial

optional arguments:
  -h, --help            show this help message and exit
  --print-number PRINT_NUMBER
                        an integer for printing repeatably

> python argparseTest.py --print-number 5
print number 1
print number 2
print number 3
print number 4
print number 5
```

argparse의 인자는 지정할 수 있는 종류가 상당히 많다.
## --help, -h
`--help` 또는 `-h`: 기본으로 내장되어 있는 옵션이다. 이 인자를 넣고 python으로 실행하면 인자 사용법에 대한 도움말이 출력된다.
```
> python argparseTest.py -h
usage: argparseTest.py [-h] [--print-number PRINT_NUMBER]
...
```

## argument 이름 정의
인자의 이름을 지정할 때 여러 이름을 짓는 것이 가능하다. 지정할 때 두 개를 연속해서 나열한다. 보통 1~2개를 지정하는데, `--help`와 `-h`같이 fullname과 약자를 하나씩 지정하는 편이다. 또 `help=`에서 description을 써줄 수 있다.  
참고로 help 메시지는 % formatting을 지원한다.
```python
parser.add_argument('--print-number', '-p', help='an integer for printing repeatably')
```

## type 지정
기본적으로 `parse_args()`가 주어진 인자들을 파싱할 때는 모든 문자를 숫자 등이 아닌 문자열 취급한다. 따라서 데이터 타입을 지정하고 싶으면 `add_argument()`에서 `type=`을 지정해 주어야 한다. default는 말한 대로 `str`이다.
    - ex) `parser.add_argument('--print-number', '-p', type=int, ...)`
    - type으로 사용 가능한 것은 한 개의 문자열을 받아들여 return 문이 있는 모든 callable 객체이다.
    - Common built-in types과 functions이 사용 가능한데, `str`, `int`, `float`, `bool`과 `open` 등이 있다. `list`와 같은 것은 불가능하다. list처럼 쓰고 싶으면 아래쪽에서 설명할 `action=append`를 이용한다.
    - `argparse.FileType()` 함수도 `type=`에 사용 가능한데, `mode=`, `bufsize=`, `encoding=`, `errors=` parameter를 취하는 함수로서 다양한 파일을 여러 옵션으로 지정할 수 있다. 예를 들어 `argparse.FileType('w')`는 쓰기 가능한 파일을 만든다. 자세한 것은 [여기](https://docs.python.org/3/library/argparse.html?highlight=argparse#type)를 참조한다.
    

## positional / optional 인자
positional 인자와 optional 인자가 있다. 인자의 이름 앞에 `-`가 붙어 있으면 optional, 아니면 positional 인자로서 필수로 지정해야 한다.  
단, positional 인자도 필수로 넣어야 하게끔 할 수 있다. `add_argument()` 함수에 `required=True`를 집어넣으면 된다. 그러나 C언어에서 `#define true false`같은 짓인 만큼 권장되지 않는다.
```python
# argparseTest.py
# ...
parser.add_argument('--foo', '-f') # optional
parser.add_argument('bar')         # positional
args = parser.parse_args()
print('args.foo:', args.foo)
print('args.bar:', args.bar)
```
```
# optional 인자는 지정하지 않아도 되고, 그럴 경우 기본값이 저장된다.
> python argparseTest.py bar_value
args.foo: None
args.bar: bar_value

# positional 인자는 반드시 값을 정해 주어야 한다.
> python argparseTest.py --foo 1
usage: argparseTest.py [-h] [--foo FOO] bar
argparseTest.py: error: the following arguments are required: bar

# optional 인자 뒤에는 반드시 저장할 값을 지정해야 한다. 
# 이는 `action=store`인 optional 인자에 해당한다. 6번 항목에서 설명하겠다.
> python argparseTest.py bar_value --foo
usage: argparseTest.py [-h] [--foo FOO] bar
argparseTest.py: error: argument --foo/-f: expected one argument

# optional 인자는 `--foo 3`또는 `--foo=3` 두 가지 방식으로 지정할 수 있다.
# positional 인자는 그런 거 없다.
> python argparseTest.py --foo=5 bar=bar_value
args.foo: 5
args.bar: bar_value

# positional 인자가 여러 개라면 순서를 반드시 지켜야 한다.
# optional 인자는 값만 잘 지정한다면 어디에 끼워 넣어도 상관없다.
> python argparseTest.py bar_value --foo 7
args.foo: 7
args.bar: bar_value
```
## default 값 지정
 값을 저장할 때 명시적으로 지정하지 않았을 때 들어가는 기본값을 설정할 수 있다. `add_argument()`에서 `default=` 옵션을 지정한다.
    - `argparse.SUPPRESS`를 적을 경우, 인자를 적지 않았을 때 None이 들어가는 것이 아닌 아예 인자 자체가 생성되지 않는다. 또한 `--help`에도 표시되지 않는다.
```python
parser.add_argument('--foo', '-f', type=int, default=5)
```
```
> python argparseTest.py
args.foo: 5

# 그러나 인자를 적어 놓고 값은 안 주면 에러가 난다. 
# 기본적으로 한 개의 값을 추가로 받아야 하기 때문이다.
# 이걸 바꾸려면 6번이나 7번 항목을 참조한다.
> python argparseTest.py --foo
usage: argparseTest.py [-h] [--foo FOO]
argparseTest.py: error: argument --foo/-f: expected one argument
```
## action의 종류 지정
인자를 정의(`add_argument()`에 의해)할 때 action을 지정할 수 있다. 액션에는 다음과 같은 것들이 있으며, 기본값은 `store`이다.
    - `store`: action을 지정하지 않으면 `store`이 된다. 인자 이름 바로 뒤의 값을 해당 인자에 대입(저장)시킨다. 
    - `store_const`: `add_argument()`에서 미리 지정되어 있는 `const=`에 해당하는 값이 저장된다. `const=`는 반드시 써 주어야 한다.
    - `store_true`, `store_false`: 인자를 적으면(값은 주지 않는다) 해당 인자에 `True`나 `False`가 저장된다.
    - `append`: 값을 하나가 아닌 여러 개를 저장하고 싶을 때 쓴다. 인자를 여러 번 호출하면 같이 주는 값이 계속 append된다.
    - `append_const`: append와 비슷하지만 사전에 지정한 const 값이 저장된다.
    - `count`: 인자를 적은 횟수만큼 값이 올라간다. 보통 `verbose` 옵션에 많이 쓴다.
    - `help`: 도움말 메시지를 출력하게 하고 종료하여 코드는 실행시키지 않는다. `--help` 역할을 대신한다.
    - `version`: `version` 인자에 사용가능하다. 버전 정보를 출력하고 종료한다.
```
parser.add_argument('--foo', action='store_const', const=10)
> python argparseTest.py --foo
args.foo: 10

# 인자를 적지 않으면 default 값(None)이 저장된다.
parser.add_argument('--foo', action='store_const', const=10)
> python argparseTest.py
args.foo: None

# default 값을 지정하면 당연히 바뀐다.
parser.add_argument('--foo', action='store_const', const=10, default=5)
> python argparseTest.py
args.foo: 5

# store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다.
# store_false의 경우 반대이다.
parser.add_argument('--foo1', action='store_true')
parser.add_argument('--foo2', action='store_true')
parser.add_argument('--foo3', action='store_false')
parser.add_argument('--foo4', action='store_false')
args = parser.parse_args()

print('args.foo1:', args.foo1)
print('args.foo2:', args.foo2)
print('args.foo3:', args.foo3)
print('args.foo4:', args.foo4)
> python argparseTest.py --foo1 --foo4
args.foo: True
args.foo: False
args.foo: True
args.foo: False

# 참고로 한 번만 호출해도 args.foo는 데이터 타입이 list가 된다. 안 하면 None이다.
parser.add_argument('--foo', action='append')
> python argparseTest.py --foo 1 --foo 123 --foo=xyz
args.foo: ['1', '123', 'xyz']
```

## attribute name: -, _ 구분
인자의 이름에는 `-`와 `_`을 쓸 수 있다. 단, python 기본 문법은 변수명에 `-`를 허용하지 않기 때문에, 인자의 이름에 `-`가 들어갔다면 `args.인자`로 접근하려면 `-`를 `_`로 바꿔 주어야 한다.
    - `--print-number`의 경우 `args.print_number`로 접근할 수 있다.
    - `--print_number`의 경우 `args.print_number`로 동일하다.

## dest: 적용 위치 지정
argument를 지정할 때 store나 action의 저장 또는 적용 위치를 바꿔서 지정할 수 있다. 예를 들어 `--foo`의 `dest=` 옵션을 `--foo-list`로 지정하면, `args.foo_list`에 값이 저장되는 식이다.
```python
parser.add_argument('--foo', action='append', dest='foo_list')
parser.add_argument('--bar', dest='bar_value')
args = parser.parse_args()

print('args.foo_list:', args.foo_list)
print('args.bar_value:', args.bar_value)

try:
    if args.foo is not None:
        print('Hmm?')
except AttributeError as e:
    print('Where are you gone?', e)
```
```
> python argparseTest.py --foo 1 --foo 123 --foo=xyz --bar ABC
args.foo_list: ['1', '123', 'xyz']
args.bar_value: ABC
Where are you gone? 'Namespace' object has no attribute 'foo'
```

## nargs: 값 개수 지정
argparse는 일반적으로 1개의 값을 추가로 받거나, `action=store_true`의 경우는 값을 추가로 받지 않는다. 이를 바꿔 주는 것이 `nargs=` 이다.
    - `N`: N개의 값을 읽어들인다.
    - `?`: 0개 또는 1개의 값을 읽어들인다. 
        - 인자와 값을 모두 적은 경우 해당 값이 저장된다.
        - 인자만 적은 경우 const 값이 저장된다.
        - 아무것도 적지 않았으면 default 값이 저장된다.
    - `*`: 0개 이상의 값을 전부 읽어들인다.
    - `+`: 1개 이상의 값을 전부 읽어들인다. 정규표현식의 것과 매우 비슷하다.
    - `argparse.REMAINDER`: 남은 값을 개수 상관없이 전부 읽어들인다.

예제는 [원문](https://docs.python.org/3/library/argparse.html?highlight=argparse#nargs)이나 [번역본](https://docs.python.org/ko/3.7/library/argparse.html#nargs)을 참조한다.

## choices: 값 범위 지정
인자와 같이 주어지는 값의 범위를 제한하고 싶으면 `choices=` 옵션을 쓰면 된다. `choices=`에 들어갈 수 있는 정의역은 list 등 iterable 객체이다(`in` 조건검사를 할 수 있으면 된다).
```
parser.add_argument('--foo', choices=range(1, 5))
> python argparseTest.py --foo 5
usage: argparseTest.py [-h] [--foo {1,2,3,4}]
argparseTest.py: error: argument --foo: invalid choice: '5' (choose from 1, 2, 3, 4)
```

## metavar: 이름 재지정
 metavar은 `help=`에서 도움말 메시지를 생성할 때 표시되는 이름을 변경할 수 있다(직접 값을 참조하는  `args.foo` 같은 경우 기본 이름 또는 `dest=`에 의해 재지정된 이름을 써야 한다).

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

## [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

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

## [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html)

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

## [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms)

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
