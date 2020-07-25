---
layout: post
title: PyTorch 사용법 - 01. 소개 및 설치
author: YouWon
categories: PyTorch
tags: [PyTorch, TensorFlow, usage]
---

---

[PyTorch 사용법 - 00. References](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)  
**[PyTorch 사용법 - 01. 소개 및 설치](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-01-introduction/)**  
[PyTorch 사용법 - 02. Linear Regression Model](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)  
[PyTorch 사용법 - 03. How to Use PyTorch](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/)  
[PyTorch 사용법 - 04. Recurrent Neural Network Model](https://greeksharifa.github.io/pytorch/2019/06/12/pytorch-usage-04-RNN-Model/)  

---

## 간단한 소개

PyTorch는 유연성과 속도를 모두 갖춘 딥러닝 연구 플랫폼이다. GPU 사용이 가능하기 때문에 속도가 상당히 빠르다.  
또 입문 난이도가 높지 않은 편이고 코드가 간결하다.  
현재는 TensorFlow의 사용자가 많지만, 그 특유의 비직관적인 구조와 난이도 때문에 요즘은 PyTorch의 사용자가 늘어나는 추세이다.

### TensorFlow와의 비교

TensorFlow는 구현 패러다임이 Define and Run인데 비해, PyTorch는 Define by Run이다.

무슨 말인가 하면, TensorFlow는 코드를 직접 돌리는 환경인 세션을 만들고, placeholder를 선언하고 이것으로 계산 그래프를 만들고(Define), 코드를 실행하는 시점에 데이터를 넣어 실행하는(Run) 방식이다.  
이는 계산 그래프를 명확히 보여주면서 실행시점에 데이터만 바꿔줘도 되는 유연함을 장점으로 가지지만, 그 자체로 비직관적이다. 그래서 딥러닝 프레임워크 중 난이도가 가장 높은 편이다.

하지만 PyTorch는 그런 어려움이 없다. 일반적인 Python 코딩이랑 크게 다를 바 없다. 선언과 동시에 데이터를 집어넣고, 세션 같은 것도 필요없이 그냥 돌리면 끝이다. 덕분에 코드가 간결하고, 난이도가 낮다.  
한 가지 단점은 아직 사용자가 적어 구글링했을 때 검색 결과가 많지 않다는 정도?

---

## 설치 방법

[여기](https://pytorch.org/)를 참조한다. 자신에게 맞는 OS, package manager, Python 버전, CUDA 버전 등을 선택하면 그에 맞는 명령어 집합이 나온다. 이를 명령창에 실행하면 설치가 진행된다.  
torchvision을 설치할 경우에 무슨 라이브러리가 없다면서 에러 메시지가 뜨긴 하는데, 사용하는 데 별 문제는 없을 것이다. 만약 자신이 그 부분을 꼭 써야 한다면 에러를 해결하고 넘어가자.

설치를 완료했으면, 명령창에 다음과 같이 입력해보자. Anadonda를 플랫폼으로 사용한다면 conda 설정은 직접 해 주어야 한다.

`python`

```python
# 이 부분은 Python Interpreter에서 입력함.
import torch  
x = torch.randn(3,5)  
print(x)
```

결과가 대략 다음과 같이 나오면 설치가 완료되었다. 숫자가 다른 것은 랜덤이니 신경 쓰지 말자.

![01_run_pytorch.PNG](/public/img/PyTorch/2018-11-02-pytorch-usage-01-Introduction/01_run_pytorch.PNG)

---

## GPU 사용을 위한 설치

GPU 사용을 위한 필수 절차는 다음과 같다. _경우의 수가 너무 많아서 스크린샷은 생략_

1. **호환성 체크**
    1. 컴퓨터에 있는 GPU의 **compute capability** 확인
        - [여기](https://developer.nvidia.com/cuda-gpus)에서 확인
    2. compute capability에 맞는 CUDA SDK 버전 확인
        - [여기](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)에서 확인
    3. Pytorch와 CUDA의 호환성 확인
        - 설치하고자 하는 PyTorch(또는 Tensorflow)가 지원하는 최신 CUDA 버전이 있다. 이보다 상위 버전의 CUDA를 설치하면 PyTorch 코드가 제대로 돌아가지 않는다.
        - [Pytorch 홈페이지](https://pytorch.org/)에서 정해주는 CUDA 버전을 설치하는 쪽이 편하다. 2020.02.13 기준 최신 버전은 10.1이다.
    4. CUDA에 맞는 cuDNN 버전 확인
        - [여기](https://developer.nvidia.com/rdp/cudnn-archive)에서 확인할 수 있다.
2. **CUDA 설치**
    1. [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)에서 원하는 CUDA를 다운받는다. 운영체제와 버전 등을 체크하고, 가능하면 Installer Type은 network가 아닌 local로 받는다. 인터넷으로 설치하면서 받는 것이 아닌 한번에 설치파일을 받는 식이다.
        - 같은 버전인데 update가 추가된 버전이 있다. 보통은 이것까지 추가로 설치해 주는 쪽이 좋다. base installer를 먼저 설치한 뒤에 추가로 설치해 주도록 하자.
    2. 설치 파일로 CUDA를 설치한다. 설치 시에는 다른 프로그램을 설치하거나 제거하는 중이면 실행이 되지 않으니 주의하자.
    3. cuda visual studio integration 관련해서 설치 실패가 뜨는 경우가 많은데, 이 부분이 필요한 코드를 실행할 일이 있다면 이 단계에서 설치해 주는 것이 좋다.
3. **cuDNN 설치**
    1. 우선 [cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)에서 사용하고자 하는 CUDA에 맞는 버전을 찾아 다운받는다.
    2. 윈도우의 경우 압축을 풀어 CUDA 설치 폴더(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1`)에 붙여넣기 하면 된다. 폴더 경로는 설치한 CUDA 버전에 따라 달라진다.
    3. Ubuntu 등의 경우는 홈페이지에 명시된 절차를 따르도록 하자.
4. **환경변수 등록**
    1. 윈도우의 경우 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`을 등록하자.
        - 실행이 잘 안 되는 경우 상위 또는 하위 폴더 몇 개를 추가 등록하면 되는 경우도 있다.
    2. Ubuntu 등의 경우는 다음과 비슷하다. 자신의 OS에 맞춰서 구글링하자.
        ```
        export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        ```
5. **설치 확인**
    1. 다음 코드를 python을 실행하여 입력해보고 `True`가 뜨면 성공한 것이다.
        ```python
        import torch
        torch.cuda.is_available()
        ```


---

## PyTorch Project 구조

프로젝트의 구조는 코딩하는 사람 마음대로이긴 하나, 기본적으로는 다음과 같은 구조를 따른다.

1. Set HyperParameter and Run
2. Load Data
3. Define and Load Models
3-1. Define util functions
4. Set Loss function(creterion) and Optimizer
5. Train Model
6. Visualize and save results

PyTorch는 각 단계에서 다음의 장점을 갖는다.
1. PyTorch가 아닌 Python의 특징인데, 여러분은 많은 Machine Learning 코드를 보면서 `python train.py --epochs 50 --batch-size 16` 등 많은 옵션을 설정할 수 있는 것을 보았을 것이다. Python의 `argparse` 패키지는 이것을 가능하게 해 준다. 
2. 데이터 로드 시 `DataLoader`라는 클래스를 제공한다. `DataLoader`를 통해 데이터를 불러오면, 이 안에서 데이터 처리에 대한 거의 모든 것을 쉽게 수행할 수 있다. 
    - 이를테면 Data Augmentation 같은 것도 전부 제공된다.
    - 여러 종류의 Data Transformation이 지원된다.
3. 일반적인 모델을 불러올 때는 다른 Deep Learning Framework도 대체로 간결하지만, PyTorch는 `torchvision`이라는 패키지에서 따로 pretrain까지 된 모델들을 제공하므로 다른 곳에서 모델을 다운로드할 필요 없이 이를 바로 쓸 수 있다.
2-1. 많은 프로그래머들이 `utils.py`에 유틸리티 함수(예를 들면 [YOLO](https://greeksharifa.github.io/paper_review/2018/10/26/YOLOv2/)에서 [IoU](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)를 구하는 함수)를 따로 빼내어 여러 가지를 한번에 정의한다. 프로젝트에서 부가적인 부분은 따로 관리하는 것이 가독성이 좋다.
4. 이 부분은 다른 Deep Learning Framework와 비슷하다.
5. Tensorflow와는 달리 Session을 설정할 필요가 없다.
6. 이 부분도 역시 비슷하다.

---

[다음 글](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)에서는 Linear Regression Model을 예로 들어서 간단한 프로젝트의 구조를 설명하도록 하겠다.

---

## References

- [Reference](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)

PyTorch에서 자주 사용되는 함수들을 정리한 글이다. 
