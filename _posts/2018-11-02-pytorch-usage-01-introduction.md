---
layout: post
title: PyTorch 사용법 - 01. 소개 및 설치
author: YouWon
categories: PyTorch
tags: [PyTorch, TensorFlow]
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

## PyTorch Project 구조

프로젝트의 구조는 코딩하는 사람 마음대로이긴 하나, 기본적으로는 다음과 같은 구조를 따른다.

1. Load 'preprocessed Data'
2. Load Model
3. Set Loss function(creterion) and Optimizer
4. Train Model
5. Display output (and graph) and save results

PyTorch는 각 단계에서 다음의 장점을 갖는다.
1. 데이터 로드 시 `DataLoader`라는 클래스를 제공한다. `DataLoader`를 통해 데이터를 불러오면, 이 안에서 데이터 처리에 대한 거의 모든 것을 쉽게 수행할 수 있다. 
    - 이를테면 Data Augmentation 같은 것도 전부 제공된다.
2. 일반적인 모델을 불러올 때는 다른 Deep Learning Framework도 대체로 간결하지만, PyTorch는 `torchvision`이라는 패키지에서 따로 pretrain까지 된 모델들을 제공하므로 다른 곳에서 모델을 다운로드할 필요 없이 이를 바로 쓸 수 있다.
3. 이 부분은 다른 Deep Learning Framework와 비슷하다.
4. Tensorflow와는 달리 Session을 설정할 필요가 없다.
5. 이 부분도 역시 비슷하다.

---

## 게시글

각 프로젝트의 구조를 설명한 게시글들을 소개하도록 하겠다.

- [Linear Regression Model](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)
- NonLinear Regression Model
- Classfication Model

---

## References

- [Reference](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)

PyTorch에서 자주 사용되는 함수들을 정리한 글이다. 
