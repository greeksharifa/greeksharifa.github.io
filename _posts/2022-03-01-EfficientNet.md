---
layout: post
title: EfficientNet 논문 설명(EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks)
author: YouWon
categories: [Computer Vision]
tags: [MobileNet, Google]
---

---


이 글에서는 Google Inc.에서 발표한 MobileNet V3 논문을 간략하게 정리한다.

---

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

논문 링크: **[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)**

Github: [https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

- 2019년 5월(Arxiv), ICML 2019
- Mingxing Tan, Quoc V. Le

최근에 모델의 크기를 키움으로써 성능을 높이는 방향의 연구가 많이 이루어졌다. 모델을 크게 만드는 것은 3가지 방법이 있는데,

1. network의 depth를 깊게 만드는 것
2. channel width(filter 개수)를 늘리는 것(width가 넓을수록 미세한 정보가 많이 담아짐)
3. input image의 해상도를 올리는 것


<center><img src="/public/img/2022-03-01-EfficientNet/fig02.png" width="100%"></center>


EfficientNet은 이 3가지의 최적의 조합을 AutoML을 통해 찾은 논문이다. 조합을 효율적으로 만들 수 있도록 하는 compound scaling 방법을 제안하며 이를 통해 더 작은 크기의 모델로도 SOTA를 달성한 논문이다.
 
---

## Abstract

- 한정된 자원으로 최대의 효율을 내기 위한 방법으로 model scaling(depth, width, resolution)을 시스템적으로 분석하여 더 나은 성능을 얻고자 한다. 
- 새로운 scaling 방법으로 compount coefficient를 제안한다.
- 이를 바탕으로 찾은 효율적인, 기본이 되는 모델 EfficientNet을 소개한다.
- ImageNet에서 기존 ConvNet보다 8.4배 작으면서 6.1배 빠르고 더 높은 정확도를 갖는다.


<center><img src="/public/img/2022-03-01-EfficientNet/fig01.png" width="70%"></center>


---

## 1. Introduction

- ConvNet의 크기를 키우는 것은 널리 쓰이는 방법이다.
- 그러나 제대로 된 이해를 바탕으로 이루어지지는 않았던 것 같다.
- 그래서 scaling하는 방법을 다시 한 번 생각해보고 연구하는 논문을 제안한다.
    - 그 방법이 *compound scaling method*이다.
- 이 방법을 [MobileNets](https://greeksharifa.github.io/computer%20vision/2022/02/01/MobileNetV1/)와 [ResNet](https://greeksharifa.github.io/computer%20vision/2021/10/24/ImageNet-CNN-models/#resnet)에서 검증해보고자 한다.
    - 그림 1이 결과를 보여주고 있다.


---

## 2. Related Work

**ConvNet Accuracy**

AlexNet 이후 ImageNet competition에서 더 깊어지고 커지면서 정확도가 [높아지는 모델](https://greeksharifa.github.io/computer%20vision/2021/10/24/ImageNet-CNN-models/)들이 여럿 발표되었다. 최근 발표되는 모델들은 ImageNet뿐만 아니라 다른 데이터셋에서도 잘 작동한다. 그러나 정확도는 높아졌지만, 사용하는 자원 역시 크게 늘어났다.

**ConvNet Efficiency**

깊은 ConvNets는 좀좀 over-parameterized된다. 효율을 높이기 위해 모델 압축하는 여러 기법이 제안되었다: SqueezeNets, [MobileNets](https://greeksharifa.github.io/computer%20vision/2022/02/01/MobileNetV1/), ShuffleNets 등.


**Model Scaling**

- ResNet(ResNet-18, ResNet-50, ResNet-200)은 깊이를 달리 하였다.
- MobileNets는 network width를 달리 하였다.
- 또한 이미지 해상도가 높아지면 (찾아낼 정보가 많아서) 정확도를 높아진다. (물론 계산량도 많이 늘어난다.)

많은 연구가 진행되었으나 어떻게 효율적인 조합을 찾는지는 아직까지 정립되지 않았다.

---

## 3. Compound Model Scaling

### 3.1. Problem Formulation

뭔가 괜히 복잡하게 써 놨는데 그냥 ConvNet을 수식화해 정리해놓은 부분이다.  $H, W, C$를 입력 tensor의 크기, $F$를 Conv layer라 하면 ConvNet은

<center><img src="/public/img/2022-03-01-EfficientNet/eq01.png" width="70%"></center>

로 표현 가능하다.

모델이 사용하는 자원이 제한된 상태에서 모델의 정확도를 최대화하는 문제를 풀고자 하는 것이므로, 이 문제는 다음과 같이 정리할 수 있다.

<center><img src="/public/img/2022-03-01-EfficientNet/eq02.png" width="70%"></center>


### 3.2. Scaling Dimensions

- **Depth**: 네트워크의 깊이가 증가할수록 모델의 capacity가 커지고 더 복잡한 feature를 잡아낼 수 있지만, vanishing gradient의 문제로 학습시키기가 더 어려워진다. 이를 해결하기 위해 Batch Norm, Residual Connection 등의 여러 기법들이 등장하였다.
- **Width**: 각 레이어의 width를 키우면 정확도가 높아지지만 계산량이 제곱에 비례하여 증가한다.
- **Resolution**: 입력 이미지의 해상도를 키우면 더 세부적인 feature를 학습할 수 있어 정확도가 높아지지만 마찬가지로 계산량이 제곱에 비례해 증가한다.

<center><img src="/public/img/2022-03-01-EfficientNet/fig03.png" width="100%"></center>

공통적으로, 어느 정도 이상 증가하면 모델의 크기가 커짐에 따라 얻는 정확도 증가량이 매우 적어진다.

### 3.3. Compound Scaling


직관적으로, 더 높은 해상도의 이미지에 대해서는, 

- 네트워크를 깊게 만들어서 더 넓은 영역에 걸쳐 있는 feature(by larger receptive fields)를 더 잘 잡아낼 수 있도록 하는 것이 유리하다. 
- 또, 더 큰 이미지일수록 세부적인 내용도 많이 담고 있어서, 이를 잘 잡아내기 위해서는 layer의 width를 증가시킬 필요가 있다.

즉, 이 depth, width, resolution이라는 세 가지 변수는 밀접하게 연관되어 있으며, 이를 같이 움직이는 것이 도움이 될 것이라고 생각할 수 있다. 

계산량은 깊이에 비례하고, 나머지 두 변수에 대해서 그 제곱에 비례하므로 다음과 같은 비율로 변수들이 움직이게 정할 수 있다.

 
<center><img src="/public/img/2022-03-01-EfficientNet/eq03.png" width="70%"></center>


이 논문에서는 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$로 맞춰서 전체 계산량은 $2^\phi$에 비례하게 잡았다.

---

## 4. EfficientNet Architecture

MnasNet에 기반한 baseline network를 사용한다. 구체적인 모양은 다음과 같다.

<center><img src="/public/img/2022-03-01-EfficientNet/tab01.png" width="70%"></center>

이 baseline network에 기반해서 시작한다.

- STEP 1: $\phi=1$로 고정하고, $\alpha, \beta, \gamma$에 대해서 작게 grid search를 수행한다. 찾은 값은 $\alpha=1.2, \beta=1.1, \gamma=1.15$로 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$이다.
- STEP 2: 이제 $\alpha, \beta, \gamma$를 고정하고 $\phi$를 변화시키면서 전체적인 크기를 키운다.

$\alpha, \beta, \gamma$를 직접 갖고 큰 모델에 실험해서 더 좋은 결과를 얻을 수도 있지만 큰 모델에 대해서는 그 실험에 들어가는 자원이 너무 많다. 그래서 작은 baseline network에 대해서 먼저 좋은 $\alpha, \beta, \gamma$를 찾고(STEP 1) 그 다음에 전체적인 크기를 키운다(STEP 2).

---

## 5. Experiments

### 5.1. Scaling Up MobileNets and ResNets

결과부터 보자.


<center><img src="/public/img/2022-03-01-EfficientNet/tab02.png" width="100%"></center>

Efficient하다.

depth, width, resolution을 어떻게 늘리는지에 대한 비교도 진행해 보았다. 섹션 3의 직관적인 설명과 같은 결과를 보이고 있다.

<center><img src="/public/img/2022-03-01-EfficientNet/tab03.png" width="70%"></center>


### 5.2. ImageNet Results for EfficientNet

추론 latency에 대한 결과를 기록해 놓았다.

<center><img src="/public/img/2022-03-01-EfficientNet/tab04.png" width="70%"></center>

8.4배 적은 연산량으로 더 높은 정확도를 갖는다는 것은 꽤 고무적이다. 결과에 따라서 18배 적거나, 아니면 5.7배, 6.1배 더 빠른 추론 시간을 보여주기도 한다. (표 1, 5 등)

<center><img src="/public/img/2022-03-01-EfficientNet/fig06.png" width="100%"></center>


### 5.3. Transfer Learning Results for EfficientNet

전이학습 dataset에 대한 결과를 기록해 놓았다.

<center><img src="/public/img/2022-03-01-EfficientNet/tab05.png" width="100%"></center>

여기도 비슷하게 몇 배 더 작고 적은 연산량으로 더 좋은 정확도를 갖는다는 내용이다. 

데이터셋에 대한 정보이다.

<center><img src="/public/img/2022-03-01-EfficientNet/tab06.png" width="70%"></center>

baseline 모델에 대해서 어떻게 scaling을 할지를 테스트해 보았다. 표 3과 같은 결과를 보여준다.

<center><img src="/public/img/2022-03-01-EfficientNet/fig08.png" width="70%"></center>

---

## 6. Discussion

어떻게 scaling을 해야 하는지 아래 그림이 단적으로 보여준다. depth, width, resolution은 서로 긴밀히 연관되어 있으며 이들을 같이 키우는 것이 자원을 더 효율적으로 쓰는 방법이다.

<center><img src="/public/img/2022-03-01-EfficientNet/fig08.png" width="70%"></center>

어쨰서 compound scaling method라 다른 방법에 비해 더 좋은지를 나타내는 그림이 아래에 있다. 이미지의 어디에 집중하고 있는지를 보여준다. (근데 attention을 딱히 적용하진 않았다.)

<center><img src="/public/img/2022-03-01-EfficientNet/fig07.png" width="100%"></center>

<center><img src="/public/img/2022-03-01-EfficientNet/tab07.png" width="70%"></center>

---

## 7. Conclusion

한정된 자원을 갖고 있는 상황에서 Depth, Width, Resolution을 어떻게 적절히 조절하여 모델의 크기와 연산량을 줄이면서도 성능은 높일 수 있는지에 대한 연구를 훌륭하게 수행하였다.


---
