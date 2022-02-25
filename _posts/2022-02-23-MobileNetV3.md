---
layout: post
title: MobileNetV3 논문 설명(Searching for MobileNetV3 리뷰)
author: YouWon
categories: [Computer Vision]
tags: [MobileNet, Google]
---

---


이 글에서는 Google Inc.에서 발표한 MobileNet V3 논문을 간략하게 정리한다.

---

# Searching for MobileNetV3

논문 링크: **[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)**

Github: [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md)

- 2019년 5월(Arxiv)
- Google Inc.
- Andrew Howard, Mark Sandler, Grace Chu et al.

모바일 기기에서 동작하는 것을 목표로 한, 성능을 별로 희생하지 않으면서도 모델을 크게 경량화하였다. [MobileNet V1](https://greeksharifa.github.io/computer%20vision/2022/02/01/MobileNetV1/) 및 [MobileNet V2](https://greeksharifa.github.io/computer%20vision/2022/02/10/MobileNetV2/)의 후속 논문이다.
 
---

## Abstract

모바일 환경에 맞춘 새로운 세대의 MobileNets(V3)을 제안한다. 

- NetAdapt 알고리즘 및
- 그 도움을 받아 Hardware-aware NAS(Network Architecture Search)와
- novel architecture advances에 기반해 있다.

이 논문은 어떻게 자동화된 탐색 알고리즘(automated search algorithm)과 네트워크 디자인을 어떻게 사용하는지 보여준다. 2가지 크기의 모델을 제안하는데,

- MobileNetV3-Large는 MobileNetV2에 비해 latency는 20% 줄이면서도 정확도는 3.2% 더 높다.
    - MobileNetV3-Large LR-ASPP는 Cityspaces segmentation에서  MobileNetV2 R-ASPP보다 비슷한 정확도를 가지면서 34% 더 빠르다.
- MobileNetV3-Small는 MobileNetV2에 비해 정확도는 비슷하면서 25% 더 빠르다.

각각 자원을 더 많이 쓰냐 적게 쓰냐의 차이가 있다. 

또 Semantic Segmentation task에서는 Lite Reduced Atrous Spatial Pyramid pooling(LR-ASPP)라는 새로운 효율적인 decoder를 제안한다.


---

## 1. Introduction

모바일 app이라면 어디나 존재하는 효율적인 신경망은 완전히 새로운 on-device 경험을 가능하게 한다. 이러한 상황에서 모바일과 같이 작은 device에 들어가는 네트워크는 서버에 전송되는 데이터를 줄여 더 빠른 속도, 더 적은 전력소모 등의 효과를 가져올 수 있다. 이렇게 사용되는 네트워크는 크기가 작으면서도 성능은 충분히 챙길 수 있는 효율성을 가져야만 하며 이 논문에서는 그러한 모델을 개발하는 것을 다룬다.

이를 위해

1. 상호보완적 탐색 기술
2. 모바일 setting에서 비선형성의 새롭고 효율적인 버전
3. 효율적인 네트워크 디자인
4. 효율적인 segmentation decoder

를 소개한다.



<center><img src="/public/img/2022-02-23-MobileNetV3/fig01.png" width="70%"></center>


<center><img src="/public/img/2022-02-23-MobileNetV3/fig02.png" width="70%"></center>

---

## 2. Related Work

정확도와 효율성(latency(반응 속도) or 실행 시간 등) 간 trade-off에 관한 연구가 최근 많이 진행되었다.

- SqueezeNet은 1x1 conv를 활용하여 parameter 수를 줄였다.
- [MobileNet V1](https://greeksharifa.github.io/computer%20vision/2022/02/01/MobileNetV1/) 및 [MobileNet V2](https://greeksharifa.github.io/computer%20vision/2022/02/10/MobileNetV2/)은 Depthwise Separable Convolution으로 연산 수(MAdds)를 크게 줄였다.
- ShuffleNet은 group conv와 shuffle 연산으로 MAdds를 더 줄였다.
- CondenseNet은 학습 단계의 group conv를 학습하고 추후를 위해 유용한 dense connection만을 남기는 방식을 채택했다.
- ShiftNet은 point-wise conv에 interleave한 shift 연산을 제안하여 값비싼 spatial conv 연산을 대체했다.


그리고 architecture 디자인 과정을 자동화하기 위한 연구도 많이 이루어졌다. RL 등이 대표적이지만 전체 상태 공간을 탐색하는 것은 매우 어렵다. 탐색 연산량을 줄이기 위해 Proxylessnas, DARTS, Fbnet 등의 논문에 발표되었다.

양자화(Quantization)은 Reduced Precision Arithmetic을 통해 네트워크의 효율성을 높이고자 하였다.

---

## 3. Efficient Mobile Building Blocks

- MobileNetV1에서는 Depthwise Separable Convolution을 제안하여 모델 크기를 크게 줄였다.
- MobileNetV2에서는 linear bottleneck \& inverted residual 구조를 제안하여 더 효율적인 구조를 제안했다.
- MnasNet은 bottleneck 구조에 squeeze and excitation 모듈에 기반한 light weight attention module을 제안했다.

MobileNet V3에서는 이러한 layer들의 조합을 building block으로 사용하여 효율적인 구조를 제안한다.


<center><img src="/public/img/2022-02-23-MobileNetV3/fig03.png" width="70%"></center>

<center><img src="/public/img/2022-02-23-MobileNetV3/fig04.png" width="70%"></center>


---

## 4. Network Search

- 각 network block을 최적화함으로써 전체 network struce를 찾기 위한 platform-aware NAS를 사용하였다. 
- filter의 수를 찾기 위해 NetAdapt 알고리즘을 사용하였다.
- 이 테크닉들은 상호보완적이며 효율적인 모델을 찾기 위해 결합하여 사용할 수 있다.

### 4.1. Platform-Aware NAS for Blockwise Search

MnasNet과 같은 접근법을 사용하지만 조금 더 작은 weight factor $w=-0.15$(원래는 $w=-0.07)를 사용하여 latency가 다를 때 정확도의 큰 변화를 막고자 했다. 

### 4.2. NetAdapt for Layer-wise Search

platform-aware NAS와 상호보완적인 접근법이다. 간략히 정리하면 다음과 같이 동작한다:

1. platform-aware NAS로 찾은 seed network 구조로 시작한다.
2. 각 step마다:
    - 새로운 *proposals*를 생성한다. 각 proposal은 architecture의 변화를 포함하며 이전 step에 비해 적어도 $\delta$만큼의 latency reduction을 만들어 낸다.
    - 각 proposal에 대해 이전 step의 사전학습된 model


---

## 5. Network Improvements


### 5.1. Redesigning Expensive Layers



### 5.2. Nonlinearities


### 5.3. Large squeezeandexcite


### 5.4. MobileNetV3 Definitions

---

## 6. Experiments


### 6.1. Classification



#### 6.1.1 Training setup



#### 6.1.2 Measurement setup



### 6.2. Results


#### 6.2.1 Ablation study 



### 6.3. Detection



### 6.4. Semantic Segmentation


---

## 7. Conclusions and future work




---