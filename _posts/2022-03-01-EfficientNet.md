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



### 3.2. Scaling Dimensions



### 3.3. Compound Scaling


 

---

## 4. EfficientNet Architecture

---

## 5. Experiments




### 5.1. Scaling Up MobileNets and ResNets




### 5.2. ImageNet Results for EfficientNet




### 5.3. Transfer Learning Results for EfficientNet



---

## 6. Discussion



---

## 7. Conclusion





---