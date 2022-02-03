---
layout: post
title: MobileNetV1 논문 설명(MobileNets - Efficient Convolutional Neural Networks for Mobile Vision Applications 리뷰)
author: YouWon
categories: [Computer Vision]
tags: [MobileNet, Google]
---

---


이 글에서는 Google Inc.에서 발표한 MobileNet V1 논문을 간략하게 정리한다.

---

# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

논문 링크: **[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)**

Github: [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

- 2017년 4월(Arxiv)
- Google Inc.
- Andrew G. Howard, Menglong Zhu et al.

모바일 기기에서 동작하는 것을 목표로 한, 성능을 별로 희생하지 않으면서도 모델을 크게 경량화하였다.

---

## Abstract


---

## 1. Introduction


<center><img src="/public/img/2021-12-14-Swin-Transformer/01.png" width="70%"></center>



---

## 2. Prior Work

**CNN and variants**: 


---

## 3. MobileNet Architecture

### 3.1. Depthwise Separable Convolution


### 3.2. Network Structure and Training




### 3.3. Width Multiplier: Thinner Models


### 3.4. Resolution Multiplier: Reduced Representation


---

## 4. Experiments



### 4.1. Model Choices



### 4.2. Model Shrinking Hyperparameters



### 4.3. SFine Grained Recognition



### 4.4. Large Scale Geolocalizaton


### 4.5. Face Attributes


### 4.6. Object Detection



### 4.7. Face Embeddings


---

## 5. Conclusion





---

