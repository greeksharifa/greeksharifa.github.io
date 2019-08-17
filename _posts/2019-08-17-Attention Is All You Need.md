---
layout: post
title: Attention Is All You Need
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [PyTorch]
---

이 글에서는 2017년 *Ashish Vaswani* 등 연구자가 발표한 Attention Is All You Need를 살펴보도록 한다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# Attention Is All You Need

논문 링크: **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**

## 초록(Abstract)

성능 좋은 변환(번역) 모델은 인코더와 디코더를 포함한 복잡한 recurrent 또는 convolutional 신경망에 기반을 두고 있다. 최고 성능을 내는 모델 역시 attention mechanism을 사용하여 인코더와 디코더를 연결한다.  
이 논문에서 recurrence와 convolution을 전부 제외하고 오직 attention mechanism에만 기반한 **Transformer**라는 간단한 모델을 제안한다. 두 기계번역 task 실험에서는 이 모델은 병렬화와 학습시간 감소와 더불어 최고 수준의 품질을 가진다는 것을 보여준다. 이 모델은 WMT 2014 영어$\rightarrow$독일어 번역 task에서 이전보다 2 높은 28.4 BLEU를 달성하였다. 여기서 이 모델은 8개의 GPU로 8일 동안 학습시켜 41.8점의 BLEU state-of-the-art 단일 모델이다.  
이 논문에서 **Transformer**는 크거나 한정된 학습 데이터를 가지고서도 성공적으로 다른 task들에 일반화될 수 있음을 보인다.

---

## 1. 서론(Introduction)

RNN, LSTM, GRU 등은 sequence 모델링과 언어모델 등 변환 문제, 기계번역 등의 문제에서 뛰어난 성과를 보였다.  
Recurrent 모델은 보통 입력과 출력의 symbol position에 따라 계산을 수행한다. 계산 단계에서 위치를 적절히 맞추기 위해 이전 상태 $h_{t-1}$과 위치 $t$의 함수인 은닉상태 $h_t$를 생성한다. 이는 근본적으로 메모리 제한으로 인해 sequence가 길수록 병렬화를 힘들게 한다. 최근 들어 모델의 성능 자체는 비약적으로 상승했지만 위의 문제는 여전히 남아 있다.

Attention mechanism은 입력과 출력 sequence의 거리에 상관없이 의존성을 모델링함으로써 다양한 과제에서의 sequence 모델링과 변환 모델에서 매우 중요한 부분이 되었다. 그러나 거의 대부분의 경우 recurrent 네트워크와 함께 사용되고 있다.

이 논문에서는, **Transformer**라는, recurrence를 제거하고 입력-출력 간 전역 의존성을 학습할 수 있는 attention mechanism만을 사용한 모델 구조를 제안한다. **Transformer**는 병렬화를 비약적으로 달성하였으며 8개의 P100 GPU만으로 딱 12시간만을 학습하여 state-of-the-art 결과를 얻을 수 있게 한다. 

---

## 2. 예측 네트워크(Prediction Network)

<center><img src="/public/img/2019-08-17-Attention Is All You Need/01.png" width="100%" alt="Deep RNN Architecture"></center>

위 그림은 이 논문에서 사용된 기본 RNN 모델의 구조이다. 


---




---

## Refenrences

논문 참조. 33개의 레퍼런스가 있다.

---

