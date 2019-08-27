---
layout: post
title: OpenAI GPT-1 - Improving Language Understanding by Generative Pre-Training
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [Paper_Review, NLP]
---

---

이 글에서는 2018년 6월 *Alec Radford* 등이 발표한 OpenAI GPT-1: Improving Language Understanding by Generative Pre-Training를 살펴보도록 한다.

코드와 사전학습(기학습)된 모델은 [여기](https://github.com/google-research/bert)에서 볼 수 있다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# OpenAI GPT-1 - Improving Language Understanding by Generative Pre-Training

논문 링크: **[OpenAI GPT-1 - Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)**

홈페이지: **[OpenAI](https://openai.com/blog/language-unsupervised/)**

Tensorflow code: **[Official Code](https://github.com/openai/finetune-transformer-lm)**

## 초록(Abstract)

자연어이해는 원문함의, 질답, 의미유사성 평가, 문서분류 등 넓은 범위의 과제로 이루어져 있다. 미분류 상태의 큰 말뭉치가 풍부함에도, 이러한 특정 과제의 학습을 위한 분류된 데이터는 부족하며, 모델이 적절히 수행하도록 만드는 것을 어렵게 한다.  
이 논문에서는 이러한 과제들에서의 큰 성능 향상은, 각 특정 과제에 맞춘 세부조정(fine-tuning) 후, 언어모델을 다양한 미분류 말뭉치로 생성적 사전학습(*generative pre-training*)을 시키는 것으로 가능하다. 이전의 접근법과는 달리 모델구조는 최소한으로 변화시키면서 효과적인 전이(transfer)를 얻기 위한 세부조정 단계에서 과제에 맞는 입력표현(input representations)을 사용했다. 그리고 이 접근법이 다양한 과제에 대해 효과적임을 보일 것이다.

이 논문에서 제시하는 과제에 대한 별다른 지식이 없는(task-agnostic) 모델은 특정과제에 특화된 구조를 사용하는 모델의 성능을 뛰어넘는데 연구된 12개의 과제 중 9개에서는 state-of-the-art를 달성하였다. 예를 들어 상식추론(*Cloze*)에서는 8.9%, QA에서는 5.7%, 원문함의에서는 1.5% 상승하였다.

---

## 1. 서론(Introduction)

원본 그대로의 텍스트에서 효과적으로 학습하는 능력은 NLP에서 지도학습에 대한 의존성을 낮추는 데 있어 매우 중요하다. 대부분의 딥러닝 방법은 수작업으로 분류된 방대한 양의 데이터를 필요로 하는데 이는 분류된 자원의 부족으로 인한 많은 범위로의 응용에 제약을 건다. 이러한 상황에서 미분류 데이터로부터 언어적 정보를 얻어낼 수 있는 모델은 힘들게 분류된 데이터를 만드는 것의 훌륭한 대안이 될 뿐만 아니라, 괜찮은 지도 방법이 있다 하더라도 비지도학습이 더 좋을 결과를 얻기도 한다. 사전학습된 단어 embedding이 그러한 예이다.

그러나 미분류 텍스트에서 단어 수준 정보 이상의 것을 얻는 것은 다음 두 가지 이유로 어렵다.

1. 어떤 최적화 목적함수가 전이(transfer)에 유용한 텍스트 표현(representation)을 배우는 데 효과적인지 불분명하다. 최근 연구들은 언어모델링이나 기계번역, 담화 일관성(discourse coherence) 등 다양한 objective에서 각 방법이 다른 과제에서는 다른 방법을 능가하는 것을 보여 왔다.
2. 학습된 표현을 다른 과제로 전이하는 가장 효과적인 방법에 대한 일치된 의견이 없다. 존재하는 방법들은 복잡한 학습전략이나 부가 학습 목적함수를 더하는 등 모델 구성에 과제에 특화된(task-specific) 변화를 주어야 한다. 이러한 불확실성은 언어처리에 대한 준지도학습

---

## 2. 관련 연구(Related work)

범용언어표현의 사전학습 연구는 긴 역사가 있다. 간단히 살펴보자.

**Semi-supervised learning for NLP**

넓은 범위에서 

**Unsupervised pre-training**

이 방법은 

**Auxiliary training objectives**

언어추론이나

---

## 3. Framework

이 framework에는 

<center><img src="/public/img/2019-08-21-OpenAI GPT-1 - Improving Language Understanding by Generative Pre-Training/01.png" width="100%" alt="Architecture"></center>


### 3.1. Unsupervised pre-training

로 원본 token을 예측한다. 이 과정의 변형은 부록 C.2에서 다룬다.

### 3.2. Supervised fine-tuning



### 3.3. Task-specific input transformations

**Textual entailment**

**Similarity**

**Question Answering and Commonsense Reasoning**


---

## 4. 실험(Experiments)

### 4.1. Setup

**Unsupervised pre-training**

**Model specifications**

**Fine-tuning details**

### 4.2. Supervised fine-tuning

Stanford Question Answering Dataset은 10만여 개의 질답 쌍으로 구성되어 있다. 질문과 그에 대한 답을 포함하는 위키피디아 지문이 주어지면, 해당 지문에서 답이 되는 부분을 찾는 과제이다.

**Natural Language Inference**

**Question answering and commonsense reasoning**

**Semantic Similarity**

**Classification**



---

## 5. 분석(Analysis)

**Impact of number of layers transferred**

**Zeroo-shot Behaviors**

**Ablation studies**


---

## 6. 결론(Conclusion)

최근 

---

## Refenrences

논문 참조. 71개의 레퍼런스가 있다.

부록은 없다.

---
