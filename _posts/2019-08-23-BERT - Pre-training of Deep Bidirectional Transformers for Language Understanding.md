---
layout: post
title: BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [Paper_Review, NLP]
---

이 글에서는 2018년 10월 *Jacob Devlin* 등이 발표한 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding를 살펴보도록 한다.

어쩐지 [ELMo](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/20/ELMo-Deep-contextualized-word-representations/)를 매우 의식한 듯한 모델명이다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

논문 링크: **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)**

Pytorch code: **[Github: dhlee347](https://github.com/dhlee347/pytorchic-bert)**

## 초록(Abstract)

이 논문에서는 

---

## 1. 서론(Introduction)



---

## 2. 관련 연구(Related work)



### 2.1. Unsupervised Feature-based Approaches

### 2.2. Unsupervised Fine-tuning Approaches

### 2.3. Transfer Learning from Supervised Data

---

## 3. BERT


**Model Architecture**

**Input/Output Representations**

### 3.1. Pre-training BERT

**Task #1: Masked LM**

**Task #2: Next Sentence Prediction(NSP)**

**Pre-training data**

### 3.2. Fine-tuning BERT



---

## 4. 실험(Experiments)

<center><img src="/public/img/2019-08-23-BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding/01.png" width="100%" alt="Results"></center>

### 4.1. GLUE

### 4.2. SQuAD v1.1

### 4.3. SQuAD v2.0

### 4.4. SWAG

---

## 5. Ablation Studies

이 섹션에서는 특정 부분을 빼거나 교체해서 해당 부분의 역할을 알아보는 ablation 분석을 수행한다. 한국어로 번역하기 참 어려운 단어이다.

### 5.1. Effect of Pre-training Tasks


### 5.2. Effect of Model Size



### 5.3. Feature-based Approach with BERT


---

## 6. 결론(Conclusion)

이 논문에서는 
---

## Refenrences

논문 참조. 레퍼런스가 많다.



---

## Appendix



---