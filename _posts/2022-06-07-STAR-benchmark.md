---
layout: post
title: STAR benchmark 논문 설명(STAR - A Benchmark for Situated Reasoning in Real-World Videos)
author: YouWon
categories: [Computer Vision]
tags: [Benchmark, Video]
---

---


이 글에서는 MIT 등 연구자들이 STAR benchmark 논문을 간략하게 정리한다.

---

# STAR: A Benchmark for Situated Reasoning in Real-World Videos

논문 링크: **[STAR: A Benchmark for Situated Reasoning in Real-World Videos](https://openreview.net/forum?id=EfgNF5-ZAjM)**

Repo: [http://star.csail.mit.edu/#repo](http://star.csail.mit.edu/#repo)  
Github: [https://github.com/csbobby/STAR_Benchmark](https://github.com/csbobby/STAR_Benchmark)

- NIPS 2021
- Bo Wu(MIT-IBM) et al.


 
---

## Abstract

- 주변의 상황으로부터 지식을 얻고 그에 따라 추론하는 것은 매우 중요하고 또한 도전적인 과제이다.
- 이 논문에서는 실세계 영상에 대해 situation abstraction, logic-grounded 질답을 통해 situated 추론 능력을 평가하는 새로운 benchmark를 제시한다.
- **STAR(Situated Reasoning in Real-World Videos)**
    - 이는 사람의 행동, 상호작용 등과 연관된 실세계 영상에 기반하여 만들어진 것으로 naturally dynamic, compositional, logical한 특성을 가진다.
    - 4가지 형태의 질문(interaction, sequence, prediction, and feasibility)을 포함한다.
    - 이 실세계 영상의 situations은 추출한 단위 entity와 relation을 연결한 hyper-graph로 구성된다.
    - 질문과 답변은 절차적으로 생성되었다.
- 여러 영상 추론 모델을 이 데이터셋에 적용하여 보았을 때 상황 추론 task에서 어려움을 겪는 것을 발견하였다.
- Diagnostic neuro-symbolic 모델을 제시하며, 이 benchmark의 challenge를 이해하기 위한 이 모델은 visual perception, situation abstraction, language understanding, and functional reasoning을 disentangle할 수 있다.



---

## 1. Introduction

그림 1과 같은 (실세계) 상황에서 우리(사람)는 어떻게 행동할지, 현실적인 결정을 무의식적으로 내릴 수 있다. 그러나 기계한테는 주어진 문맥과 상황을 모두 고려하여 결정을 내린다는 것은 꽤 어려운 문제이다.

<center><img src="/public/img/2022-06-07-STAR-benchmark/fig01.png" width="100%"></center>

- 상황을 formulae의 집합으로 두고 가능한 logic 규칙을 만들어 이해하려는 시도가 있었으나 모든 가능한 logic rule을 만드는 것은 불가능하며 현실성이 떨어진다.

현존하는 비디오 이해 모델들을 도전적인 task에서 테스트한 결과 성능이 매우 낮아짐을 확인하였다. 이 모델들은 추론 자체보다는 시각적인 내용과 질답 간 연관성을 leverage하는 데에 집중하고 있었다.

이 논문에서는 STAR benchmark를 제안한다.

- 4종류의 질문을 포함한다: interaction question, sequence question, prediction question, and feasibility question.
    - 각 질문은 다양한 장면과 장소에서 얻어진 action 중심 situation과 연관되어 있으며 각 situation은 여러 action과 연관되어 있다.
- 현존하는 지식과 상황에 따라 유동적으로 변화하는 지식을 표현하기 위해 entity와 relation으로 구조화된 표현으로 추상화하였다(situation hypergraphs).
- 시각적 추론 능력에 집중하기 위해 (자연어) 질문은 간결한 형태의 template에 맞춰 생성되었다.
- 보조 용도로, (더욱 어려운) 사람이 만든 질문을 포함하는 STAR-Humans도 같이 제공한다.
- 다른 데이터셋과 비교한 결과는 Table 1에 있다.

또한, **Neuro-Symbolic Situated Reasoning (NS-SR)**라는, 실세계 situated 추론을 위한 neural-symbolic 구조를 갖는 diagnostic model을 제안한다. 이는 질문에 답변하기 위해 구조화된 situation graph와 situation으로부터 얻은 dynamic clues를 활용한다. 

이 논문이 기여한 바는,

- interaction, sequence, prediction, and feasibility questions에 집중하여, 실세계 영상에서 situated reasoning 문제를 형식화했다.
- situated reasoning을 위해 잘 설계된 benchmark인 STAR을 구성하였다. 
    - 3가지 측면(visual perception, situation abstraction and logic reasoning)에서 annotation이 설계되었다.
    - 각 영상은 situation hyper-graph로 grounded되어 있으며 각 질문은 functional program으로 연관되어 있다.
- 여러 SOTA 방법을 STAR로 테스트해 보았고 '사람에게는 자명한 상황'에서 모델은 실수가 많음을 보였다.
- Diagnostic neuro-symbolic framework을 설계하였고 더욱 강력한 추론 모델을 위한 연구 방향을 제시하였다.



---

## 2. Related Work

- **Visual Question Answering**
- **Visual Reasoning**

모델의 추론 능력을 진단하기 위한 여러 데이터셋이 있다: CLEVR, GQA, MarioQA, COG, CATER, CLEVRER, etc.

- **Situation Formalism**

<center><img src="/public/img/2022-06-07-STAR-benchmark/tab01.png" width="100%"></center>

---

## 3. Situated Reasoning Benchmark

situations abstraction과 logical reasoning을 결합하였고, 아래 3가지 가이드라인을 따라 benchmark를 구축했다.

1. 추상화를 위한 bottom-up anotations에 기반한 계층적 graph로 표현되는 situations
2. situated reasoning을 위한 질문과 선택지 생성은 정형화된 질문, functional programs, 공통 situation data types에 grouded됨
3. situated reasoning이 situation graphs에 대해 반복적으로 수행할 수 있음

만들어진 데이터셋의 metadata는 다음과 같다. 

- 60K개의 situated reasoning 질의
- 240K개의 선택지
- 22K개의 trimmed situation video clip으로 구성된다.
- 144K개의 situation hypergraph(structured situation abstraction)
- 111 action predicates
- 28 objects
- 24 relationships
- train/val/test = 6:1:1
- 더 자세한 내용은 부록 2, 3 참조


### 3.1. Situation Abstraction

**Situations**

Situation은 STAR의 핵심 컨셉으로 entity, event, moment, environment를 기술한다. [Charades dataset](https://prior.allenai.org/projects/charades)으로부터 얻은 action annotation과 9K개의 영상으로 situation을 만들었다. 영상들은 주방, 거실, 침실과 같은 11종류의 실내환경에서의 일상생활이나 활동을 묘사한다. 각 action별로 영상을 나눌 수 있으며 영상 역시 이에 맞춰서 나눌 수 있다.  

각 action은 (1) action precondition과 (2) effect로 나눌 수 있다. 
1. action precondition은 환경의 초기 static scene을 보여주기 위한 첫 frame이다.
2. action effect는 하나 또는 여러 개의 action의 process를 기술한다.

질문의 종류에 따라서는:  
- interaction, sequence 타입의 question은 완전한 action segment를 포함한다.
- prediction, feasibility 타입의 question은 불완전한 action segment를 포함하거나 아예 포함하지 않는다.


**Situation Hypergraph**




### 3.2. Questions and Answers Designing

**Question Generation**


**Answer Generation**


**Distractor Generation**


**Debiasing and Balancing Strategies**


**Grammar Correctness and Correlation**


**Rationality and Consistency**


---

## 4. Baseline Evaluation




### 4.1. Comparison Analysis




---

## 5. Diagnostic Model Evaluation

### 5.1. Model Design


**Video Parser**


**Transformers-based Action Transition Model**


**Language Parser**



**Program Executor**




### 5.2. Result Analysis


**Situation Abstraction**
**Visual Perception**
**Language Understanding**
**Without Ground-Truths**


---

## 6. Conclusion

한정된 자원을 갖고 있는 상황에서 Depth, Width, Resolution을 어떻게 적절히 조절하여 모델의 크기와 연산량을 줄이면서도 성능은 높일 수 있는지에 대한 연구를 훌륭하게 수행하였다.


---
