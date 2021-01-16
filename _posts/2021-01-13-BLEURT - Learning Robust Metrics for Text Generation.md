---
layout: post
title: BLEURT - Learning Robust Metrics for Text Generation
author: YouWon
categories: [Machine Learning]
tags: [Paper_Review, NLP, Evaluation_Metric]
---

---

이 글에서는 2020년 ACL에 Google Research 팀의 *Thibault Sellam* 등이 게재한 **BLEURT: Learning Robust Metrics for Text Generation**를 살펴보도록 한다.

[구글 AI 블로그](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)에서 이 논문에 대한 설명을 볼 수 있다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# BLEURT: Learning Robust Metrics for Text Generation

논문 링크: **[BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696)**

홈페이지: **[구글 AI 블로그](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)**

Tensorflow code: **[Official Code](https://github.com/google-research/bleurt)**

## 초록(Abstract)

텍스트 생성은 지난 몇 년간 상당한 발전을 이루었다. 그러나 아직 그 평가 방법은 매우 뒤떨어져 있는데, 가장 자주 사용되는 BLEU나 ROUGE는 사람의 판단과는 낮은 연관성을 갖는다(즉, 사람이 보기에 별로 적절치 않다). 이 논문에서는 BLEURT라는 새 평가 방법을 제안하는데, BERT에 기반한 학습된 평가방법으로 수천 개 정도의 학습 예시만으로도 사람의 판단 방식을 모델링할 수 있다. 이 접근법의 핵심은 수백만 개의 합성된 예시를 모델을 일반화하기 위해 사용하는 새로운 사전학습 방식이라는 것이다. BLEURT는 WMT Metrics와 WebNLG Competition 데이터셋에서 state-of-the-art 결과를 얻었다. 바닐라 BERT 기반 방식과는 다르게, BLEURT는 학습 데이터가 드물고 기존 분포에서 벗어나도 훌륭한 결과를 보여준다. 

---

## 1. 서론(Introduction)

지난 몇 년간, 자연어생성 분야에서의 연구는 번역, 요약, 구조화된 데이터 → 텍스트 생성, 대화 및 이미지 캡션 등을 포함한 많은 문제에서 encoder-decoder 신경망을 통해 상당한 발전을 이루었다. 하지만, 평가 방법의 부족으로 인해 발전이 적지 않게 지연되었다.

인간 평가(Human Evaluation)는 시스템의 품질을 측정하는 데 있어서 종종 최선의 지표가 되지만, 이는 매우 비싸며 상당히 시간이 많이 소요되는 작업으로 매일 모델 개발의 pipeline에 넣을 수는 없다. 그래서, NLG 연구자들은 보통 계산이 빠르고 그럭저럭 괜찮은 품질의 결과를 주는 자동 평가방법(automatic evaluation metrics)을 사용해 왔다. 이 논문에서는 문장 수준의, 참조 기반 평가 방법으로, 어떤 *후보* 문장이 *참조* 문장과 얼마나 비슷한지를 측정하는 방법을 제시한다. 

1세대 평가방법은 문장 간의 표면적 유사도를 측정하기 위해 수동으로 만들어졌다. BLEU와 ROUGE라는 두 개의 방법이 N-gram 중첩(overlap)에 기반하여 만들어졌다. 이러한 평가방법은 오직 어휘의 변화에만 민감하며, 의미적 또는 문법적인 측면의 변화는 제대로 측정하지 못하였다. 따라서, 이러한 방식은 사람의 판단과는 거리가 멀었으며, 특히 비교할 시스템이 비슷한 정확도를 가질 때 더욱 그렇다.

NLG 연구자들은 이 문제를 *학습된* 구성 요소를 이 평가방법에 집어넣음으로써 다뤄 왔다. WMR Metrics Shard Task라는, 번역 평가에서 자주 사용되는 평가방법이 있다. 최근 2년간은 신경망에 기반한 RUSE, YiSi, ESIM이 많이 사용되었다. 최근의 방법들은 다음 가지로 나뉜다:

1. 완전히 학습된 평가방법.
    - BEER, RUSE, ESIM
    - 일반적으로 end-to-end 방식으로 학습되었으며, 보통 수동으로 만든 feature나 학습된 embedding에 의존한다.
    - 훌륭한 표현능력(expressivity)를 가진다.
    - 유창성, 충실함, 문법, 스타일 등 task-specific한 속성을 가지도록 튜닝할 수 있다.
2. Hybrid 평가방법. 
    - YiSi, BERTscore
    - 학습된 요소(contextual embeddings)를 수동으로 만든 논리(token alignment 규칙)와 결합한다.
    - 강건성(Robustness)를 확보할 수 있다.
    - 학습 데이터가 적거나 없는 상황에서 좋은 결과를 얻을 수 있다.
    - train/test 데이터가 같은 분포에 존재한다는 가정을 하지 않는다.

사실, IID 가정은 *domain drift* 문제에 의해 NLG 평가에서 특히 문제가 되었다. 이는 평가방법의 주 목적이지만, *quality drift* 때문이기도 하다: NLG 시스템은 시간이 지남에 따라 더 좋아지는 경향을 보이며, 따라서 2015년에 순위 데이터로 학습한 모델은 2019년의 최신 모델을 구별하지 못할 수 있다(특히 더 최근 것일수록). 이상적인 학습된 평가방법은 학습을 위해 이용가능한 순위 데이터를 완전히 활용하고, 분포의 이탈(drift)에 강간한 것을 모두 확보하는 것이다. 즉 *추론extrapolate*할 수 있어야 한다.

이 논문에서 통찰한 바는 표현능력과 강건성을 인간 순위에 미세조정하기 전 대량의 합성 데이터에서 사전학습하는 방식으로 결합하는 것이 가능하다는 것이다.  
여기서 BERT에 기반한 텍스트 생성 평가방법으로 BLEURT를 제안한다. 핵심은 새로운 사전학습 방법으로, 어휘적 그리고 의미적으로 다양한 감독 신호(supervision signals)를 얻을 수 있는 다양한 Wikipedia 문장에서 임의의 변화(perturbation)을 준 문장들을 사용하는 방법이다.

BLEURT를 영어에서 학습하고 다른 일반화 영역에서 테스트한다. 먼저 WMT Metrics Shared task의 모든 연도에서 state-of-the-art 결과를 보인다. 그리고 스트레스 테스트를 하여 WMT 2017에 기반한 종합평가에서 품질 이탈에 대처하는 능력을 측정한다. 마지막으로, data-to-text 데이터셋인 WebNLG 2017으로부터 얻는 3개의 task에서 다른 도메인으로 쉽게 조정할 수 있음을 보인다. Ablation 연구는 이 종합 사전학습 방법이 IID 세팅에서 그 성능을 증가시키며, 특히 학습 데이터가 편향되어 있거나, 부족하거나, 도메인을 벗어나는 경우에 더 강건함을 보인다. 

코드와 사전학습 모델은 [온라인](https://github.com/google-research/bleurt)에서 볼 수 있다.

---

## 2. 서두(Preliminaries)

$x = (x_1, ..., x_r)$는 $r$개의 token을 갖는 참조 문장이며 $\tilde{x} = (\tilde{x}_1, ..., \tilde{x}_p)$는 길이 $p$의 예측 문장이다. $y$가 예측 문장이 참조 문장과 관련하여 얼마나 좋은지를 사람이 정한 값이라 할 때 크기 $N$의 학습 데이터셋은 다음과 같이 쓴다.

$$ \{(x_i, \tilde{x}_i, y_i)\}^N_{n=1} $$

학습 데이터가 주어지면, 우리의 목표는 사람의 평가를 예측하는 함수 $f : (x, \tilde{x}) \rightarrow y$를 학습하는 것이다.


---

## 3. 품질 평가를 위한 미세조정 BERT(Fine-Tuning BERT for Quality Evaluation)

적은 양의 순위(rating) 데이터가 주어졌을 때, 이 task를 위해 비지도 표현을 사용하는 것이 자연스럽다. 이 모델에서는 텍스ㅌ 문장의 문맥화된 표현을 학습하는 [BERT](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/23/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/)를 사용하였다. $x$와 $\tilde{x}$가 주어지면, BERT는 문맥화된 벡터의 sequence를 반환하는 [Transformer](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/17/Attention-Is-All-You-Need/)이다:

$$ v_{[\text{CLS}]}, v_{x_1}, ..., v_{x_r}, v_1, ..., v_{\tilde{x}_p} = \text{BERT}(x, \tilde{x})  $$

$ v_{[\text{CLS}]}$는 특수 토큰 $\text{[CLS]}$의 표현이다. 여기서 순위를 예측하기 위해 $\text{[CLS]}$ 벡터에 선형 layer를 하나 추가했다:

$$ \hat{y} = f(x, \tilde{x}) = W\tilde{v}_{[\text{CLS}]} + b$$

$W$와 $b$는 각각 weight matrix와 bias이다. 위의 선형 layer와 BERT parameter는 대략 수천 개의 예시를 사용하여 미세조정(fine-tuned)된다. Regression Loss로 다음을 쓴다.

$$ l_{\text{supervised}} = \frac{1}{N} \Sigma^N_{n=1}\Vert y_i - \hat{y}\Vert^2 $$

이 접근법이 상당히 간단하지만, Section 5에서 

---

## 4. 합성 데이터에서 사전학습(Pre-Training on Synthetic Data)



### 4.1. 문장 쌍 생성(Generating Sentence Pairs)


**Mask-filling with BERT:**

**Backtranslation:**

**Dropping words:**

<center><img src="/public/img/2021-01-13-BLEURT - Learning Robust Metrics for Text Generation/01.png" width="80%" alt="Examples"></center>


### 4.2. 사전학습 신호(Pre-Training Signals)

**Automatic Metrics:**


**Backtranslation Likelihood:**

**Textual Entailment:**



### 4.3. 모델링(Modeling)






---

## 5. 실험(Experiments)




**Our Models:**


### 5.1. WMT Metrics Shared Task



**Datasets and Metrics:**

**Models:**

**Results:**

**Takeaways:**



### 5.2 Robustness to Quality Drift




**Methodology:**

**Results:**

**Takeaways:**

### 5.3 WebNLG Experiments

**Dataset and Evaluation Tasks:**


**Systems and Baselines:**

**Results:**

**Takeaways:**


### 5.4 Ablation Experiments






---

## 6. 관련 연구(Related Work)


---

## 7. 결론(Conclusion)




**Acknowledgements**

언제나 있는 감사의 인사

---

## Refenrences

논문 참조. 많은 레퍼런스가 있다.


---

## Appendix A Implementation Details of the Pre-Training Phase

### A.1. Data Generation


**Random Masking:**

**Backtranslation:**

**Word dropping:**



### A.2 Pre-Training Tasks

**Automatic Metrics:**

**Backtranslation Likelihood:**

**Normalization:**



### A.3 Modeling


**Setting the weights of the pre-training tasks:**



## Appendix B Experiments–Supplementary Material



### B.1 Training Setup for All Experiments



### B.2 WMT Metric Shared Task

**Metrics.**

**Training setup.**

**Baselines.**




### B.3 Robustness to Quality Drift

**Data Re-sampling Methodology:**


### B.4 Ablation Experiment–How Much



---
