---
layout: post
title: ELMo - Deep contextualized word representations
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [Paper_Review, NLP]
---

이 글에서는 2018년 2월 *Matthew E. Peters* 등이 발표한 Deep contextualized word representations를 살펴보도록 한다.

참고로 이 논문의 제목에는 ELMo라는 이름이 들어가 있지 않은데, 이 논문에서 제안하는 모델의 이름이 ELMo이다.  
Section 3에서 나오는 이 모델은 **E**mbeddings from **L**anguage **Mo**dels이다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# ELMo - Deep contextualized word representations

논문 링크: **[Deep contextualized word representations](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)**

홈페이지: **[OpenAI Blog](https://openai.com/blog/language-unsupervised/)**

## 초록(Abstract)

이 논문에서는 단어 사용의 복잡한 특성(문법 및 의미)과 이들이 언어적 문맥에서 어떻게 사용되는지(다의성)를 모델링하는, 새로운 종류의 ***맥락과 깊게 연관된* 단어표현(*Deep contextualized* word representation)**을 소개한다. 이 논문에서의 word vector는 큰 말뭉치에서 학습된 deep bidirectional language model(**biLM**)의 내부 상태로부터 학습한다. 이 표현(representation)은 이미 존재하는 모델에 쉽게 불일 수 있으며 이로써 QA 등 6개의 도전적인 NLP 문제에서 상당히 향상된 state-of-the-art 결과를 얻을 수 있음을 보였다. 또한 기학습된(pre-trained) 네트워크의 깊은 내부를 살펴보는 분석도 보인다.

---

## 1. 서론(Introduction)

**기학습된 단어 표현(Pre-trained word representations)**은 많은 자연어이해 모델에서 중요한 요소였다. 그러나 문법, 의미, 다의성을 학습한 높은 품질의 representation을 얻는 것은 어려운 일이다. 이 논문에서는 쉽게 다른 모델에 적용가능하며 성능도 뛰어난 **Deep contextualized word representation**을 소개한다.

이 representation은 (문장 내) 각 token이 전체 입력 sequence의 함수인 representation를 할당받는다는 점에서 전통적인 단어 embedding과 다르다. 이를 위해 이어붙여진 **language model(LM)**로 학습된 bidirectional LSTM(biLM)로부터 얻은 vector를 사용한다. 이 때문에 이를 **ELMo(Embeddings from Language Models) representation**이라 부른다. 이는 LSTM layer의 최종 layer만을 취한 것이 아닌, 각 layer 결과를 가중합하여 얻어지며 이것이 성능이 더 좋다.  
LSTM의 낮은 단계의 layer(입력과 가까운 층)는 품사 등 문법 정보를, 높은 단계의 layer(출력과 가까운 층)는 문맥 정보를 학습하는 경향이 있다.

많은 실험에서 ELMo representation이 매우 뛰어남을 보여 주었는데, 상대적으로 에러율을 20% 줄이기도 하였다. 

---

## 2. 관련 연구(Related work)

기학습된 단어 벡터(pretrained word vectors)는 많은 NLP 모델에서 매우 중요한 역할을 했다. 그러나 미리 학습된 단어 벡터는 다의어도 한 개의 벡터로 표현하기 때문에 문맥 정보를 고려하지 못한다.  
이를 극복하기 위한 방안으로 보조단어 정보를 활용하거나 각 단어당 여러 벡터를 만드는 방법이 고려되었다. 이 논문의 방법(ELMo representation)은 보조정보로부터의 이점을 가지며 또한 명시적으로 여러 벡터를 만들 필요도 없다.

문맥의존 표현을 학습하는 다른 연구로는 다음이 있다.

- 양방향 LSTM을 사용하는 context2vec(Melamud et al., 2016)
- 표현 안에 pivot word 자체를 포함하는 CoVe(McCann et al., 2017)

Deep biRNN의 낮은 단계의 layer를 사용하여 dependency parsing(Hashimoto et al., 2017)이나 CCG super tagging(Søgaard and Goldbert, 2016) 등의 문제에서 성능 향상을 시킨 연구도 있었다.  
Dai and Le(2015)와 Ramachandran et al.(2017)에서는 언어모델(LM)로 인코더-디코더 쌍을 기학습시키고 특정 task에 fine-tune시켰다.  

이 논문에서는 미분류된 데이터로부터 biLM을 기학습시킨 후 weights를 고정시키고 task-specific한 부분을 추가하여 leverage를 증가시키고 풍부한 biLM representation을 얻을 수 있게 하였다.

---

## 3. ELMo: Embeddings from Language Models

다른 단어 embedding과는 다르게 ELMo word representation은 전체 입력 sequence의 함수이다. 이는 글자수준 합성곱(character convolutions, Sec. 3.1)로부터 얻은 biLM의 가장 위 2개의 layer의 선형함수(가중합, Sec. 3.2)으로 계산된다. 이는 준지도학습과 더불어 biLM이 대규모에서 기학습되며(Sec 3.4) 쉽게 다른 NLP 모델에 붙일 수 있도록(Sec 3.3) 해준다.

### 3.1. Bidirectional language models

$N$개의 token $(t_1, t_2, ..., t_N)$이 있을 때, 전방언어모델(forward language model)은 $(t_1, ..., t_{k-1})$이 주어졌을 때 token $t_k$가 나올 확률을 계산한다:

$$ p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k \vert t_1, t_2, ..., t_{k-1}) $$

최신 언어모델은 token embedding이나 문자단위 CNN을 통해 맥락-독립적 token representation $x_k^{NM}$을 계산하고 이를 전방 LSTM의 $L$개의 layer에 전달한다.  
각 위치 $k$에서, 각 LSTM layer는 맥락-의존적 representation $\overrightarrow{h}_{k, j}^{LM}(j = 1, ..., L)$을 출력한다.  
LSTM의 최상위 layer LSTM 출력 $\overrightarrow{h}_{k, L}^{LM}$은 Softmax layer와 함께 다음 token을 예측하는 데 사용된다.

후방(backward) LSTM은 거의 비슷하지만 방향이 반대라는 것이 다르다. 식의 형태는 똑같지만 뒤쪽 token을 사용해 확률을 계산하고 token을 예측한다.

$$ p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k \vert t_{k+1}, t_{k+2}, ..., t_N) $$

즉 $(t_{k+1}, ..., t_N)$이 주어졌을 때 representation $\overleftarrow{h}_{k, j}^{LM}$을 계산한다.

biLM은 이 둘을 결합시킨 로그우도를 최대화한다.

$$ \sum_{k=1}^N \Big( \text{log} \ p(t_k \vert t_1, ..., t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) + \text{log} \ p(t_k \vert t_{k+1}, ..., t_N; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s) \Big)$$

$\Theta_x$는 token representation, $\Theta_s$는 Softmax layer이며 이 둘은 LSTM의 parameter과는 다르게 고정된다.

### 3.2. ELMo

ELMo는 biLM의 중간 layer representation을 task-specific하게 결합한다. biLM의 $L$-layer는 각 token $t_k$당 $2L+1$개의 representation을 계산한다.

$$ h_{k, j}^{LM}: \text{token layer}, h_{k, j}^{LM} = [\overrightarrow{h}_{k, j}^{LM}; \overleftarrow{h}_{k, j}^{LM}] $$

일 때

$$ R_k = \{ x_k^{LM}, \overrightarrow{h}_{k, j}^{LM}, \overleftarrow{h}_{k, j}^{LM} \vert j = 1, ..., L \} = \{ h_{k, j}^{LM} \vert j = 0, ..., L \} $$

위의 식은 위치 $k$에서 $R_k$는 $1+L+L=2L+1$개의 representation으로 이루어져 있다는 뜻이다.  

Downstream model로의 포함을 위해, ELMo는 $R$의 모든 layer를 하나의 벡터 $ELMo_k = E(R_k; \Theta_e)$로 압축시킨다.  
가장 단순한 예로 ELMo가 단지 최상위 레이어를 택하는 $E(R_k) = h_{k, L}^{LM}$는 TagLM이나 CoVe의 것과 비슷하다.

더 일반적으로, 모든 biLM layer의 task-specific한 weighting을 계산한다:

$$ ELMo_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k, j}^{LM} $$

$s^{task}$는 softmax-정규화된 가중치이고 scalar parameter $\gamma^{task}$는 전체 ELMo 벡터의 크기를 조절하는 역할을 한다. $\gamma$는 최적화 단계에서 중요하다.  
각 biLM layer에서의 활성함수는 다른 분포를 갖는데, 경우에 따라 가중치를 정하기 전 각 biLM layer에 정규화를 적용하는 데 도움이 되기도 한다.


### 3.3. Using biLMs for supervised NLP tasks



### 3.4. Pre-trained bidirectional language model architecture


---

<center><img src="/public/img/2019-08-20-ELMo - Deep contextualized word representations/01.png" width="100%" alt="Transformer Architecture"></center>

## 4. 평가(Evaluation)


### Question Answering
### Textual entailment
### Semantic rol;e labeling
### Coreference resolution
### Named entity extraction
### Sentiment analysis


---

## 5. 분석(Analysis)

### 5.1. Alternate layer weighting schemes
### 5.2. Where to include ELMo?
### 5.3. What information is captured by the biLM's representations?
#### Word sense disambiguation
#### POS tagging
#### Implications for supervised tasks
### 5.4. Sample efficiency

### 5.5. Visualization of learned weights


---

## 6. 결론(Conclusion)




---

## Refenrences

논문 참조. 61개의 레퍼런스가 있다.

---

