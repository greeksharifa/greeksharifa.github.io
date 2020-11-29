---
layout: post
title: Attention Is All You Need(Attention 논문 설명)
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [Paper_Review, NLP]
---

---

이 글에서는 2017년 6월(v1) *Ashish Vaswani* 등이 발표한 Attention Is All You Need를 살펴보도록 한다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# Attention Is All You Need

논문 링크: **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**

Pytorch code: **[Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)**

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

## 2. 배경(Background)

연속적 계산을 줄이려는 노력은 Extended Neural GPU, ByteNet, ConvS2S 등의 모델을 탄생시켰으나 이들은 전부 CNN을 기본 블록으로 사용한다. 이러한 모델들은 임의의 위치의 input-output 사이의 관련성을 파악하기 위해서는 거리에 따라(선형 또는 로그 비례) 계산량이 증가하며, 이는 장거리 의존성을 학습하기 어렵게 한다.  
Transformer는, 이를 상수 시간의 계산만으로 가능하게 하였다.

intra-attention으로도 불리는 Self-attention은 sequence의 representation을 계산하기 위한 단일 sequence의 다른 위치를 연관시키는 attention mechanism이다. Self-attention은 많은 과제들에서 사용되었으며 성공적이었다.

End-to-end 메모리 네트워크는 sequence-aligned recurrence 대신 recurrent attention mechanism에 기반하였으며 simple-language QA와 언어모델링 task 등에서 좋은 성과를 내었다.

그러나, Transformer는 RNN이나 convolution 없이 오직 attention에 전적으로 의존한 첫 번째 변환 모델이다. 앞으로 이 모델에 대한 설명이 이어질 것이다.

---

## 3. 모델 구성(Model Architecture)

Transformer는 크게 인코더와 디코더로 나뉘며, 인코더는 입력인 symbol representations $(x_1, ..., x_n)$을 continuous representations $z = (z_1, ..., z_n)$으로 매핑한다. $z$가 주어지면, 디코더는 한번에 한 원소씩 출력 sequence $(y_1, ..., y_n)$를 생성한다.  
각 단계는 자동회귀(auto-regressive)이며, 다음 단계의 symbol을 생성할 때 이전 단계에서 생성된 symbol을 추가 입력으로 받는다. 

Transformer는 인코더와 디코더 모두에서 쌓은 self-attention과 point-wise FC layer를 사용하며, 그 구성은 아래 그림에 나타나 있다.

<center><img src="/public/img/2019-08-17-Attention Is All You Need/01.png" width="100%" alt="Transformer Architecture"></center>

### 3.1. Encoder and Decoder Stacks

인코더는 $N = 6$ 개의 동일한 레이어로 구성되며, 각 레이어는 아래 두 개의 sub-layer로 이루어져 있다. 

- multi-head self-attention mechanism
- simple, position-wise fully connected feed-forward network

각 sub-layer의 출력값은 LayerNorm($x$ + Sublayer($x$))이고, Sublayer($x$)는 sub-layer 자체로 구현되는 함수이다. 이 residual connection을 용이하게 하기 위해, embedding layer를 포함한 모델의 모든 sub-layer는 $d_{model} = 512$차원의 출력값을 가진다.

디코더 역시 $N = 6$ 개의 동일한 레이어로 구성되지만, 각 레이어는 인코더의 것과 동일한 두 개의 sub-layer 외에 한 가지를 더 가진다.

- encoder stack의 출력값에 multi-head attention을 수행하는 sub-layer

인코더와 비슷하게 residual connection이 각 sub-layer의 정규화 layer 뒤에 있다. 그리고 디코더가 출력을 생성할 때 다음 출력에서 정보를 얻는 것을 방지하기 위해 **masking**을 사용한다. 이는 $i$번째 원소를 생성할 때는 $1 \sim i-1$번째 원소만 참조할 수 있도록 하는 것이다.

### 3.2. Attention

Attention 함수는 *query + key-value* $\rightarrow$ *output* 으로의 변환을 수행한다. query, key, value, output은 모두 벡터이다. output은 value들의 가중합으로 계산되며, 그 가중치는 query와 연관된 key의 호환성 함수(compatibility function)에 의해 계산된다.

#### 3.2.1. Scaled Dot-Product Attention

이 이름은 Attention을 계산하는데 dot-product를 쓰고, 그 결과에 scale 조정을 하기 때문에 이렇게 붙여졌다.

<center><img src="/public/img/2019-08-17-Attention Is All You Need/02.png" width="100%" alt="Scaled Dot-Product Attention & Multi-head Attention"></center>

입력은 $d_k$차원의 query와 key, $d_v$차원의 value로 구성된다.   
query와 모든 key의 내적(dot product)을 계산하고, 각각 $\sqrt{d_k}$로 나누고, value의 가중치를 얻기 위해 softmax 함수를 적용한다.

실제로는, query들에 대해 동시에 계산하기 위해 이를 행렬 $Q$로 묶는다. 모든 key와 value 역시 각각 행렬 $K$와 $V$로 표현된다. 이제 $Q, K, V$의 attention을 구하는 식은 다음과 같다.

$$ Attention(Q, K, V) = \text{softmax} \Big( \frac{QK^T}{\sqrt{d_k}} \Big) V $$

가장 널리 쓰이는 attention 함수는 다음 두 가지다:

- Additive attention: 단일 hidden layer의 feed-forward 네트워크를 사용하여 호환성 함수를 계산한다. $d_k$가 작을 때 성능이 더 좋다.
- Dot-product attention: $d_k$가 더 클 때는 빠른 행렬곱 알고리즘에 힘입어 더 빠르고 더 공간 효율적이다.


#### 3.2.2. Multi-Head Attention

$d_{model}$차원 key, value, query로 단일 attention function을 쓰는 것보다 query, key, value를 각각 $d_k, d_k, d_v$차원으로 각각 다르게 $h$번 학습시키는 것이 낫다. 여기서 $h$번 학습시킨다는 것은 단지 반복을 한다는 것이 아니라, 각 sub-layer에 동일한 부분이 $h$개 존재한다는 뜻이다. 위 그림의 오른쪽을 보자.  
이렇게 각각 따로 계산된 $h$쌍의 $d_v$차원 출력은 이어붙인(concatenate) 후 한번 더 선형 함수에 통과시켜(projected) 최종 출력값이 된다.

식으로 나타내면 다음과 같다.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O, where \ head_i=\text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

여기서 $ W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, W_i^K \in \mathbb{R}^{d_{model} \times d_k}, W_i^V \in \mathbb{R}^{d_{model} \times d_v}, W^O \in \mathbb{R}^{hd_v \times d_{model}} $이며, 논문에서는 $h=8, d_k=d_v=d_{model}/h = 64$를 사용하였다.  
각 head의 차원이 줄었기 때문에 단일 head attention과 계산량은 비슷하다.

#### 3.2.3. Applications of Attention in our Model

- "encoder-decoder attention" layer에서, query는 이전 디코더 layer에서 오며 memory key와 value는 encoder의 출력에서 온다. 이는 디코더가 입력의 모든 위치(원소)를 고려할 수 있도록 한다.
- 인코더는 self-attention layer를 포함한다. 여기서 모든 key, value, query는 같은 곳(인코더의 이전 layer의 출력)에서 온다. 따라서 인코더의 각 원소는 이전 layer의 모든 원소를 고려할 수 있다.
- 이는 디코더에서도 비슷하다. 그러나 auto-regressive 속성을 보존하기 위해 디코더는 출력을 생성할 시 다음 출력을 고려해서는 안 된다. 즉 이전에 설명한 **masking**을 통해 이전 원소는 참조할 수 없도록 한다. 이 masking은 dot-product를 수행할 때 $-\infty$로 설정함으로써 masking out시킨다. 이렇게 설정되면 softmax를 통과할 때 0이 되므로 masking의 목적이 달성된다.

### 3.3. Position-wise Feed-Forward Networks

인코더와 디코더의 각 layer는 FC feed-forward 네트워크를 포함하는데, 이는 각 위치마다 동일하게 적용되지만 각각 따로 적용된다. 이는 ReLU 활성함수와 2개의 선형변환을 포함한다. kernel size가 1인 CNN과 같다.

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

각 레이어에 이 부분은 독립적인 parameter를 사용한다. 논문에서는 $d_{model}=512, d_{ff} = 2048$을 사용했다.

### 3.4. Embeddings and Softmax

다른 모델들과 비슷하게 embedding을 사용하였다. 이 모델에서는 2개의 embedding layer와 pre-softmax 선형변환 사이에 같은 weight 행렬을 사용했다. Embedding layer에는 $\sqrt{d_{model}}$을 곱한다.

### 3.5. Positional Encoding

이 모델에는 recurrence도 convolution도 사용되지 않기 때문에 sequence에 있는 원소들의 위치에 대한 정보를 따로 넣어 주어야 한다. 그래서 인코더와 디코더 stack의 밑부분에 **positional encodings**를 입력 embedding에 추가하였다. 이는 embedding과 갈은 $d_{model}$차원을 가지며, 따라서 더할 수 있다. 모델에서 사용된 것은 사인/코사인 함수이다.

$$ \quad PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) $$

$$ \ PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}}) $$

$pos$는 위치(position)이고 $i$는 차원이다.

가능한 여러 함수 중 사인함수 버전을 선택한 이유는 학습 때보다 더 긴 sequence를 만나도 추정이 가능하기 때문이다.

---

## 4. 왜 Self-Attention인가(Why Self-Attention)

$(x_1, ..., x_n) \rightarrow (z_1, ..., z_n)$에 self-attention이 적합한 이유는

1. layer 당 전체 계산량이 적고
2. 계산이 병렬화될 수 있다. 즉 병렬적으로 한번에 많은 계산을 할 수 있는데, recurrence의 경우 순차적으로 계산해야 하기 때문에 계산의 병렬화가 거의 불가능하다.
3. 장거리 의존성(long-range 또는 long-term dependency) 때문이다.

장거리 의존성을 학습할 수 있는 중요 요인은 네트워크 상에서 횡단할 수 있는 경로의 길이인데, 길이가 짧을 때는 다 비슷하므로 최대 길이를 중점적으로 살펴보았다.

<center><img src="/public/img/2019-08-17-Attention Is All You Need/03.png" width="100%" alt="Why Self-Attention"></center>

위 표에서 볼 수 있듯 장거리 의존성의 학습 속도(또는 능력)에서 self-attention이 가장 좋다.

---

## 5. 학습(Training)

Parameter | Descrption
-------- | --------
DataSet(German) | WMT 2014 English-German dataset(4.5M쌍의 문장, 37000 vocab)
DataSet(French) | WMT 2014 English-French dataset(36M쌍의 문장, 32000 vocab)
Batch size | 25000
Hardware | 8개의 P100 GPU
Schedule | Base Model: 12시간=10만 step $\times$ 0.4초/step, Big Model: 36시간=30만 step
Optimizer | Adam($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9} $)
Learning Rate | $lrate = d_{model}^{-0.5} \cdot \min ($step\_num$^{-0.5}$, step\_num $\cdot$ warmup\_steps $^{-1.5}) $ 
warmup\_steps | 4000
Regularization | Residual Dropout($P_{drop} = 0.1$)

---

## 6. 결과(Results)

<center><img src="/public/img/2019-08-17-Attention Is All You Need/04.png" width="100%" alt="Result 1"></center>

<center><img src="/public/img/2019-08-17-Attention Is All You Need/05.png" width="100%" alt="Result 2"></center>

Machine Translation, Model Variations, English Constituency Parsing에 대한 실험 결과이다. Base Model만 해도 충분히 최고 성능을 보여주며, 특히 Big Model의 경우 state-of-the-art를 상당한 수준으로 경신하는 성능을 보여 주었다.  
이외에 따로 요약이 필요하지는 않아 자세한 조건이나 성능, 설명은 생략하도록 하겠다. 필요하면 논문 참조하는 편이 낫다.

### 결과: 부록

원래는 부록에 있는 자료이지만 결과 섹션으로 가져왔다. 

아래 그림에서는 *making* 이라는 단어가 *making...more difficult* 라는 구를 만드는 데 중요한 역할을 하는 것을 보여준다.
<center><img src="/public/img/2019-08-17-Attention Is All You Need/06.png" width="100%" alt="Attention Visaulizations"></center>

여러 개의 attention을 시각화한 자료는 다음 두 그림에서 확인할 수 있다.

<center><img src="/public/img/2019-08-17-Attention Is All You Need/07.png" width="100%" alt="Attention Head Visaulizations 1"></center>

<center><img src="/public/img/2019-08-17-Attention Is All You Need/08.png" width="100%" alt="Attention Head Visaulizations 1"></center>

---

## 7. 결론(Conclusion)

(여러 번 나온 말이지만) **Transformer**는 recurrence와 convolution을 모두 제거한, 오직 attention에만 의존하는 새로운 종류의 모델이다. 이 모델은 계산량을 줄이고 병렬화를 적용해 학습 속도가 훨씬 빠를 뿐만 아니라 그 성능 또한 state-of-the-art를 달성하는 수준에 이르렀다.  
또한 이러한 attention에 기반한 모델은 다른 task들에 적용할 수도 있다. 비단 텍스트뿐만 아니라 이미지, 오디오나 비디오 등의 상대적으로 큰 입력-출력을 요하는 task들에 효과적으로 사용할 수 있을 것이다.

이 모델을 학습하고 평가한 코드는 [여기](https://github.com/tensorflow/tensor2tensor)에서 찾아볼 수 있다.

---

## Refenrences

논문 참조. 40개의 레퍼런스가 있다.

---

