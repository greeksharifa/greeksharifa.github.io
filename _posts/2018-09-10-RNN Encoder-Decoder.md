---
layout: post
title: RNN Encoder-Decoder
author: Youyoung
categories: Paper_Review
tags: [RNN, Paper_Review]
---

### Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation  
> 본 글은 Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio가 2014년에 Publish한 위 논문을 리뷰한 것이다.  
  

**Abstract**  
Encoder의 역할은 a sequence of symbols를 고정된 길이의 vector representation으로 나타내는 것이고, Decoder의 역할은 그 representation을 다시 sequence of symbols로 나타내는 것이다.  
  
두 모델은 jointly train되는데, 그 목적은 source sequence가 주어졌을 때,  
target sequence가 나타날 조건부 확률을 최대화하는 것이다.  
= (Maximize the conditional probability of a target sequence given a source sequence.)

  
**Introduction**  
본 논문은 SMT(Statistical Machine Translation)의 영역에서 모델을 분석하는 것을 주요 목적으로 하고 있다.  
  
연구자들은 memor capacity와 학습 효율을 향상시키기 위해 다소 복잡한(sophiscated) Hidden Unit을 사용하였으며, 영어를 프랑스어로 번역하는 과정을 통해 그 성능을 평가하였다.  
  
본 논문에서 제안한 RNN Encoder의 경우 phrase table에서 언어학적 규칙(regularities)을 잡아내는 역할을 수행하였으며 이는 사실 전반적인 번역 성능의 향상을 꾀하는 과정의 일부로 평가된다. Decoder의 경우 phrase의 continuous space representation을 학습하는데, 이는 phrase의 의미론적(semantic)이고 통사론적(syntactic) 구조를 저장하는 역할을 수행한다.  

**RNN Encoder-Decoder**  
기본적인 RNN은 sequence에서 다음 symbol을 예측하는 방향으로 학습됨으로써 sequence의 확률 분포를 학습한다.  
이 때 시간t에 대한 output은 $ p(x_t | x_{t-1}, ..., x_1) $와 같은 조건부 확률로 표현된다.  
따라서 sequence **x**의 확률은 아래와 같이 표현할 수 있다.  

$$ p(x) = \prod_{t=1}^T p(x_t | x_{t-1}, ..., x_1) $$  

(위 $p(x)$의 x는 **x** vector이며, 위와 같은 식으로 표현되는 이유는 곱셈정리에 의한 것임)  
이전의 symbol들에 근거하여 다음 symbol을 예측한다고 볼 수 있다.  
  
본 논문에서 제안하는 RNN Encoder-Decoder 모델은 다소 새로운 구조를 갖고 있다. Encoder는 input sequence **x**의 원소 symbol들을 연속적으로 읽어 들인다.  
(reads each symbol of an input sequence **x** sequentially)  
  
이 과정 속에서 시간 t의 hidden state는 아래와 같이 업데이트 된다.
  
$$h_{<t>} = f(h_{<t-1>}, x_t)$$  
  
즉, 이전 hidden state와 시간t의 새로운 input $x_t$에 의해 업데이트 되는 것이다.  
모든 reading이 끝나고 나서 나면 RNN의 hidden state는 모든 input sequence에 대한 summary **c**이다.  

<center><img src="/public/img/Paper_Review/2018-09-10-RNN-Encoder-Decoder/r1.jpg" width="50%"></center>  

Decoder는 주어진 hidden state $h_{<t-1>}$을 바탕으로  
다음 symbol $ y_{<t>} $를 예측함으로써 output sequence를 생성하도록 학습된다.  
  
다만 여기서 주목할 점은, 기본 RNN과 달리 새로운 hidden state는 summary **c**와  
이전 output symbol $ y_{t-1} $에도 conditioned 되어 있다는 것이다.  
즉 아래와 같이 표현될 수 있다.  
  
$$ h_{<t>} = f(h_{<t-1>}, y_{t-1}, c) $$
  
다음 symbol의 조건부 확률 분포는 아래와 같이 나타낼 수 있다.  
  
$$ P(y_t | y_{t-1}, ..., y_1, c) = g(h_{<t>}, y_{t-1}, c)$$  
  
정리하자면, RNN Encoder-Decoder의 두 성분은 아래의 조건부 로그 가능도를 최대화하도록 결합하여 학습된다.  
  
$$ \max_{\theta} {1 \over N} \sum_{n=1}^N log p_{\theta} (y_n|x_n) $$  
  
여기서 $\theta$는 모델 parameter의 집합을 의미하며,  
$y_n$은 output sequence를 $x_n$은 input sequence를 의미한다.  
  
이렇게 학습된 RNN Encoder-Decoder는 크게 2가지 방법으로 활용될 수 있다.  
1. 주어진 input sequence에 대해 새로운 target sequence를 생성할 수 있다.  
2. input & output sequences 쌍의 적합성을 평가할 수 있다. (Score)  
  
2.3절인 *Hidden Unit that Adatively Remembers and Forgets*부분은 LSTM Unit의 기본 형식을 따르고 있기 때문에 식에 대한 리뷰는 생략하겠다. 다만 사용된 용어만을 살펴 보면 아래과 같다.  
  
Reset Gate $r_j$, Update Gate $z_j$, Proposed Unit $h_j^{t}$  
  
효과에 대해 설명하자면,  
1. Reset Gate가 0에 가까워질 때, 시간 t의 **Candidate hidden state**는 이전 hidden state를 잊고(ignore, forget) 시간 t의 현재 input x로 Reset하게 된다.  
2. Updated Gate는 이전(시간 t-1) hidden state의 정보가 얼마나 현재(시간 t) hidden state에 영향을 줄 것인가를 결정한다. 이를 통해 lont-term 정보를 효과적으로 보존(rembember)한다.  
3. 각각의 hidden unit은 Reset/Update Gate를 각각 갖고 있기 때문에, different time scales에 나타나는 종속성(dependencies)를 포착(capture)하는 법을 학습하게 된다. short-term dependencies를 포착하는 법을 학습한 unit들의 Reset Gate는 자주 활성화 될 것이며, long-term dependencies를 포착하는 법을 학습한 unit들의 Update Gate는 자주 활성화될 것이다.  

**Statistical Machine Translation: SMT**  
앞에서도 설명하였듯이 SMT의 기본 목표는 주어진 source sentence에 대하여 translation을 찾는 것이고, 식으로 표현하자면 아래와 같다.  

$$ p(f|e) \propto p(e|f) * p(f) $$

일단 Phrase Pairs를 평가하는 측면에서 본 모델을 살펴보도록 하겠다.  
RNN Encoder-Decoder를 학습시킬 때, 기존 말뭉치들에서 각각의 phrase pair들의 출현 빈도는 고려하지 않는다.  
그 이유는 2가지로 풀이 된다.  
1. 계산량 감소를 위해서이다.   
2. 본 모델이 단순히 출현빈도 Rank에 영향을 받지 않게 하기 위함이다.  

사실 phrase 속에 존재하는 translation probability는 이미 원래 corpus의 phrase pairs의 출현 빈도를 반영한다. 모델이 학습 과정 속에서 이러한 출현 빈도에 따른 어떤 규칙을 학습하는 것이 아니라, 언어학적 규칙(linguistic regularities)을 학습하도록 하는 것이 핵심이라고 할 수 있다.  
(learning the manifold of plausible translations)  

**Experiments**  
대규모의 데이터가 구축되었지만 실제 학습을 위해서 본 논문의 저자는 source & target vocab을 가장 자주 등장한 15,000개의 단어로 한정하였다. 이는 전체 데이터셋의 93%를 커버한다고 밝혔다.  

학습과정에 대한 자세한 사항은 논문을 참조하기 바란다.  

**Conclusion**  
결과적으로 RNN Encoder-Decoder는 phrase pairs 내에 있는 언어학적 규칙을 포착하고, 적절하게 구성된 target phrases 또한 제안하는 데에도 좋은 성능을 보이는 것으로 확인되었다. 

**Structure of Encoder**  
source phrase X와 Y의 형태는 아래와 같다.  
  
$$ X = (x_1, x_2, ... , x_N), Y = (y_1, y_2, ... , y_N) $$  
  
X.shape = (N, K), Y.shape = (N, K)  
  
물론 여기서 각 세로 벡터는 one-hot vector이다.  
source phrase의 각 단어는 500차원의 벡터로 임베딩된다.  
Encoder의 Hidden state는 1000개의 unit을 갖고 있다.  
(시간 t의 hidden state $h_{<t-1>}$의 shape = (1000, 1))  
  
위 hidden state가 계산되는 과정을 살펴보면,  
1. Reset Gate  
$$ r = \sigma(W_r e(x_t) + U_r h_{<t-1>}) $$  
  
(1000, 1) = (1000, 500) X (500, 1) + (1000, 1000) X (1000, 1)  
  
2. Update Gate  
$$ z = \sigma(W_z e(x_t) + U_z h_{<t-1>}) $$  
  
shape은 위와 같다.
  
3. Candidate  
$$ \tilde{h}^{<t>} = tanh(W e(x_t) + U(r \odot h_{<t-1>} )) $$  
  
(1000, 1) = (1000, 500)X(500, 1) + (1000, 1000)X(1000, 1)$\odot$(1000, 1)  
  

4. Hidden State  
$$ h^{<t>} = z h^{<t-1>} + (1-z) \tilde{h}^{<t>}$$  
  

5. Representatino of the source phrase: 농축된 정보  
$$ c = tanh(V h^{<t>}) $$
  

**Structure of Decoder**  
Soon to be updated



