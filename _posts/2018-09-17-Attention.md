---
layout: post
title: Attention
author: Youyoung
categories: Paper_Review
tags: [NLP, Paper_Review]
---

### Neural Machine Translation by Jointly Learning to Align and Translate  
> 본 글은 Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio가 2014년에 Publish한 위 논문을 리뷰한 것이다.  
  

### Introduction  
Basic Encoder-Decoder 모델은 source sentence의 모든 정보를 fixed-length vector로 압축하는 방식을 골자로 한다.  
그러나 이 모델은 긴 문장을 대상으로 할 때 어려움을 겪는 것이 일반적이다.  
이를 해결하기 위해 본 논문에서 제안된 새로운 모델은 **target word**를 예측하는 것과 관련된 source sentence의 부분을 자동적으로 soft-search하여 이와 같은 문제를 해결해낸다.  
(learns to align and translate jointly)  
  
> encodes the input sentence into sequence of vectors and  
> chooses a subset of these vectors adaptively while decoding the translation.  


### Background: NMT  
translation의 핵심은 source sentence **x**가 주어졌을 때의 **y**의 조건부확률을 최대화하는 target sentence **y**를 찾는 것이다.  
  
$$ arg\max_{y} p(\vec{y}|x) $$  
  
번역 모델에 의해 이러한 조건부 분포가 학습되면, source sentence가 주어졌을 때 상응하는 번역된 문장은 위의 조건부 확률을 최대화하는 문장을 찾음으로써 generate된다.  

*RNN Encoder-Decoder*는 이전 리뷰에서 다룬 적이 있으므로 생략하도록 한다.  
  

### Learning to align and translate  
모델의 구성 요소를 하나하나 살펴보기 이전에 notation에 관한 정리를 진행하겠다.  
  
$y_i$: i time에서의 target word  
$s_i$: i time에서의 디코더 Hidden State  
$c_i$: i time에서의 context vector = annotations의 가중합  
$\alpha_{ij}$: attention weight = normalized score = 연결 확률  
$h_j$: j time에서의 인코더 Hidden State = annotations  
$e_{ij}$: attention score = unnormalized score  
$f, g$ = 비선형 함수  

명확히 하자면, subscript **i**는 디코더를 명시하며, subscript **j**는 인코더를 명시한다.  
  
이제 모델의 구성 요소를 살펴볼 것이다.  
먼거 타겟 word $y_i$를 예측하기 위한 조건부 확률은 아래와 같이 정의된다.  

$$ p(y_i|y_1, ..., y_{i-1}, \vec{x}) = g(y_{i-1}, s_i, c_i) $$  
  
이 중 디코더의 i time Hidden State인 $s_i$를 먼저 살펴보면,  

$$ s_i = f(s_{i-1}, y_{i-1}, c_i) $$  
  
Basic Encoder-Decoder 모델과 달리 target word를 예측하기 위한 조건부 확률은 분리된 context vector $c_i$에 의존한다.  
  
**Context Vector** $c_i$는 annotations $h_j$의 가중합이다.  

$$ c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j $$  
  
여기서 $h_j$는 j time annotation으로, input sequence의 i번째 단어 주위 부분에 강하게 집중하여 input sequence에 대한 정보를 담게 된다.  


*Bidirectional RNN*  
이 $h_j$는 forward RNN의 Hidden States와 backward RNN의 Hidden States를 세로로 합친 열벡터이다.  

$$ h_j = [\overrightarrow{h_j}^T | \overleftarrow{h_j}^T]^T $$  
  
이러한 방식으로 $h_j$는 두 방향 모두로 words들을 요약한 정보를 담게 된다.  

이제 **attention weight** $a_{ij}$가 어떻게 계산되는지 살펴보겠다.  

$$ a_{ij} = \frac{ exp(e_{ij}) } {\sum_{k=1}^{T_x} exp(e_{ik}) } $$  
  
이 $a_{ij}$는 **Normalized Score**라고 할 수 도 있다. 왜냐하면 softmax함수의 확률로서 계산되기 때문이다.  
  
**Unnormalized Score**인 $e_{ij}$는 아래와 같이 계산된다.  

$$ e_{ij} = a(s_{i-1}, h_j) $$  
  
여기서 a함수는 **alignment model**이다. 이 a를 다른 component와 함께 학습되는 순전파 신경망으로서 모수화한다.  
이 alignment model은 j time 인풋이 i time 아웃풋과 얼마나 유사한지를 평가하게 된다.  
또한 이 모델은 잠재변수로 설정되지 않고, soft alignment를 직접적으로 계산하여 cost function의 gradient가 역전파될 수 있도록 하게 만든다.  
계산 방법은 마지막 부분에서 설명하도록 하겠다.  
  

위 설명을 보면, 결국 i번째 **Context Vector**인 $c_i$는 expected annotation over all the annotations with probabilities $\alpha_{ij}$라고 할 수 있다.  
이 $\alpha_{ij}$는 다음 Hidden State인 $s_i$를 결정하고 target word $y_i$를 generate하는 데에 있어 $h_j$의 중요성을 결정하는 역할을 하게 된다.  
  
즉 이는 일종의 **attention**이라는 개념으로 설명될 수 있다.  
디코더는 source sentence의 어떤 부분에 **집중**해야 하는지 결정하게 되는 것이다.  
  

### Conclusion  
제안된 모델은 다음 target word를 generate하는 데에 관련이 있는 정보에만 집중하며 이 때문에 source sentence의 길이에 상당히 robust하다.  
다만 unknown or rare words를 다루는 데 있어서는 좀 더 보완이 필요하다.  
  

### Details about Model Architecture
이 부분에서는 Appendix에 나와 있는 수식들을 종합하여, 본 논문에서 제안한 RNNSearch라는 모델의 구조에 대해 정리하도록 하겠다.  
논문이 굉장히 친절하여 Matrix의 차원이 정확하고 자세하게 나와있으므로 반드시 참고할 필요가 있다.  
  
1. 상수에 대한 설명은 아래와 같다.  
$m$: word embedding 차원, 본 모델에선 620   
$n$: 인코더/디코더 Hidden Units의 수, 본 모델에선 1000  
$n'$: Alignment Model 내에서의 Hidden Units의 수, 본 모델에선 1000  
$l$: , 본 모델에선 500  
$T_x$: source sentence의 길이  
$K_x$: source language의 vocab_size  
  

2. 벡터들의 크기는 아래와 같다.  
$y_i$: (k, 1)  
$s_i$: (n, 1)  
$h_i$: (n, 1)  
$v_a$: (n', 1)  
$z_i$: (n, 1)  
$r_i$: (n, 1)  


3. 행렬들의 크기는 아래와 같다. W, U, C는 모두 Parameter Matrix이다.  
$X$: ($T_x$, $K_x$)  
$Y$: ($T_y$, $K_y$)  
$E$: (m, K), x와 결합할 때는 $K_x$, y와 결합할 때는 $K_y$  
$W$: (n, m), $W, W_z, W_r$에 한정  
$W_a$: (n', n), Alignment 모델에서 사용  
$U$: (n, n), $U, U_z, U_r$에 한정  
$U_a$: (n', 2n), Alignment 모델에서 사용  
$C$: (n, 2n), $C, C_z, C_r$에 한정  
  
**Encoder**  
source sentence Matrix **X**는 번역 대상인 하나의 문장을 뜻한다.  
각각의 열벡터는 $\vec{x_i}$로 표기되며 이 벡터의 크기는 $K_x$로,  
source language의 vocab_size를 의미한다.  

인코더의 Bidirectional RNN은 아래와 같이 계산된다.  
(Bias항은 잠시 삭제한다.)  
  
$$ h_j = (1 - z_i) \odot h_{j-1} + z_i \odot \tilde{h_j} $$  

위에서 $z_i$가 Update Gate이며, 각 Hidden State가 이전 activation을 유지하느냐 마느냐를 결정한다.  

$$ \tilde{h_j} = tanh(W*Ex_j + U[r_j \odot h_{j-1}]) $$  

위에서 $r_j$가 Reset Gate이며, 이전 State의 정보를 얼마나 Reset할지 결정한다.  

$$ z_j = \sigma(W_z * Ex_j + U_z * h_{j-1}) $$  

$$ r_j = \sigma(W_r * Er_j + U_x * h_{j-1}) $$  
  
위에서 계산한 식은 $\overrightarrow{h_j}$, $\overleftarrow{h_j}$ 모두에게 통용되며,  
이를 stack하여 annotation $h_j$를 만들게 되는 것이다.  
  
  

**Decoder**  
디코더의 Hidden State인 $s_i$ 역시 계산 방식은 유사하다.  
  
$$ s_i = (1 - z_i) \odot s_{i-1} + z_i \odot \tilde{s_i} $$  

$$ \tilde{s_i} = tanh(W*Ex_i + U[r_i \odot s_{i-1}] + C*c_i) $$  

$$ z_i = \sigma(W_z * Ex_i + U_z * s_{i-1} + C_zc_i) $$  

$$ r_i = \sigma(W_r * Er_j + U_x * s_{i-1} + C_rc_i) $$  
  
$$ c_i = \sum_{j=1}^{T_x}a_{ij}h_j $$  

$$ a_{ij} = \frac{ exp(e_{ij}) } {\sum_{k=1}^{T_x} exp(e_{ik}) } $$  
  
$$ e_{ij} = v_a^T tanh(W_a * s_{i-1} + U_a * h_j) $$  
  
최종적으로 Decoder State $s_{i-1}$, Context Vector $c_i$, 마지막 generated word $y_{i-1}$을 기반으로, target word $y_i$의 확률을 아래와 같이 정의한다.  
  
$$ p(y_i|s_i, y_{i-1}, c_i) \propto exp(y_i^T W_o t_i) $$  

즉 오른쪽 편에 있는 스칼라값에 정비례한다는 뜻이다.  
잠시 행렬의 차원을 정의하고 진행하겠다.  

$W_o$: ($K_y$, $l$)  
$U_o$: ($2l$, n)  
$V_o$: ($2l$, m)  
$C_o$: ($2l$, 2n)  
이들은 모두 Parameter이다.  

이제 $t_i$를 정의할 것인데, 그 전에 두 배 크기인 candidate $\tilde{t_i}$를 먼저 정의하겠다.  

$$ \tilde{t_i} = U_o * s_{i-1} + V_o * Ey_{i-1} + C_oc_i $$  
  
차원을 맞춰보면 위 벡터는 크기가 ($2l$, 1)인 것을 알 수 있을 것이다.  
이제 이 벡터에서 아래와 같은 maxout과정을 거치면,  

<center><img src="/public/img/Paper_Review/2018-09-27-Attention/a1.png" width="50%"></center>

$t_i$는 아래와 같이 정의된다.  

$$ t_i = [ max(\tilde{t_{i, 2j-1}}, \tilde{t_{i, 2j}}) ]_{j=1, ..., l}^T $$  
  
아주 멋지다.  
**The End**  