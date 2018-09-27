---
layout: post
title: Attention
author: Youyoung
categories: Paper_Review
tags: [NLP, Paper_Review]
---

### Neural Machine Translation by Jointly Learning to Align and Translate  
> 본 글은 Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio가 2014년에 Publish한 위 논문을 리뷰한 것이다.  
  

**Introduction**  
Basic Encoder-Decoder 모델은 source sentence의 모든 정보를 fixed-length vector로 압축하는 방식을 골자로 한다.  
그러나 이 모델은 긴 문장을 대상으로 할 때 어려움을 겪는 것이 일반적이다.  
이를 해결하기 위해 본 논문에서 제안된 새로운 모델은 **target word**를 예측하는 것과 관련된 source sentence의 부분을 자동적으로 soft-search하여 이와 같은 문제를 해결해낸다.  
(learns to align and translate jointly)  
  
> encodes the input sentence into sequence of vectors and  
> chooses a subset of these vectors adaptively while decoding the translation.  


**Background: NMT**  
translation의 핵심은 source sentence **x**가 주어졌을 때의 **y**의 조건부확률을 최대화하는 target sentence **y**를 찾는 것이다.  
  
$$ arg\max_{y} p(\vec{y}|x) $$  
  
번역 모델에 의해 이러한 조건부 분포가 학습되면, source sentence가 주어졌을 때 상응하는 번역된 문장은 위의 조건부 확률을 최대화하는 문장을 찾음으로써 generate된다.  

*RNN Encoder-Decoder*는 이전 리뷰에서 다룬 적이 있으므로 생략하도록 한다.  
  

**Learning to align and translate**  
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
계산 방법은

### Appendix 부분 집어넣기

위 설명을 보면, 결국 i번째 **Contet Vector**인 $ c_i $는 expected annotation ovr all the annotations with probabilities $ \alpha_{ij} $라고 할 수 있다.  
이 $ \alpha_{ij} $는 다음 Hidden State인 $ s_i $를 결정하고 target word $ y_i $를 generate하는 데에 있어 $ h_j $의 중요성을 결정하는 역할을 하게 된다.  
  
즉 이는 일종의 **attention**이라는 개념으로 설명될 수 있다.  
디코더는 source sentence의 어떤 부분에 **집중**해야 하는지 결정하게 되는 것이다.  
  

**Conclustion**  
제안된 모델은 다음 target word를 generate하는 데에 관련이 있는 정보에만 집중하며 이 때문에 source sentence의 길이에 상당히 robust하다.  
다만 unknown or rare words를 다루는 데 있어서는 좀 더 보완이 필요하다.  




