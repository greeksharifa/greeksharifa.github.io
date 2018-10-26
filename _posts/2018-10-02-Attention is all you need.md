---
layout: post
title: Attention is all you need
author: Youyoung
categories: Paper_Review
tags: [NLP, Paper_Review]
---

### Attention is all you need
> 본 글은 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin가 2017년에 Publish한 위 논문을 간략히 리뷰한 것이다.  
  

### Introduction  
본 논문은 RNN이나 CNN 구조를 차용하지 않고 오직 attention만을 이용한 **Transformer**라는 새로운 모델을 제시한다.  
  
### Model Architecture  
Encoder는 동일한 6개의 layer로 이루어져있고 각각의 layer는 2개의 sub-layer를 가진다.  
sub-layer는 Multi-Head Attention과 Feed-forward Network로 이루어져있고, 두 layer의 아웃풋은  
이전 단계의 output(Residual Connection)과 결합하여 **Add & Normalization** layer를 거치게 된다.  

Decoder 역시 이와 구조가 비슷하지만, 다만 인코더와 달리 순차적으로 결과를 만들어내야 하므로,  
**Masking** 기법을 사용한다. 이 기법을 통해 position i보다 이후에 있는 position에 attention을 주지 않게 된다.  
즉, position i에 대한 예측은 이전 output들에게만 의존하는 것이다.  

Attention Function은 query와 key-value paris를 output에 mapping하는 기능으로 이해할 수 있는데,  
이 output은 일종의 value의 가중합으로 생각할 수 있다. 왜냐하면 각 value에 해당하는 weight이 query와 key의
compatibility(호환) function으로 계산되기 때문이다. 이 function은 query와 key의 유사도(Similiarity)를 계산한다.  

논문 중반부에 등장하는 여러 방법론에 대해서는 자세히 설명하진 않겠지만,  
핵심적인 아이디어 중 하나인 Scaled Dot-product Attention과 Multi-Head Attention에 관해서는 언급을 하겠다.  

**Scaled Dot-Product Attention**  
Dot-product Attention에서 사실 Scaling만 해준 Case이다.  
  
먼저 Matrix를 정의하고 시작하면,  

Matirx Notation | Description
---------       | ---------
Q               | Query, (Q 행 수, $d_k$)
K               | Key,   (K 행 수, $d_k$)
V               | Value, (K 행 수, $d_v$)

Encoder-Decoder 시스템에서 Query는 이전 디코더 layer의 아웃풋이고,  
Key와 Value는 인코더 아웃풋으로 생각하면 된다.  
Self-Attention의 경우 Q=K=V이다.  (Self-Attention 리뷰 참조)  

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V  $$  
  
Q라는 Matrix를 구성하는 q라는 query vector가 있다고 할 때,  
그 q와 가장 유사한 K 내부의 key vector를 찾는 것이 dot-product의 역할이다.  
수많은 key들 중 q와 가장 유사한 key를 담았다면, 이에 상응하는 value를 찾아내어  
Matrix Multiplication을 해주면 Attention을 구할 수 있다.  
중간에 scaling을 재주지 않으면 연산량이 크게 증가하게 된다.  


**Multi-Head Attention**  

$$ MultiHead(Q, K, V) = Concat[head_1, ..., head_h] W^O $$  

여기서  

$$ head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V) $$

Matrix의 차원은 아래와 같다.  

Matirx Notation | Description
---------       | ---------
W_i^Q           | ($d_{model}$, $d_k$)
W_i^K           | ($d_{model}$, $d_k$)
W_i^V           | ($d_{model}$, $d_v$)
W_i^O           | ($h * d_v$, $d_{model}$)

여기서 $d_k = d_v = d_{model} / h = 64$이고,  
$h=8, d_{model}=512$ 이다.





