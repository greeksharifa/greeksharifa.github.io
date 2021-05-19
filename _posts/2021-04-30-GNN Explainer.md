---
layout: post
title: GNN Explainer 설명
author: Youyoung
categories: [Machine_Learning]
tags: [Machine_Learning, Paper_Review]
---

본 글에서는 2019년에 발표된 **GNN Explainer: Generating Explanations for Graph Neural Networks**라는 논문에 대한 Review를 진행할 것이다. 본 논문은 GNN의 내부를 파헤쳐 학습 및 예측 결과에 대한 이해 가능한 설명을 결과물로 내놓는 과정에 대해 설명하고 있다. **GNN**에 대한 다른 알고리즘들에 대해 알아보고 싶다면 [GraphSAGE](https://greeksharifa.github.io/machine_learning/2020/12/31/Graph-Sage/), [PinSAGE](https://greeksharifa.github.io/machine_learning/2021/02/21/Pin-Sage/), [GCMC](https://greeksharifa.github.io/machine_learning/2020/12/06/GCMC/) 등을 참조해도 좋을 것이다.  


---
# GNN Explainer: Generating Explanations for Graph Neural Networks 리뷰  
## 1. Introduction  
(전략)  

여러 강점에도 불구하고 GNN의 경우 쉽게 그 예측 값에 대해 인간이 이해할 수 있는 쉬운 설명을 제공하지는 않는다. 그럼에도 불구하고 예측 값을 이해하는 것은 매우 중요한데, 왜냐하면 모델에 대한 믿음을 강화하고 모델의 투명성을 강화하며, 실제 세계에서 이를 적용하는 데 있어 모델이 만들어 낸 구조적인 결함을 수정하고 네트워크의 특성을 파악하는 데에 큰 도움을 주기 때문이다.  

본 논문에서는 GNN에 의해 생성된 예측 값을 설명하는 방법인 `GNN Explainer`에 대해 소개할 것이다. `GNN Explainer`는 학습된 GNN과 그 예측 값을 바탕으로 예측에 있어 가장 큰 영향력을 발휘한 Node Feature의 일부와 Input Graph의 subgraph의 형식으로 **설명**을 제공하게 된다.  

`GNN Explainer`는 GNN이 학습된 전체 그래프의 rich subgraph로서 설명을 구체화하기 때문에 sub-graph는 GNN의 예측 값과 상호적인 정보를 최대화하게 된다. 이는 **Mean Field Variational Appoximation**을 형성하고 GNN의 연산 graph에서 중요한 subgraph를 선택하는 실제 값의 **Graph Mask**를 학습하는 과정을 통해 이루어진다. 동시에 `GNN Explainer`는 중요하지 않은 Node Feature를 걸러내는 **Feature Mask**또한 학습하게 된다.  

## 2. Related Work  

(논문 참고)  

## 3. Formulating explanations for GNN  
기호에 대해 잠시 짚고 넘어가자.  

|기호|설명|
|:--------:|:--------:|
|$G$| Graph |
|$E$| Edge |
|$V$| Node |
|$\mathcal{X}$| d차원의 Node Features |
|$f$| V에 속하는 모든 Node를 C개의 Class 중 하나로 연결시키는 Label 함수 |
|$\Phi$| GNN 모델 |

### 3.1. Background on GNN  
수식에 관해서는 논문 원본을 참조하기 바란다. GNN은 먼저 모든 Node 쌍에 대해 Neural Message를 계산한다. 그리고 각 Node $v_i$ 에 대하여 GNN은 $N_{v_i}$, 즉 $v_i$ 의 이웃 Node들로부터 Message를 통합한다. 이렇게 통합된 메시지는 Parameter Matrix 및 비선형 함수와 함께 계산되어 Update 과정을 거친다. $L$ 개의 Layer 계산이 끝난 Node $v_i$ 의 최종 임베딩은 $\mathbf{z_i} = h_i^L$ 의 형상을 가진다.  

본 논문의 `GNN Explainer`는 앞서 설명한 과정인 Message, Aggregation, Update의 측면에서 구성되었다.  

### 3.2. GNN Explainer: Problem formulation  

<center><img src="/public/img/Machine_Learning/2021-04-30-GNN Explainer/01.JPG" width="80%"></center>  

GNN의 이웃 통합에 의해 정의된 Node $v$ 의 연산 그래프는 Node $v$ 에서의 예측 값 $\hat{y}$ 을 생성하기 위해 GNN이 사용하는 모든 정보를 결정한다는 사실이 본 논문의 핵심 인사이트이다. 즉, $v$ 의 연산 그래프는 GNN이 어떻게 Embedding $\mathbf{z}$ 를 생성하는지 알려주는 셈이다.  

또 기호에 대해 짚고 넘어가겠다.  

|기호|설명|
|:--------:|:--------:|
|$G_c (v)$| 연산 그래프 |
|$A_c (v)$| 인접 행렬 |
|$X_c (v)$| Node Features |
|$P_{\Phi} (Y \vert G_c, X_c)$| Node가 어떤 C 클래스에 속할 확률 |

GNN의 예측 값이라는 것은 결국 $\hat{y} = \Phi (G_c(v), X_c(v))$ 에 의해 결정된다. 즉 위 3가지 요소만 있으면 예측 값이 확정된다는 뜻이고 예측 값을 설명하기 위해서는 $G_c(v), X_c(v)$ 만 고려하면 된다는 뜻이다.  

식으로 설명하면 예측 값을 결정하기 위해 $G_S, X_S^F$ 가 필요한데, 전자의 경우 앞서 설명한 것처럼 연산 그래프의 작은 부분 그래프를 의미하고, $X_S$ 는 $G_S$ 와 관련된 Feature를 의미하며, $X_S^F$ 는 그 Feature 들 중에서 중요한 Feature들을 의미하게 된다.  


## 4. GNN Explainer  
예측 값 집합에 대한 설명을 제공할 때 `GNN Explainer`는 그 집합 속에 존재하는 각 설명들을 통합하여 자동적으로 프로토타입과 함께 요약하게 된다. 본 섹션에서는 `GNN Explainer`가 Link Prediction과 Graph Classification과 같은 ML Task에서 어떻게 사용되는지 소개할 것이다.  

### 4.1. Single-instance Explanations  
Node $v$ 가 주어졌을 때, 우리의 목적은 GNN의 예측값 $\hat{y}$ 을 만드는 데 있어 중요한 subgraph $G_S \subseteq G_c$ 와 관련된 Feature $X_S = \{ x_j \vert v_j \in G_S \}$ 를 찾아내는 데에 있다.  

일단 $X_S$ 는 d차원의 Node Feature를 담고 있는 작은 부분 집합이라고 하자. 추후에 **설명**을 위해 필요한 Node Feature의 차원은 몇 인지 자동적으로 결정하는 법에 대해서도 설명할 것이다.  

우리는 **Importance, 중요도**의 개념을 공통 정보 $M I$ 를 사용하여 표현하고, 아래와 같이 최적화 프레임워크에 따라 `GNN Explainer`를 공식화할 수 있다. (식 1)  

$$ \max_{G_S} M I (Y, (G_S, X_s)) = H(Y) - H(Y \vert G=G_S, X=X_S) $$  

Node $v$ 에 대하여 $M I$ 는 $v$ 의 연산 그래프가 explanation subgraph $G_S$ 로 제한되고, Node Feature가 $X_S$ 로 제한될 때 예측 값 $\hat{y} = \Phi(G_c, X_c)$ 의 확률값의 변화를 측정하게 된다.  

예를 들어, $v_j \in G_c(v_i), v_j \neq v_i$ 의 상황이 있다고 해보자. 
이는 $v_i$ 의 연산그래프에 속해있는 $v_j$ 라는 Node가 존재함을 가정한 것이다. 만약 $v_j$ 를 $v_i$ 의 연산 그래프에서 제거했을 때 예측 값의 확률을 낮춘다면 (0에 가깝게 만든다면) 이 Node $v_j$ 는 $v_i$ 의 예측을 위해 아주 중요한 설명을 제공하는 Node임을 의미한다. 반대의 경우도 마찬가지이다.  

식1을 다시 보자. 우리는 **entropy term** $H(Y)$ 가 상수임을 알 수 있는데, 왜냐하면 GNN을 학습할 때 $\Phi$ 는 고정되어 있기 때문이다. 따라서 결과적으로 예측된 Label 분포 $Y$ 와 explanation $(G_S, X_S)$ 사이의 공통 정보를 최대화하는 것은 결국 아래와 같은 조건부 **entropy**를 최소화하는 것과 같다. (식2)  

$$ H(Y \vert G=G_S, X=X_S) = -\mathbb{E}_{Y \vert G_S, X_S} [log P_{\Phi} (Y \vert G=G_S, X=X_S)] $$  

예측 값 $\hat{y}$ 을 위한 explanation은 결국 GNN 연산 그래프가 $G_S$ 로 한정될 때 발생하는 $\Phi$ 의 불확실성을 최소화하는 $G_S$ 라고 할 수 있다. 즉, $G_S$ 가 $\hat{y}$ 의 확률을 최대화하는 것이다.  

Compact한 explanation을 얻기 위해, 우리는 $G_S$ 의 크기를 $\vert G_S \vert \leq K_M$ 과 같이 제한하여 $G_S$ 가 최대 $K_M$ 개의 Node를 갖도록 하였다. 이는 `GNN Explainer`가 예측에 있어 가장 공통적인 정보를 제공하는 $K_M$ 개의 Edge를 취함으로써 $G_c$의 Noise를 감소시키는 효과를 가져온다.  

**GNN Explainer's optimization framework**  
$G_c$가 $\hat{y}$ 를 위한 진짜 explanation인 subgraph $G_S$ 를 지수 함수적으로 많이 갖게 된다면 `GNN Explainer`의 직접적인 최적화는 어렵다. 따라서 본 논문에서는 subgraph $G_S$ 를 위한 **Fractional Adjacency Matrix**를 고려하였다.  

이는 $A_S \in [0, 1]^{n \times n}$ 로 표현할 수 있으며, 모든 $j, k$ 에 대해 $A_S[j, k] \leq A_c[j, k]$ 라는 subgraph 제한을 두었다. 이는 $C_e$ 가 Edge Type의 수를 의미할 때 $G_S \in [0, 1]^{C_e \times n \times n}$ 와 같은 설정을 적용한 것이다.  

이러한 `Continuous Relaxation`은 $G_c$의 subgraph 분포의 **Variational Approximation**이라고 해석할 수 있다. 특히, 만약 우리가 $G_S \sim \mathcal{G}$ 를 **Random Graph Variable**로 취급한다면 식2의 목적함수는 아래와 같이 바뀌게 된다. (식3)  

$$ \min_{\mathcal{G}} \mathbb{E}_{G_S \sim \mathcal{G}} H(Y \vert G=G_S, X=X_S) $$  

Convexity 가정과 함께, Jensen의 부등식은 아래와 같은 Upper Bound를 제공한다. (식4)  

$$ min_{\mathcal{G}} H(Y \vert G=\mathbb{E}_{\mathbf{G}} [G_S], X=X_S) $$  

실제로는 신경망의 복잡함 때문에 위와 같은 가정은 옳지 않다. 하지만 실험적으로 본 결과, 이 목적함수를 정규화와 함께 최소화한 것이 종종 양질의 explanation을 가능하게 하는 Local Minimum으로 귀결되었다는 것을 알 수 있었다.  

$\mathbb{E}_{\mathcal{G}}$ 를 추정하기 위해서 우리는 **Mean-field Variational Approximation**을 사용하였으며 $\mathcal{G}$ 를 아래와 같이 다변량 베르누이 분포를 통해 분해하였다.  

$$ P_{\mathcal{G}} (G_S) = \prod_{(j, k) \in G_c} A_S[j, k] $$  

이를 통해 **Mean-field Approximation** 에 상응하는 기댓값을 추정할 수 있고, 그 결과 $A_S$ 를 얻을 수 있고, 이 때 $(j, k)$ 번재 원소는 Edge $(v_j, v_k)$ 가 존재하는지에 대한 기댓값을 나타내게 된다.  

우리는 GNN의 Non-convexity에도 불구하고 불연속성을 증대시키기 위해 정규화를 함께 적용한 이러한 근사값이 훌륭한 Local Minima로 수렴한다는 사실을 확인하였다. 

식4에서의 조건부 Entropy는 $\mathbb{E}_{\mathcal{G}}$ 가 인접 행렬의 연산 그래프를 **masking** 함으로써 최적화하는 방법을 취함으로써 최적화될 수 있다. 이를 식으로 나타내면 아래와 같다.  

$$ A_c \odot \sigma(M) $$  

이 때 $M \in \mathbb{R}^{n \times n}$ 은 우리가 학습해야 하는 Mask를 의미한다.  


### 4.2. Joint learning of graph structural and node feature information  


하나의 Label C를 갖는 모든 instance에 대한 subgraph를 탐색



### 4.3. Multi-instance explanations through graph prototypes  


### 4.4. GNN Explainer model extensions  


## 5. Experiments  



## 6. Conclusion  















---
# References  
1) [논문 원본](https://arxiv.org/abs/1903.03894)  


