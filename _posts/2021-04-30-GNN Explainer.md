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

본 논문에서는 GNN에 의해 생성된 예측 값을 설명하는 방법인 `GNN Explainer`에 대해 소개할 것이다. `GNN Explainer`는 학습된 GNN과 그 예측 값을 바탕으로 예측에 있어 가장 큰 영향력을 발휘한 Node Feature의 일부와 Input Graph의 Sub-graph의 형식으로 **설명**을 제공하게 된다.  

`GNN Explainer`는 GNN이 학습된 전체 그래프의 rich sub-graph로서 설명을 구체화하기 때문에 sub-graph는 GNN의 예측 값과 상호적인 정보를 최대화하게 된다. 이는 **Mean Field Variational Appoximation**을 형성하고 GNN의 연산 graph에서 중요한 sub-graph를 선택하는 실제 값의 **Graph Mask**를 학습하는 과정을 통해 이루어진다. 동시에 `GNN Explainer`는 중요하지 않은 Node Feature를 걸러내는 **Feature Mask**또한 학습하게 된다.  

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
















---
# References  
1) [논문 원본](https://arxiv.org/abs/1903.03894)  


