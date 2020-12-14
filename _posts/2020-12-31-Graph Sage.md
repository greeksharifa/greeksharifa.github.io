---
layout: post
title: Graph Sage (Inductive Representation Learning on Large Graphs) 설명
author: Youyoung
categories: [Machine_Learning]
tags: [Machine_Learning, Recommendation System, Paper_Review]
---

본 글에서는 2017년에 발표된 **Inductive Representation Learning on Large Graphs**란 논문에 대한 Review를 진행할 것이다. 논문 리뷰가 끝난 후에는 [Uber Engineering 블로그 글](https://eng.uber.com/uber-eats-graph-learning/?fbclid=IwAR2Ow1M7gZi2KjL7t2aecLV8-db0Ph3BioJtUOXGLk6s2ekurbLXdOuEUi4)에서 **Graph Sage**를 어떻게 추천 시스템에 적용하였는지에 대해서도 다루도록 하겠다. 

---
# 1. Inductive Representation Learning on Large Graphs Paper Review  
## 1.1. Introduction  
큰 Graph에서 Node의 저차원 벡터 임베딩은 다양한 예측 및 Graph 분석 과제를 위한 Feature Input으로 굉장히 유용하다는 것이 증명되어 왔다. Node 임베딩의 기본적인 아이디어는 Node의 Graph 이웃에 대한 고차원적 정보를 Dense한 벡터 임베딩으로 만드는 차원 축소 기법을 사용하는 것이다. 이러한 Node 임베딩은 Downstream 머신러닝 시스템에 적용될 수 있고 Node 분류, 클러스터링, Link 예측 등의 문제에 도움이 될 수 있다.  

하지만 이전의 연구들은 고정된 단일 Graph로부터 Node를 임베딩하는 것에 집중하였는데 실제 현실에서는 새로운 Node와 (sub) Graph가 빠르게 생성되는 것이 일반적이다. (Youtube를 생각해보라!) 고정된 Graph에서 추론을 행하는 것을 **transductive**한 경우라고 부르고 틀에서 벗어나 새로운 Node에 대해서도 합리적인 추론을 행할 수 있는 경우를 **inductive**라고 부른다. Node 임베딩을 생성하기 위한 **inductive**한 접근은 또한 같은 형태의 feature가 존재할 때 Graph 전체에 대해 **일반화된 결과**를 제공하게 된다.  

이러한 **inductive** Node 임베딩 문제는 굉장히 어렵다. 왜냐하면 지금껏 본 적이 없는 Node에 대해 일반화를 하는 것은 이미 알고리즘이 최적화한 Node 임베딩에 대해 새롭게 관측된 subgraph를 맞추는 (align) 작업이 필요하기 때문이다. **inductive** 프레임워크는 반드시 Node의 Graph 내에서의 지역적인 역할과 글로벌한 위치 모두를 정의할 수 있는 Node의 이웃의 구조적인 특성을 학습해야 한다.  

Node 임베딩을 생성하기 위한 대부분의 기존 접근은 본질적으로 **transductive**하다. 다수의 이러한 접근 방법들은 행렬 분해 기반의 목적함수를 사용하여 각 Node에 대해 임베딩을 직접적으로 최적화하고 관측되지 않은 데이터를 일반화하지 않는다. 왜냐하면 이러한 방법들은 고정된 단일 Graph에 대해 예측하기 때문이다.  

이러한 접근법들은 **inductive**한 방법에서 작동하도록 수정될 수 있는데, 이를 위해서는 굉장한 연산량이 수반되고 새로운 예측이 만들어지기 전에 추가적인 Gradient Descent 단계를 필요로 한다. 최근에는 **Graph Convolution** 연산을 이용한 방법들이 논의되었다. 지금까지는 **Graph Convolutional Network** 또한 **transductive**한 환경에서만 적용되었다. 본 논문에서 우리는 이 GCN을 inductive unsupervised learning으로 확장하고 단순한 합성곱 이상의 학습 가능한 Aggregation 함수를 사용하여 GCN을 일반화하는 프레임워크를 제안할 것이다.  

<center><img src="/public/img/Machine_Learning/2020-12-31-Graph Sage/01.JPG" width="80%"></center>  

본 논문에서는 **Inductive Node Embedding**을 위해 일반화된 프레임워크, `Graph Sage`를 제안한다. 이름은 SAmple과 aggreGatE를 결합하였다. 좀 끼워 맞춘 느낌이 들긴 하다. 행렬 분해에 기반한 임베딩 접근법과 달리 관측되지 않은 Node에서도 일반화할 수 있는 임베딩 함수를 학습하기 위해 Node Feature(텍스트, Node 프로필 정보, Node degree 등)를 Leverage한다. 학습 알고리즘에 Node Feature를 통합함으로써 우리는 이웃한 Node Feature의 분포와 각 Node의 이웃에 대한 위상적인 구조를 동시에 학습할 수 있다. 풍부한 Feature를 가진 Graph에 집중하여 우리의 접근법은 또한 (Node Degree와 같이) 모든 Graph에 존재하는 구조적인 Feature를 활용할 수 있다. 따라서 본 논문의 알고리즘은 Node Feature가 존재하지 않는 Graph에도 적용될 수 있다.  

각 Node에 대한 고유의 임베딩 벡터를 학습하는 대신에, 우리는 Node의 지역 이웃으로부터 Feature 정보를 규합하는 **Aggregator Function**의 집합을 학습한다. 중요한 포인트이다. 왜냐하면 이 컨셉을 통해 각 Node에 귀속된 임베딩 벡터의 한계를 돌파할 수 있기 때문이다. 


## 1.3. Proposed method: GraphSAGE  
우리의 접근법의 가장 중요한 아이디어는 Node의 지역 이웃으로부터 Feature Information을 통합하는 방법에 대해 학습한다는 것이다. 먼저 1.3.1에서는 `GraphSAGE` 모델의 파라미터가 이미 학습되어 있다고 가정하고  `GraphSAGE`의 임베딩 생성 알고리즘에 대해 설명할 것이다. 이후 1.3.2에서는 `Stochastic Gradient Descent`와 `Backpropagation` 기술을 통해 모델이 어떻게 학습되는지 설명할 것이다.  

### 1.3.1. Embedding Generation (i.e. forward propagation) Algorithm  
본 섹션에서는 일단 모델의 파라미터가 모드 학습되었고 고정되어 있다고 가정하고 Embedding Generation 혹은 Propgation 알고리즘에 대해 설명할 것이다. 일단 2종류의 파라미터가 있다.  

첫 번째는 $K$ 개의 **Aggregator Function**으로, $AGGREGATE_k, \forall k \in \{1, ..., K\}$ 라고 표현되며, 이 함수는 Node 이웃으로부터 정보를 통합하는 역할을 수행한다.  

두 번째는 **Weight Matrices**의 집합으로, $\mathbf{W}^k, \forall k \in \{1, ..., K\}$ 라고 표현되며, 이들은 모델의 다른 layer나 *search depth* 사이에서 정보를 전달하는데 사용된다. 다음은 파라미터 학습 과정을 나타낸 것이다.  

<center><img src="/public/img/Machine_Learning/2020-12-31-Graph Sage/02.JPG" width="80%"></center>  

위에서 확인할 수 있는 직관은 각 iteration 혹은 search depth에서 Node는 그의 지역 이웃으로부터 정보들을 모으고, 이러한 과정이 반복되면서 Node는 Graph의 더 깊은 곳으로부터 정보를 증가시키면서 얻게 된다는 것이다.  

알고리즘1은 Graph 구조와 Node Features가 Input으로 주어졌을 때의 임베딩 생성 과정에 대해 기술하고 있다. 아래에서는 Mini-batch 환경에서 어떻게 일반화할 수 있을지 설명할 것이다. 알고리즘1의 바깥 Loop의 각 단계를 살펴보면, $\mathbf{h}^k$ 는 그 단계에서의 Node의 Representation을 의미한다.  

첫 번째로, 각 Node $v$ 는 그것의 바로 이웃(Immediate Neighborhood)에 속하는 Node들의 Representation을 하나의 벡터 $\mathbf{h}_{\mathcal{N}(v)}^{k-1}$ 로 합산한다. 이 때 이 합산 단계는 바깥 Loop의 이전 반복 단계(k-1)에서 생성된 Representation에 의존하고 $k=0$ 일 때는 Input Node Feature가 Base Representation의 역할을 하게 된다.  

이웃한 Feature 벡터들을 모두 통합한 다음, 모델은 Node의 현재 Representation $\mathbf{h}_v^{k-1}$ 과 $\mathbf{h}_{\mathcal{N}(v)}^{k-1}$ 을 쌓은 뒤 비선형 활성화 함수를 통과시킨다. 최종적으로 depth $K$ 에 도달하였을 때의 Representation은 $\mathbf{z}_v = \mathbf{h}_v^K, \forall v \in \mathcal{V}$ 이라고 표현하겠다.  

사실 **Aggregator Function**은 다양하게 변형될 수 있으며, 여러 방법에 대해서는 1.3.3에서 다루도록 하겠다.  

알고리즘1을 Mini-batch 환경으로 확장하기 위해서는 우리는 먼저 depth $K$ 까지 필요한 이웃 집합을 추출해야 한다. 이후 안쪽 Loop(알고리즘1의 3번째 줄)를 실행하는데, 이 때 모든 Node를 반복하는 것이 아니라 각 depth에서의 반복(recursion)을 만족하는데 필요한 Represention에 대해서만 계산한다. (Appendix A 참조)  

**Relation to the Weisfeiler-Lehman Isomorphism Test**  
`GraphSAGE` 알고리즘은 개념적으로 Graph Isomorphism(동형 이성)을 테스트하는 고전적일 알고리즘에서 영감을 얻어 만들어졌다. 만약 위에서 확인한 알고리즘1에서 $K= \vert \mathcal{V} \vert$ 로 세팅하고 **Weight Matrices**를 단위 행렬로 설정하고 비선형적이지 않은 적절한 Hash 함수를 Aggregator로 사용한다면, 알고리즘1은 `Naive Vertex Refinement`라고 불리는 **Weisfeiler-Lehman: WL Isomorphism Test**의 Instace라고 생각할 수 있다.  

이 테스트는 몇몇 경우에는 들어맞지 않지만, 여러 종류의 Graph에서는 유효하다. `GraphSAGE`는 Hash 함수를 학습 가능한 신경망 Aggregator로 대체한 WL Test의 연속형 근사에 해당한다. 물론 `GraphSAGE`는 Graph Isomorphism을 테스트하기 위해서 만들어진 것이 아니라 유용한 Node 표현을 생성하기 위함이다. 그럼에도 불구하고 `GraphSAGE`와 고전적인 WL Test 사이의 연결성은 Node 이웃들의 위상적인 구조를 학습하기 위한 본 알고리즘의 디자인의 기저에 깔려 있는 이론적 문맥을 이해하는 데에 있어 큰 도움을 준다.  

**Neighborhood Definition**  
본 연구에서 우리는 알고리즘1에서 기술한 것처럼 모든 이웃 집합을 사용하지 않고 고정된 크기의 이웃 집합을 샘플링하여 사용하였다. 이렇게 함으로써 각 Batch의 계산량을 동일하게 유지할 수 있었다.  

다시 말해, $\mathcal{N}(v)$ 을 집합 $\{u \in \mathcal{V}: (u, v) \in \mathcal{E}\}$ 에서 각 반복 $k$ 에서 고정된 크기로 균일하게 뽑은 Sample이라고 정의할 수 있다.  

이러한 샘플링 과정이 없으며, 각 Batch의 메모리와 실행 시간은 예측하기 힘들며 계산량 또한 엄청나다.  

`GraphSAGE`의 per-batch space와 time complexity는 $O(\prod_{i=1}^K S_i)$ 로 고정되어 있으며, $S_i, i \in \{1, ..., K\}$와 $K$ 는 User-specified 상수이다.  

실제 적용할 때, $K=2$ 와 $S_1 * S_2 <= 500$ 으로 했을 때 좋은 성능을 보였다. (자세한 사항은 1.4.4를 참조)  

### 1.3.2. Learning the Paremeters of GraphSAGE  







---
# Reference  
1) [논문 원본](https://arxiv.org/abs/1706.02216)  
