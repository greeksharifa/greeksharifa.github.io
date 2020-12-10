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

하지만 이전의 연구들은 고정된 단일 Graph로부터 Node를 임베딩하는 것에 집중하였는데 실제 현실에서는 새로운 Node와 (sub) Graph가 빠르게 생성되는 것이 일반적이다. (Youtube를 생각해보라!) 고정된 Graph에서 추론을 행하는 것을 `transductive`한 경우라고 부르고 틀에서 벗어나 새로운 Node에 대해서도 합리적인 추론을 행할 수 있는 경우를 `inductive`라고 부른다. Node 임베딩을 생성하기 위한 `inductive`한 접근은 또한 같은 형태의 feature가 존재할 때 Graph 전체에 대해 **일반화된 결과**를 제공하게 된다.  

이러한 `inductive` Node 임베딩 문제는 굉장히 어렵다. 왜냐하면 지금껏 본 적이 없는 Node에 대해 일반화를 하는 것은 이미 알고리즘이 최적화한 Node 임베딩에 대해 새롭게 관측된 subgraph를 맞추는 (align) 작업이 필요하기 때문이다. `inductive` 프레임워크는 반드시 Node의 Graph 내에서의 지역적인 역할과 글로벌한 위치 모두를 정의할 수 있는 Node의 이웃의 구조적인 특성을 학습해야 한다.  

Node 임베딩을 생성하기 위한 대부분의 기존 접근은 본질적으로 `transductive`하다. 다수의 이러한 접근 방법들은 행렬 분해 기반의 목적함수를 사용하여 각 Node에 대해 임베딩을 직접적으로 최적화하고 관측되지 않은 데이터를 일반화하지 않는다. 왜냐하면 이러한 방법들은 고정된 단일 Graph에 대해 예측하기 때문이다.  

이러한 접근법들은 `inductive`한 방법에서 작동하도록 수정될 수 있는데, 이를 위해서는 굉장한 연산량이 수반되고 새로운 예측이 만들어지기 전에 추가적인 Gradient Descent 단계를 필요로 한다. 최근에는 **Graph Convolution** 연산을 이용한 방법들이 논의되었다. 지금까지는 **Graph Convolutional Network** 또한 `transductive`한 환경에서만 적용되었다. 본 논문에서 우리는 이 GCN을 inductive unsupervised learning으로 확장하고 단순한 합성곱 이상의 학습 가능한 Aggregation 함수를 사용하여 GCN을 일반화하는 프레임워크를 제안할 것이다.  

<center><img src="/public/img/Machine_Learning/2020-12-31-Graph Sage/01.JPG" width="80%"></center>  

본 논문에서는 **Inductive Node Embedding**을 위해 일반화된 프레임워크, **Graph Sage**를 제안한다. 이름은 SAmple과 aggreGatE를 결합하였다. 좀 끼워 맞춘 느낌이 들긴 하다. 




---
# Reference  
1) [논문 원본](https://arxiv.org/abs/1706.02216)  
