---
layout: post
title: Graph Pooling - gPool, DiffPool, EigenPool 설명
author: Youyoung
categories: [Machine_Learning]
tags: [Machine_Learning, Paper_Review]
---

본 글에서는 총 3가지의 **Graph Pooling**에 대해 핵심 부분을 추려서 정리해 볼 것이다. `gPool`, `DiffPool`, `EigenPool`이 그 주인공들이며, 상세한 내용은 글 하단에 있는 논문 원본 링크를 참조하길 바란다.  

[Github](https://github.com/ocasoyy/pytorch-gnn-research)에 관련 코드 또한 정리해서 업데이트할 예정이다.  

---
# gPool(Graph U-nets) 설명  
graph pooling은 간단하게 말하자면 full graph에서 (목적에 맞게) 중요한 부분을 추출하여 좀 더 작은 그래프를 생성하는 것으로 생각할 수 있다. subgraph를 추출하는 것도 이와 맥락을 같이 하기는 하지만, 보통 subgraph 추출은 그때 그때 학습에 필요한 아주 작은 부분 집합을 추출하는 것에 가깝고, graph pooling은 전체 그래프의 크기와 밀도를 줄이는 과정으로 생각해볼 수 있다.  

단순한 그림 예시는 아래와 같다.  

<center><img src="/public/img/Machine_Learning/2021-09-09-GraphPooling/01.PNG" width="65%"></center>  

locality 정보를 갖고 있는 grid 형식의 2D 이미지를 다룬 CNN과 달리 그래프 데이터에 바로 pooling과 up-sampling을 수행하는 것은 어려운 작업이다. 본 논문에서는 U-net의 구조를 차용하여 그래프 구조의 데이터에 pooling과 up-sampling(unpooling) 과정을 적용하는 방법에 대해 소개하고 있다.  

**pooling** 과정은 `gPool` layer에서 수행되며, 학습 가능한 projection 벡터에 project된 scalar 값에 기반하여 일부 node를 선택하여 작은 그래프를 만드는 것이다. **unpooling** 과정은 `gUnpool` layer에서 수행되며, 기존 `gPool` layer에서 선택된 node의 position 정보를 바탕으로 원본 그래프를 복원하게 된다.  

모든 node feature는 projection을 통해 1D 값으로 변환된다.  

$$ y_i = \frac{\mathbf{x_i} \mathbf{p}}{\Vert \mathbf{p} \Vert} $$  

$\mathbf{x_i}$ 는 node feature 벡터이고, 이 벡터는 학습 가능한 projection 벡터 $\mathbf{p}$ 와의 계산을 통해 하나의 scalar 값으로 변환된다. 이 때 $y_i$ 는 projection 벡터 방향으로 투사되었을 때 node $i$ 의 정보를 얼마나 보존하고 있는지를 측정하게 된다. 연산 후에 이 값이 높은 $k$ 개의 node를 선택하면 **k-max pooling** 과정이 이루어지는 것이다.  

참고로 논문에서 모든 계산은 full graph 기준으로 이루어진다. 계산 과정은 아래와 같다.  

$$ \mathbf{y} = \frac{X^l \mathbf{p}^l}{\Vert \mathbf{p}^l \Vert} $$  

$$ idx = rank(\mathbf{y}, k) $$  

$$ \tilde{\mathbf{y}} = \sigma(\mathbf{y}(idx)) $$  

$$ \tilde{X^l} = X^l(idx, :) $$  

$$ A^{l+1} = A^l (idx, idx) $$  

$$ X^{l+1} = \tilde{X}^l \odot (\tilde{\mathbf{y}} \mathbf{1}_C^T) $$  

$\tilde{X}^l$ 이 $(k, C)$ 의 형상을 가졌고, $\tilde{\mathbf{y}}$ 는 $(k, 1)$ 의 형상을, $\mathbf{1}$ 은 $(C, 1)$ 의 형상을 갖고 있다.  

$(\tilde{\mathbf{y}} \mathbf{1}_C^T)$ 는 아래와 같이 생겼다.  

$$
\begin{bmatrix}
y_1 \dots y_1  \\
\vdots \ddots \vdots \\
y_k \dots y_k
\end{bmatrix}
$$

과정을 그림으로 보면 아래와 같다.  

<center><img src="/public/img/Machine_Learning/2021-09-09-GraphPooling/02.PNG" width="90%"></center>  

`gUnpooling` 과정은 아래와 같이 extracted feature 행렬과 선택된 node의 위치 정보를 바탕으로 graph를 복원하는 방식으로 이루어진다.  

$$ X^{l+1} = distribute(0_{N, C}, X^l, idx) $$  

이렇게 pooling 과정을 진행하다 보면 node 사이의 연결성이 약화되어 중요한 정보 손실이 많이 일어날 수도 있다. 이를 완화하기 위해 논문에서는 Adjacency Matrix를 2번 곱하여 Augmentation 효과를 취한다. 또한 self-loop의 중요성을 강조하기 위해 Adjacency Matrix에 Identity Matrix를 2번 더해주는 스킬이 사용되었다. 2가지 방법 모두 세부 사항을 조금 달리하면서 여러 GNN 논문에서 자주 등장하는 기법이다.  

지금까지 설명한 `gPooling`과 `gUnpooling`, 그리고 GCN을 결합하면 **Graph U-Nets**가 완성된다.  

<center><img src="/public/img/Machine_Learning/2021-09-09-GraphPooling/04.PNG" width="70%"></center>  

이 모델은 node 분류 및 그래프 분류에서 사용될 수도 있으며, task에 따라 graph에서 중요한 정보를 추출하는 등의 목적으로 사용될 수 있다.  

---
# DiffPool(Hierarchical Graph Representation Learning with Differentiable Pooling) 설명  


---
# EigenPool(Graph Convolutional Networks with EigenPooling) 설명  


---
# References  
1) [gPool 논문 원본](https://arxiv.org/pdf/1905.05178.pdf)  
2) [DiffPool 논문 원본](https://arxiv.org/pdf/1806.08804.pdf)  
3) [EigenPool 논문 원본](https://arxiv.org/pdf/1904.13107.pdf)  
