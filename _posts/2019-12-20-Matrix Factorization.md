---
layout: post
title: 추천 시스템의 기본 - 02. Matrix Factorization 논문 리뷰
author: Youyoung
categories: Machine_Learning
tags: [Machine_Learning, Recommendation System, Matrix Factorization, Latent Factor Collaborative Filtering]
---
본 글은 2009년에 발표된 **Matrix Factorization Techniques for Recommender Systems** 논문을 리뷰하고 간단히 요약 정리한 글이다. 논문 원본은 [이곳](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)에서 다운 받을 수 있다.  

---
## 1. Introduction  
`컨텐츠 기반 필터링`은 각 사용자나 아이템에 대해 프로필을 만들고, 그 특성을 구체화하는 방식으로 이루어진다. 반면 위 방식의 대안이라고 할 수 있는 `협업 필터링`은 어떤 명시적(Explicit) 프로필을 만들지 않고, 이전 구매 기록이나 제품 평가 기록 등 과거 사용자 행동에만 의존해서 시스템을 구성한다. 이 방식은 유저-아이템 간의 상관관계를 찾아내는 것이 주 목적이라고 할 수 있다.  

`협업 필터링`은 **Domain-free** 즉, 특별히 이 분야에 대한 지식이 필요하지 않다는 장점을 가진다. 반면 새로운 사용자와 아이템을 다루기에 부적합하다는 **Cold Start Problem**이라는 한계를 갖고 있다.  

`협업 필터링`은 **근접 이웃 방법**과 **잠재 요인 방법**로 나뉜다. 후자의 경우 평점 패턴에서 20~100가지의 factor(요인)을 추론하는 것을 목적으로 한다.  

---
## 2. MF Methods and A Basic MF Model  
`잠재 요인 협업 필터링`을 구현하는 가장 좋은 방법 중 하나는 **Matrix Factorization**이다. 
기본적으로 이 방법은 평점 패턴으로부터 추론한 요인 벡터들을 통해 사용자와 아이템의 특성을 잡아낸다. 이 때 사용자와 아이템 사이의 강한 관련성이 있다면 추천이 시행된다. 이 방법은 확장성, 높은 정확도, 유연성이라는 장점을 가진다.  

추천 시스템은 여러 종류의 Input Data를 활용할 수 있다. 물론 가장 좋은 것은 양질의 **명시적 피드백**(Explicit Feedback)이 될 것인데, 이는 영화 평점이나 좋아요/싫어요와 같은 아이템에 대한 사용자의 선호 결과를 의미한다. 일반적으로 이러한 피드백은 그리 많이 이루어지지 않기 때문에, 이를 행렬로 정리하면 희소(Sparse) 행렬이 될 수 밖에 없다.  

만약 이러한 명시적 피드백 조차 활용할 수 없을 때는, 추천 시스템은 **암시적 피드백**(Implicit Feedback)을 이용하여 사용자의 선호를 파악하게 된다. 이는 구매내역이나 검색기록, 검색 패턴, 커서의 움직임 등을 의미하며 이를 통해 사용자의 선호를 파악하는 것이 목표라고 할 수 있겠다.  

Matrix Factorization(이하 MF 또는 행렬 분해) 모델은 사용자와 아이템 모두를 차원 f의 결합 잠재요인 공간에 매핑하는데, 사용자-아이템 상호작용은 이 공간에서 내적으로 모델링 된다.  

아이템 i는 $ q_i $로, 사용자 u는 $ p_u $라는 벡터로 표현된다. 이 둘의 내적은 **사용자-아이템 사이의 상호작용**을 반영하며 이는 곧 아이템에 대한 사용자의 전반적인 관심을 표현한다고 볼 수 있다. 식은 아래와 같다.  

$$ \hat{r_{ui}} = q^{T}_i p_u $$  

이 모델은 사실 **SVD**(Singular Vector Decomposition)과 매우 유사한데, 추천 시스템에서는 결측값의 존재로 이 SVD를 직접적으로 사용하는 것은 불가능하다. 결측값을 채워 넣는 것 역시 효율적이지 못하고 데이터의 왜곡 가능성 때문에 고려하기 힘들다.  

따라서 오직 관측된 평점만을 직접적으로 모델링하는 방법이 제시되었으며, 이 때 과적합을 방지하기 위해 규제 항이 포함되었다. 요인 벡터 $ q_i, p_u $를 학습하기 위해 시스템은 관측된 평점 세트를 바탕으로 아래 식을 최소화하는 것을 목적으로 한다.  

$$ \min_{q, p} \sum_{(u, i) \in K} ( r_{ui} - q^T_i p_u  ) + \lambda (\Vert{q_i}\Vert^2 + \Vert{p_u}\Vert^2) $$  

이 때, **K**는 $ r_{ui} $가 측정된(known) 값일 때의 (u, i) 세트를 의미한다. 결과적으로 이 모델은 알려지지 않은 평점을 예측하는 것이 목적이기 때문에 과적합을 방지해야 하고, 이를 위해 규제항이 필요하고 $ \lambda $가 이 규제의 정도를 제어한다. $ \lambda $는 주로 Cross-Validation에 의해 결정된다.  

---
## 3. Learning Algorithms and Adding Biases  



---
## 4. Additional Input Sources and Temporal Dynamics  


---
## 5. Inputs with varying confidence levels  


---
## 6. Netflix Prize Competition   




---
## Reference  
> 
