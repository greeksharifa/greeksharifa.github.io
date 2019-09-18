---
layout: post
title: Contextual Bandit and Tree Heuristic
author: Youyoung
categories: Machine_Learning
tags: [Machine Learning, Paper_Review, Contextual Bandit]
---

## 1. Contextual Bandit의 개념  
Contextual Bandit 문제를 알기 위해선 Multi-Armed Bandit 문제의 개념에 대해 숙지하고 있어야 한다.  
위 개념에 대해 알기를 원한다면 [여기](https://sumniya.tistory.com/9)를 참고하기 바란다.  
  
Multi-Armed Bandit 문제에서 Context 개념이 추가된 Contextual Bandit 문제는 대표적으로 추천 시스템에서 활용될 수 있다. 단 전통적인 추천 시스템을 구축할 때는 Ground Truth y 값, 즉 실제로 고객이 어떠한 상품을 좋아하는지에 대한 해답을 안고 시작하지만, Contextual Bandit과 관련된 상황에서는 그러한 이점이 주어지지 않는다.  
  
그림을 통해 파악해보자.  
  
<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/01.jpg" width="60%"></center>  

<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/02.jpg" width="60%"></center>  

첫 번째 그림은 전통적인 추천시스템에 관한 것이고, 두 번째 그림은 Contextual Bandit 문제와 관련된 것이다.  

온라인 상황에서 우리가 고객에게 어떠한 상품을 제시하였을 때, 고객이 그 상품을 원하지 않는다면 우리는 새로운 시도를 통해 고객이 어떠한 상품을 좋아할지 파악하도록 노력해야 한다. 이것이 바로 **Exploration**이다.  

만약 고객이 그 상품에 호의적인 반응을 보였다면, 이 또한 중요한 데이터로 적재되어 이후에 동일 혹은 유사한 고객에게 상품을 추천해 주는 데에 있어 이용될 것이다. 이 것이 **Exploitation**이다.  

위 그림에 나와 있듯이, Contextual Bandit 문제 해결의 핵심은, Context(고객의 정보)를 활용하여 Exploitation과 Exploration의 균형을 찾아 효과적인 학습을 진행하는 것이다.  

---

## 2. Lin UCB  
Lin UCB는 **A contextual-bandit approach to personalized news article recommendation**논문에 처음 소개된 알고리즘으로, Thompson Sampling과 더불어 Contextual Bandit 문제를 푸는 가장 대표적이고 기본적인 알고리즘으로 소개되어 있다.  

이 알고리즘의 기본 개념은 아래와 같다.  

<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/03.jpg" width="60%"></center>  

Context Vector를 어떻게 구성할 것인가에 따라 Linear Disjoint Model과 Linear Hybrid Model로 구분된다. Hyperparameter인 Alpha가 커질 수록 Exploration에 더욱 가중치를 두게 되며, 결과는 이 Alpha에 다소 영향을 받는 편이다.  

본 알고리즘은 이후 Tree Heuristic과의 비교를 위해 테스트 용으로 사용될 예정이다.  

---

## 3. Tree Heuristic

### 3.1 Tree Boost
Tree Heuristic에 접근하기 위해서는 먼저 그 전신이라고 할 수 있는 Tree Boost 알고리즘에 대해 알아야 한다. 본 알고리즘은 **A practical method for solving contextual bandit problems using decision trees** 논문에서 소개되었다.  

<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/04.jpg" width="60%"></center>  


### 3.2 Tree Heuristic


<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/05.jpg" width="60%"></center>  


---
$$ D = {(x_i, y_i)} (|D| = n, x_i \in \mathbb{R^m}, y_i \in \mathbb{R}) $$
  
$ \vec{x_i} $라는 i번째 데이터가 Input으로 들어왔을 때, 각각의 Tree가 Decision Rule을 통해 산출한 **score = output = $ f_k(x_i) $** 을 모두 더한 값을 아래의 식과 같이 **최종 output = $ \hat{y_i} $** 으로 출력하게 된다.  

> link: [https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html]
