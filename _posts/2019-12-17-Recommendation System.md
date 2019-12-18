---
layout: post
title: 추천 시스템의 기본 - 01. 잠재요인 협업필터링 (Latent Factor Collaborative Filtering)
author: Youyoung
categories: Machine_Learning
tags: [Machine_Learning, Recommendation System, Matrix Factorization, Latent Factor Collaborative Filtering]
---

## 0. Introduction
추천시스템은 이제는 너무 많은 산업에서 도입하고 있는 시스템이기에 웬만큼 참신하지 않은 이상 새롭게 들리지 않는 것이 현실이다. 그러나 소비자의 입장에서 추천시스템을 보는 것과, 이 시스템의 개발자가 추천시스템을 바라 보는 것에는 큰 차이가 있다. 성공적으로 추천 엔진을 도입한 산업, 기업들이 있는 반면 여러 가지 어려움으로 인해 실질적인 효과가 떨어지는 산업, 기업도 있기 마련이다.  

사용자(User)의 행동 양식, 인구학적(Demographic) 정보, 아이템(Item)의 특성, 외부 변수 등 수많은 변인들을 관리하고 분석해서 사용자에게 가장 알맞는 아이템을 추천해주는 일은 분명 쉬운 일은 아니다. 이러한 어려움을 극복하기 위해 연구자들은 과거부터 여러 종류의 추천 시스템을 개발해왔는데, 지금부터 그에 대해 조금씩 알아보고자 한다.  

추천 시스템을 만드는 방법에는 굉장히 다양한 방식이 존재하지만, 본 글에서는 가장 핵심이 되는 방법론들에 대해서만 간단히 언급하고자 한다. 추천 시스템은 크게 `컨텐츠 기반 필터링(Content Based Filtering)` 방식과 `협업 필터링(Collaborative Filterin)` 방식으로 나뉜다. 협업 필터링은 또 `최근접 이웃(Nearest Neighbor) 협업 필터링`과 `잠재 요인(Latent Factor) 협업 필터링`으로 나뉜다.  

과거에는 `컨텐츠 기반 필터링`과 `최근접 이웃 협업 필터링`이 더욱 주목을 받았지만, 2009년에 있었던 **넷플릭스 추천 컨테스트**에서 **행렬 분해(Matrix Factorization)**를 이용한 `잠재 요인 협업 필터링` 방식이 우승을 차지하면서, 연구자들은 이 방식에 큰 관심을 갖게 되었다. 현재로서는 많은 경우에 이 방식이 우위를 차지하지만, 상황에 따라서는 다른 방식이 더 좋은 결과를 낼 때도 많고, 하이브리드 형식으로 결합하는 방식 또한 좋은 효율을 보여주는 경우도 많다.  

앞으로 총 4개의 시리즈로 이어질 추천 시스템에 관한 글들은, 위에서 언급한 `잠재 요인 협업 필터링`과 이 방법론에서 출발하여 발전된 알고리즘에 대해 다룰 예정이다. 간단히 순서를 보면 아래와 같다.  

> 01. 잠재요인 협업필터링  
> 02. Matrix Factorization Techiques for Recommender Systems 논문 리뷰  
> 03. Factorization Machines 설명  
> 04. Field-aware Factorization machines 설명  

**Matrix Factorization** 개념에 **Support Vector Machine**의 개념을 결합한 것이 **Factorization Machines**이다. 여기서 더 나아가 개별 feature들의 메타정보(field)를 알고리즘에 반영한 것이 **Field-aware Factorization Machines**이다. 줄여서 각각 **FM**과 **FFM**이라고 부르는 것이 일반적이다.  

로지스틱 모델과 달리 **FFM**은 가중치를 latent vector화 했기 때문에 연산량과 메모리 사용량이 더 많은 단점이 있지만, 최근 여러 논문에서는 system tuning을 통해 실제 광고 서빙에 사용하는 데 큰 지장이 없음을 밝혔다. 여력이 될 때 더욱 최신 연구들에 대해서도 글을 추가하도록 할 것이다.  

---
## 1. 추천 시스템의 개요  
### 1.1. 컨텐츠 기반 필터링  


### 1.2. 최근접 이웃 협업 필터링  
#### 1.2.1. 사용자 기반 최근접 이웃 협업 필터링  


#### 1.2.2. 아이템 기반 최근접 이웃 협업 필터링  


### 1.3. 잠재 요인 협업 필터링  



---
## 2. Singular Vector Decomposition  



---
## 3. 잠재 요인 협업 필터링의 Matrix Factorization  



---
## 4. 간단한 예제  



---
## 5. Movielens 데이터를 이용한 예제  



---
## 6. Surprise 모듈을 활용한 예제  







---
## Reference  
> 파이썬 머신러닝 완벽 가이드, 권철민, 위키북스
> [카카오 리포트](https://brunch.co.kr/@kakao-it/84)
> [Surprise 모듈 문서](https://surprise.readthedocs.io/en/stable/getting_started.html)
> [SVD 설명](https://rfriend.tistory.com/185)

