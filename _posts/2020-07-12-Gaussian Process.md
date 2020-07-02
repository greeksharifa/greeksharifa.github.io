---
layout: post
title: Gaussian Process 설명
author: Youyoung
categories: Bayesian_Statistics
tags: [Machine_Learning, Bayesian_Statistics]
---

**Gaussian Process**는 Random(Stochastic) Process의 한 예이다. 이 때 Random Process는 시간이나 공간으로 인덱싱된 Random Variable의 집합을 의미한다. **GP**의 정의는 아래와 같다.  

> Stochastic process such that every finite collection of those random variables has a multivariate normal distribution.

이는 또한 이 Random Variable들의 선형 결합 일변량 정규분포를 따른다는 말과 동일한 설명이고, GP의 일부를 가져와도 이는 항상 다변량 정규분포를 따른다는 것을 의미한다. 또한 다변량 정규분포는 주변 분포와 조건부 분포 역시 모두 정규분포를 따르기 때문에 이를 계산하기 매우 편리하다는 장점을 지닌다.  

**GP**는 일종의 `Bayesian Non-parametric method'으로 설명되는데, 이 때 Non-parametric이라는 것은 parameter의 부재를 의미하는 것이 아니라 parameter가 무한정 (infinite) 있다는 것을 의미한다.  

지금부터는 **GP**에 대해 이해하기 위해 단계적으로 설명을 진행할 것이다.  


---
## 1. Modeling with Gaussians  
다변량 정규분포를 생각해보자. 

$$ p(x, y) = \mathcal{N}( 
    \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix},
    \begin{bmatrix} \Sigma_x \Sigma_{xy} \\ \Sigma_{xy}^T \Sigma_y \end{bmatrix}) $$  

앞서 언급하였듯이 이 다변량 정규분포를 이루는 확률변수의 어떠한 부분집합에 대해서도 주변 분포와 조건부 분포 모두 정규분포를 따른다. **GP**는 여기서 한발 더 나아가서, 이러한 다변량 정규분포를 무한 차원으로 확장시키는 개념으로 생각하면 된다.  

이 무한의 벡터를 일종의 함수로 생각할 수도 있을 것이다. 연속형 값을 인풋으로 받아들이는 함수를 가정하면, 이는 본질적으로 input에 의해 인덱싱된 값들을 반환하는 무한한 벡터로 생각할 수 있다. 이 아이디어를 무한 차원의 정규분포에 적용하면 이것이 바로 **GP**의 개념이 된다.

따라서 **Gaussian Process**는 함수에 대한 분포라고 표현할 수 있다. 다변량 정규분포가 평균 벡터와 공분산 행렬로 표현되는 것처럼, **GP** 또한 평균 함수와 공분산 함수를 통해 다음과 같이 정의된다.  

$$ P(X) \sim GP(m(t), k(x, x\prime)) $$  

**GP**에 있어서 `Marginalization Property`는 매우 중요한 특성이다. 우리가 관심 없거나 관측되지 않은 수많은 변수에 대해 Marginalize할 수 있다.  

**GP**의 구체적 예시를 다음과 같이 들 수 있을 것이다. 실제로 이는 가장 흔한 설정이다.  

$$ m(x) = 0 $$  
$$ k(x, x\prime) = \theta_1 exp( - \frac{\theta_2}{2} ( x - x\prime)^2 ) $$  

여기서 공분한 함수로 **Squared Exponential**을 사용하였는데, $x$와 $x\prime$이 유사한 값을 가질 수록 1에 수렴하는 함수이다. (거리가 멀수록 0에 가까워짐) 평균 함수로는 0을 사용하였는데, 사실 평균 함수로 얻을 수 있는 정보는 별로 없기에 단순한 설정을 하는 것이 가장 편리하다.  

유한한 데이터 포인트에 대해 **GP**는 위에서 설정한 평균과 공분산을 가진 다변량 정규분포가 된다.  

---
## 2. Gaussian Process Regressor  



---
## 3. Fitting Gaussian Process with Python  
### 3.1. scikit-learn  
회귀 문제에서는 공분산 함수(kernel)를 명시함으로써 `GaussianProcessRegressor`를 사용할 수 있다. 이 때 적합은 주변 우도의 로그를 취한 값을 최대화하는 과정을 통해 이루어진다. 이 Class는 평균 함수를 명시할 수 없는데, 왜냐하면 평균 함수는 0으로 고정되어 있기 때문이다.  

분류 문제에서는 `GaussianProcessClassifier`를 사용할 수 있을 것이다. 언뜻 생각하면 범주형 데이터를 적합하기 위해 정규 분포를 사용하는 것이 이상하다. 이는 `Latent Gaussian Response Variable`을 사용한 뒤 이를 unit interval(다중 분류에서는 simplex interval)로 변환하는 작업을 통해 해결할 수 있다. 이 알고리즘의 결과는 일반적인 머신러닝 알고리즘에 비해 부드럽고 확률론적인 분류 결과를 반환한다.  

**GP**의 Posterior는 정규분포가 아니기 때문에 Solution을 찾기 위해 주변 우도를 최대화하는 것이 아니라 `Laplace 근사`를 이용한다.  



---
## 4. Role of Gaussian Process in Bayesian Optimization  


---
## Reference
1) [GP 설명 블로그](https://blog.dominodatalab.com/fitting-gaussian-process-models-python/)
2) [GP 회귀 논문](https://arxiv.org/abs/1505.02965)



