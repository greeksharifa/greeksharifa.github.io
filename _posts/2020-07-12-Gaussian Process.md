---
layout: post
title: Gaussian Process 설명
author: Youyoung
categories: Bayesian_Statistics
tags: [Machine_Learning, Bayesian_Statistics]
---

<center><img src="/public/img/Machine_Learning/2020-07-12-Gaussian Process/01.png" width="70%"></center>  

**Gaussian Process**에 대해 알아보자!  

**Gaussian Process**는 Random(Stochastic) Process의 한 예이다. 이 때 Random Process는 시간이나 공간으로 인덱싱된 Random Variable의 집합을 의미한다. **GP**의 정의는 아래와 같다.  

> Stochastic process such that every finite collection of those random variables has a multivariate normal distribution.

이는 또한 이 Random Variable들의 선형 결합 일변량 정규분포를 따른다는 말과 동일한 설명이고, GP의 일부를 가져와도 이는 항상 다변량 정규분포를 따른다는 것을 의미한다. 또한 다변량 정규분포는 주변 분포와 조건부 분포 역시 모두 정규분포를 따르기 때문에 이를 계산하기 매우 편리하다는 장점을 지닌다.  

**GP**는 일종의 `Bayesian Non-parametric method`으로 설명되는데, 이 때 Non-parametric이라는 것은 parameter의 부재를 의미하는 것이 아니라 parameter가 무한정 (infinite) 있다는 것을 의미한다.  

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
## 2. Gaussian Process Regression  
본 Chapter는 아래 Reference에 명시되어 있는 논문의 일부를 번역한 내용을 포함한 것임을 밝힌다.  

### 2.1. Components of Gaussian Process  
앞에서 공분산 함수 $ k(x, x\prime) $에 대해서 설명하였는데, 본 논문에 맞추어 Notation을 살짝 변형하도록 하겠다.  

$$ k(x, x\prime) = \sigma_f^2 exp( - \frac{( x - x\prime)^2}{2l^2} ) $$  
  
정말 기호만 살짝 바뀌었다. $x$가 $x\prime$과 유사할 수록 $f(x)$라 $f(x\prime)$과 **상관성**(Correlation)을 가진다고 해석할 수 있다. 이것은 좋은 의미이다. 왜냐하면 함수가 smooth해지고 이웃한 데이터 포인트끼리 더욱 유사해지기 때문이다.  

만약 그 반대의 경우 2개의 데이터 포인트는 서로 마주칠 수도 없다. 즉 새로운 $x$값이 삽입될 때, 이와 먼 곳에 있는 관측값들은 그다지 큰 영향을 미칠 수 없다. 이러한 분리가 갖는 효과는 사실 **length parameter**인 $l$에 달렸있는데, 이 때문에 이 공분산 함수는 상당한 유연성을 지니는 식이 된다.  

하지만 데이터는 일반적으로 **Noise**를 포함하고 있다. 이 때문에 언제나 측정 오차는 발생하기 마련이다. 따라서 $y$ 관측값은 $f(x)$에 더불어 **Gaussian Noise**를 포함하고 있다고 가정하는 것이 옳다.  

$$ y = f(x) + \mathcal{N}(0, \sigma_n^2) $$  

많이 보았던 회귀식 같아 보인다. 이 **Noise**를 공분산 함수안에 집어넣으면 아래와 같은 형식을 갖추게 된다.  

$$ k(x, x\prime) = \sigma_f^2 exp(- \frac{( x - x\prime)^2}{2l^2} ) + \sigma_n^2 \delta(x, x\prime) $$  

여기서 $\delta(x, x\prime)$은 `Kronecker Delta Function`이다.  

많은 이들은 **GP**를 사용할 때 $\sigma_n$을 공분산 함수와 분리해서 생각하지만 사실 우리의 목적은 
**y*** 를 예측하는 것이지 정확한 **f*** 를 예측하는 것이 아니기 때문에 위와 같이 설정하는 것이 맞다.  

**Gaussian Process Regression**을 준비하기 위해 모든 존재하는 데이터포인트에 대해 아래와 같은 공분한 함수를 계산하도록 하자.  

<center><img src="/public/img/Machine_Learning/2020-07-12-Gaussian Process/02.JPG" width="80%"></center>  

$K$의 대각 원소는 $\sigma_f^2 + \sigma_n^2$ 이고, 비대각 원소 중 끝에 있는 원소들은 $x$ 가 충분히 큰 domain을 span할수록 0에 가까운 값을 갖게 된다.  

### 2.2. How to Regress using Gaussian Process  
**GP**에서 가장 중요한 가정은 우리의 데이터가 다변량 정규 분포로부터 추출된 Sample로 표현된다는 것이므로 아래와 같이 표현할 수 있다.  

$$ \begin{bmatrix} \mathbf{y} \\ y* \end{bmatrix} = \mathcal{N}(0,
    \begin{bmatrix} K, K_*^T \\ K_*, K_{**} \end{bmatrix}) $$  

우리는 물론 조건부 확률인 다음 식에 대해 알고 싶다.  

$$ p( y_*| \mathbf{y} ) $$  

이 확률은 데이터가 주어졌을 때 $y_*$ 에 대한 예측의 확실한 정도를 의미한다. 본 논문 원본 Appendix에는 앞으로의 과정에 대한 증명이 담겨 있다. 일단 진행하자.  

$$ y_*|\mathbf{y} \sim \mathcal{N}( K_*K^{-1}\mathbf{y}, K_{**} - K_* K^{-1} K_*^T ) $$  

정규분포이므로, $y_*$ 에 대한 **Best Estimate**는 평균이 될 것이다.  

$$ \bar{y}_* = K_*K^{-1}\mathbf{y} $$  

그리고 분산 또한 아래와 같다.  

$$ var(y_*) = K_{**} - K_* K^{-1} K_*^T $$  

이제 본격적으로 예제를 사용해보자. Noise가 존재하는 데이터에서 다음 포인트 $x_*$ 에서의 예측 값은 얼마일까?  

<center><img src="/public/img/Machine_Learning/2020-07-12-Gaussian Process/03.JPG" width="80%"></center>  

6개의 관측값이 다음과 같이 주어졌다.  

```python
x = [-1.5, -1, -0.75, -0.4, -0.25, 0]
```

Noise의 표준편차 $\sigma_n$ 이 0.3이라고 하자. $\sigma_f$ 와 $l$ 을 적절히 설정하였다면 아래와 같은 행렬 $K$를 얻을 수 있다.  

<center><img src="/public/img/Machine_Learning/2020-07-12-Gaussian Process/04.JPG" width="50%"></center>  

공분산 함수를 통해 아래 사실을 알 수 있다.  

$$ K_{**} = 3 $$  

$$ K_* = [0.38, 0.79, 1.03, 1.35, 1.46, 1.58] $$  

$$ \bar{y}_* = 0.95 $$  

$$ var(y_*) = 0.21 $$  

$$ x* = 0.2 $$  

그런데 매번 이렇게 귀찮게 구할 필요는 없다. 엄청나게 많은 데이터 포인트가 존재하더라도 이를 한번에 큰 $K_*$ 과 $K_{**}$ 을 통해 계산해버리면 그만이다.  

만약 1000개의 Test Point가 존재한다면 $K_{**}$ 는 (1000, 1000)일 것이다.  

**95% Confidence Interval**은 아래 식으로 구할 수 있고 이를 Graph로 표현하면 아래 그림과 같다.  

$$ \bar{y}_* \pm 1.96\sqrt{var(y_*)} $$  

<center><img src="/public/img/Machine_Learning/2020-07-12-Gaussian Process/05.JPG" width="80%"></center>  


### 2.3. GPR in the Real World  







---
## 3. Fitting Gaussian Process with Python  
### 3.1. scikit-learn 이용  
먼저 scikit-learn 라이브러리를 이용해보자.  

회귀 문제에서는 **공분산 함수**(kernel)를 명시함으로써 `GaussianProcessRegressor`를 사용할 수 있다. 이 때 적합은 주변 우도의 로그를 취한 값을 최대화하는 과정을 통해 이루어진다. 이 Class는 평균 함수를 명시할 수 없는데, 왜냐하면 평균 함수는 0으로 고정되어 있기 때문이다.  

분류 문제에서는 `GaussianProcessClassifier`를 사용할 수 있을 것이다. 언뜻 생각하면 범주형 데이터를 적합하기 위해 정규 분포를 사용하는 것이 이상하다. 이는 `Latent Gaussian Response Variable`을 사용한 뒤 이를 unit interval(다중 분류에서는 simplex interval)로 변환하는 작업을 통해 해결할 수 있다. 이 알고리즘의 결과는 일반적인 머신러닝 알고리즘에 비해 부드럽고 확률론적인 분류 결과를 반환한다.  

**GP**의 Posterior는 정규분포가 아니기 때문에 Solution을 찾기 위해 주변 우도를 최대화하는 것이 아니라 `Laplace 근사`를 이용한다.  



### 3.2. PyMC3  
다음은 Bayesian 방법론을 위한 라이브러리, PyMC3를 이용해보자.  





---
## 4. Role of Gaussian Process in Bayesian Optimization  





---
## Reference
1) [GP 설명 블로그](https://blog.dominodatalab.com/fitting-gaussian-process-models-python/)  
2) [GP 회귀 논문](https://arxiv.org/abs/1505.02965)  
3) [Tensorflow Github](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Process_Regression_In_TFP.ipynb?fbclid=IwAR1GBOFx9znZVisebDIEv9jPLmBX7KfTCESOli7eUOQLEQdl83do9mdu5ys)  


