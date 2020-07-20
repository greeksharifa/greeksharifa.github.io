---
layout: post
title: Monte Carlo Approximation (몬테카를로 근사 방법) 설명
author: Youyoung
categories: Bayesian_Statistics
tags: [Machine_Learning, Bayesian_Statistics]
---

이번 글에서는 Monte Carlo 근사 방법에 대해 간단히 알아볼 것이다. 사실 개념 자체는 간단하니 가볍게 파악해보록 하자. 본 글은 Kevin P.Murphy의 **Machine Learning: A Probabilistic Perspective** 내용을 번역 및 요약한 것임을 밝힌다.   

---
# 1. Monte Carlo Approximation  
일반적으로 변수 변환 공식을 사용하여 확률 변수의 함수의 분포를 계산하는 것은 굉장히 어렵다. 그러나 몬테카를로 근사법은 이에 대한 확실한 대안이 될 수 있다.  

어떤 분포로부터 $S$ 개의 Sample $x_1, ..., x_s$ 를 추출하였다고 하자. 이 Sample에 기반하여 `Empirical Distribution`인 $[f(x_i)]_{i=1}^S$ 를 이용하여 우리는 $f(X)$ 라는 분포를 근사할 수 있다. 이를 **몬테카를로 적분**이라고 부른다.  

확률 변수의 어떠한 함수에 대해서도 기댓값을 구할 수 있는 것이 **몬테카를로 근사법**이다. Sample을 뽑고 이에 대해 `Arithmetic Mean`을 취하면 된다.  

$$ E[f(X)] = \int f(x) p(x) dx \approx \frac{1}{S} \Sigma_{i=1}^S f(x_i) $$  

f 함수를 변환시켜 다음과 같은 여러 특성도 얻을 수 있다.  

$$ \bar{x} = \frac {1}{X} \Sigma_{i=1}^S x_i \to E[X] $$  

$$ \Sigma_{i=1}^S (x_i - \bar{x})^2 \to var[X] $$  

$$ \frac{1}{S} |{x_i \le c}| \to p(X \le c) $$  

$$ median {x_1, ..., x_s} \to median(X) $$  

---
# 2. Example: estimating $\pi$ by Monte Carlo Integration  
**몬테카를로 적분**하면 떠오르는 가장 기본적인 예제이다. 원의 면적을 구해보자. 원의 면적을 $I$ 라고 해보자. Indicator Function을 사용하여 나타내면 아래와 같다.  

$$ I = \int_{-r}^{r} \int_{-r}^{r} \mathbb{I} (x^2 + y^2 \le r^2) dxdy $$  

Indicatior Funciton을 $f(x, y)$ 라고 하고, 원 밖에 점이 찍히면 0, 안에 찍히면 1의 값을 갖는 함수라고 하자. $p(x), p(y)$ 는 $-r, r$ 구간 내에 존재하는 균일 분포 함수이다. 따라서 $p(x) = p(y) = \frac{1}{2r}$ 이다.  

$$ I = (2r)(2r) \int \int f(x, y) p(x) p(y) dx dy $$  

$$ \approx 4r^2 \frac{1}{S} \Sigma_{i=1}^S f(x_i, y_i) $$  


---
# 3. Accuracy of Monte Carlo Approximation  
기본적으로 몬테카를로 근사법은 표본의 크기가 클수록 정확도가 높아진다. 실제 평균을 $\mu = E[f(X)]$ 라고 하자. 그리고 몬테카를로 근사법에 의한 근사 평균을 $\hat{\mu}$ 라고 하자. 독립된 표본이라는 가정 하에  

$$ \hat{\mu} - \mu \to \mathcal{N} (0, \frac{\sigma^2}{S}) $$  

$$ \sigma^2 = var[f(X)] = E[f(X)^2] - E[f(X)]^2 $$  

위는 중심극한정리에 의한 결과이다. 물론 $\sigma^2$ 는 미지의 값이다. 그러나 이 또한 **몬테카를로 적분**에 의해 근사될 수 있다.  

$$ \hat{\sigma}^2 = \frac{1}{S} \Sigma_{i=1}^S (f(x_i) - \hat{\mu})^2 $$  

$$ P(\mu - 1.96 \frac{\hat{\sigma}}{\sqrt{S}} \le \hat{\mu} \le \mu + 1.96 \frac{\hat{\sigma}}{\sqrt{S}}) \approx 0.95 $$  

이 때 $\sqrt{ \frac{\hat{\sigma^2}} {S}}$ 는 **Numerical(Empirical) Standard Error** 라고 불리며, 이는 $\mu$ 의 추정량에 대한 불확실성에 대한 추정치이다. 만약 우리가 95%의 확률로 $\pm \epsilon$ 내부에 위치할 만큼 정확한 Report를 원한다면 우리는 $1.96 \sqrt{ \frac{\hat{\sigma}^2}{S}} \le \epsilon$ 을 만족시키는 $S$ 개의 Sample이 필요하다.  

