---
layout: post
title: AB Test Sample Size 구하기  
author: Youyoung
categories: [Machine_Learning]
tags: [Machine_Learning, AB Test]
---

이번 포스팅에서는 AB Test를 진행할 때 거의 필수적으로 고려해야 하는 Sample Size를 구하는 과정에 대해 정리해본다.  

---
# AB Test Sample Size 구하기  
## 1. Basic  
귀무가설과 대립가설이 아래와 같다고 해보자.  

$$ H_0: \mu = \mu_0 $$  

$$ H_a: \mu > \mu_0 $$  

실험의 특성을 고려하여 유의수준과 검정력을 설정하면 이를 통해 최소 Sample Size를 추정할 수 있다. 1 - 검정력인 $\beta$ 는 $\mu = \mu_a$, \mu_a > \mu_0$ 과 같이 $\mu$ 를 특정 값으로 지정하였을 때 구할 수 있다.  

Reject Region(기각역)이 아래와 같이 정의된다고 하자. $\hat{\theta}$ 는 물론 추정량을 의미한다.  

$$ RR = \{ \hat{\theta}: \hat{\theta} > k \} $$  

$k$ 는 기각역이 시작되는 point이다.  

2종 오류가 발생할 확률인 $\beta$ 는 대립가설이 참일 때 귀무가설을 적절히 기각하지 않을 확률을 의미한다. 이를 위 수식과 이어서 설명하면, 아래와 같이 표현할 수 있다.  

$$ \beta = P(\hat{\theta} \leq k, \theta = \theta_a) $$  

위 사실들을 고려하여 다시 한 번 $\alpha$ 와 $\beta$ 를 구해보자.  

$$ \alpha = P(\bar{Y} > k, \mu = \mu_0) $$  

$$ = P( \frac{\bar{Y} - \mu_0}{\sigma / \sqrt{n}} > \frac{k - \mu_0}{\sigma / \sqrt{n}}, \mu = \mu_0 ) $$  

$$ = P(Z > \mathcal{z}_{\alpha}) $$  

$$ \beta = P(\bar{Y} \leq k, \mu = \mu_a) $$  

$$ = P( \frac{\bar{Y} - \mu_a}{\sigma / \sqrt{n}} \leq \frac{k - \mu_a}{\sigma / \sqrt{n}}, \mu = \mu_a ) $$  

$$ = P(Z \leq - \mathcal{z}_{\beta}) $$  

위 2개 식에서 우리는 아래 사실들을 정리할 수 있다.  

$$ \frac{k - \mu_0}{\sigma / \sqrt{n}} = \mathcal{z}_{\alpha} $$  

$$ \frac{k - \mu_a}{\sigma / \sqrt{n}} = -\mathcal{z}_{\beta} $$  

위 식을 $k$ 에 대해 정리하면 다음과 같다.  

$$ k = \mu_0 + \mathcal{z}_{\alpha} (\frac{\sigma}{\sqrt{n}}) = \mu_a - \mathcal{z}_{\beta} (\frac{\sigma}{\sqrt{n}}) $$  

따라서 위 식에서 $k$ 를 제외하고 $n$ 에 대해 다시 정리하면, **Sample Size for an Upper-tail $\alpha$ - level Test**를 얻을 수 있다.  

$$ n = \frac{(\mathcal{z}_{\alpha} + \mathcal{z}_{\beta})^2 \sigma^2}{(\mu_a - \mu_0)^2} $$  

이 때 분모에서 제곱항 내부에 위치한 값을 **Effect Size**라고 표현하며, 실험에서 확인하고 싶은 유의미한 차이를 의미한다.  

$$ \delta = \mu_a - \mu_0 $$  

그리고 위 공식은 $n$ 이 충분히 클 때 성립한다.  


## 2. Proportion Test  
애플리케이션 상에서 AB Test를 할 때에는 비율의 차이를 metric으로 두는 경우가 많다. 예를 들어 새로운 모델을 배포하였을 때 Conversion Rate이 충분히 상승하였는지 알고 싶을 수 있다. 이 경우 귀무가설과 대립가설은 아래와 같다.  

$$ H_0: p = p_0 $$  

$$ H_a: p > p_0 $$  

이 경우도 결국 똑같이 구할 수 있다. 신청 여부에 따른 분포이기 때문에 본 Test에서는 이항 분포가 사용된다. 그렇다면 분산은 아래와 같이 구할 수 있다.  

$$ \sigma_0^2 = p_0 (1-p_0) $$  

새로운 모델에서의 목표 Conversion Rate을 $p_a$ 라고 하면 **Effect Size**는 $p_a - p_0$ 가 될 것이다. 그리고 표본 평균의 분산은 아래와 같이 구할 수 있다.  

$$ \sigma^2 = p_0(1-p_0) + p_a(1-p_a) $$  

위 사실을 모두 종합하여 Sample Size를 구해보면 다음과 같다.  

$$ n = \frac{(\mathcal{z}_{\alpha} + \mathcal{z}_{\beta})^2 (p_0(1-p_0) + p_a(1-p_a)) }{(p_a - p_0)^2} $$  

만약 아래와 같은 실험 설정이라면,  

$$ \alpha = 0.05, 1 - \beta = 0.8, p_0 = 0.02, p_a = 0.022 $$  

필요한 최소 Sample Size는 63215이다.  


---
## References  
1) Mathematical Statistics with Applications(Denis D. Wackerly, ...) Ch 10.4  
2) [Comparing Two Proportions – Sample Size](https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/)  

