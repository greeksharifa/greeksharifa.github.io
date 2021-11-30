---
layout: post
title: Variational Inference (변분 추론) 설명
author: Youyoung
categories: Bayesian_Statistics
tags: [Machine_Learning, Bayesian_Statistics]
---

본 글에서는 정보 이론과 관련이 있는 **Kullback-Leibler Divergence**와 이를 기반으로 한 **Variational Inference**에 대해 간략히 정리해보고자 한다. 시작에 앞서, 변분 추론은 근사 추정의 대표적인 방법이라는 점을 밝히고 싶으며, 본 글에서 소개된 변분 추론 기법은 **Vanilla Variational Inference**라고 부를 수 있는 **CAVI**이다. **CAVI**의 단점을 보완한 다양한 변분 추론 기법들이 연구되었으며 이에 대한 내용은 후속 글에서 다루도록 할 것이다.  

## 1. Kullback-Leibler Divergence  
정보이론에서 정보량은 불확실성이 커질수록 많아지는 것으로 정의한다. **Shannon Entropy**는 확률의 값에 $log$ 를 씌우고 -1을 곱해준 값으로, 모든 사건의 정보량의 Expectation을 의미한다. 확률 분포 P에 대한 섀넌 엔트로피는 아래와 같이 정의할 수 있다.  

$$ H(P) = E_{X \sim P}[-logP(x)] $$  

**Kullback-Leibler Divergence**는 분포의 유사한 정도를 나타내는 척도인데, 비대칭적이기 때문에 거리를 나타내는 척도로는 사용될 수 없다. 그보다는 **정보량의 차이**를 나타내는 것이라고 이해하면 좋다.  

어떤 확률 분포 $P(x)$ 가 주어졌을 때, 이 분포와 $Q(x)$ 라는 분포와의 차이를 알고 싶다면 쿨백-라이블리 발산 값을 구하면 된다. 이 값은 **Cross-Entropy**에서 **Self-information**을 뺀 값으로 정의되는데, **Cross-Entropy**는 아래와 같이 정의된다.  

$$ H(P, Q) =  E_{X \sim P}[-log(Q(x))] = -\Sigma_x P(x) log Q(x) $$  

$H(P)$ 로 표현되는 **Self-Information**은 그 확률변수가 갖고 있는 정보량을 의미하는데,  

이산확률변수라면 $-\Sigma_x P(x) logP(x)$로 표현할 수 있다.  

이제 쿨백-라이블리 발산 식을 보자.  

$$ D_{KL}(P||Q) = H(P, Q) - H(P)$$  

$$ = -\Sigma_x P(x) log Q(x) + \Sigma_x P(x) logP(x) $$  

$$ = -\Sigma_x [P(x) log Q(x) - P(x)logP(x)] $$  

$$ = -\Sigma_x P(x) [logQ(x)-logP(x)] $$  

$$ = -\Sigma_x P(x) log(\frac{Q(x)}{P(x)}) $$  

$$ = E_{X \sim P}[-log \frac{Q(x)}{P(x)}] $$  

쿨백-라이블리 발산을 최소화하는 것은 기본적으로 두 확률 분포 사이의 차이를 최소화하는 것이다. 일반적으로 **P**를 우리가 갖고 있는 데이터의 분포, 혹은 계산하기 편리한 쉬운 분포라고 한다면 **Q**는 모델이 추정한 분포 또는 확률적으로 계산하기 어려운 분포라고 생각할 수 있다.  

머신러닝 관점에서 생각해본다면 **Self-Information** 부분은 학습 과정에 있어서 어떤 변동을 겪지는 않는다. 따라서 쿨백-라이블리 발산을 최소화하는 것은 크로스-엔트로피를 최소화하는 것과 동일한 의미를 지닌다.  

두 분포가 동일하면 쿨백-라이블리 발산은 0의 값을 가지고 다를수록 점점 큰 값을 가진다.

---
## 2. Variational Inference  
### 2.1. ELBO  
$\mathbf{x}$ 란 확률 변수가 있고, 이 변수의 특성의 상당 부분은 잠재 변수인 $\mathbf{z}$ 에 의해 설명된다고 하자. 우리는 이 때 우리는 $\mathbf{x}$ 의 실현 값인 데이터가 존재할 때 $\mathbf{z}$ 의 분포, 즉 **Posterior** $p(\mathbf{z}|\mathbf{x})$ 를 알고 싶다. 그런데 **Posterior**는 많은 경우에 Numerical 계산이 불가능하다. 따라서 우리는 이 **Posterior**를 알기 쉬운 분포 $q(\mathbf{z})$ 로 바꾸고 싶다.  

$$ p(z|x) \to q(z) $$  

변분추론은 이렇게 **Posterior**를 다루기 쉬운 분포로 근사하는 방법론을 의미한다.  

이 때 $q(z)$ 는 어떤 함수의 집합 $Q$ 의 한 원소라고 생각할 수 있다.  

용어를 잠시 정리해보자.  

$p(x)$ 는 **Marginal Probability** 또는 **Evidence**를, $p(z)$ 는 **Prior**를 의미한다.  
$p(x|z)$ 는 **Likelihood**를, $p(z|x)$ 는 **Posterior**를 의미한다.  

위에서 확인한 쿨백-라이블리 발산을 이용하여 변분추론을 설명하면, 변분추론은 아래와 같이 쿨백-라이블리 발산 값을 최소화하는 $Q$ 집합 내의 *q* 함수를 찾는 것이 된다.  

$$ q^*(x) = argmin_{q \in Q} D_{KL}(q(z)||p(z|x)) $$  

이전 Chapter에서 쿨백-라이블리 발산은 분포의 유사도를 측정하는 Index라고 하였다. 이 식을 위 관점을 다시 한 번 분석해보자.  

$$ D_{KL}(q(z)||p(z|x)) = \int q(z) log \frac{q(z)}{p(z|x)} dz $$  

$$ = \int q(z) log \frac{q(z)p(x)}{p(x|z)p(z)} $$  

$$ = \int q(z) log \frac{q(z)}{p(z)}dz + \int q(z) logp(x) dz - \int q(z) log p(x|z) dz $$  

$$ = D_{KL}(q(z)||p(z)) + logp(x) - E_{z \sim q(z)}[logp(x|z)] $$  

이렇게 사후확률과 다루기 쉬운 분포 사이의 쿨백-라이블리 발산은 총 3가지 항으로 분해할 수 있다.  

쿨백-라이블리 발산은 그 정의 때문에 0 이상의 값을 가진다. 즉 Non-negative이다. 따라서 제일 마지막 줄은 아래와 같이 다시 표현할 수 있다. (이 부분은 Jensen의 부등식을 통해서도 추론할 수 있다.)  

$$ 0 \le D_{KL}(q(z)||p(z)) + logp(x) - E_{z \sim q(z)}[logp(x|z)] $$  

$$ logp(x) \ge E_{z \sim q(z)}[logp(x|z)] - D_{KL}(q(z)||p(z)) $$  

우항을 **ELBO**(Evidence Lower BOund)라고 부른다. **Evidence**의 하한선이라는 의미이다. 

**Variational Density** $q(z)$ 와 **Posterior** 사이의 쿨백-라이블리 발산 값 부터 다시 표현해보면,  

$$ D_{KL}(q(z)||p(z|x)) = D_{KL}(q(z)||p(z)) + logp(x) - E_{z \sim q(z)}[logp(x|z)] $$  

$$ logp(x) = ELBO + D_{KL}(q(z)||p(z|x)) $$  
  
위 식을 보면 **Evidence**는 **ELBO**와 **쿨백-라이블리 발산**의 합으로 구성된다는 것을 알 수 있다. 이는 매우 중요한 수식이다. 일단은 쿨백-라이블리 발산을 최소화하는 것은 곧 **ELBO**를 최대화하는 것과 의미가 같다는 것은 쉽게 파악할 수 있다. 다만 **ELBO**와 쿨백-라이블리 발산 모두 *q* 함수에 의존적이기 때문에 단순히 한 쪽을 최소화하는 *q* 함수를 찾았다고 해서 이것이 반드시 **Evidence**의 값을 최소화한다고 말하기는 어렵다는 부분은 잊으면 안된다.  
  
다음 Chapter 부터는 최적화 방법에 대해 설명할 것인데, 전통적인 방법인 **CAVI**를 통해 설명을 진행하도록 하겠다.  

----
### 2.2. CAVI: Cooridinate Ascent mean field Variational Inference  
(**Mean-Field Variational Inference**)  

그렇다면 *q* 함수는 대체 어떤 함수인가? 아주 클래식한 방법으로 설명하자면, **Mean Field Variational Family**를 언급해야 할 것이다. 잠재 변수 $\mathbf{z}$ 가 모두 독립적이라고 할 때, *q* 함수는 아래와 같이 분해될 수 있다.  

$$ q(\mathbf{z}) = \prod_j q_j(z_j) $$  

이렇게 Variational 분포를 각각의 곱으로 분해하고 나면, 우리는 각 Factor에 대해 **Coordinate Ascent Optimization**을 적용할 수 있다. **CAVI**는 한 쪽을 고정한 채, Mean-Field Variational Density의 각 Factor를 반복적으로 최적화한다. 이 알고리즘은 **ELBO**의 Local Optimum으로 이끈다.  

**ELBO**를 다시 확인해보자.  

$$ ELBO = E_{z \sim q(z)}[logp(\mathbf{x}|z)] - D_{KL}(q(z)||p(z)) $$  

$$ = \int_z q(z) log p(\mathbf{x}, z) - q(z)logq(z) dz $$  

$$ = E_{z \sim q(z)} [logp(\mathbf{x}, z)] - E_{z \sim q(z)}[logq(z)] $$  

**CAVI**의 핵심 아이디어는 *q* 함수가 분해될 수 있다는 사실을 이용하는 것이다. $z_j$ 를 j번째 잠재 변수라고 하자. (아래의 - 기호는 그 index를 제외한 것을 의미한다.) 이 때 우리가 알고 싶은 것은, $\mathbf{x}$ 와 $\mathbf{z}_{-j}$ 가 모두 주어졌을 때 $z_j$ 의 Complete 조건부 확률이다. 이는 아래와 같이 표현할 수 있다.  

$$ logp(z_j|\mathbf{z}_{-j}, \mathbf{x}) $$  

그런데 앞서 했던 가정에 따라 모든 잠재 변수는 독립적이다. 따라서 위 식은 아래와 같다.  

$$ = logp(z_j, \mathbf{z}_{-j}, \mathbf{x}) $$  

지금부터 이 사실을 염두에 두고 위에서 보았던 **ELBO** 식을 $q_j$ 의 관점에서 풀어쓸 것이다. 아래에서 나오는 $l$ 기호는 $j$ 가 아닌 Index를 의미한다. (j번째 잠재 변수가 아닌 나머지 Variational Factors: $q_l(\mathbf{z}_l)$ )  

$$ ELBO = E_q[logp(\mathbf{x}, z_j, \mathbf{z}_{-j})] - E_{q_l}[logq_l(\mathbf{z}_l)] $$  

Iterative Expectation을 이용하면,  

$$ (E[A] = E[E[A|B]]) $$  

$$ = E_j [E_{-j} [logp(\mathbf{x}, z_j, \mathbf{z}_{-j})|z_j]] - E_{q_j}[logq_j] + Const $$  

첫 항의 안쪽 부분을 보자. 기댓값의 정의에 따라 다음과 같이 식을 전개할 수 있다.  

$$ E_{-j} [logp(\mathbf{x}, z_j, \mathbf{z}_{-j})|z_j] = \int_{-j} logp(\mathbf{x}, z_j, \mathbf{z}_{-j}) q(\mathbf{z}_{-j}|z_j) dq_{-j} $$  

$$ = \int_{-j} logp(\mathbf{x}, z_j, \mathbf{z}_{-j}) q(\mathbf{z}_{-j}) dq_{-j}$$  

$$ = E_{-j} [logp(\mathbf{x}, z_j, z_{-j})] $$  

최종적으로 $q_j$ 에 대한 **ELBO**는 아래와 같다.  

$$ ELBO = E_j[ E_{-j} [logp(\mathbf{x}, z_j, \mathbf{z}_{-j})]] - E_j[logq_j] + Const $$  

첫 번재 항을 최대로 하는 것이 $q_j$ 에 대한 **ELBO**를 최대화하는 길이다. 따라서 $q_j$ 에 대한 **Optimal Solution**은 아래와 같이 표현할 수 있다.  

$$ q^*_j{z_j} \propto exp( E_{-j} [logp(\mathbf{x}, z_j, \mathbf{z}_{-j})] ) $$  

쿨백-라이블리 발산이 아래와 같이 표현되었다는 사실을 기억해보자.  

$$ D_{KL}(Q(x)||P(x)) = E_{X \sim P}[-log \frac{Q(x)}{P(x)}] $$  

위 쿨백-라이블리 발산의 개념을 적용해보면, $q_j$ 에 대한 **ELBO** 식은 $q^{*}_j z_j, q_j(z_j)$ 사이의 Negative 쿨백-라이블리 발산 값을 의미한다.  

따라서 이를 해석해보면, `j번째 잠재변수의 Variational Density`를 `j번째 잠재변수의 최적화된 Variational Density`와 유사하게 만드는 것이 $q_j$ 의 **ELBO**를 최대화하는 것이고, 이러한 과정을 모든 j에 대해, **ELBO**가 수렴할 때까지 반복한다면 우리가 원하는 *q* 함수를 얻을 수 있다는 의미가 된다.  

지금까지 설명한 부분을 정리해보자.  

<center><img src="/public/img/Machine_Learning/2020-07-14-Variational Inference/01.JPG" width="60%"></center>  

**CAVI**의 일반적인 절차는 아래와 같다.  
1) Variational 분포 q를 설정한다.  
2) 각 잠재 변수의 Gradient를 잡아 각 $q_j$ 를 최적화한다.  
3) ELBO를 계산한다.  
4) ELBO가 수렴할 때 까지 위 과정을 반복한다.  

**CAVI**는 클래식하고 좋은 방법론이지만, Non-Convex 최적화 문제에서 `Global Optimum`에 도달할 것이라고 보장해주지는 못한다. 즉, 충분히 쿨백-라이블리 발산 값을 최소화하지 못할 수도 있다는 뜻이다. 또한 MCMC와 같은 Posterior Estimation 보다는 (최적화 방법이기에) 속도가 빠르지만 한 쪽을 고정하고 다른 쪽을 교대로 계산하는 방법을 채택하고 있기 때문에 기본적으로 속도가 아주 빠르지는 않다는 단점도 지니고 있다.  

이에 대한 보완책으로 여러 방법론이 대두되었는데, Stochastic Gradient Descent, Convex Relaxation, Monte Carlo Sampling 등의 개념을 활용한 알고리즘들이 등장하였다. 글 서두에서도 밝혔듯이 이러한 알고리즘들에 대해서는 후속 글에서 다루도록 하겠다.  


---
## Reference  
1) [변분추론 설명 블로그1](https://ratsgo.github.io/generative%20model/2017/12/19/vi/)  
2) [변분추론 설명 블로그2](https://zhiyzuo.github.io/VI/)  
3) [패턴인식-머신러닝 책 정리 사이트](http://norman3.github.io/prml/docs/chapter09/4.html)  
4) [변분추론 논문](https://arxiv.org/abs/1601.00670)