---
layout: post
title: Variational AutoEncoder 설명
author: Youyoung
categories: [Generative Model]
tags: [Machine Learning, Paper_Review]
---

본 글의 주제는  2014년에 발표된 생성 모델인 Variational AutoEncoder에 대해 설명하고 이를 코드로 구현하는 내용을 담고 있다. **VAE**에 대해서 알기 위해서는 **Variational Inference** (변분 추론)에 대한 사전지식이 필요하다. 이에 대해 알고 싶다면 [이 글](https://greeksharifa.github.io/bayesian_statistics/2020/07/13/Variational-Inference/)을 참조하길 바란다.  

본 글은 크게 3가지 파트로 구성되어 있다. Chapter1에서는 VAE 논문을 리뷰할 것이다. Chapter2에서는 논문에서 언급되었던 몇 가지 개념에 대해 상세히 서술할 것이다. Chapter3에서는 Tensorflow를 통해 VAE를 구현할 것이다.  


---
# 1. Auto-Encoding Variational Bayes 논문 리뷰  
## 1.1. Introduction  
연속형 잠재 변수와 파라미터가 다루기 힘든 사후 분포를 갖는 방향성 확률 모델에 대해 효율적인 근사 추론 및 학습을 수행할 수 있는 방법이 없을까? **Variational Bayesian** 접근법은 다루기 힘든 사후 분포에 대한 근사의 최적화를 내포한다.  

불행히도, 일반적인 Mean-Field 접근법은 근사적 사후 분포에 대해 기댓값의 분석적 해결법을 요구하는데 이는 보통 굉장히 intractable한 방법이다. 본 논문은 **Variational Lower Bound**(ELBO)의 `Reparameterization`이 Lower Bound의 미분 가능한 불편향 estimator를 만드는 방법에 대해 보여줄 것이다. 이 **Stochastic Gradient Variational Bayes: SGVB estimator**는 연속형 잠재변수나 파라미터를 갖고 있는 대부분의 모델에 대해 효율적인 근사 사후 추론을 가능하게 하며, 표준 Stochastic Gradient Ascent 스킬을 사용하여 최적화하기에 굉장히 편리하다.  

iid 데이터셋이고, 데이터 포인트 별로 연속형 잠재변수를 갖고 있는 경우에 대해 본 논문은 `Auto-Encoding VB` 알고리즘을 제안한다. 이 알고리즘에서는 **Simple Ancestral Sampling**을 이용하여 근사 사후 추론을 하는 인식 모델을 최적화하기 위해 **SGVB estimator**를 사용하여 추론과 학습을 효율적으로 해낸다. 이 과정은 MCMC와 같이 데이터포인트 별로 반복적인 추론을 행하여 많은 연산량을 요구하지 않는 장점을 가진다.  

학습된 근사 사후 추론 모델은 recognition, denoising, representation, visualization의 목적으로 활용될 수 있다. 본 알고리즘이 인식(recognition) 모델에 사용될 때, 이를 `Variational Auto-Encoder`라고 부를 것이다.  

---
## 1.2. Method  
본 섹션에서는 연속형 잠재 변수를 내포하는 다양한 방향성 그래픽 모델(DIrected Graphical Model)에서 Stochastic 목적 함수인 **Lower Bound Estimator**를 끌어내는 과정을 설명할 것이다. 데이터포인트 별 잠재변수는 iid한 상황이라는 가정 하에 본 논문에서는 (전역) 파라미터에 대해 Maximum Likelihood와 Maximum A Posteriori 추론을 수행하고 잠재변수에 대해 `Variational Inference`를 수행할 것이다. 이러한 방법은 온라인 러닝에도 사용될 수 있지만 본 논문에서는 간단히 하기 위해 고정된 데이터셋을 사용할 것이다.  

### 1.2.1. Problem Scenario  
N개의 Sample을 가진 $X$ 라는 데이터가 있다고 해보자. 본 논문은 이 데이터가 관측되지 않은 연속형 확률 변수 $\mathbf{z}$ 를 내포하는 어떤 Random Process에 의해 형성되었다고 가정한다.  

이 과정은 2가지 단계로 구성된다.  

$$ \mathbf{z}^{(i)} \sim Prior: p_{\theta^*}(\mathbf{z}) $$  

$$ \mathbf{x}^{(i)} \sim Conditional Dist: p_{\theta^*}(\mathbf{x}|\mathbf{z}) $$  

(여기서 $\mathbf{z}$는 원인, $\mathbf{x}$는 결과라고 보면 이해가 쉬울 것이다.)  

이 때 우리는 위 2개의 확률이 모두 아래 두 분포의 **Parametric Families of Distributions**에서 왔다고 가정한다.  

$$ p_{\theta}(\mathbf{z}), p_{\theta}(\mathbf{x}|\mathbf{z}) $$  

이들의 확률밀도함수는 거의 모든 $\theta, \mathbf{z}$에 대해 미분가능하다고 전제한다.  

불행히도, 이러한 과정의 많은 부분은 우리가 직접 확인하기 어렵다. True 파라미터인 $\theta^*$ 와 잠재 변수의 값 $\mathbf{z}^{(i)}$ 는 우리에게 알려져 있지 않다.  

본 논문은 주변 확률이나 사후 확률에 대한 단순화를 위한 일반적인 가정을 취하지 않고, 아래에서 제시한 상황처럼 분포가 intractable하고 큰 데이터셋을 마주하였을 경우를 위한 효율적인 알고리즘에 대해 이야기하고자 한다.  

**1) Intractability**  

$$ \int p_{\theta}(\mathbf{x}) p_{\theta}(\mathbf{x}|\mathbf{z}) d\mathbf{z} $$  

(1) Marginal Likelihood $p_{\theta}(\mathbf{x})$ 의 적분은 위 식으로 표현되는데, 이 식이 intractable한 경우가 존재한다. 이 경우는 Evidence가 적분이 불가능한 경우를 의미한다.  

(2) True Posterior Density가 intractable한 경우 (EM알고리즘이 사용될 수 없음)  

True Posterior Density는 아래와 같다.  

$$ p_{\theta}(\mathbf{z}|\mathbf{x}) = p_{\theta}(\mathbf{x}|\mathbf{z}) p_{\theta}(\mathbf{z})/p_{\theta}(\mathbf{x}) $$  


(3) 어떠한 합리적인 Mean-Field VB 알고리즘을 위한 적분이 불가능한 경우  

이러한 Intractability는 굉장히 흔하며, 복잡한 Likelihood 함수를 갖는 신경망 네트워크에서 발견할 수 있다.  

$$ Likelihood: p_{\theta}(\mathbf{x}|\mathbf{z}) $$  
  

**2) A Large Dataset**  

데이터가 너무 크면 배치 최적화는 연산량이 매우 많다. 우리는 작은 미니배치나 데이터포인트에 대해 파라미터 업데이트를 진행하고 싶은데, Monte Carlo EM과 같은 Sampling Based Solution은 데이터 포인트별로 Sampling Loop를 돌기 때문에 너무 느리다.  


위 시나리오에서 설명한 문제들에 대해 본 논문은 아래와 같은 해결책을 제시한다.  

첫 번째로, 파라미터 $\theta$ 에 대한 **효율적인 근사 ML/MAP 추정**을 제안한다. 이 파라미터들은 숨겨진 Random Process를 흉내내고 실제 데이터를 닮은 인공적인 데이터를 생성할 수 있게 해준다.  

두 번째로, 파라미터 $\theta$ 의 선택에 따라 관측값 $\mathbf{x}$ 가 주어졌을 때 **잠재 변수 $\mathbf{z}$ 에 대한 효율적인 근사 사후 추론**을 제안한다.  

세 번째로, 변수 $\mathbf{x}$ **에 대한 효율적인 근사 Marginal Inference**를 제안한다. 이는 $\mathbf{x}$ 에 대한 prior이 필요한 모든 추론 task를 수행할 수 있게 해준다.  

위 문제를 해결하기 위해 아래와 같은 **인식 모델**이 필요하다.  

$$ q_{\phi}(\mathbf{z}|\mathbf{x}) $$  

이 모델은 intractable한 True Posterior의 근사 버전이다.  

기억해야 할 것이, Mean-Field Variational Inference에서의 근사 Posterior와는 다르게 위 인식 모델은 꼭 계승적(factorial)일 필요도 없고, 파라미터 $\phi$ 가 닫힌 형식의 기댓값으로 계산될 필요도 없다.  

본 논문에서는 인식 모델 파라미터인 $\phi$ 와 생성 모델 파라미터인 $\theta$를 동시에 학습하는 방법에 대해 이야기할 것이다.  


|구분|기호|
|--------|--------|
|인식 모델 파라미터| $\phi$ |
|생성 모델 파라미터| $\theta$ |


코딩 이론의 관점에서 보면, 관측되지 않은 변수 $\mathbf{z}$ 는 잠재 표현 또는 *code*라고 해석될 수 있다. 본 논문에서는 인식 모델을 **Encoder**라고 부를 것이다. 왜냐하면 데이터 포인트 $\mathbf{x}$ 가 주어졌을 때 이 **Encoder**가 데이터 포인트 $\mathbf{x}$ 가 발생할 수 code $\mathbf{z}$의 가능한 값에 대한 분포를 생산하기 때문이다.  

비슷한 맥락에서 우리는 생성 모델을 **확률적 Decoder**라고 명명할 것인데, 왜냐하면 code $\mathbf{z}$ 가 주어졌을 때 이 **Decoder**가 상응하는 가능한 $\mathbf{x}$ 의 값에 대해 분포를 생성하기 때문이다.  

$$ Encoder: q_{\phi}(\mathbf{z}|\mathbf{x}) $$  

$$ Decoder: p_{\theta}(\mathbf{x}|\mathbf{z}) $$  


### 1.2.2. The Variational Bound  
Marginal Likelihood는 각 데이터 포인트의 Marginal Likelihood의 합으로 구성된다. 이를 식으로 표현하면 아래와 같다.  

$$ log p_{\theta}(\mathbf{x}^{(1)}, ..., \mathbf{x}^{(N)}) = \sum_{i=1}^N log p_{\theta} (\mathbf{x}^{(i)}) $$  

그런데 데이터 포인트 하나에 대한 Marginal Likelihood는 아래와 같이 재표현이 가능하다.  

$$ log p_{\theta} (\mathbf{x}^{(i)}) = D_{KL} (q_{\phi}(\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta} (\mathbf{z}|\mathbf{x}^{(i)})) + L(\theta, \phi; \mathbf{x}^{(i)}) $$  

우변의 첫 번째 항은 True Posterior의 근사 `KL Divergence`이다. 이 KL Divergence는 음수가 아니기 때문에, 두 번째 항인 $L(\theta, \phi; \mathbf{x}^{(i)})$ 는 i번째 데이터 포인트의 Marginal Likelihood의 `Varitaional Lower Bound`라고 한다. 변분 추론 글을 읽어보고 왔다면 알겠지만, 이는 Evidence의 하한 값을 뜻하기도 하기 때문에 `ELBO`라고 부르기도 한다.  

부등식으로 나타내면 아래와 같다.  

$$ log p_{\theta} (\mathbf{x}^{(i)}) \geq L(\theta, \phi; \mathbf{x}^{(i)}) = E_{q_{\phi} (\mathbf{z}|\mathbf{x})} [-logq_{\phi}(\mathbf{z}|\mathbf{x}) + log p_{\theta}(\mathbf{x}, \mathbf{z})] $$  

이 식은 또 아래와 같이 표현할 수 있다.  

$$ \mathcal{L} (\bm{\theta}, \phi; \mathbf{x}^{(i)}) = -D_{KL} (q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta} (\mathbf{z}) ) + E_{q_{\phi} (\mathbf{z} | \mathbf{x}^{(i)})} [log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}) ] $$  

우리는 **Lower Bound** $L(\theta, \phi; \mathbf{x}^{(i)})$ 를 Variational 파라미터와 생성 파라미터인 $\phi, \theta$ 에 대하여 미분하고 최적화하고 싶은데, 이 Lower Bound의 $\phi$ 에 대한 Gradient는 다소 복잡하다.  

이러한 문제에 대한 일반적인 `Monte Carlo Gradient Estimator`는 아래와 같다.  

$$ \triangledown_{\phi} E_{q_{\phi} (\mathbf{z}|\mathbf{x})} [f(\mathbf{z})] = E_{q_{\phi} (\mathbf{z})} [f( \mathbf{z} \triangledown_{q_{\phi} (\mathbf{z})} log q_{\phi} (\mathbf{z}) ] \simeq \frac{1}{L} \Sigma_{l=1}^L f(\mathbf{z}) \triangledown_{q_{\phi} (\mathbf{z}^{(l)})} log q_{\phi} (\mathbf{z}^{(l)}) $$  

$$ Where: \mathbf{z}^{(l)} \sim q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)}) $$  

그런데 이 Gradient Estimator는 굉장히 큰 분산을 갖고 있어서 우리의 목적에 적합하지 않다.  


### 1.2.3. The SGVB estimator and AEVB algorithm  
이 섹션에서는 **Lower Bound**의 실용적인 추정량과 파라미터에 대한 그 추정량의 미분 값을 소개할 것이다. 

$$ q_{\phi} (\mathbf{z}|\mathbf{x}) $$  

일단 우리는 위와 같은 **근사 Posterior**를 가정한다. 다만 x에 대한 조건부를 가정하지 않은 케이스인 $q_{\phi}(\mathbf{z})$ 와 같은 경우에도 같은 테크닉을 적용할 수 있음에 유의하자. Posterior를 추론하는 **Fully Variational Bayesian** 방법론은 본 논문의 부록에 소개되어 있다.  

위에서 설정한 **근사 Posterior**에 대해 **1.2.4** 섹션에서 정한 certain mild conditions 하에 우리는 그 **근사 Posterior**를 따르는 확률 변수 $\tilde{\mathbf{z}}$ 를 `Reparameterize` 할 수 있는데, 이는 Auxiliary Noise 변수 $\epsilon$ 의 **Diffrentiable Transformation**($g_{\phi} (\epsilon, \mathbf{x})$)를 통해 이루어진다.  

$$ \tilde{\mathbf{z}} = q_{\phi} (\mathbf{z}|\mathbf{x})  $$  

$$ \tilde{\mathbf{z}} \sim g_{\phi} (\epsilon, \mathbf{x}), \epsilon \sim p(\epsilon) $$  

다음 Chapter를 보면 적절한 $p(\epsilon)$ 분포와 $g_{\phi} (\epsilon, \mathbf{x})$ 함수를 선정하는 일반적인 방법에 대해 알 수 있다. 이제 이 Chapter 서두에서 언급한 **근사 Posterior**에 대해 어떤 함수 $f(\mathbf{z})$ 기댓값의 `Monte Carlo 추정량`을 다음과 같이 쓸 수 있다.  

$$ E_{q_{\phi} (\mathbf{z}|\mathbf{x})} [f(\mathbf{z})] = E_{p(\epsilon)} [f(g_{\phi} (\epsilon, \mathbf{x}^{(i)}))] \approx \frac{1}{L} \Sigma_{l=1}^L f(g_{\phi} (\epsilon^{(l)}, \mathbf{x}^{(i)})) $$  

이는 실제로 계산하기 어려운 어떤 함수에 대해 $L$ 개의 Sample를 뽑아 이에 대한 근사치로 추정량을 구하는 방법이다.  

이제 이 테크닉을 **ELBO**에 대해 적용하면 **SGVB: Stochastic Gradient Variational Bayes** 추정량을 얻을 수 있다.  

$$ \tilde{\mathcal{L}}^A (\theta, \phi, \mathbf{x}^{(i)}) \simeq \mathcal{L} (\theta, \phi, \mathbf{x}^{(i)}) $$  

$$ \tilde{\mathcal{L}}^A (\theta, \phi, \mathbf{x}^{(i)}) = \frac{1}{L} \Sigma_{l=1}^L log p_{\theta} (\mathbf{x}^{(i)}, \mathbf{z}^{(i, l)}) - logq_{\phi} (\mathbf{z}^{(i, l)}|\mathbf{x}^{(i)}) $$  

$$ g_{\phi} (\epsilon^{(i, l)}, \mathbf{x}^{(i)}), \epsilon^{(l)} \sim p(\epsilon) $$  

잠시 아래 식에서 쿨백-라이블리 발산에 주목해보자.  

$$ \mathcal{L} (\theta, \phi; \mathbf{x}^{(i)}) = -D_{KL} (q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta} (\mathbf{z}) ) + E_{q_{\phi} (\mathbf{z} | \mathbf{x}^{(i)})} [log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}) ] $$  

이 쿨백-라이블리 발산 값은 종종 analytic 하게 적분될 수 있는데 이렇게 되면 오직 `Expected Reconstruction Error`만이 샘플링에 의한 추정을 필요로하게 된다. `Expected Reconstruction Error` 항은 아래 식을 의미한다.  

$$ E_{q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)})} [ logp_{\theta} (\mathbf{x}^{(i)}|\mathbf{z}) ] $$  

쿨백-라이블리 발산 항은 **근사 Posterior**를 **Prior** $p_{\theta}(z)$ 에 가깝게 만들어서 $\phi$ 를 규제하는 것으로 해석될 수 있다.  

$$ KL: p_{\theta} (\mathbf{z}|\mathbf{x}) \to p_{\theta}(\mathbf{z}) $$

이러한 과정을 **SGVB** 추정량의 두 번째 버전으로 이어지는데, 이 추정량은 일반적인 추정량에 비해 작은 분산을 갖고 있다.  

$$ \tilde{\mathcal{L}}^B (\theta, \phi, \mathbf{x}^{(i)}) = -D_{KL} (q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta} (\mathbf{z}) ) + \frac{1}{L} \Sigma_{l=1}^L log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}^{(i, l)}) $$  

이 때  

$$ \mathbf{z}^{(i, l)} = g_{\phi} (\epsilon^{(i, l)}, \mathbf{x}^{(i)}), \epsilon^{(l)} \sim p(\epsilon) $$  

$N$ 개의 데이터 포인트를 갖고 있는 데이터셋 $X$ 에서 복수의 데이터 포인트가 주어졌을 때 우리는 미니배치에 기반하여 전체 데이터셋에 대한 **Marginal Likelihood Lower Bound**의 추정량을 구성할 수 있다.  

$$ \mathcal{L} (\theta, \phi, X) \simeq \tilde{\mathcal{L}}^M (\theta, \phi, X^M) = \frac{N}{M} \Sigma_{i=1}^M \tilde{\mathcal{L}} (\theta, \phi, \mathbf{x}^{(i)}) $$  

이 때  

$$ X^M = [\mathbf{x}^{i}]_{i=1}^M $$  

미니배치 $X^M$ 은 전체 데이터셋 $X$ 에서 랜덤하게 뽑힌 샘플들을 의미한다. 본 논문의 실험에서, 미니 배치 사이즈 $M$ 이 (예를 들어 100) 충분히 크면 데이터 포인트 별 sample의 크기인 $L$ 이 1로 설정될 수 있다는 사실을 알아 냈다.  

$$ \triangledown_{\theta, \phi} \tilde{\mathcal{L}} (\theta; X^M) $$  

위와 같은 derivate가 도출될 수 있고, 이에 따른 Gradients는 **SGD**나 **Adagrad**와 같은 확률적 최적화 방법과 연결되어 사용될 수 있다.  

다음은 기본적인 Stochastic Gradients를 계산하는 방법에 대한 내용이다.  

<center><img src="/public/img/Machine_Learning/2020-07-31-Variational AutoEncoder/03.JPG" width="100%"></center>  

아래 목적 함수를 보면 **Auto-Encoder**와의 연결성이 더욱 뚜렷해진다.  

$$ \tilde{\mathcal{L}}^B (\theta, \phi, \mathbf{x}^{(i)}) = -D_{KL} (q_{\phi} (\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta} (\mathbf{z}) ) + \frac{1}{L} \Sigma_{l=1}^L log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}^{(i, l)}) $$  

**Prior**로부터 나온 **근사 Posterior**에 대한 쿨백-라이블리 발산 값인 첫 번째 항은 `Regularizer`의 역할을 하며, 두 번째 항은 `Expected Negative Reconstruction Error`의 역할을 하게 된다.  

$g_{\phi} (.)$ 라는 함수는 데이터 포인트 $\mathbf{x}^{(i)}$ 와 Random Noise Vector $\epsilon^{(l)}$ 을 데이터 포인트 $\mathbf{z}^{(i, l)}$ 을 위한 **근사 Posterior**로 부터 추출된 Sample로 매핑하는 역할을 수행한다.  

$$ \tilde{\mathbf{z}} = q_{\phi} (\mathbf{z}|\mathbf{x}), \tilde{\mathbf{z}} \sim g_{\phi} (\epsilon, \mathbf{x}), \epsilon \sim p(\epsilon) $$  

그 후, 이 Sample $\mathbf{z}^{(i, l)}$ 은 아래 함수의 Input이 된다.  

$$ log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}^{(i, l)}) $$  

이 함수는 $\mathbf{z}^{(i, l)}$ 이 주어졌을 때 생성 모델 하에서 데이터 포인트 $\mathbf{x}^{(i)}$ 의 확률 밀도 함수와 동일하다. 이 항은 Auto-Encoder 용어에서 `Negative Reconstruction Error`에 해당한다.  


### 1.2.4. The Reparamaterization Trick  

$$ q_{\phi} (\mathbf{z}|\mathbf{x}) $$  

위에 제시된 **근사 Posterior**로부터 Sample을 생성하기 위해 본 논문에서는 다른 방법을 적용하였다. 본질적인 Parameterization 트릭은 굉장히 단순하다. $\mathbf{z}$ 가 연속형 확률 변수이고 아래와 같이 어떤 조건부 분포를 따른다고 하자.  

$$ \mathbf{z} \sim q_{\phi} (\mathbf{z}|\mathbf{x}) $$  

그렇다면 이제 이 확률 변수 $\mathbf{z}$ 를 다음과 같은 `Deterministic 변수`라고 표현할 수 있다.  

$$ \mathbf{z} = g_{\phi} (\epsilon, \mathbf{x}) $$  

이 때 $\epsilon$ 은 독립적인 Marginal $p(\epsilon)$ 을 가지는 보조 변수이고, $g_{\phi} (.)$ 함수는 $\phi$ 에 의해 parameterized 되는 vector-valued 함수이다.  

이 `Reparameterization`은 매우 유용하다. 왜냐하면 **근사 Posterior**의 기댓값을 $\phi$에 대해 미분 가능한 기댓값의 Monte Carlo 추정량으로 재표현하는 데에 사용될 수 있기 때문이다.  

증명은 다음과 같다. $z = g_{\phi} (\epsilon, \mathbf{x})$ 라는 `Deterministic Mapping`이 주어졌을 때 우리는 다음 사실을 알 수 있다.  

$$ q_{\phi} (\mathbf{z}|\mathbf{x}) \prod_i d z_i = p(\epsilon) \prod_i d\epsilon_i $$  

이는 각 Density에 derivatives를 곱한 것이 서로 같다는 것을 의미한다. 참고로 위에서 $dz_i$ 라고 표기한 것은, 무한소에 대한 수식이기 때문이며 사실 $d\mathbf{z} =  \prod_i dz_i$ 이다. 따라서 아래와 같은 식을 구성할 수 있다.  

$$ \int q_{\phi} (\mathbf{z}|\mathbf{x}) f(\mathbf{z}) d\mathbf{z} = \int p(\epsilon) f(\mathbf{z}) d\mathbf{z} = \int p(\epsilon) f(g_{\phi} (\epsilon, \mathbf{x}) ) d\epsilon $$  

마지막 식을 보면 $\mathbf{z}$ 를 `Deterministic` 하게 표현하여 오로지 $\epsilon, \mathbf{x}$ 로만 식을 구성한 것을 알 수 있다. 이제 미분 가능한 추정량을 구성할 수 있다.  

$$ f(g_{\phi} (\epsilon, \mathbf{x}) ) d\epsilon \simeq  \frac{1}{L} \Sigma_{l=1}^L f( g_{\phi} (\mathbf{x}, \epsilon^{(l)}) ) $$  

사실 이는 이전 Chapter에서 **ELBO**의 미분 가능한 추정량을 얻기 위해 적용하였던 트릭이다.  

예를 들어 단변량 정규분포의 케이스를 생각해보자.  

$$ z \sim p(z|x) = \mathcal{N} (\mu, \sigma^2) $$  

이 경우 적절한 `Reparameterization`은 $z = \mu + \sigma \epsilon$이다. 이 때 $\epsilon$은 $\mathcal{N} (0, 1)$을 따르는 보조 Noise 변수이다. 그러므로 우리는 다음 식을 얻을 수 있다.  

$$ E_{\mathcal{N} (z; \mu, \sigma^2)} [f(z)] = E_{\mathcal{N} (\epsilon; 0, 1)} [f(\mu + \sigma \epsilon)] \simeq \frac{1}{L} \Sigma^L_{l=1} f(\mu + \sigma \epsilon^{(l)}), \epsilon^{(l)} \sim \mathcal{N} (0, 1)  $$  

그렇다면 어떤 **근사 Posterior**에 따라 미분 가능한 변환 함수 $g_{\phi}(.)$ 와 보조 변수 $\epsilon \sim p(\epsilon)$ 은 어떻게 선택하는가? 기본적인 방법은 다음과 같다.  

첫 번째로, Tractable Inverse CDF를 선택한다. $\epsilon \sim \mathcal{U} (\mathbf{0}, \mathbf{I})$ 일 때, $g_{\phi} (\epsilon, \mathbf{x})$ 는 **근사 Posterior**의 Inverse CDF라고 해보자. 예를 들면, Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Reciprocal, Gompertz, Gumbel, Erlang 분포가 있다.  

두 번째로, 정규 분포 예시와 유사하게 어떠한 **location-scale** family of distributions에 대해 우리는 location=0, scale=1인 표준 분포를 보조 변수 $\epsilon$ 으로 선택할 수 있다.  

그렇다면 $g(.)$ = location + scale * $\epsilon$ 이 될 것이다. 예를 들면, Laplace, Elliptical, T, Logistic, Uniform, Triangular, Gaussian이 있다.  

마지막으로 분포를 결합하는 방식을 채택할 수 있다. 종종 확률 변수를 보조 변수의 다른 변환으로 표현할 수 있다. 예를 들어 Log-Normal, Gamma, Dirichlet, Beta, Chi-Squared, F가 있다.  

이 3가지 방법이 모두 통하지 않는다면, Inverse CDF의 좋은 근사법은 PDF에 비해 많은 연산량과 소요 시간을 요한다.  


---
## 1.3. Example: Variational Auto-Encoder  
이번 Chapter에서는 확률론적 Encoder (생성 모델의 Posterior의 근사) 와 `AEVB` 알고리즘을 통해 파라미터 $\phi$ 와 $\theta$ 가 Jointly 최적화되는 신경망에 대한 예시를 다루도록 할 것이다.  

잠재 변수에 대한 **Prior**는 `Centered Isotropic Multivariate Gaussian` $p_{\theta} (\mathbf{z}) = \mathcal{N} (\mathbf{z}; \mathbf{0}, \mathbf{I})$ 라고 하자. 이 경우에서는 **Prior**에 파라미터가 없다는 사실을 염두에 두자. 

$$ p_{\theta} (\mathbf{x} | \mathbf{z}) $$  

위 분포는 다변량 정규 분포 혹은 베르누이 분포(각각 실수 데이터, 또는 Binary 데이터 일때)로 설정하고, 분포의 파라미터는 **Multi Layer Perceptron**에서 $\mathbf{z}$ 로 계산된다.  

$$ p_{\theta} (\mathbf{z} | \mathbf{x}), q_{\phi} (\mathbf{z} | \mathbf{x}) $$  

이 경우 위 식의 왼쪽 부분인 **True Posterior**는 intractable하다. 위 식의 오른쪽 부분인 **근사 Posterior**에 더 많은 자유가 있기 때문에, 우리는 **True(but intractable) Posterior**가 Approximately Diagonal Covariance를 가진 근사 정규분포를 따른다고 가정한다. 이 경우 우리는 `Variational Approximate Posterior`가 Diagonal Covariance 구조를 가진 다변량 정규분포라고 설정할 수 있다.  

$$ log q_{\phi} (\mathbf{z} | \mathbf{x}^{(i)}) = log \mathcal{N} (\mathbf{z}; {\mu}^{(i)}, {\sigma}^{2(i)} {I}) $$  

이 때 **근사 Posterior**의 평균과 표준편차는 **MLP**의 Output이다. 이 때 MLP는 `Variational Parameter`인 $\phi$ 와 데이터 포인트 $\mathbf{x}^{(i)}$ 의 비선형적 함수라고 생각할 수 있다.  

**1.2.4** 에서 설명하였듯이, 우리는 다음과 같이 **근사 Posterior**에서 Sampling을 한다.  

$$ \mathbf{z}^{(i, l)} \sim q_{\phi} (\mathbf{z} | \mathbf{x}^{(i)}) $$  

이 때 아래 사실을 이용한다.  

$$ \mathbf{z}^{(i, l)} = g_{\phi} (\mathbf{x}^{(i)} , {\epsilon}^{(l)} ) = {\mu}^{(i)} + {\sigma}^{(i)} \odot {\epsilon}^{(l)}, {\epsilon}^{(l)} \sim \mathcal{N} (\mathbf{0}, \mathbf{I}) $$  

이 모델에서 **Prior**와 **근사 Posterior**는 정규 분포를 따른다. 그러므로 우리는 이전 Chapter에서 보았던 추정량을 사용할 수 있는데, 이 때 쿨백-라이블리 발산 값은 추정 없이 계산되고 미분될 수 있다.  

$$ \mathcal{L} (\theta, \phi, \mathbf{x}^{(i)}) \simeq \frac{1}{2} \Sigma_{j=1}^J (1 + log( \sigma_j^{(i)} )^2 - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2  ) + \frac{1}{L} \Sigma_{l=1}^L log p_{\theta} (\mathbf{x}^{(i)} | \mathbf{z}^{(i, l)}) $$  

이 때  

$$ \mathbf{z}^{(i, l)} = g_{\phi} (\mathbf{x}^{(i)} , {\epsilon}^{(l)} ) = {\mu}^{(i)} + {\sigma}^{(i)} \odot {\epsilon}^{(l)}, {\epsilon}^{(l)} \sim \mathcal{N} (\mathbf{0}, \mathbf{I}) $$  

$$ log p_{\theta} (\mathbf{x}^{(i)}, \mathbf{z}^{(i, l)}) $$  

바로 위에 있는 `Decoding Term`은 우리가 모델링하는 데이터의 종류에 따라 베르누이 또는 정규분포 **MLP**가 된다.  


---
## 1.4. Related Work  
`Wake-Sleep` 알고리즘은 연속적인 잠재 변수를 갖고 있는 같은 문제를 갖고 있는 상황에 적용할 수 있는 다른 방법론이다. 본 논문에서 제시된 방법과 마찬가지로 이 알고리즘은 **True Posterior**를 근사하기 위해 인식 모델을 사용한다. 그러나 이 이 알고리즘의 경우 동시에 발생하는 2개의 목적함수를 최적화해야 하기 때문에 **Marginal Likelihood**의 최적화에 적합하지 않다. 이 알고리즘은 이산적인 잠재변수를 다루는 모델에 적합하다. `Wake-Sleep` 알고리즘의 데이터 포인트 별 계산 복잡도는 `AEVB`와 같다.  

`Stochastic Variational Inference`는 주목 받는 알고리즘이다. [이 논문](https://icml.cc/2012/papers/687.pdf)에서는 2.1 장에서 Naive Gradient Estimator의 높은 분산을 감소시키기 위해 Control Variate Scheme을 소개하였고, 이를 **Posterior**의 **Exponential Family Approximation**에 적용하였다.  

`AEVB` 알고리즘은 Variational Objective로 학습되는 `Directed Probabilistic Model`과 `Auto-Encoder` 사이의 연결성을 밝혀준다. 선형적인 `Auto-Encoder`와 `Generative Linear-Gaussian` 모델의 특정 종류 사이의 연결성은 오래 전부터 알려져왔다. [이 논문](https://papers.nips.cc/paper/1398-em-algorithms-for-pca-and-spca.pdf)은 **PCA**가 아래 조건에 해당하는 `Linear-Gaussian Model`의 **Maximum Likelihood**라는 점을 밝히고 있다. (특히 매우 작은 $\epsilon$ 일 때) 

$$ p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I}), p(\mathbf{x}|\mathbf{z}) = \mathcal{N}( \mathbf{x}; \mathbf{Wz}, \epsilon \mathbf{I} ) $$  

`Auto-Encoder`와 관련된 연구 중에서 [이 논문](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)은  unregularized auto-encoder는 input $X$ 와 잠재 표현 $Z$ 의 Mutual Information의 **Maximum Lower Bound**라는 것을 보였다. Mutual Information을 최대화하는 것은 결국 조건부 엔트로피를 최대화하는 것과 같은데, 이 조건부 엔트로피는 autoencoding 모델 하에서 데이터의 expected loglikelihood에 의해 lower bounded된다. (`Negative Reconstruction Error`)  

그러나 이 Reconstruction 기준은 유용한 표현을 학습하기에는 sufficient하지 않다. denoising, contractive, sparse autoencoder 변인들이  `Auto-Encoder`로 하여금 더 유용한 표현을 학습하도록 제안되었다.  

`SGVB` 목표함수는 **Variational Bound**에 의해 좌우되는 규제화 항을 포함하며, 일반적인 불필요한 규제화 하이퍼 파라미터들은 포함하지 않는다.  

Predictive Sparse Decomposition과 같은 Encoder-Decoder 구조 역시 본 논문에서 영감을 받은 방법론이다. **Generative Stochastic Network**를 발표한 [이 논문](https://arxiv.org/abs/1306.1091)에서는 Noisy autoencoder가 데이터 분포로부터 Sampling을 하는 **Markov Chain**의 Transition Operator를 학습한다는 내용이 소개되어 있다. 다른 논문에서는 Deep Boltzmann 머신과 함께 인식 모델이 사용되기도 하였다. 이러한 방법들은 비정규화된 모델들에 타겟팅되어 있거나 희소 코딩 모델에 있어 제한적이며, 본 논문에서 제안된 알고리즘은 `Directed Probabilisitic Model`의 일반적인 경우를 학습할 수 있기 때문에 차별화 된다.  


---
## 1.5. Experiments  






---
# 2. 이론에 대한 보충 설명  





---
# 3. Tensorflow로 VAE 구현  



---
# Reference  
1) https://ratsgo.github.io/generative%20model/2018/01/27/VAE/  
2) https://www.youtube.com/watch?v=SAfJz_uzaa8  
3) https://taeu.github.io/paper/deeplearning-paper-vae/
4) 