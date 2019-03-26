---
layout: post
title: GAN의 개선 모델들(Semi-supervised GAN, catGAN, ACGAN, LSGAN, WGAN, WGAN_GP, DRAGAN, infoGAN, EBGAN, BEGAN, SGAN)
author: YouWon
categories: [Generative Model]
tags: [GAN, Machine Learning, CNN, Generative Model, Paper_Review]
---

---

이 글에서는 LSGAN, WGAN, WGAN_GP, DRAGAN, infoGAN, ACGAN, EBGAN, BEGAN 등에 대해 알아보도록 하겠다.

각각에 대해 간단히 설명하면, 

- **catGAN(Categorical GAN):** D가 real/fake만 판별하는 대신 class label/fake class을 출력하도록 바꿔서 unsupervised 또는 semi-supervised learning이 가능하도록 하였고 또한 더 높은 품질의 sample을 생성할 수 있게 되었다.
- **Semi-supervised GAN:** catGAN과 거의 비슷하다. original GAN과는 달리 DCGAN을 기반으로 만들어졌다.
- **:** 
- **:** 
- **:** 
- **:** 
- **:** 
- **:** 
- **:** 
- **:** 



논문을 적절히 번역 및 요약하는 것으로 시작한다. 많은 부분을 생략할 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# catGAN

논문 링크: **[catGAN](https://arxiv.org/abs/1511.06390)**

데이터의 전체 또는 일부가 unlabeled인 경우 clustering은 $p_x$를 직접 예측하는 generative model과 분포를 예측하는 대신 데이터를 직접 잘 구분된 카테고리로 묶는 discriminative model로 나누어지는데, 이 모델에서는 이 두 아이디어를 합치고자 했다.  
논문에서 이 **catGAN**은 original GAN이 $real, fake$만 구분하던 것을 real인 경우에는 그 class가 무엇인지까지 구분하게($C_1, C_2, ..., C_N, C_{fake}$)했다는 점에서 original GAN의 일반화 버전이라고 하였으며, 또한 [RIM(Regularized Information Maximization)](https://papers.nips.cc/paper/4154-discriminative-clustering-by-regularized-information-maximization)에서 regularization이 추가가 되었듯 catGAN에선 G가 D에 대한 regularization을 하기 때문에 RIM의 확장판이라고도 하였다.

<center><img src="/public/img/2019-03-20-advanced-GANs/catGAN1.png" width="50%"></center>

RIM에서 최적의 unsupervised classifier의 목적함수로 엔트로피를 사용하였듯 catGAN도 목적함수로 엔트로피 개념을 사용한다. 아래는 논문에 나온 그림이다.

<center><img src="/public/img/2019-03-20-advanced-GANs/catGAN2.png" width="100%"></center>

왼쪽에서 초록색은 G(generate라고 되어 있다), 보라색은 D를 의미한다. 여기서 H는 엔트로피이다.  

오른쪽 그림을 보면, D의 입장에서는:

- i) real data는 실제 class label을 딱 하나 갖고 있기 때문에 해당하는 label일 확률만 1에 가깝고 나머지는 0이어야 한다. 따라서 엔트로피( $ H[p(y \vert x, D)] $ )를 최소화한다.
- ii) fake data의 경우 특정 class에 속하지 않기 때문에 class label별로 확률은 비슷해야 한다. 따라서 엔트로피$H[p(y \vert x, G(z))]$를 최대화한다.
- iii) 학습 sample이 특정 class에 속할 확률이 비슷해야 한다는 가정을 했기 때문에, input data $x$에 대한 marginal distribution(주변확률분포)의 엔트로피($H[p(y \vert D)]$)가 최대가 되어야 한다. 

G의 입장에서는:

- D를 속여야 하기 때문에 G가 만든 가짜 데이터는 가짜임에도 특정 class에 속한 것처럼 해야 한다. 즉, D의 i) 경우처럼 엔트로피($H[p(y \vert x, G(z))]$)를 최소화한다.
- 생성된 sample은 특정 class에 속할 확률이 비슷해야 하기 때문에 marginal distribution의 엔트로피($H[p(y \vert D)]$)가 최대화되어야 한다.

따라서 D와 G의 목적함수를 정리하면,

$$ L_D = max_D ~~~ H_{\chi}[p(y| D)] - \mathbb{E}_{x\sim \chi} [H[p(y|x, D)]] + \mathbb{E}_{z\sim P(z)}[H[p(y|G(z), D)]] $$

$$ L_G = min_G ~~~ H_G[p(y| D)] + \mathbb{E}_{z\sim P(z)}[H[p(y|G(z), D)]] $$

다만 $L_D$의 마지막 항을 직접 구하는 것은 어렵기 때문에, $z \sim P(z) $를 $M$개 뽑아 평균을 계산하는 몬테카를로 방법을 쓴다.

위 목적함수를 사용하여 실험한 결과는 다음과 같다.

<center><img src="/public/img/2019-03-20-advanced-GANs/catGAN3.png" width="100%"></center>

Unsupervised catGAN은 9.7%의 error를 보이는 데 반해 $n=100$만의 labeled data가 있는 버전의 경우 error가 1.91%까지 떨어진다. $n=1000$, $n=전체$인 경우 error는 점점 떨어지는 것을 볼 수 있다. 즉, 아주 적은 labeled data를 가진 semi-supervised learning이라도 굉장히 쓸모있다는 뜻이다.

또한 k-means나 RIM과 비교했을 때 두 원을 잘 분리해내는 것을 볼 수 있다.

<center><img src="/public/img/2019-03-20-advanced-GANs/catGAN4.png" width="100%"></center>

MNIST나 CIFAR-10 데이터도 잘 생성해내는 것을 확인하였다.

<center><img src="/public/img/2019-03-20-advanced-GANs/catGAN5.png" width="100%"></center>

---

# Semi-supervised GAN

논문 링크: **[Semi-supervised GAN](https://arxiv.org/abs/1606.01583)**

위의 catGAN과 거의 비슷한 역할을 한다. 전체적인 구조도 비슷하다.

논문 자체가 짧고 목적함수에 대한 내용이 없어서 자세한 설명은 생략한다. 특징을 몇 개만 적자면, 

- original GAN과는 달리 sigmoid 대신 softmax를 사용하였다. $N+1$개로 분류해야 하니 당연하다.
- DCGAN을 기반으로 작성하였다.
- D가 classifier의 역할을 한다. 그래서 논문에서는 D/C network라고 부른다(D이자 C).
- classifier의 정확도는 sample의 수가 적을 때 CNN보다 더 높다는 것을 보여주었다. sample이 많을 때는 거의 같았다.
- original GAN보다 생성하는 이미지의 품질이 좋다.

<center><img src="/public/img/2019-03-20-advanced-GANs/semiGAN.png" width="60%"></center>

---

# ACGAN

논문 링크: **[ACGAN](https://arxiv.org/abs/1610.09585)**


---

# LSGAN

논문 링크: **[LSGAN](https://arxiv.org/abs/1611.04076)**


---

# WGAN

논문 링크: **[WGAN](https://arxiv.org/abs/1701.07875)**


---

# Improved WGAN

논문 링크: **[WGAN](https://arxiv.org/abs/1704.00028)**


---

# DRAGAN

논문 링크: **[DRAGAN](https://arxiv.org/abs/1705.07215)**


---

# infoGAN

논문 링크: **[infoGAN](https://arxiv.org/abs/1606.03657)**


---

# EBGAN

논문 링크: **[EBGAN](https://arxiv.org/abs/1609.03126)**


---

# BEGAN

논문 링크: **[BEGAN](https://arxiv.org/abs/1703.10717)**


---

# SGAN

논문 링크: **[SGAN](https://arxiv.org/abs/1712.02330)**


---

# 이후 연구들

GAN 이후로 수많은 발전된 GAN이 연구되어 발표되었다. 

---
