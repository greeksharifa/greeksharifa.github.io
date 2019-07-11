---
layout: post
title: Deep Learning Tutorial(딥러닝 튜토리얼) 01. 소개
author: YouWon
categories: [Deep Learning Tutorial]
tags: [Deep Learning]
---

---

**[Deep Learning Tutorial(딥러닝 튜토리얼) 01. 소개](https://greeksharifa.github.io/)**  



---

이 글에서는 Deep Learning(딥러닝)을 소개하고 그 기초를 다룬다.

---

## Deep Learning(딥러닝)이란?

### 직관적인 이해

여러분은 A 회사의 주식 가격을 예측하고자 한다. 그러기 위해서 A 회사에 대한 정보를 수집하였다.

- A의 설립일자
- A의 재작년 수익
- A의 작년 수익
- A의 대표의 나이
- A의 본사가 위치한 국가의 소득수준

그리고 여러분은 수학에서 $x$를 입력하면 $y$가 나오는 함수처럼, 이 정보들을 가지고 주식 가격을 추정해보려고 한다. 위 5개의 요인 중 어떤 것이 중요할 지는 모르지만 대충 다음과 같이 그래프를 그렸다고 하자.

<center><img src="/public/img/2019-06-10-Deep-Learning-01/01.png" width="100%" alt="A의 주가를 예측하라!"></center>

(저 그래프가 정말 맞는지는 우선 논외로 한다. 이걸 잘 설계하는 것이 딥러닝에서는 **매우** 중요하다)

모든 딥러닝이 이렇게 흘러가지는 않지만, 딥러닝은 대충 이런 것이다. 약간 더 자세히 설명하면,

- 입력($\mathbf{X} = $ *{A의 설립일자, A의 재작년 수익, A의 작년 수익, A의 대표의 나이, A의 본사가 위치한 국가의 소득수준}* )을 받아서
- 학습을 시켜놓은 **네트워크(심층신경망, DNN, Deep Neural Network)** 에 집어넣으면
- 출력($\mathbf{\hat{Y}} = $ *{A의 주식 가격}*)을 내놓는

이런 네트워크를 설계하고, 학습시키고, 테스트하는 그런 과정을 포함한다.

### 그래서 Deep Learning이 뭔데?

간단히 얘기하자면 Deep Neural Network(심층신경망)을 설계하고 학습시켜 다음 출력을 생성하는 것이다.

[국문 위키피디아](https://ko.wikipedia.org/wiki/%EB%94%A5_%EB%9F%AC%EB%8B%9D#cite_ref-1)를 인용해보자.

> 딥 러닝(영어: deep learning), 심층학습(深層學習)은 여러 비선형 변환기법의 조합을 통해 높은 수준의 추상화(abstractions, 다량의 데이터나 복잡한 자료들 속에서 핵심적인 내용 또는 기능을 요약하는 작업)를 시도하는 기계학습(machine learning) 알고리즘의 집합[1] 으로 정의되며, 큰 틀에서 사람의 사고방식을 컴퓨터에게 가르치는 기계학습의 한 분야

[영문 위키피디아](https://en.wikipedia.org/wiki/Deep_learning)도 인용해보자.

> Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised.

해석하면,

> 딥러닝(심층구조학습 또는 구조적학습)은 인공신경망에 근거한 넓은 범위의 기계학습방법의 한 부분이다. 학습 방식에는 지도(감독)을 받거나, 지도을 일부만 받거나, 받지 않는 방법이 있다.

**기계학습(Machine Learning)**은 컴퓨터가 스스로 학습하여 예측모형을 개발하는 인공지능의 한 분야이다.

하나의 용어를 설명하려면 더 많은 용어들을 설명해야 한다. 바로 지도학습으로 넘어가자.

#### 지도학습(Supervised Learning)은 또 무엇인가?

다른 이름으로는 감독학습, 교사학습으로도 불린다.

이번엔 A의 주가 말고 그냥 아라비아 숫자를 생각해보자. 여러분은 다음과 같은 과제를 받았다.

> 손으로 쓴 숫자 이미지가 주어지면, 해당 이미지에는 0~9 중 어떤 숫자가 쓰여 있는지 판별하라.

<center><img src="/public/img/2019-06-10-Deep-Learning-01/02.png" width="100%" alt="각 숫자 이미지에는 레이블이 있다."></center>

위의 각 숫자 이미지에는 **Label**이 달려 있는 것을 확인할 수 있다. 지도학습은 이와 같이 각 데이터에 레이블이 있는 상태에서 학습을 시작하는 방법이다. 즉 모든 데이터($X$, 여기서는 숫자 이미지)에 레이블($Y$, 여기서는 0 ~ 9 중 하나의 숫자)이 주어져 있는 경우이다.

#### 그럼 비지도 학습(Unsupervised Learning)은?

무감독 학습, 비교사 학습이라고도 한다.

간단하다. 위의 데이터에서 이미지만 주어지고 레이블은 주어지지 않는 경우이다. 이런 경우에는 보통 clustering(군집화) 등 비슷한 이미지끼리 그룹화하는 등의 task를 수행하게 된다. 위의 숫자 이미지라면 0은 0끼리, 1은 1끼리 그롭화하는 것을 생각할 수 있겠다.  
물론 이것말고 비지도 학습의 종류는 많다.

#### 그럼 준지도 학습(Semi-supervised Learning)이란?

일부의 데이터에만 레이블 $Y$가 주어져 있는 경우이다. 

#### 왜 비지도 학습같이 어려운 것을 하는가?

네트워크의 학습 관점에서, 정답(레이블)이 주어져 있는 경우가 대개 학습이 훨씬 쉽다. 보통 쉬운 순서대로 지도학습, 준지도학습, 비지도학습 순이다.  
그런데 왜 비지도 학습 같은 것을 하는가?

<center><img src="/public/img/2019-06-10-Deep-Learning-01/03.png" width="80%" alt="각 숫자 이미지에는 레이블이 없다."></center>


현실에서 데이터는 엄청나게 많지만 그것에 레이블을 다는 작업은 보통 수동으로 한다(...). 그래서 레이블이 없는 경우가 거의 대부분이며, 많은 연구자들이 기를 쓰고 semi-supervised learning이라도 할 수 있도록 소수의 데이터에라도 레이블을 추가하거나 아니면 아예 컴퓨터가 알아서 레이블링을 하도록 학습을 시키는 이유이기도 하다.

---

## Deep Learning의 역사

### Perceptron(퍼셉트론)

딥러닝의 근간인 인공신경망(ANN, Artificial Neural Network)의 시초는 [F. Rosenblatt 가 1958년 발표](https://psycnet.apa.org/record/1959-09865-001)한 퍼셉트론(perceptron)이다.

<center><img src="/public/img/2019-06-10-Deep-Learning-01/04.png" width="80%" alt="Perceptron"></center>

$$ \hat{y} = g(\sum_{i=1}^n{w_i x_i + b}) $$

웬만한 식에서 $\ \hat{}$ 이 붙은 것($\hat{y}$ 등)은 네트워크 또는 모델이 내놓은 예측치를 의미한다. 이와 대비되는 것으로 실제 정답($y$)이 있다.

수학 시간에서 봤을 함수 $y = ax + b$와 비슷한 상태이다. 다른 점이 있다면

- $x$가 하나가 아닌 여러 개($x_1, x_2, ..., x_n$)이며
- 가중치는 $a$가 아닌 $w_1, w_2, ..., w_n$)으로 표시되고
- Activation function($g$)가 있다.

Activation function에 대해서는 나중에 설명하도록 한다.

즉 $n$개의 입력값들의 선형 결합에 어떤 특정 함수를 적용하여 $y$라는 값을 예측하겠다는 것인데, 이 모형은 XOR같이 간단한 것조차 구분하지 못했기 때문에 거의 30년간 인공신경망 연구는 묻히게 된다.

Perceptron은 선형 결합(Linear combination)으로 계산되기 때문에, $x$의 개수가 많아져 다차원의 공간에서 Perceptron이 어느 값 이상이냐 미만이냐로 나누는 것은 곧 다차원의 공간에서 hyperplane으로 나눈다는 것을 의미한다. XOR을 표현하기 위한 2차원 공간($x_1, x_2$만 사용)에서는 hyperplane이 직선으로 나타나기 때문에 우리가 보기가 쉬워진다.

<center><img src="/public/img/2019-06-10-Deep-Learning-01/05.png" width="100%" alt="XOR"></center>

<center><img src="/public/img/2019-06-10-Deep-Learning-01/06.png" width="100%" alt="XOR"></center>

> 출처: http://www.cs.stir.ac.uk/courses/ITNP4B/lectures/kms/2-Perceptrons.pdf

위 그림과 같이 XOR은 한 직선으로 구분해내는 것이 불가능하다. 

### MLP(Multi-Layer Perceptron, 다층 퍼셉트론)

위에서 설명한 것은 Sinle-Layer Perceptron이다. 즉, 퍼셉트론이 한 층으로만 되어 있다는 것인데, 이를 여러 층으로 쌓으면 위에서 본 XOR을 퍼셉트론이로 구분해내는 것이 가능해진다.

<center><img src="/public/img/2019-06-10-Deep-Learning-01/08.png" width="100%" alt="Multi-Layer Perceptron"></center>

<center><img src="/public/img/2019-06-10-Deep-Learning-01/07.png" width="100%" alt="Multi-Layer Perceptron"></center>

> 출처: https://gomguard.tistory.com/178

자세한 것은 [여기](https://gomguard.tistory.com/178)를 참조하면 될 것 같다.



---

[다음 글](https://greeksharifa.github.io/)에서는 더 살펴보도록 한다.
