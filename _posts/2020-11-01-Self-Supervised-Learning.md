---
layout: post
title: Self-Supervised Learning(자기지도 학습)
author: YouWon
categories: [Self-Supervised Learning]
tags: [Self-Supervised Learning, Paper_Review, Unsupervised Learning]
---

이 글에서는 Self-Supervised Learning(자기지도 학습)에 대해 알아본다. Self-Supervised Learning은 최근 Deep Learning 연구의 큰 트렌드 중 하나이다.

Self-Supervised Learning의 기본적인 개념과 여러 편의 논문을 간략히 소개하고자 한다.

---

# Self-Supervised Learning

일반적으로 Supervised Learning(지도학습)이 높은 성능의 모델을 만드는 것이 유리하지만, 수많은 데이터에 label을 전부 달아야 한다는 점에서 데이터셋 모으기가 어려우며 따라서 활용하는 방법도 제한적일 수밖에 없다.  
이와 같은 문제를 해결하고자 나온 방법이 

- 아주 일부분만 label이 존재하는 데이터셋을 사용하는 Semi-Supervisd Learning(준지도 학습)
- label이 전혀 없는 데이터셋을 사용하는 Unsupervised Learning(비지도 학습)

이고, 최근 주목받는 연구 방법이 Self-Supervised Learning(자기지도 학습)이다. 보통 Self-Supervised Learning을 연구할 때, 다음과 같은 과정을 따른다:

1. Pretext task(연구자가 직접 만든 task)를 정의한다.
2. Label이 없는 데이터셋을 사용하여 1의 Pretext task를 목표로 모델을 학습시킨다.
    - 이때, 데이터 자체의 정보를 적당히 변형/사용하여 (label은 아니지만) 이를 supervision(지도)으로 삼는다. 
3. 2에서 학습시킨 모델을 Downstream task에 가져와 weight는 freeze시킨 채로 transfer learning을 수행한다(2에서 학습한 모델의 성능만을 보기 위해).
4. 그래서 처음에는 label이 없는 상태에서 직접 supervision을 만들어 학습한 뒤, transfer learning 단계에서는 label이 있는 ImageNet 등에서 Supervised Learning을 수행하여 2에서 학습시킨 모델의 성능(feature를 얼마나 잘 뽑아냈는지 등)을 평가하는 방식이다. 

여기서 Self-Supervised Learning의 이름답게 label 등의 직접적인 supervision이 없는 데이터셋에서 스스로 supervision을 만들어 학습하기 때문에, supervision이 전혀 없는 Unsupervised Learning의 분류로 보는 것은 잘못되었다는 [시각](https://www.facebook.com/722677142/posts/10155934004262143/)이 있다. 

> I now call it "self-supervised learning", because "unsupervised" is both a loaded and confusing term.
In self-supervised learning, the system learns to predict part of its input from other parts of it input. In other words a portion of the input is used as a supervisory signal to a predictor fed with the remaining portion of the input.
Self-supervised learning uses way more supervisory signals than supervised learning, and enormously more than reinforcement learning. **That's why calling it "unsupervised" is totally misleading**. That's also why more knowledge about the structure of the world can be learned through self-supervised learning than from the other two paradigms: the data is unlimited, and amount of feedback provided by each example is huge.
Self-supervised learning has been enormously successful in natural language processing. For example, the BERT model and similar techniques produce excellent representations of text.
BERT is a prototypical example of self-supervised learning: show it a sequence of words on input, mask out 15% of the words, and ask the system to predict the missing words (or a distribution of words). This an example of masked auto-encoder, itself a special case of denoising auto-encoder, itself an example of self-supervised learning based on reconstruction or prediction. But text is a discrete space in which probability distributions are easy to represent.
So far, similar approaches haven't worked quite as well for images or videos because of the difficulty of representing distributions over high-dimensional continuous spaces.
Doing this properly and reliably is the greatest challenge in ML and AI of the next few years in my opinion.
\- Yann Lecun

게시물을 보면 예전에는 이러한 학습 방식을 "주어진" supervision이 없기 때문에 Unsupervised Learning이라 불렀지만, 사실은 모델이 "스스로" supervision을 만들어 가며 학습하기 때문에 Self-Supervised Learning이라 부르는 것이 맞다고 설명하는 내용이다.  
그리고 (label은 없어도) 데이터는 무궁무진하며, 그로부터 얻을 수 있는 feedback 역시 엄청나기 때문에 "스스로" supervision을 만드는 Self-Supervised Learning 방식이 매우 중요하게 될 것이라고 언급한다.

---

# Image Representation Learning

## [Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)


<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/01.png" width="100%" alt="Examples"></center>

이 논문에서는 **Examplar**를 pretext task로 한 **Examplar-CNN**이라고 하는 모델을 소개한다. 

$N$개의 이미지에서 중요한 부분, 정확히는 considerable gradients를 가지고 있는 32 $\times$ 32 크기의 patch를 하나씩 추출한다(Gradient 크기의 제곱에 비례하는 확률로 patch를 선정함). 이렇게 추출한 **Seed patch**를 갖고 여러 가지 data augmentation을 진행한다. 위 그림의 오른쪽에서 볼 수 있듯이 translation, scaling, rotation, contrast, color 등을 이동하거나 조정하여 만든다.

분류기는 Data augmentation으로 얻어진 patch들은 하나의 class로 학습해야 한다. 여기서 Loss는 다음과 같이 정의된다.

$$ L(X) = \sum_{\text{x}_i\in X} \sum_{T \in T_i} l(i, T\text{x}_i) $$

$l(i, T\text{x}_i)$은 Transformed sample과 surrogate true lable $i$ 사이의 loss를 의미한다. 즉 하나의 이미지가 하나의 class를 갖게 되는데, 이는 데이터셋이 커질수록 class의 수도 그만큼 늘어난다는 문제를 갖고 따라서 학습하기는 어렵다는 단점을 안고 있다. 

아래는 실험 결과이다.


<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/04.png" width="100%" alt="Examples"></center>



---

## [Unsupervised Visual Representation Learning by Context Prediction](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf)

- [Code](http://graphics.cs.cmu.edu/projects/deepContext/)

이 논문에서는 **context prediction** 종류의 pretext task를 제안하는데, 간단하게 말하면 이미지에 $ 3 \times 3 = 9$개의 patch를 가져와, 중간 patch와 다른 1개의 patch를 보여 주고, 다른 1개의 patch가 중간 patch의 어느 뱡향(왼쪽 위, 위, 오른쪽 등 8방향)에 위치하는지를 찾는 문제이다. 아래 그림을 보면 이해가 빠를 것이다:

<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/05.png" width="100%" alt="Examples"></center>

<details>
    <summary>위 그림에서 무엇이 정답인지 보기</summary>
    <p>Answer key: Q1: Bottom right Q2: Top center</p>
</details>

두 장의 이미지를 한 번에 입력으로 받아야 하기 때문에 아래와 같이 patch를 받는 부분을 두 부분으로 만들었다.

<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/06.png" width="60%" alt="Examples"></center>

그리고 "자명한" 해를 쉽게 찾을 것을 방지하기 위해, 9개의 patch들이 딱 붙어 있지 않도록 하였으며 중심점의 위치도 랜덤하게 약간 이동하여 patch를 추출하여 자명한 경우를 최대한 피하고자 하였다(하지만 충분치는 않았다고 한다). 

아래는 실험 결과이다.

<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/07.png" width="100%" alt="Examples"></center>

---




- [Code](http://www.cs.cmu.edu/~xiaolonw/unsupervise.html)


---

<center><img src="/public/img/2020-11-01-Self-Supervised-Learning/0.png" width="100%" alt="Examples"></center>

