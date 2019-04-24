---
layout: post
title: DANs(Dual Attention Networks for Multimodal Reasoning and Matching)
author: YouWon
categories: [Computer Vision]
tags: [Attention Mechanism, Paper_Review, VQA]
---

---

이 글에서는 네이버랩스(Naver Corp.)에서 2017년 발표한 논문인 Dual Attention Networks for Multimodal Reasoning and Matching에 대해 알아보고자 한다.  
네이버랩스는 인공지능 국제대회 'CVPR 2016: VQA Challenge'에서 2위를 차지하였고, 해당 챌린지에서 DAN(Dual Attention Networks)라는 알고리즘을 개발하였다. 이어 이 알고리즘을 조금 더 일반화하여 2017년 발표한 논문이 이 논문이다.

VQA가 무엇인지는 [여기](https://greeksharifa.github.io/computer%20vision/2019/04/17/Visual-Question-Answering/)를 참조하면 된다.

간단히, DANs은 


논문을 적절히 번역 및 요약하는 것으로 시작한다. 많은 부분을 생략할 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# DANs(Dual Attention Networks for Multimodal Reasoning and Matching)

논문 링크: **[DANs(Dual Attention Networks for Multimodal Reasoning and Matching)](https://arxiv.org/abs/1611.00471)**

## 초록(Abstract)

vision과 language 사이의 세밀한 상호작용을 포착하기 위해 우리는 visual 및 textual attention을 잘 조정한 Dual Attention Networks(DANs)를 제안하고자 한다. DANs는 이미지와 텍스트 모두로부터 각각의 중요한 부분에 여러 단계에 걸쳐 집중(attend / attention)하고 중요한 정보를 모아 이미지/텍스트의 특정 부분에만 집중하고자 한다. 이 framework에 기반해서, 우리는 multimodal reasoning(추론)과 matching(매칭)을 위한 두 종류의 DANs를 소개한다. 각각의 모델은 VQA(Visual Question Answering), 이미지-텍스트 매칭에 특화되어 state-of-the-art 성능을 얻을 수 있었다.

---

## 서론(Introduction)

Vision과 language는 실제 세계를 이해하기 위한 인간 지능의 중요한 두 부분이다. 이는 AI에도 마찬가지이며, 최근 딥러닝의 발전으로 인해 이 두 분야의 경계조차 허물어지고 있다. VQA, Image Captioning, image-text matching, visual grounding 등등.

최근 기술 발전 중 하나는 attention mechanism인데, 이는 이미지 등 전체 데이터 중에서 중요한 부분에만 '집중'한다는 것을 구현한 것으로 많은 신경망의 성능을 향상시키는 데 기여했다.   
시각 데이터와 텍스트 데이터 각각에서는 attention이 많은 발전을 가져다 주었지만, 이 두 모델을 결합시키는 것은 연구가 별로 진행되지 못했다.  

VQA같은 경우 "(이미지 속) 저 우산의 색깔은 무엇인가?" 와 같은 질문에 대한 답은 '우산'과 '색깔'에 집중함으로써 얻을 수 있고, 이미지와 텍스트를 매칭하는 task에서는 이미지 속 'girl'과 'pool'에 집중함으로써 해답을 얻을 수 있다.

<center><img src="/public/img/2019-04-17-Dual-Attention-Networks/01.png" width="80%"></center>

이 논문에서 우리는 vision과 language의 fine-grained 상호작용을 위한 visual 모델과 textual 모델 두 가지를 잘 결합한 Dual Attention Networks(DANs)를 소개한다. 

---

## 관련 연구(Related Works)

- **Structures losses for image modeling:** 


---

## 방법(Method)



---

## 실험(Experiments)







---

## 결론(Conclusion)



### Acknowledgments

~~매우 많다 ㅎㅎ~~

---

## 참고문헌(References)

논문 참조!

--- 




---

## 부록



---