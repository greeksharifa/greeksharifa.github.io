---
layout: post
title: KnowIT VQA - Answering Knowledge-Based Questions about Videos(KnowIT VQA 논문 설명)
author: YouWon
categories: [Computer Vision]
tags: [Paper_Review, VQA, Task_Proposal, AAAI2020]
---

---

이 글에서는 KnowIT VQA: Answering Knowledge-Based Questions about Videos에 대해 알아보고자 한다.  

[VQA task](https://greeksharifa.github.io/computer%20vision/2019/04/17/Visual-Question-Answering/)는 이미지(Visual, 영상으로도 확장 가능)와 그 이미지에 대한 질문(Question)이 주어졌을 때, 해당 질문에 맞는 올바른 답변(Answer)을 만들어내는 task이다.  

KnowIT VQA는 VQA 중에서도 Video를 다루는 QA이며, 전통적인 VQA와는 달리 이미지 등 시각자료에서 주어지는 정보뿐만 아니라 외부 지식(상식 베이스, Knowledge Base)이 있어야만 문제를 풀 수 있는 질의들로 이루어진 새로운 task이다. 

KVQA 홈페이지는 https://knowit-vqa.github.io/ 이다.  
데이터셋 중 Annotation도 [같은 링크](https://knowit-vqa.github.io/)에서 다운받을 수 있다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# KnowIT VQA: Answering Knowledge-Based Questions about Videos

논문 링크: **[KnowIT VQA: Answering Knowledge-Based Questions about Videos](https://arxiv.org/abs/1910.10706)**

## 초록(Abstract)

지식기반 그리고 Video QA를 결합한 새로운 영상이해(video understanding) task를 제안한다. 먼저, 유명한 시트콤(참고: 빅뱅이론이다)에 대해 24,282개의 사람이 만든 질답 쌍을 포함하는 video dataset인 KnowIT VQA를 소개한다. 데이터셋은 시각적, 텍스트, 시간적 일관성 추론을 지식기반 질문과 결합하여 이는 시리즈를 봄으로써 얻을 수 있는 경험을 필요로 한다. 그리고 시트콤에 대한 세부적인 지식과 함께 시각적, 텍스트 비디오 자료를 모두 사용하는 영상이해 모델을 제안한다. 이 논문의 주된 결과는 1) (외부) 지식의 결합이 Video QA에 대한 성능을 크게 높여주는 것과 2) KnowIT VQA에 대한 성능은 사람의 정확도에 비해 많이 낮다는 것(현 video 모델링의 한계에 대한 연구의 유용성을 내포함)이다.

---

## 서론(Introduction)

VQA는 자연어와 이미지 이해를 함께 평가하는 중요한 task로 지난 몇 년간 많은 종류의 VQA가 제안되고 연구되어왔다. 특히 초기 연구와 그 모델들은 주로 이미지의 내용에만 집중하는 형태를 보여 왔다(질문의 예: 안경을 쓴 사람이 몇 명이나 있는가?).

단지 이미지-이미지에 대한 질문을 묻는 것은 한계가 명확했다.

1. 이미지의 features는 오직 이미지의 정적인 정보만을 잡아낼 뿐 video에 대한 시간적인 일관성이나 연관성은 전혀 고려하지 않았다(예: 그들은 어떻게 대화를 끝내는가?).
2. 시각 자료(이미지나 영상)는 그 자체로는 외부 지식을 필요로 하는 질문에 대답하는 충분한 정보를 포함하지 않는다.

이러한 한계를 해결하기 위해 단순 이미지와 질문만을 사용하는 Visual QA에서 1) 영상을 사용하는 VideoQA, 외부 지식을 활용하는 KBVQA 등이 등장하였다. 하지만, 여러 종류의 질문을 다루는 일반적인 framework는 여전히 나오지 않았다.

이 논문의 기여하는 바는 여기서부터 출발한다. 영상에 대한 이해와, 외부 지식에 기반한 추론을 모두 활용하는 방법을 제안한다. 

1. 영상을 활용하기 위해 유명 시트콤인 빅뱅이론에다 annotation을 더하여 KnowIT VQA dataset을 만들었다. 질문들은 시트콤에 익숙한 사람만이 대답할 수 있는 수준으로 만들어졌다. 
2. 특정 지식을 continuous representation으로 연결하여 각 질문에 내재된 의도를 추론하는 모듈과 얻어진 지식을 영상 및 텍스트 내용과 결합하여 추론하는 multi-modal 모델을 제안한다.


<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/01.png" width="80%" alt="ovieQA Dataset"></center>




---

## 관련 연구(Related Work)

- **Video Question Answering**은 행동인식, 줄거리 이해, 시간적 연관성을 추론하는 등의 시간적 정보를 해석하는 문제들을 다뤄 왔다. 영상의 출처에 따라 시각자료는 자막이나 대본 등 텍스트 자료와 연관되어 해석을 위한 추가 수준의 문맥을 제공한다. 대부분의 연구는 영상이나 텍스트 자체의 정보를 추출할 뿐 두 modality의 결합은 제대로 활용하지 않았다. MovieFIB, TGIF-QA, MarioVQA는 행동인식, Video Context QA는 시간적 일관성을 다룬다. 
    - 오직 몇 개의 데이터셋, PororoQA와 TVQA만이 multi-model video representation을 함께 해석하는 benchmark를 제안하였다. 하지만 영상자료를 넘어서는 추가 지식에 대한 고려는 이루어지지 않았다.
- **Knowledge-Based Visual Question Answering**: 시각자료와 텍스트만 가지고서는 그 자체에 내재된 정보 외에 추가 지식은 전혀 쓸 수가 없다. 그래서 외부 지식을 포함하는 VQA가 여럿 제안되었다. KBVQA(Knowledge-based VQA)는 외부 지식을 활용하는 대표 VQA 중 하나이다. 이는 지식을 포함하는 데이터셋이지만 초기 단계에 머물러 있다. 
    - 그 중 일부 dataset은 질문의 형태가 단순하거나 외부 지식이 정해진 형태로만 제공되었다. KB-VQA는 template에서 만들어진 질문-이미지 쌍을 활용했으며, R-VQA는 각 질문에 대한 relational facts만을 사용했다. FVQA는 entity 식별, OK-VQA는 자유형식의 질문을 사용하였으나 지식에 대한 annotatino이 없었다. 
    - KnowIT-VQA는 영상을 활용하며, 지식을 특정 데이터에서 직접 추출한 것이 아닌 일반지식을 가져왔기에 이러한 단점들을 해결하였다고 한다.


---

## KnowIT VQA 데이터셋(KnowIT VQA Dataset)


TV Show는 인물, 장면, 줄거리의 일반적인 내용들을 사전에 알 수 있기에 영상이해 task에서 현실 시나리오를 모델링하는 좋은 과제로 여겨진다. 그래서 이 논문에서는 유명 시트콤인 빅뱅이론을 선정하여 KnowIT VQA dataset으로 만덜었고, 이는 영상과 show에 대한 지식기반 질답을 포함한다. 


KnowIT VQA 데이터셋과 다른 데이터셋을 비교한 표를 아래에서 볼 수 있다:

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/02.png" width="100%" alt="ovieQA Dataset"></center>



### Video Collection



### QA Collection



### Knowledge Annotations



### Data Splits



### Dataset Comparison


---

## 인간 평가(Human Evaluation)

### Evaluation Design



---

### Evaluation on the questions



### Evaluation on the knowledge

---

## ROCK Model


### Knowledge Base


### Knowledge Base Cleaning



### Knowledge Retrieval Module



### Prior Score Computation



### Video Reasoning Module



### Language Representation




### Answer Prediction



---

## 실험 결과(Experimental Results)



### Answers



### QA



### SUbs, QA



### Vis, Sub, QA



### Knowledge




### Knowledge Retrieval Results







---

## 결론(Conclusion)

얼굴 식별과 VQA에서의 진전에도 불구하고 Knowledge-aware VQA에서 더 많은 연구가 필요함을 보였다. 기존의 Facenet이나 memory network는 수많은 distractor가 존재하거나 규모가 커진 경우에는 제대로 성능을 내지 못하였다.

이 논문에서는 Knowledge-aware VQA인 KVQA를 제안하였고 평가방법과 baseline을 제시하였다. KVQA의 현재 버전은 KG의 중요 entity인 사람에만 국한되어 있다. 하지만 Wikidata와 같은 KG는 기념물이나 사건 등 여러 흥미로운 entity에 대한 정보도 갖고 있기 때문에 추후 확장할 수 있다. 이 분야에 많은 연구가 이루어지길 희망한다.

**Acknowledgements**

New Energy and Industrial Technology Development Organization

---

## 참고문헌(References)

논문 참조!

--- 


<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/0.png" width="100%" alt="ovieQA Dataset"></center>
