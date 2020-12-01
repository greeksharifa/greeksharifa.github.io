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


<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/01.png" width="80%" alt="KnowIT VQA"></center>




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




### Video Collection

영상은 빅뱅이론 TV show에서 가져왔으며, 이는 각 20분 정도의 207개의 episode들로 구성된다. 텍스트 데이터는 DVD에 있는 자막(대본)을 그대로 [가져왔다](https://bigbangtrans.wordpress.com/). 각각의 대사는 시간과 화자(speaker)의 순서에 맞게 주석을 달았다. 영상은 scene으로 나누어지며 이는 또 20초 정도의 clip으로 분리되어, 총 12,264개의 clip이 있다.

### QA Collection

Amazon Mechanical Turk (AMT)를 사용하였으며 빅뱅이론에 대해 깊은 이해도를 갖는 사람들로 하여금 질답을 생성하도록 했다. 목표는 show에 친숙한 사람들만이 질문에 대답할 수 있도록 하는 질문(그리고 답)을 만드는 것이었다. 각 clip에 대해, 영상과 자막을 각 episode의 대본과 요약과 연결시키도록 했다.  각 clip은 질문, 정답, 3개의 오답과 연관된다. 

### Knowledge Annotations

여기서 "knowledge"(지식)이란 주어진 영상 clip에 직접 포함되지 않는 정보를 가리킨다. 질답 쌍은 추가 정보로 annotated되어:

- **Knowledge**: 짧은 문장으로 표현되는 질문에 대답하기 위해 필요한 정보. 
    - 질문: 왜 Leonard는 Penny를 점심에 초대했는가?
    - Knowledge: Penny는 이제 막 이사를 왔다.
    - 정답: 그는 Penny가 빌딩에서 환영받는다고 느끼기를 원했다.
- **Knowledge Type**: Knowledge가 특정 episode에서 왔는지, 아니면 show의 많은 부분에서 반복되는지(recurrent)를 말한다. 6.08%의 Knowledge만이 반복되며 나머지 9개의 season에 속하는 Knowledge는 거의 균등하다. 분포와 Knowledge의 예시를 아래 그림에서 볼 수 있다.

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/03.png" width="100%" alt="KnowIT VQA"></center>

- **Question Type**: 범용 모델의 개발을 장려하기 위해 test set에서만 제공되며, 4가지 종류로 구분된다:
    - visual-based(22%): video frame 안에서 답을 찾을 수 있다.
    - textual-based(12%): 자막에서 답을 찾을 수 있다.
    - temporal-based(4%): 현재 video clip의 특정 시간에서 답을 예측 가능하다.
    - knowledge-based(62%): clip만으로는 답을 찾을 수 없다(외부 지식이 필요하다).

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/04.png" width="70%" alt="KnowIT VQA"></center>


### Data Splits

12,087개의 video clip으로부터 24,282개의 sample을 추출하였으며 train/val/test set으로 구분되었다. 일반적인 QA dataset처럼 정답은 오답보다 살짝 더 길이가 긴 편향이 존재한다.

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/05.png" width="70%" alt="KnowIT VQA"></center>


### Dataset Comparison


KnowIT VQA 데이터셋과 다른 데이터셋을 비교한 표를 아래에서 볼 수 있다:

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/02.png" width="100%" alt="KnowIT VQA"></center>

- QA 생성이 더 까다롭기 때문에 KBVQA의 크기가 작다.
- KnowIT VQA는 질문의 수가 24k개로 2.4k의 KB-VQA, 5.8k의 FVQA, 14k의 OK-VQA에 비해 훨씬 많다.
- TVQA와 영상이 일부 겹치지만 KnowIT VQA는 더 많은 clip을 사용하였다.

---

## 인간 평가(Human Evaluation)

다음의 목적에 따라 평가를 진행하였다:

1. video clip이 질문에 답하는 것과 연관이 있는지
2. 질문이 답변하는데 Knowledge를 *실제로* 필요로 하는지
3. **Knowledge**가 답변하는 데 정말로 도움이 되는지
4. 모델 비교를 위한 human performance baseline을 설정하기 위해


### Evaluation Design

AMT를 사용하였고, worker를 1) 9개의 시즌 전부를 적어도 한 번은 본 *masters* 그룹과 2) 빅뱅이론을 조금도 본 적 없는 *rookies*로 구분하였다. 질문에 대한 평가와 knowledge에 대한 평가를 중심으로 진행하였다.

---

### Evaluation on the questions

위의 두 그룹은 각각 **Blind**(질답만 제공), **Subs**(질답과 자막 제공), **Video**(질답, 자막, 영상 clip 제공)으로 구분되었다. worker들은 4지선다형 중 맞는 답과 그 답을 선택한 이유를 6지선다형으로 풀게 하였다.

- **Subs**와 **Video** 그룹의 차이는 **video**의 영향을 보여준다.
- *masters*와 *rookies* 그룹의 차이는 **knowledge**의 중요성을 보여주며 KnowIT VQA는 show를 보지 않았을 때 굉장히 어려운 task임을 보여준다.


### Evaluation on the knowledge

모아진 **Knowledge**의 품질과 질문 간 연관성을 분석하였다. *rookies*들에게 test set의 질문에 대답하도록 하였다. 각 질문과 선택지에 대해 자막과 video clip이 제공되었다. 답을 한 후에는 연관된 **Knowledge**를 보여주고 다시 답변하도록 하였다. 결과적으로 **Knowledge**가 주어지기 전후로 정확도는 0.683에서 0.823으로 증가하였다. 이는 답변을 하는 데 있어 **Knowledge**의 연관성(그리고 그 중요성)을 보여 준다.

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/06.png" width="70%" alt="KnowIT VQA"></center>

---

## ROCK Model

**ROCK (Retrieval Over Collected Knowledge)** 모델의 구조는 다음과 같다.

<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/07.png" width="100%" alt="KnowIT VQA"></center>

- Knowledge Base(KB)에 있는 show 정보를 표현하는 언어 instance에 기초한다.
- 언어와 시공간적 영상 표현을 합하여 질문에 대해 추론하고 답을 예측한다.

### Knowledge Base

시청자가 시리즈를 볼 때 정보를 습득하는 것을 모방한다. show에 대해 특정 정보에 접근해야 하기 때문에, **Knowledge**의 AMT worker 주석에 의존한다.

모아진 **Knowledge**는 자연어 문장의 형태로 주어진다. 'Raj가 Penny 집에서 한 것은 무엇인가?'라는 질문에서 주석 **Knowledge**는 다음과 같다:

> Raj는 Missy에게 데이트하자고 요청하길 원했는데, Howard와 Leonard는 이미 그녀에게 물어봤지만 실패했기 때문이지만, (그가 먹었던) 약효가 사라졌고 그는 그럴(요청할) 수 없었다.

Knowledge Graphs와 같은 복잡한 구조에서 어떻게 정보를 가져오는지 불분명하기에, KB를 만들어서, $K = \{w_j|j=1, ..., N \}$, knowledge instance $w$는 자연어 문장으로 표현된다. 거의 중복되는 instance를 제거하였고, $N$은 24,282에서 19,821로 줄어들었다.

### Knowledge Base Cleaning

거의 중복되는 문장을 제거하기 위해 **Knowledge** instance 사이의 유사도를 측정하였다. 각 $w_j \in K$에 대해 다음과 같은 입력 문장을 만들었다:

$$ w_j^{'} = [\text{CLS}] + w_j + [\text{SEP}] $$

[CLS]는 문장의 시작을, [SEP]은 끝을 나타내는 token이다. 이를 60개의 token으로 나눈 뒤 BERT를 통과시켜 고차원의 projection을 얻었다. 

$$ \bold{p}_j = \text{BERT}_{\bold{p}}(\tau_j) $$

두 문장의 유사도는 cosine 유사도를 사용하였으며 다음과 같다.

$$ \beta_{i, j} = sim(\bold{p}_i, \bold{p}_j) $$

그리고 무향 그래프 $V = \{w_j | j=1, ..., N\}$를 만들고 유사도 $\beta$가 0.998 초과인 경우에만 edge를 만들었다. 해당 그래프에서 크기 $L$ 이상의 clique를 찾아 임의로 1개만 남기고 나머지는 삭제하였다.

### Knowledge Retrieval Module

질문과 답안 선택지 $q_i, a_i^c, \quad c \in \{0, 1, 2, 3\}$를 사용하여 Knowledge base $K$에 질의를 한다.  
그리고 연관도 점수 $s_{ij}$에 따라 $w_j \in K$의 순위를 매긴다. 


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


<center><img src="/public/img/2020-11-24-KnowIT VQA-Answering Knowledge-Based Questions about Videos/0.png" width="100%" alt="KnowIT VQA"></center>
