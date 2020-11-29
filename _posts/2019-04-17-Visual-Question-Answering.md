---
layout: post
title: VQA(Visual Question Answering, VQA 논문 설명)
author: YouWon
categories: [Computer Vision]
tags: [Paper_Review, VQA, Task_Proposal]
---

---

이 글에서는 VQA: Visual Question Answering에 대해 알아보고자 한다.  

VQA task는 이미지(Visual, 영상으로도 확장 가능)와 그 이미지에 대한 질문(Question)이 주어졌을 때, 해당 질문에 맞는 올바른 답변(Answer)을 만들어내는 task이다.  

아래는 [서울대학교 공대뉴스광장](https://eng.snu.ac.kr/node/16080)을 인용하였다.

> VQA Challenge는 2016년 CVPR을 시작으로 매년 개최되며, 1년마다 발전된 기술을 평가하고 시상하고 있다. 2017년부터는 같은 질문에 비슷한 이미지를 보여주고 다른 답변을 하는 데이터를 VQA 2.0 데이터셋 통해 수집한 후 인공지능의 유효성을 엄밀히 평가한다.  
> 예를 들어 ‘누가 안경을 쓰고 있나?’라는 질문에 비슷한 이미지가 주어지면 ‘남자’ 또는 ‘여자’의 답을 가질 수 있도록 데이터의 분포를 고려하는 것. VQA 2.0 데이터셋은 20만 개의 이미지에 대해 110만 개의 질문과 1100만 이상의 답을 가지며, VQA 1.0보다 1.8배의 데이터를 가지고 있다.

VQA Challenge는 컴퓨터비전패턴인식학회(IEEE Computer Vision and Pattern Recognition, CVPR) 워크샵 중 하나이며, [VQA Homepage](https://visualqa.org/)에서 매년 열린다. 관심 있으면 클릭해 보자.

국내 연구팀의 대표적인 성과로는 2016년 네이버랩스 2위, 2018년 서울대 장병탁교수팀 2위가 있겠다.

VQA Challenge라고 하는 것은 Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, Devi Parikh 등의 연구자가 일종의 Challenge로서 제안한 것이기 때문에, 이를 중심으로 설명한다. 그렇기 때문에 논문이기도 하면서 동시에 새로운 task를 제안하겠다는 느낌이 강하다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.

---

# VQA: Visual Question Answering

논문 링크: **[VQA: Visual Question Answering)](https://arxiv.org/abs/1505.00468)**

## 초록(Abstract)

이 논문에서는 VQA task를 제안한다. VQA task는 이미지(Visual, 영상으로도 확장 가능)와 그 이미지에 대한 질문(Question)이 주어졌을 때, 해당 질문에 맞는 올바른 답변(Answer)을 만들어내는 task이다.  
VQA를 성공적으로 수행하기 위한 시스템은 이미지 captioning을 하는 시스템보다 더 높은 수준의 이미지 이해도와 복잡한 추론능력을 가져야 한다. 또한 (간단한 수준의 답변만 하는 것은 좋지 않기 때문에 이를) 자동으로 평가하는 것도 가능해야 한다. 우리는 25만 장의 이미지와, 76만 개의 질문, 1000만 개의 답과 그에 대한 정보를 제공한다. 많은 기준과 방법들은 사람의 수행능력과 비교한다. VQA Demo는 CloudCV에서 볼 수 있다.

참고) 2019.04.17 현재 논문에 링크된 CloudCV Demo는 404 error가 뜨는 중이다.

---

## 서론(Introduction)

Computer Vision(CV), Natural Language Processing (NLP), Knowledge Representation & Reasoning (KR)를 결합한 이미지 캡셔닝(captioning)은 지난 몇 년간 급격히 발전해 왔다. 그러나 이 task는 별로 "AI-complete"하지 못하다(그다지 인공"지능"스럽지 않다).  
그러면 "AI-complete"하지 못하다는 것은 무엇인가? 이 논문에서는 좀 더 자유로운 형식에 열린 형태인 VQA(Visual Question Answering)을 제안하고자 한다. 이러한 답변을 제대로 하기 위해서는 다양한 AI 능력들이 필요하다:

- 세밀한 인식("이 피자엔 어떤 종류의 치즈가 있는가?")
- 물체 감지("얼마나 많은 자전거가 있는가?")
- 행동인식("남자는 울고 있는가?")
- 지식기반 추론("이것은 채식주의자를 위한 피자인가?")
- 상식 추론("이 사람은 20/20 시력을 갖고 있는가?", "이 사람은 회사를 원하는가?" 참고: 20/20은 1.0/1.0과 같음)

또한 VQA 시스템은 자동으로 평가가 가능해야 한다. 이 논문에서는 열린 문제(open-ended, 답변의 가능성이 다양함)와 다지선다형(multiple-choice) task를 둘 다 본다. 다지선다형 문제는 열린 문제와는 다르게 단지 정해진 답변 중 옳은 것을 고르기만 하면 된다.

데이터셋은 COCO 데이터셋에 5만 개를 더 추가했다. 데이터 수는 초록에도 나와 있다. 또한 이미지 캡셔닝이랑 무엇이 다른지에 대한 설명도 나와 있다.


---

## 관련 연구(Related Works)

- **VQA Efforts:** Visual Question Answering은 이전에도 다뤄진 적이 있긴 한데, 여기서 제안하는 것보다 훨씬 제한된 환경과 제한된 데이터셋 안에서 다룬다. 물체의 종류도 적고, 답변의 단어 등도 제한적이다. 이 VQA task는 그렇지 않다. free-form, open-ended이다.
- **Text-based Q&A:** 이 문제는 NLP와 텍스트 처리 분야에서 잘 연구되었다. VQA 기술에 도움이 될 몇 가지 접근법이 있다. 이 경우 질문은 텍스트를 기반으로 이루어진다. VQA는 text와 vision 모두에 의존한다. 
- **Describing Visual Content:** 이미지 태깅, 이미지 캡셔닝, 비디오 캡셔닝 등이 VQA와 관련이 있다. 그러나 그 캡션은 vision에 특화된 것이 아닌 지나치게 일반적인(많은 이미지에 대해 동일한 캡션을 써도 말이 됨) 경우가 있다.
- **Other Vision+Language Tasks:** 이미지 캡셔닝보다 평가가 쉬운 coreference resolution, generating referring expressions 등의 task가 있다.


---

## VQA 데이터셋(VQA Dataset Collection)

사실 이미지 한장이면 충분할 듯 하다.

<center><img src="/public/img/2019-04-17-Visual-Question-Answering/01.png" width="100%"></center>

잘 안 보이니까 일부만 확대하겠다.

<center><img src="/public/img/2019-04-17-Visual-Question-Answering/02.png" width="100%"></center>

- 약 20만 장의 현실 이미지와 약 5만 장의 추상적인 이미지가 있다.
- Training / Validation / Test 셋이 나누어져 있다. 그 나누는 비율도 정해져 있다(추상 이미지의 경우 20K/10K/20K). subsplit은 없다.
- 이미 MS COCO 데이터셋은 이미지당 5개의 한 문장짜리 캡션이 있으므로, 추상 이미지에도 그만큼 붙여서 만들었다.
- 흥미롭고, 다양하고, 잘 만들어진 질문을 모으는 것은 매우 중요한 문제이다.
    - "저 고양이의 색깔은 무엇인가?", "지금 몇 개의 의자가 이미지에 있는가?" 같은 질문은 너무 단순하다.
    - 그러나 우리는 "상식"을 필요로 하는 질문을 원한다. 또, 상식"만"으로 대답할 수 있는 질문은 안 된다.
        - 예를 들면 "사진의 저 동물은 어떤 소리를 낼 것 같은가?" 같은 질문이다.
        - "콧수염은 무엇으로 만들어지는가?" 같은 질문은 의미 없다.
    - 그래서 총 76만 개 정도의 질문을 확보하였다.
- 많은 질문들에 대해서는 yes/no만 해도 충분하다. 그러나 그렇지 않은 것들도 있다.
- 열린 형태(open-ended) 질문들은 다음 metric에 의해 평가된다.
    - $ \text{accuracy} = min({\text{그 답변을 한 사람의 수} \over 3}, 1) $
- 다지선다형(객관식) 문제는 18개의 선택지가 있다.
    - 이외에도 다양한 형태의 문제가 존재한다.



## VQA 데이터셋 분석(VQA Dataset Analysis)

데이터의 정확한 수, 질문의 종류 및 수, 답변의 종류 및 수, 질답의 길이 등에 대한 분포 등이 수록되어 있다.

- 질문에는 "What is...", "Is there...", "How many...", "Does the..." 같은 질문들이 있다. 질문의 길이는 4~8단어가 대부분이다.
- 답변에는 yes/no, 색깔, left/right 등의 답변이 많다. 1 / 2 / 3단어인 경우가 대략 90%, 6%, 2.5% 정도씩 있다.
- 상식을 필요로 하는 질문은 위에서 설명한 대로 당연이 이미지에서도 정보를 얻어야 답변이 가능하다.


task를 제안하는 것인만큼 데이터에 대한 정보가 매우 자세하다. 아래 그림 같은 정보도 있다. 여러 종류의 질문에 대해 답변이 어떤 단어가 어떤 비율로 있는지 등을 나타낸다.

<center><img src="/public/img/2019-04-17-Visual-Question-Answering/03.png" width="100%"></center>


---

## VQA 기준선과 방법(VQA Baselines and Methods)

### Baselines

- **random:** 무작위로 답변을 선택한다.
- **prior("yes"):** "yes" 답변이 가장 많기 때문에 항상 yes를 답으로 내놓는다.
- **per Q-type prior:** 각 질문 종류별로 답변 중 최빈값을 답으로 내놓는다.
- **nearest neighbor:** 가장 유사한 K개의 질문을 뽑아 그 답변들 중 최빈값을 답으로 내놓는다.

### Methods

- **Image Channel:** 이미지를 위한 embedding을 제공한다.
    - I: VGGNet의 마지막 hidden 레이어가 4096차원의 embedding으로 사용된다.
    - norm I: 위와 비슷하나 $l_2$ 정규화된 활성함수를 사용
- **Question Channel:** 질문을 위한 embedding을 제공한다.
    - Bag-of-Words Question(BoW Q): 질문의 최빈 1000개의 단어와 30차원의 BoW를 사용하여 1030차원의 질문 embedding을 만든다.
    - LSTM Q: 1024차원이다.
    - deeper LSTM Q: 2048차원이다.
- **Multi-Layer Perceptron(MLP):** 
    - BoW Q + I에 대해서는 단지 concatenate한다.
    - LSTM Q + I, deeper LSTM Q + norm I에 대해서는 이미지 embedding은 차원을 맞추기 위해 1024차원으로 변환된 후 LSTM embedding과 element-wise하게 곱해진다.


### Results

방법에 따라서는 28.13%/30.53%(각각 open-ended와 multiple-choice)를 나타낸 것부터 58.16%/63.09%를 나타낸 모델(deeper LSTM Q + norm I)까지 결과는 다양하다.  
따라서 적어도 60%는 넘어야 의미 있는 VQA 시스템이라고 할 수 있을 것이다.

---

## VQA Challenge and Workshop

CVPR 2016에서부터 1년 간격으로 열린다. 테스트 서버도 준비되어 있다.

---

## 결론 및 토의(Conclusion and Discussion)

이 논문에서는 VQA task를 제안하였고, 그에 맞는 데이터를 제공하였다.  
우리는 VQA가 자동평가가 가능한 "AI-complete" 문제를 풀기 위한 한계를 끌어올리기에 적합하다고 생각한다. 이를 위한 노력에 드는 시간도 가치가 있다고 여겨진다.

---

## 참고문헌(References)

논문 참조!

--- 

결론 이후에도 많은 정보가 있으니 참조하면 좋다. 매우 흥미로운 것들이 많다.  
대부분은 데이터의 분포에 관한 설명 및 시각화한 그림들이다.

---
