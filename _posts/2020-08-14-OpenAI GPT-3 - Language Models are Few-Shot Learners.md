---
layout: post
title: OpenAI GPT-3 - Language Models are Few-Shot Learners
author: YouWon
categories: [NLP(Natural Language Processing) / RNNs]
tags: [Paper_Review, NLP]
---

---

이 글에서는 2020년 5월 *Tom B. Brown* 등이 발표한 OpenAI GPT-3: Language Models are Few-Shot Learners를 살펴보도록 한다.

[GPT-3을 이용한 API](https://beta.openai.com/)가 공개되어 있다.

중요한 부분만 적을 예정이므로 전체가 궁금하면 원 논문을 찾아 읽어보면 된다.  
이 논문은 총 페이지수가 75페이지 정도는 된다..

---

# OpenAI GPT-3 - Language Models are Few-Shot Learners

논문 링크: **[OpenAI GPT-3 - Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**

API: **[GPT-3을 이용한 API](https://beta.openai.com/)**

Github for Data: **[GPT-3](https://github.com/openai/gpt-3)**

## 초록(Abstract)

최근 연구들은, 방대한 텍스트 말뭉치(corpus)로 사전학습(pre-training)한 후 특정 task에 맞춰 미세조정(find-tuning)하는 방법을 통해, 많은 NLP task에서 상당한 발전을 이루었다. (그러나, ) 이는 그 모델의 구조에 있어서는 task 종류에 민감하지 않지만(task-agnostic), (학습) 방법은 여전히 수천, 수만의 예시 데이터를 통해 어느 task에 특화된(task-specific) 미세조정 단계를 요구한다. 이와는 대조적으로, 사람은 일반적으로 단지 몇 개의 예시, 혹은 간단한 지시사항만으로도 (현 NLP 시스템에게는 여전히 많이 어려운) 새로운 언어 task를 수행할 수 있다.  
이 논문에서 우리는 언어모델의 크기를 키우는 것이 task에 대한 일반성과(task-agnostic), few-shot 성능을 높이고, 미세조정 접근법을 사용한 이전의 state-of-the-art와도 비등한 성능을 확보할 수 있음을 보인다. 구체적으로, 이전의 그 어떤 비-희박 언어모델보다 10배는 많은 1750억 개의 인자를 가지는 자기회귀(auto-regressive) 언어모델인 ***GPT-3*** 를 학습시켜, few-shot 세팅에서 성능을 측정하였다. 모든 종류의 task에 대해, GPT-3는 어떤 gradient의 update나 미세조정을 거치지 않고 오직 few-shot 설명(문제 설명 등)을 취하였다. GPT-3는 번역, 질답(QA), *cloze* task 등의 많은 NLP 데이터셋과, 문장에서 새로운 단어를 쓰는 단어 해석, 3자리 연산, domain adaptation 등등 수많은 task에서 강력한 수행능력을 얻었다. 그와 동시에, GPT-3가 few-shot 학습을 할 때 여전히 어려워하는 몇몇 데이터셋과 더불어 거대한 웹 데이터셋으로 학습할 때 GPT-3가 방법론적인 문제를 맞이하는 데이터셋(이 무엇이 있는지)을 확인하였다. 마지막으로, GPT-3는 어떤 기사를 사람 또는 기계가 썼는지 판별하는 문제에서 사람이 봐도 어려움을 느낄 정도의 기사를 써낼 수 있음을 보인다. 그리고, GPT-3가 넓게 보아 어떤 사회적 영향력을 가질 수 있는지를 논의한다.

---

## 1. 서론(Introduction)

최근 몇 년간 NLP 시스템에서 사전학습된 언어 표현(representations)을 사용하는 추세가 있었고, downstream transfer을 위한 task-agnostic한 방향으로 적용되었다. 먼저, 단어 벡터를 사용하는 단일 레이어 표현을 학습시켜 task-specific한 모델 구조에 입력되고, 더 강력한 표현을 얻기 위해 다중 레이어 표현을 활용하는 RNN과 문맥적 state가 사용되었다(여전히 task-specific한 모델 구조를 가졌음). 그리고 더 최근에는 사전학습된 재귀, 또는 transformer 언어 모델이 task-specific한 모델 구조의 필요성을 제거하고 직접 미세조정하는 방식이 사용되었다.

이러한 최근의 패러다임은 독해, 질답, 원문함의 등등 수많은 어려운 NLP task들에서 상당한 발전을 이루어 냈으며, 새로운 모델구조와 알고리즘에 기반하여 더 많은 진전을 이루었다. 하지만, 이 방법의 큰 한계는 모델구조가 task-agnostic하더라도, 여전히 task-specific한 데이터셋과 task-specific한 미세조정 단계를 필요로 한다는 것이다: 특정 task에서 더 강력한 성능을 위해서는 일반적으로 해당 task에 초점이 맞춰진 수천~수만개의 데이터에 대해서 미세조정을 진행해야 한다. 이러한 한계를 없애는 것은 여러 이유에서 가치가 있다.

1. 현실적인 관점에서, 새 task마다 레이블링이 전부 되어 있는 큰 데이터셋을 필요로 하는 것은 언어모델의 활용성을 제한한다. 문법교정에서 파생되는 어떤 것이든, 추상 개념의 예시를 생성하는 것과, 짧은 이야기를 비평하는 것 등을 포함하여 광범위한 분야에서 유용한 언어task들이 있다. 이러한 많은 task들에 대해 각각 그에 맞는 큰 규모의 감독학습용 데이터셋을 구하는 것은 어렵다(그 과정이 모든 새로운 task마다 반복되어야 하는 경우에는 특히 더).
2. 학습 데이터에 존재하는 거짓 상관관계를 활용할 수 있는 가능성이 모델의 표현력과 학습 분포의 협소함에 따라 크게 증가한다. 이는 사전학습과 미세조정 패러다임에 문제를 야기하는데, 모델이 사전학습 동안에 정보를 습득할 수 있도록 큰 크기를 갖게 설계되었지만, 아주 좁은 task 분포에 미세조정(국한)된다. 예를 들어 [Pretrained transformers improve out of distribution robustness](https://arxiv.org/abs/2004.06100)는 더 큰 모델은 분포 외 데이터를 반드시 더 잘 일반화하지는 않는다. 모델은 학습 (데이터) 분포에 너무 맞춰져 있고 그 밖의 것은 잘 일반화하지 못하기 때문에 사전학습-미세조정 패러다임 하에서는 일반화가 잘 이루어질 수 없다는 증거가 존재한다. 따라서 특정 벤치마크에서 미세조정된 모델의 성능은 해당 부분에서는 인간 수준일지 몰라도 보다 근본적인 task에서는 실제 성능이 과장되었을 수 있다.
3. 인간은 대부분의 언어 task를 배우기 위해 대규모 감독학습용 데이터셋이 필요하지 않다 - 자연어로 된 간단한 지시문 혹은 아주 적은 수의 예시만 있어도 어떤 사람이 새로운 task를 충분히 능숙하게 수행하도록 만들 수 있다(예: 이 문장이 기쁜 혹은 슬픈 무언가를 말하는지 선택하라. 또는, 여기 용감한 행동을 하는 사람 예시 2개가 있다. 용감한 행동의 세 번째 예시를 들라). 현재 NLP 기술에서 이런 개념적인 한계를 제쳐놓더라도, 이러한 적응 능력은 현실적으로 이점이 있다 - 이는 사람을 균일하게 여러 task와 기술들을 섞거나 전환하게 할 수 있다(긴 대화문에 무언가 더 추가하는 것 등). 더 넓은 곳에서 유용하게 쓰이려면, NLP 시스템을 (사람만큼) 유동적이고 일반성을 갖도록 할 것 이다.

이러한 문제들을 다루는 가능성 있는 방법은 언어모델의 문맥에서, 모델이 학습하는 동안 여러 기술과 패턴인식 능력을 키우고, 추론 시간에는 이를 원하는 task에 빠르게 적용시키거나 인식시키는 방법인 **meta-learning**이다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/01.png" width="100%" alt="Examples"></center>

무감독 사전학습 동안, 언어모델은 여러 기술들과 패턴인식 능력을 키워 이를 추론 시간에 사용한다. 각 sequence에 대해 forward-pass 안에서 일어나는 내부 반복 과정을 *문맥 내 학습* 이라고 부른다. 이 다이어그램에서 문장들은 사전하습 동안 모델이 데이터 표현을 볼 수 있게 하지는 않지만, 모델은 어떤 하위 작업들이 한 개의 sequence 내에서 일어난다는 사실은 알 수 있다.

우리가 **문맥 내 학습**이라고 부르는 이것을 통해 시도하려는 최근 연구는 사전학습된 언어모델의 텍스트 입력을 task specification의 형태로 사용한다: 이 모델은 자연어 지시문과 task 설명이라는 조건 속에서 '다음에 무엇이 올 것인지를 예측'한다. 

이 방법은 처음에는 가능성을 보였지만, 여전히 미세조정에 비하면 갈 길이 멀다 - 예를 들어 *Language
models are unsupervised multitask learners* 는 Natural Questions에서 4%만을, 55 F1 CoQa를 달성하였는데 이는 최신 결과보다 35점이나 뒤떨어진 결과이다. Meta-learning은 명백히 언어 task를 푸는 현실적인 방법으로서 실행 가능하기 위한 상당한 개선을 요구한다.

언어모델링의 다른 최신 경향은 앞으로 갈 길을 제시할 수 있다. 최근 몇 년간 [Transformer](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/17/Attention-Is-All-You-Need/) 언어 모델의 크기(parameter의 수)는 [1억 개](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/21/OpenAI-GPT-1-Improving-Language-Understanding-by-Generative-Pre-Training/)부터 3, 15, 80, 110, [170억 개](https://msturing.org/)까지 증가하였다. 크기가 증가할 때마다 텍스트 합성과 downstream NLP task에서 상당한 성능 개선을 보여주었고, 이러한 log loss는 많은 downstream task와 관련하여 scale에 따라 개선되는 경향이 뚜렷하다. 문맥 내 학습이 모델의 parameter 안에서 많은 기술과 task를 습득하기 때문에, 문맥 내 학습 능력은 scale에 따라 그 능력이 더 증가한다고 보는 것은 설득력이 있다.

이 논문에서, 우리는 1750억 개의 parameter를 가지는 자기회귀 언어모델(**GPT-3**)을 학습함으로서 이 가설을 테스트하고, 그 문맥 내 학습능력을 측정한다. 구체적으로, GPT-3을 학습셋에 직접 포함되어 있지 않은 task에 대해 빠르게 적응할 수 있는지를 테스트하도록 고안된 여러 최신 task를 포함하여 20개 이상의 NLP 데이터셋에 대해 평가를 진행한다. 각 task에 대해, GPT-3를 3가지 조건에서 평가한다: 

- few-shot learning, 혹은 모델의 10/100개의 문맥창(context window)에 맞는 설명 또는 예시(demonstration)을 허용하는 문맥 내 학습 조건,
- one-shot learning, 딱 한 개의 예시만을 허용하는 조건,
- zero-shot learning, 어떤 예시도 허용되지 않고, 모델에 주어지는 것은 오직 자연어로 된 지시문인 조건.

GPT-3은 전통적인 미세조정 조건에서 평가할 수도 있지만, 이는 추후 연구로 남겨둔다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/02.png" width="100%" alt="Examples"></center>

위 그림은 우리가 연구한 조건에서, 모델이 단어에서 관련 없는 기호를 제거하도록 하는 task에서 few-shot learning 결과를 보여준다. 모델 성능은 자연어 지시문이 포함되면, 모델 문맥에 주어지는 예시의 수($K$)가 증가하면 높아진다. Few-shot learning 성능은 모델 크기에 따라서도 크게 증가한다. 모델의 크기와 문맥 내 예시의 수와 관련한 일반적인 경향은 우리가 연구하는 대부분의 task에 대해서도 성립한다. 우리는 이러한 "학습" 곡선은 어떤 가중치 업데이트나 미세조정을 거치지 않았으며, 단지 조건으로 주어지는 예시의 수를 늘렸을 뿐이다.

대략, NLP task들에서 GPT-3은 zero-shot과 one-shot 조건에서 훌륭한 결과를, few-shot 조건에서도 SOTA와 비슷하거나 경우에 따라서는 넘어서는 결과를 보여주었다. 예로, GPT-3은 CoQA, zero-shot에서 81.5 F1을(심지어 기존 SOTA는 미세조정 모델이다), few-shot에서는 85.0 F1을 달성했다. 비슷하게, TriviaQA에서는 zero-shot에서는 64.3%, one-shot에서는 68.0%, few-shot에서는 71.2%로, few-shot의 경우는 같은 closed-book 세팅에서 SOTA를 달성한 미세조정 모델의 것과 같다.

GPT-3은 또한 단어해독(순서 맞추기), 연산 수행, 정의된 것을 단 한 번만 보고서 문장에서 새로운 단어를 사용하는 등 즉석에서 추론하는 task와 빠른 적응력을 측정하는 task들에서 one-shot과 few-shot에서 숙련된 결과를 내놓음을 보여주었다. 또한 few-shot 세팅에서, GPT-3은 사람이 보기에도 주어진 기사가 인간 혹은 기계가 썼는지 분간하기 어려운 기사를 생성해낼 수 있다.

그와 동시에, GPT-3이 few-shot에서 어려움을 겪는 몇몇 task를 확인하였다. 여기에는 자연어 추론문제인 ANLI, 독해 데이터셋인 RACE와 QuAC 등이 포함된다. 이러한 한계를 포함하여 GPT-3의 장단점을 보여줌으로써, 우리는 언어모델에서 few-shot learning의 연구를 촉진하고 어떤 개선이 가장 필요한지 관심을 모을 수 있을 것이다.

전체 결과의 느낌은 아래 그림에서 볼 수 있다. zero-shot 성능은 모델 사이즈에 따라 천천히 증가하는 것에 비해, few-shot은 더 가파르게 증가하며, 더 큰 모델일수록 문맥 내 학습에서 월등함을 보여준다. 논문에서 그림 3.8을 보면 SuperGLUE에서 더 자세한 분석을 볼 수 있다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/03.png" width="100%" alt="Examples"></center>

또, **데이터 오염**(학습 데이터셋과 테스트 데이터셋이 겹치는 문제)에 대해서도 체계적으로 연구했다 - *Common Crawl* 등을 통해 얻은 데이터셋에서 거대한 모델을 학습시킬 때 생기는 문제로, 웹에서 모은 데이터가 있기 때문에 가질 수 있는 문제이다(즉, train/test set이 우연히 겹치는 부분이 적지 않을 수 있다). 이 논문에서 데이터 오염과 그 왜곡 효과를 측정하는 체계적 도구를 개발했다. GPT-3의 성능은 대부분의 데이터셋에서 데이터 오염에 미미한 영향만을 받았지만, 우리는 약간의 데이터셋에서 오염이 충분히 큰 영향을 가질 수 있음을 보이고, 또 그 심각도에 따라 그러한 데이터셋에는 별표(*)를 하여 결과에 포함하지 않았다.

위의 모든 것에 더하여, 우리는 zero, one, few-shot 세팅에서 GPT-3의 성능을 비교하기 위해 1.25억~130억 개의 parameter를 가지는 작은(?) 모델을 학습시켰다. 폭넓게, 대부분의 데이터셋에서 모든 3가지 조건에서 상대적 smooth scaling을 찾았다: 주목할 만한 패턴은 zero, one, few-shot 성능은 종종 모델 크기에 따라 증가하며, 이는 더 큰 모델은 더 능숙한 mera-learner임을 시사한다.

마지막으로, GPT-3에 의해 보여진 광범위한 역량에서, 우리는 편향성, 공정성, 나아가 사화적 영향력과, 에 점에서 GPT-3의 특징에 대한 예비 분석을 시도하고 논의할 것이다.

이 논문의 남은 부분은 다음과 같이 구성된다. Section 2에서는 GPT-3을 학습시키고 평가하는 접근법과 방법을 소개한다. Section 3에서는 zero, one, few-shot 세팅에서 전체 범위의 task에 대한 결과를 보여준다. Section 4에서는 데이터 오염에 대한 문제를, Section 5에서는 GPT-3의 한계에 대해 논한다. Section 6에서는 GPT-3의 영향력을, Section 7에서는 관련 연구를 보고 Section 8에서는 결론을 다룬다.

---

## 2. 접근법(Approach)

모델, 데이터, 학습 등 기본적인 접근법은 [Language models are unsupervised multitask learners](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/28/OpenAI-GPT-2-Language-Models-are-Unsupervised-Multitask-Learners/)와 비슷하지만, 모델의 크기를 키웠고, 데이터셋의 크기와 다양성, 학습량을 전부 늘렸다. 문맥 내 학습도 위 논문(GPT-2)와 비슷하지만, 이 논문에서는 문맥 내 학습을 위해 세팅을 다르게 하는 체계적인 방법을 보인다. 그래서, GPT-3을 평가하거나 원칙적으로 평가 가능할 수 있게 하는 여러 세팅들을 정의하고 대조하는 것으로 이 section을 시작한다. 이러한 세팅은 task-specific한 데이터에 얼마나 의존하려는 경향이 있는지를 보는 것이라 할 수 있다. 구체적으로, 4가지로 나누어 볼 수 있다:

1. **미세조정(Fine-Tuning, FT)**은 최근에 가장 일반적인 접근법으로, 사전학습된 모델을 원하는 task에 맞도록 감독학습 데이터셋으로 학습시키는 과정을 포함한다. 보통 수천~수만 개의 레이블링된 예시를 필요로 한다. 
    - 이러한 미세조정(fine-tuning)의 주된 장점은 많은 벤치마크에서 강력한 성능을 가지는 것이다. 
    - 주된 단점은 모든 task마다 큰 데이터셋을 새로이 필요로 하며, 분포 외의 데이터에 대해서는 일반화를 잘 못하며, 학습 데이터에 거짓/비논리적인 특성이 있는 경우 이를 흡수할 수도, 사람에 비해 불공정한 비교로 이어질 수도 있다.
    - 이 논문에서는 task-agnostic한 성능을 가지는 것이 목적이기 때문에 GPT-3은 미세조정을 진행하지 않는다. 단, 나중에는 추후 연구로 괜찮은 방향이기에 미세조정을 사용할 수도 있다.
2. **Few-Shot(FS)**은 모델이 추론 시간에서 단 몇 개의 예시만을 볼 수 있되 가중치 업데이트는 허용되지 않는 조건이다. 그림 2.1에서 보듯이, 일반적인 데이터셋에서 예시는 문맥과 원하는 답이 있고(예로는 영어-독일어 번역), few-shot은 단 $K$개의 문맥과 답이 주어진다. 이후 마지막으로 단 한 개의 문맥이 주어지면, 모델은 (정확한) 답을 생성해 내야 한다. 
    - 우리는 보통 $K$는 10~100 정도로 설정했고, 이는 모델의 문맥창($n_{ctx} = 2048$)에 잘 맞을만한 개수이다. 
    - few-shot의 주된 장점은 task-specific한 데이터에 대한 필요를 크게 줄여주며(즉, 몇 개 없어도 됨) 지나치게 크고 좁은 분포를 갖는 미세조정용 데이터셋을 학습할 가능성을 줄일 수 있다. 
    - 주된 단점은 이 방법은 미세조정 모델의 SOTA에는 한참 뒤떨어지는 성능을 갖는다는 점이다. 또한, 적은 수라 해도 여전히 task-specific한 데이터를 필요로 한다.
    - 이름에서 알 수 있듯이, 언어 모델에서 few-shot learning은 기계학습에서 다른 문맥에서 사용된 few-shot learning과 연관이 있다 - 둘 다 넓은 분포를 갖는 task에 기반한 학습 방법이며(이 경우에는 사전학습 데이터에서) 새로운 task에 빠르게 적응하는 방법이다.
3. **One-Shot(IS)**은 few-shot과 비슷하나 단 한 개의 예시와, task에 대한 자연어 지시문이 제공된다는 점이 다르다. one-shot이 few나 zero-shot과 다른 점은 이 방법이 사람이 소통하는 방법과 가장 흡사한 방법이기 때문이다. 
    - 예를 들어, Mechanical Turk와 같이, 사람에게 데이터셋을 만들어내라는 요청을 할 경우, 보통 task에 대해 하나의 예시를 주게 된다(물론 지시문과 함께). 이와 대조적으로, 예시가 아예 없다면 task의 내용이나 형식에 대해 소통하는 것이 어려울 수 있다.
4. **Zero-Shot(0S)**은 one-shot과 비슷하지만 단 하나의 예시도 없으며, 모델은 단지 task에 대한 지시문만을 받는다. 
    - 이 방법은 최대의 편의성을 갖는데, robustness나, 거짓 상관관계 등을 걱정할 필요가 없다.
    - 단, 가장 어려운 조건이다.
    - 어떤 경우에는 사람조차도 예시가 없으면 task에 대해 제대로 이해하지 못할 수도 있고, 따라서 이 조건은 "불공정할 정도로 어렵다"고 할 수 있다. 
    - 예를 들어, 누군가 '200m 달리기를 위한 세계기록 표를 만들라'고 한다면, 이 요청은 상당히 모호할 수 있는데, 표가 어떤 형식을 가져야 하고, 어떤 내용이 들어가야 하는지에 대한 명확한 설명이 없기 때문이다. 
    - 그럼에도 불구하고, 적어도 zero-shot의 조건은 사람이 task를 수행하는 것과 가장 가까운 방식이다 - 예로, 아래 그림에서 사람은 단지 텍스트 지시문만을 보고도 무엇을 해야 할지 알 수 있을 것이다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/04.png" width="100%" alt="Examples"></center>

위 그림은 영어-독일어 번역 예시에 대한 4가지 방법을 보여준다. 이 논문에서는 zero, one, few-shot에 집중하는데, 경쟁 상대로 보는 것이 아닌 비교 상대로 보기 위함이며, 다른 문제 세팅에서 특정 벤치마크에서의 성능과 sample의 효율성 사이에서 균형을 찾는다. 특히 few-shot 결과는 미세조정 모델보다 아주 약간 못한 결과를 보임을 강조한다. 궁극적으로, 하지만, 사람이 하는 것과 거의 비슷한 one-shot이나 zero-shot에서는 추후 연구의 중요한 목표로 둔다.

Section 2.1~2.3에서는 모델, 학습 데이터, 학습 과정을 자세히 설명한다. Section 2.4에서는 어떻게 few, one, zero-shot 평가를 진행했는지를 자세히 말한다.

### 2.1 Model and Architectures

모델 초기화, 사전 정규화, 내부 서술된 되돌릴 수 있는 토큰화 과정을 포함하여 [GPT-2](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/28/OpenAI-GPT-2-Language-Models-are-Unsupervised-Multitask-Learners/)와 동일한 구조를 갖지만, *Sparse Transformer*와 비슷하게, Transformer 레이어 내에서 밀집/희박한 국소 집중 패턴을 번갈아 사용하였다. 모델 크기에 따른 기계학습 성능의 의존도를 살펴보기 위해, 1.25억 개부터 1750억 개의 parameter를 가지는 8가지 다른 크기의 모델을 학습시켰고, 가장 큰 마지막 것은 GPT-3라 부르는 모델이다.  이전 연구에서 충분한 학습 데이터를 갖고 있으면 validation loss는 크기에 대한 함수로 부드러운 멱법칙을 따를 것이라 하였다; 여러 다른 크기의 학습 모델은 validation loss와 downstream 언어 과제들에 대한 가설을 모두 검증할 수 있게 해 준다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/05.png" width="100%" alt="Examples"></center>

위 표는 8가지 모델의 크기와 구조를 보여준다. $n_{params}$는 학습가능한 parameter의 전체 개수, $n_{layers}$는 레이어 수, $d_{model}$은 각 bottleneck 레이어 안에 있는 unit의 수(이 논문에서, 항상 $d_{ff} = 4 \times d_{model}$이다), $d_{head}$는 각 attention head의 차원이다. 모든 모델은 $n_{ctx} = 2048$ 토큰을 가진다.  
각 GPU 노드 등 데이터 이동을 줄이기 위해 깊이와 너비에 따라 여러 GPU에 모델을 나누었다. 또한 각 모델의 정확한 parameter 수는 계산효율성과 GPU의 부하 균형에 맞게 정했다.  이전 연구는 validation loss는 합리적인 범위 내에서는 parameter의 작은 차이에 크게 민감하지 않음을 시사한다.


### 2.2 Training Dataset

언어모델을 위한 데이터셋은 빠르게 확장되어 거의 1조 개의 단어로 구성된 Common Crawl 데이터셋에서 정점을 찍고 있다. 데이터셋의 이 크기는 같은 데이터를 두 번 쓰지 않아도 이 논문의 가장 큰 모델을 학습시키에도 충분하다. 하지만, 필터링을 전혀 또는 거의 거치지 않은 Common Crawl 데이터는 조정된 데이터셋에 비해 낮은 품질을 갖는 경향이 있다. 따라서, 데이터셋의 품질을 높이기 위한 3가지 방법이 사용되었다:

1. 고품질 출처와 연관성이 있는 것만을 받아 정제하고
2. 과적합을 정확히 측정하기 위한 온전성을 남겨두고 중복을 피하기 위해 문서 수준에서 중복 제거 작업을 수행하였고,
3. 다양성을 증가시키기 위해 고품질 출처로 알려진 말뭉치를 추가하고 섞어서 사용하였다.

자세한 것은 부록 A를 참조하라. 추가 데이터셋으로는 WebText, Books1와 Books2, 영어 위키피디아가 있다.

아래 표는 혼합된 데이터셋 구성을 보여준다. CommonCrawl의 경우 45TB의 데이터셋을 정제하여 570GB로 만들었다(4천억 개의 byte pair encoded 토큰으로 구성됨). 학습에서, 데이터의 사용은 데이터셋의 크기에 비례하지 않고, 고품질일수록 많이 선택되었다. 이는 고품질과 과적합 사이의 trade-off가 있는 것이다.

<center><img src="/public/img/2020-08-14-OpenAI GPT-3 - Language Models are Few-Shot Learners/06.png" width="100%" alt="Examples"></center>

인터넷에서 가져온 데이터로 사전학습한 언어모델에서 가장 큰 방법론적 문제는, 특히 굉장한 양의 내용을 기억하려는 큰 모델에서, 사전학습 동안 무심코 본 정보를 test나 dev set에서 다시금 마주하게 되는 (데이터) 오염 문제이다. 이러한 오염을 줄이기 위해서, 논문에서 살펴보는 모든 벤치마크의 dev/test set와 겹치는 어떤 부분이든 제거하려는 노력을 하였다.  
안타깝게도, 일부 겹치는 부분을 무시하는 버그기 필터링 과정에서 있었고, 학습의 비용 문제로 인해 다시 모델을 학습하는 것은 비현실적이었다. 그래서 이 영향을 Section 4에서 살펴보고, 데이터 오염을 더욱 공격적으로 제거하는 추후 연구를 할 것이다. 

### 2.3 Training Process



### 2.4 Evaluation


---

## 3. 결과(Results)



### 3.1. Language Modeling, Cloze, and Completion Tasks



### 3.2. Closed Book Question Answering



### 3.3. Translation




### 3.4. Winograd-Style Tasks




### 3.5. Common Sense Reasoning




### 3.6. Reading Comprehension




### 3.7. SuperGLUE



### 3.8. NLI




### 3.9. Synthetic and Qualitative Tasks



---

## 4. 벤치마크를 외웠는지 측정하고 예방하기(Measuring and Preventing Memorization Of Benchmarks)



---

## 5. 한계(Limitations)



---

## 6. 광범위한 영향(Broader Impacts)



---

## 7. 관련 연구(Related Work)

이 논문

---

## 8. 결론(Conclusion)

큰 크기


**Acknowledgements**

언제나 있는 감사의 인사

---

## Appendix A: Details of Common Crawl Filtering



---

## Appendix B: Details of Model Training



---

## Appendix C: tails of Test Set Contamination Studies



---

## Appendix D: Total Compute Used to Train Language Models



---

## Appendix E: Human Quality Assessment of Synthetic News Articles



---

## Appendix F: Additional Samples from GPT-3



---

## Appendix G: Details of Task Phrasing and Specifications



---

## Appendix H: Results on All Tasks for All Model Sizes



---

## Refenrences

논문 참조. 많은 레퍼런스가 있다.


---