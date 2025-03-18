---
layout: post
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (LLM-as-a-Judge) 요약 설명
author: Youwon
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena` 논문의 핵심 포인트만 간단히 정리한다.

- 2023년 12월(Arxiv), NeurIPS 2023
- Lianmin Zheng et al.
- UC Berkeley, UC San Diego, Carnegie Mellon University, Stanford, MBZUAI
- [논문 링크](https://arxiv.org/pdf/2306.05685)  
- [Github](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)  
- [Huggingface](https://huggingface.co/spaces/lmsys/mt-bench)
- [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench)
- [Chatbot Arena](https://lmarena.ai/)
- [Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)

---

## 요약

- Chat assistant에 기반한 LLM 평가는 open-ended 특성상 평가 방법이 많이 없다.
- 이 논문에서는 두 가지 benchmark를 제안한다.
    - MT-Bench: a multi-turn question set
    - Chatbot Arena: a crowdsourced battle platform
- 이 논문에서 LLM-as-a-Judge는 크게 3가지 방법으로 평가할 수 있다.
    - 1. Pairwise comparison: 두 개의 LLM 답변을 놓고 (별개의) LLM Judge가 어느 답변이 더 좋은지 투표하는 방식
    - 2. Single answer grading: 하나의 LLM 답변을 놓고 LLM Judge가 점수를 매기는 방식 (1-10 scale 등)
    - 3. Reference-guided grading: 특정한 경우에, 정답을 알고 있는 상황에서 답변을 평가하는 방식
- 또한 LLM-as-a-Judge의 한계점도 제시하며 몇 가지 해결책을 제시한다.
    - 1. Position bias: 첫 번째 답변이 더 좋은 점수를 받는 경향이 있다. 
      - Swapping positions: 이는 답변 두 개의 순서를 바꿔서 제시한 뒤 특정 답변이 둘 다 좋은 점수를 받을 때만 우수한 답변으로 평가하는 방식으로 해결할 수 있다.
    - 2. Verbosity bias: 더 긴 답변이 더 좋은 점수를 받는 경향이 있다.
      - Few-shot judge: Few-shot 방식으로 해결할 수 있다. 하지만 비용이 비싸져서 이 논문에서는 Zero-shot 방식 평가한다.
    - 3. Self-enhancement bias: 자기 자신의 답변이 더 좋은 점수를 받는 경향이 있다.
      - Chain-of-thought and reference-guided judge: CoT 방식은 일부 문제를 해결할 수 있지만 LLM Judge에게 따로 물어봤을 때는 정답을 맞추는 문제도 답변을 평가하라 하면 잘못된 평가를 내리기도 한다. 그래서 먼저 LLM Judge에게 정답을 물어보고, 이를 reference 삼아 다시 평가하도록 하는 방법을 사용한다.
    - 4. Fine-tuning a judge model: Vicuna-13B 모델을 arena data에 훈련시켜서 모델을 개선한다.

MT-Bench는 다음처럼 multi-turn으로 이루어진 질문 세트로 LLM의 multi-turn 대화 성능과 instruction-following 능력을 평가한다.

<center><img src="/public/img/2025-03-18-LLM-as-a-Judge/tab01.png" width="80%"></center>

Chatbot Arena는 다음처럼 두 개의 LLM 답변을 놓고 사용자는 어느 답변이 더 좋은지 투표하는 방식으로 LLM의 성능을 평가한다.

<center><img src="/public/img/2025-03-18-LLM-as-a-Judge/fig17.png" width="80%"></center>

답변 평가는 아래 그림과 같이 진행한다.

<center><img src="/public/img/2025-03-18-LLM-as-a-Judge/fig01.png" width="80%"></center>


참고:

- Metrics: We define the agreement between two types of judges as the probability of randomly selected individuals (but not identical) of each type agreeing on a randomly selected question. See more explanation in Appendix D.3. Average win rate is the average of win rates against all other players. These metrics can be computed with or without including tie votes.

