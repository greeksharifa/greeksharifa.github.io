---
layout: post
title: Tree of Thoughts - Deliberate Problem Solving with Large Language Models (ToT) 요약 설명
author: Youwon
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 `Tree of Thoughts: Deliberate Problem Solving with Large Language Models` 논문의 핵심 포인트만 간단히 정리한다.

- 2023년 5월(Arxiv), NeurIPS 2023
- Shunyu Yao et al.
- Princeton University, Google DeepMind
- [논문 링크](https://arxiv.org/pdf/2305.10601)  
- [Github](https://github.com/princeton-nlp/tree-of-thought-llm)

---

## 요약

- Exploration, strategic lookahead 등이 필요한 복잡한 추론 문제를 풀기 위해 tree of thoughts (ToT) 방법을 제안한다.
- CoT를 generalize한 방법
- 서로 다른 여러 reasoning paths를 탐색하고 다음 action을 결정할지를 self-evaluate한다.
- Game of 24, Creative Writing, Mini Crosswords 등의 문제를 풀었고
- GPT-4로 CoT를 할 때 4% 정도의 성능인 것을 74%까지 달성했다.

<center><img src="/public/img/2025-03-19-Tree-of-Thoughts/fig01.png" width="80%"></center>


- 4가지 부분으로 구성된다.
  - Thought decomposition: 적당한 크기의 thought로 분해하는 작업
  - Thought generator $G$: i.i.d thought를 CoT를 통해 생성하거나 (sample) 한 번에 여러 개의 thought를 생성함(propose)
  - State evaluator $V$: 현재 state에서 최적의 thought (state)를 평가하고 선택하는 작업
    - Value: 각 state를 독립적으로 평가함 (1-10 scale 또는 sure/likely/impossible 등). 논문에서는 task별로 다른 방식을 사용하였음.
    - Vote: 여러 state를 비교해서 best state를 선택함.
  - Search algorithm: tree에서 어떤 경로 탐색을 쓸 것인지에 관한 것인데 논문에서는 BFS (최대 $b=5$)와 DFS만 사용.

- 각 task별 thought의 정의:
<center><img src="/public/img/2025-03-19-Tree-of-Thoughts/tab01.png" width="80%"></center>

- Game of 24:
<center><img src="/public/img/2025-03-19-Tree-of-Thoughts/fig02.png" width="80%"></center>

- Creative Writing:
<center><img src="/public/img/2025-03-19-Tree-of-Thoughts/fig04.png" width="80%"></center>

- Mini Crosswords:
<center><img src="/public/img/2025-03-19-Tree-of-Thoughts/fig06.png" width="80%"></center>
