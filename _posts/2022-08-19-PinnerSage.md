---
layout: post
title: PinnerSage(Multi-modal user embedding framework for recommendations at pinterest) 요약 설명
author: Youyoung
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 `PinnerSage` 논문의 핵심 포인트만 간추려서 기술합니다.  

- [논문 링크](https://arxiv.org/abs/2007.03634)  

본 논문은 multi-embedding based user representation scheme에 대한 내용을 담고 있습니다.  

논문에서는 기존 연구에서 한계점으로 다음 내용들을 언급합니다.  

user-item을 같은 공간에 임베딩했을 때 item은 보통 1가지 종류를 갖는데 반해(예: 영화), 고 품질의 user embedding은 user의 다양한 취미, 스타일, 흥미를 반영해야 한다는 어려움이 있습니다. 이를 보완하기 위해 복수의 user embedding을 만들 수는 있지만 여러 한계점이 존재합니다.  

user, item을 jointly 학습하게 되었을 때, user가 interact 했던 item 들은 자연스럽게 가까운 거리를 갖게 되는데, 예를 들어 어떤 user가 신발, 미술작품, SF를 좋아한다고 해서 이들 item이 가까운 거리를 갖는 것은 상식적으로 말이 되지 않습니다.

embedding을 1개로 제한한 상황에서 (single embedding) 이들을 merge했을 때 이상한 결과가 도출될 가능성이 존재합니다. 예를 들어 신발, 미술작품, SF를 merge하면 이와 전혀 상관없는 item이 등장할 수 있는 것입니다. 이는 single embedding이 item의 여러 측면을 표현하지 못함을 의미합니다.

`PinnerSage`는 다음과 같은 과정을 거치게 됩니다.  

- item embedding은 미리 다른 모델에 의해 학습되어 fixed됨. 본 논문에서는 graph 기반의 PinSage가 이에 해당됨  
- user가 반응한 과거 90일 치의 item 목록을 가져와서 clustering을 실시함. 이 때 clustering은 Ward(계층적 군집화)으로 이루어짐  
- 각 item이 cluster에 배정되었으면 cluster를 대표하는 representation을 medoid를 통해 설정함  
- 모든 cluster에 대해 추론하는 것은 불가능하기 때문에 cluster importance score를 계산하여 3개의 cluster를 추출함. 이 score는 frequency와 recency를 반영하며 두 factor의 균형을 조절할 수 있는 hyperparameter가 존재함  
- 추출된 3개의 representation이 곧 user embedding에 해당하며 ANN을 통해 수 많은 item embedding 사이의 유사도를 계산하여 가장 적합한 item을 user 별로 추천하게 됨  

논문에서는 AB Test를 통해 single embedding을 사용하는 것에 비해 유의미한 개선이 있었음을 증명하였습니다. 그리고 논문 후반부에 언급되는 추천 시스템 구조는 꽤 도움이 많이 되는 정보를 담고 있는데, 그 중에서도 daily batch inference와 lightweight online inference를 분리하여 진행한 것이 실질적으로 운영에 큰 도움이 되었을 것으로 판단합니다.  

본 논문은 굉장히 많은 item embedding 사이의 유사도를 계산해야 하고, single item embedding이 충분한 표현력을 갖지 못한다고 판단될 때 실질적으로 활용 가능성이 매우 높은 방안을 제시했다는 점에서 굉장히 인상적이었습니다.  
