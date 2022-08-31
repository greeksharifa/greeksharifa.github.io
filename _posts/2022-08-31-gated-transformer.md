---
layout: post
title: GTN(Gated Transformer Networks for Multivariate Time Series Classification) 요약 설명
author: Youyoung
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 Gated Transformer Networks for Multivariate Time Series Classification
 논문의 핵심 포인트만 간추려서 기술합니다.  

- [논문 링크](https://arxiv.org/abs/2103.14438)  

본 논문에서는 MTS(Multivariate Time Series) 문제를 풀기 위해 transformer 구조를 활용하는 방법에 대해 기술하고 있습니다.  

Multivariate Time Series task의 핵심은 `stpe-wise(temporal)` 정보와 `channel-wise(spatial)` 정보를 모두 잘 포착하는 것입니다.  

본 논문은 step-wise encoder와 channel-wise encoder라는 2개의 transformer tower를 만들고 이들 output을 단순히 concatenate하는 것이 아니라 gating layer로 연결하여 최종 결과물을 산출하는 구조를 고안하였습니다.  

<center><img src="/public/img/Paper_Review/gtn.PNG" width="70%"></center>  

개인적으로 짧고 간결하고 좋은 아이디어라고 생각합니다. 다만 실험 결과를 보면 `GTN`이 꼭 우수한 것은 아니라는 결과가 나오고 이에 대해 여러 가능성을 검토한 설명이 추가되어 있긴 하지만 어떠한 확실한 정보를 주고 있지 않다는 점이 아쉬운 점으로 다가왔습니다.  

MTS 문제 해결을 위한 시스템을 설계할 때 하나의 선택지로 충분히 활용할 수 있을 것으로 보입니다.  
