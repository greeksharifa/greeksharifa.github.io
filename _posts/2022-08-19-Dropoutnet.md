---
layout: post
title: DropoutNet(Addressing Cold Start in Recommender Systems) 요약 설명
author: Youyoung
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 `DropoutNet` 논문의 핵심 포인트만 메모해 둡니다. 

- [논문 링크](https://papers.nips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html)  

- cold start 문제를 좀 더 잘 풀기 위해 dropout 구조를 활용함
- denosing autoencoder에서 영감을 받았다고 하며, 무작위로 user 혹은 item의 content feature를 0으로 masking하여 학습함
- 위 방법 자체를 dropout 구조라고 명명하며, cold start 문제를 더 잘 풀기 위해 objective function에 항을 추가하는 이전의 여러 방법보다 간단한 방법이라고 함
- 학습/예측을 위한 추천 모델 자체에는 특별한 부분은 없음
