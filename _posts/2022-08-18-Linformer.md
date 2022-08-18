---
layout: post
title: Linformer(Self-Attention with Linear Complexity) 요약 설명
author: Youyoung
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 `Linformer` 논문의 핵심 포인트만 간추려서 기술합니다.  

- [논문 링크](https://arxiv.org/pdf/2006.04768.pdf)  
- [lucidrians github](https://github.com/lucidrains/linformer)  

논문에서는 **self-attention**이 O(n^2)의 time & space complexity를 갖기 때문에 seq가 길수록 bottleneck의 원인이 된다고 지적합니다.  

경험적으로 그리고 수식적으로 증명해보면, attention matrix는 low-rank matrix로 분해할 수 있다고 설명합니다. k << seq_len인 k의 size를 갖는 matrix E를 K, V에 곱함으로써 이를 구현할 수 있습니다. (seq_len, d) -> (seq_len, k)  

실험 결과를 보면 Linformer는 대체적으로 standard Transformer와 비견할 만한 성능을 보입니다.  

요약해보면, sequence의 길이가 꽤 길거나 하는 등의 이유로 training speed를 향상시키면서도 model의 representational capacity를 유지하고 싶을 때, 특별한 제약 조건이 없다면 충분히 시도해 볼 수 있는 방법이라고 판단됩니다. 물론 실제로 데이터에 적용해보기 전까지는 이에 대한 효과를 장담하기는 어렵습니다.  
