---
layout: post
title: ScaleNorm - Transformers without Tears(Improving the Normalization of Self-Attention) 요약 설명
author: Youyoung
categories: [Paper_Review]
tags: [Machine_Learning, Paper_Review]
---

이번 글에서는 Transformers without Tears(Improving the Normalization of Self-Attention)
 논문의 핵심 포인트만 간추려서 기술합니다.  

- [논문 링크](https://arxiv.org/abs/1910.05895)  

본 논문에서는 Transformer에서 사용되는 **Normalization layer**를 개선하는 방법에 대해 기술하고 있습니다.  

논문에서 여러 내용을 다루고 있지만 핵심 포인트는 2가지 입니다. 일단 normalization layer는 `PRENORM`이어야 한다는 것입니다. original Transformer에서는 `POSTNORM`을 사용하고 있는데, 실험 결과를 보면 `PRENORM`이 backpropagation을 더 효율적으로 진행할 수 있도록 도와줍니다.  

그리고 [이 논문](https://proceedings.neurips.cc/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html)에 따르면 `Batch Normalization`의 효율성은 internal covariate shift를 감소시킨 것에서 얻어진 것이 아니라 loss landscape를 smooth하게 만듦으로써 달성된다고 합니다. 따라서 본 논문에서는 `LAYERNORM` 대신 `SCALENORM`이라는 구조를 제안합니다.  

$$ SCALENORM(x: g) = g \frac{x}{\Vert x \Vert} $$  

l2 normazliation을 적용한 것인데 여기서 $g$ 는 학습 가능한 scalar입니다. 식에서 알 수 있듯이 `LAYERNORM`에 비해 파라미터 수가 훨씬 적습니다. 논문의 실험 결과에 따르면 (데이터셋에 따라 다르겠지만) 학습 속도를 약 5% 정도 향상시켰다고 합니다.  

이 `SCALENORM`은 어떻게 보면 d 차원의 벡터를 학습가능한 radius g를 활용하여 d-1 차원의 hypersphere로 project 시키는 것으로 해석할 수도 있겠습니다. 이러한 아이디어는 각 sublayer의 activation이 이상적인 **global scale**을 갖는다는 인사이트를 담고 있습니다.  

논문에는 이 `SCALENORM`과 다른 논문에서 제시된 `FIXNORM`을 결합한 구조도 설명하고 있습니다. `SCALENORM`의 $g$ 는 $\sqrt{d}$ 로 초기화됩니다.  

실험 결과는 논문 본문을 참고해주세요.  
