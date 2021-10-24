---
layout: post
title: CNN 기본 모델(ImageNet Challenge - AlexNet, ZFNet, VGG, GoogLeNet, ResNet, InceptionNet)
author: YouWon
categories: [Computer Vision]
tags: [CNN, Paper_Review, ImageNet]
---

---

이 글에서는 ImageNet Challenge에서 2012~2015년까지 좋은 성능을 낸 대표적인 모델인 AlexNet, ZFNet, VGG, GoogLeNet(InceptionNet), ResNet을 살펴본다.

ImageNet에서 사람의 오류율(이미지 class를 잘못 분류한 비율)은 5.1%로 나와 있음을 참고한다.  
각각에 대해 간단히 설명하면, 

- **[AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf):** ImageNet Challenge에서 사실상 최초로 만든 Deep Neural Network 모델이다(이전에는 Shallow model이 대세였다). 2012년, 8 layers, 오류율 16.4%.
- **[ZFNet](https://arxiv.org/pdf/1311.2901.pdf):** AlexNet을 최적화한 모델이다. 2013년, 8 layers, 오류율 11.7%.
- **[VGGNet](https://arxiv.org/pdf/1409.1556.pdf):** Filter의 size를 크게 줄이고 대신 더 깊게 layer를 쌓아 만든 모델이다. 2014년, 19 layers, 오류율 7.3%.
- **[GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf):** InceptionNet 으로도 부른다. 이름답게 Google에서 만들었으며 한 layer에서 여러 크기의 Conv filter를 사용하여 성능을 높였다. 2014년, 19 layers, 오류율 6.7%.
- **[ResNet](https://arxiv.org/pdf/1512.03385.pdf):** Residual Network를 개발한 논문으로 identity mapping을 추가하는 기술을 통해 깊이가 매우 깊어진 모델의 학습을 가능하게 하였다. 2015년, 152 layers, 오류율 3.6%. 이때부터 사람의 정확도를 능가하였다고 평가받는다.
- **[Inception v2,3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf), [v4](https://arxiv.org/pdf/1602.07261.pdf):** GoogleNet(Inception)을 개량한 버전으로 ResNet보다 더 높은 성능을 보여준다. 2016년.


---

# AlexNet

논문 링크: **[AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)**

2012 NIPS에서 발표된 ImageNet Challenge에서의 (사실상) 최초의 Deep Neural Network이다.

<center><img src="/public/img/2021-10-24-ImageNet-CNN-models/AlexNet.png" width="100%"></center>

8개의 layers를 쌓은 지금으로서는 상당히 간단한 모델이지만 당시에는 꽤 복잡한 모델이었다.

8개의 layers라는 말은 학습가능한 parameter가 있는 layer의 개수를 의미하며 learnable parameters가 없는 pooling layer 등은 제외한다.

입력 이미지 크기는 `[224, 224, 3]`이며 이 이미지는

- `conv - pooling - layer norm`을 총 2번 거친 다음
- `conv` layer를 3번 연속 통과하고
- `pooling` layer를 1번 통과한 뒤
- `fully connected` layer를 3번 통과하여 최종적으로 1000개의 class일 확률을 예측하는 `[1000, 1]` 벡터를 출력한다.

전체 구조를 정리하면 다음과 같다.

Layer    | Input size       | Filters                                 | Output size
-------- | --------         | --------                                | -------- 
Conv 1   | [224, 224, 3]    | 96 [11 x 11] filters, stride=4, pad=1.5 | [55, 55, 96]