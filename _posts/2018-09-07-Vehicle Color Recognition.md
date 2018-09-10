---
layout: post
title: Vehicle Color Recognition
author: Youyoung
categories: Paper_Review
tags: [CNN, Paper_Review]
---

### Vehicle Color Recognition using CNN  
> 본 글은 Reza Fuad Rachmadi, I Ketut Eddy Purnama가 2018년에 Publish한 위 논문을 리뷰한 것이다.  
  

**Introduction**  
본 논문의 목표는 Color Distribution에 기반하여 CNN을 통해 색깔을 classify하는 것이다. RGB의 경우 정확히 분포로 분류할 수 없기 때문에 CIE Lab과 HSV로 변환 후 분석에 들어가도록 하였다. 데이터셋은 Vehicle(자동차)로 정하였다. (p1)  
  
Color Detection이 쉽지 않은 이유는 여러가지가 있지만 대표적인 예로, 날씨의 변화, 영상/이미지의 품질 차이, 문양/패턴의 차이 등을 들 수 있다.  
또한 이와 더불어 명백한 단색을 제외하고는 사람에 따라 색깔 인식 내지는 분류를 하는 방식이 다르기 때문이다.  
이렇기 때문에 사실 그 물체의 어떤 부분을 확인하여 색깔 구분의 핵심으로 삼는지가 굉장히 중요한 방법이다. 예를 들어 '차'의 색깔을 구분한다고 할 때, 타이어의 색깔로 구분을 짓는 사람은 별로 없을 것이다. 그보다는 차체의 보닛을 중심으로 색깔을 구분할 것이다.  
이 때문에 Region Selection 역시 색깔 구분에 있어 핵심적인 역할을 수행한다.  

**CNN Architecture**  
구조는 일반적인 CNN에서 특별히 변화된 부분이 없다. 사실 Alexnet과 거의 유사한 형태를 지녔고, 마지막에 많은 수의 Parameter를 포함하는 Fully Connected Layer를 삽입하였기 때문에, 학습에 시간이 조금 소요될 것으로 예상되었다.  

**Results and Discusssion**  
결과는 나쁘지 않은 수준이다. 이 논문의 경우 특별히 다른 방법을 적용하였다기 보다는 기본 CNN을 적용하여 8개의 단색에 대해 Classification을 수행하였기 때문에, 기본적인 구조로도 충분한 결과를 얻은 것으로 보인다.  
다만 Gray와 Green 색깔 구분에 있어 약간의 어려움을 겪었고 (정확도가 대략 10% 가까이 떨어짐), Gray와 White 색깔 구분 역시도 약간의 어려움을 겪은 것으로 결과가 나타났다.  
이는 사실상 색깔이란 것이 이산형으로 보기 어려운 구분 방식을 따르기 때문으로 풀이된다.  
논문은 구분 논리에 대해 자세히 설명하고 있지는 않지만, 첫 번째 CNN Layer가 low-level feature를 추출한다는 설명을 제공하고 있다. 이는 다른 Object Detection과 크게 다르지 않은 논리이다.  
이후 Lyaer들은 차량의 전면 부분의 이미지를 추출하여 Color Detection의 핵심적인 부분으로 삼고 있는 것을 확인할 수 있었으며, 이는 사람의 인식 방법과 유사함을 알 수 있다.  

**추가적 논의**  
이 부분은 논문에 게재된 내용이 아니고, 필자가 다른 작은 프로젝트를 수행하면서 다른 접근법에 대해 생각해보다가 이해한 내용을 덧붙인 것이다.  

Superpixel이란 개념이 있다. Segmentation에 있어 상당히 많이 사용되는 개념인데, 유사한 pixel 값들을 하나의 평균 값으로 묶어서 표현하는 것이다.  
Segment의 수에 따라 마치 blur처리를 한 듯한 느낌이 들기도 한다.  
  
이를 구현하는 방법은 여럿 있겠지만 skimage라는 파이썬 패키지가 매우 유용하다.  

```python
image_path = '~/target.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)/255.

# n_segments: 몇 개의 구역으로 나누고 싶은가?
labels = segmentation.slic(img, compactness=30, n_segments=100)
labels = labels + 1
regions = regionprops(labels)
input_img = color.label2rgb(labels, img, kind='avg')

plt.imshow(input_img)
plt.show()
```
위와 같은 코드를 이미지에 적용하면 Superpixel을 구현할 수 있다. 이렇게 유사한 Pixel들을 묶어 색깔의 다양성을 줄이고 나면, 색깔 판별에 좀 더 도움이 될 것으로 판단하였다.  

이후에 색깔을 추출해 내는 방법은 크게 2가지가 있을 것으로 생각된다.  
먼저 이후에 CNN을 적용하여 위 논문과 유사한 방식으로 색깔을 하나 판별해 내는 것인데, 이 역시 괜찮은 성능을 보이는 것으로 확인되었다.  
그런데 만약 색깔이 여러개라면? 단색이 아니라, 줄무늬 내지는 물방울 문양을 가진 옷을 판별해야 하다면?  
이 때는 Superpixel화 된 이미지 데이터에 Kmeans Clustering을 적용하여 유사한 색깔들을 소수의 군집으로 묶고, 이 중 높은 빈도수를 보이는 색깔들을 중심 색깔들로 추출할 수 있을 것이다.  
  
아래 블로그에서 예시를 찾을 수 있었다.  
https://technology.condenast.com/story/handbag-brand-and-color-detection



