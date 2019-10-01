---
layout: post
title: Imbalanced Learning
author: Youyoung
categories: Machine_Learning
tags: [Machine_Learning, Paper_Review, Imbalanced Learning]
---

## 1. Imbalanced Learning (불균형 학습) 개요  

비정상 거래 탐지와 같은 케이스의 경우, 정상적인 거래 보다는 정상 범위에서 벗어난 것으로 판단되는 거래 기록의 비중이 현저하게 작을 것이다. 그런데 보통의 알고리즘으로 이러한 비정상 거래를 찾아내기 에는 이러한 데이터의 불균형이 중요한 이슈로 작용하는데, 본 글에서는 이러한 불균형 학습과 관련된 논의를 해보고자 한다.   

알고리즘 자체로 Class 불균형을 해소하는 방법을 제외하면, Over-Sampling과 Under-Sampling 방법이 가장 대표적인 방법이라고 할 수 있다.  

### 1.1. Over-Sampling  
Over-Sampling은 부족한 데이터를 추가하는 방식으로 진행되며, 크게 3가지로 구분할 수 있다.  

첫 번째 방법은 무작위 추출인데, 단순하게 랜덤하게 부족한 Class의 데이터를 복제하여 데이터셋에 추가하는 것이다.  

두 번째 방법은 위와 달리 기존 데이터를 단순히 복사하는 것에 그치지 않고, 어떠한 방법론에 의해 합성된 데이터를 생성하는 것이다. 이후에 설명할 **SMOTE** 기법이 본 방법의 대표적인 예에 해당한다.  

세 번째 방법은 어떤 특별한 기준에 의해 복제할 데이터를 정하고 이를 실행하는 것이다.  

### 1.2. Under-Sampling  
Over-Sampling과 반대로 Under-Sampling은 정상 데이터의 수를 줄여 데이터셋의 균형을 맞추는 것인데, 주의해서 사용하지 않으면 매우 중요한 정보를 잃을 수도 있기 때문에 확실한 근거를 바탕으로 사용해야 하는 방법이다.  

Under-Sampling의 대표적인 예로는 RUS가 있고, 이는 단순히 Random Under Sampling을 뜻한다.  

---
## 2. SMOTE 기법  
SMOTE는 Synthetic Minority Oversampling TEchnique의 약자로, 2002년에 처음 등장하여 현재(2019.10)까지 8천 회가 넘는 인용 수를 보여주고 있는 Over-Sampling의 대표적인 알고리즘이다.  

알고리즘의 원리 자체는 간단하다. Boostrap이나 KNN 모델 기법을 기반으로 하는데, 그 원리는 다음과 같다.  

- 소수(위 예시에선 비정상) 데이터 중 1개의 Sample을 선택한다. 이 Sample을 기준 Sample이라 명명한다.  
- 기준 Sample과 거리(유클리드 거리)가 가까운 k개의 Sample(KNN)을 찾는다. 이 k개의 Sample 중 랜덤하게 1개의 Sample을 선택한다. 이 Sample을 KNN Sample이라 명명한다.  
- 새로운 Synthetic Sample은 아래와 같이 계산한다.
  $$X_{new} = X_i + (X_k - X_i) * \delta$$
  
  $X_{new}$: Synthetic Sample
  $X_i$: 기준 Sample
  $X_k$: KNN Sample
  $\delta$: 0 ~ 1 사이에서 생성된 난수

본 과정을 일정 수 만큼 진행하면 아래 그림과 같이 새로운 합성 데이터가 생성됨을 알 수 있다.  

<center><img src="/public/img/Machine_Learning/2019-10-01-Imbalanced Learning/01.png" width="100%"></center>  




MSMOTE
Borderline SMOTE
Adasyn


---

[여기](https://sumniya.tistory.com/9)
<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/02.JPG" width="100%"></center>

|알고리즘|10% Dataset<br /><br />(58,100)|20% Dataset<br /><br />(116,200)|50% Dataset<br /><br />(290,500)|100% Dataset<br /><br />(581,000)|비고|
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|Lin UCB|0.7086<br /><br />(23.66초)|0.7126<br /><br />(49.39초)|0.7165<br /><br />(137.19초)|0.7180<br /><br />(5분 39초)|alpha=0.2|
|Tree Heuristic|0.7154<br /><br />(100.65초)|0.7688<br /><br />(6분 48초)|0.8261<br /><br />(2463.70초)|0.8626<br /><br />(2시간 37분)|3000 trial이<br /><br />지날 때 마다 적합|

## Reference
> [참고 블로그](https://mkjjo.github.io/python/2019/01/04/smote_duplicate.html)
> [Tree Heuristic 논문](http://auai.org/uai2017/proceedings/papers/171.pdf)

