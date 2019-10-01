---
layout: post
title: Imbalanced Learning
author: Youyoung
categories: Machine_Learning
tags: [Machine_Learning, Paper_Review, Imbalanced Learning]
---

## 1. Imbalanced Learning (불균형 학습) 개요  






---

[여기](https://sumniya.tistory.com/9)
<center><img src="/public/img/Machine_Learning/2019-09-18-Contextual Bandit and Tree Heuristic/02.JPG" width="100%"></center>

|알고리즘|10% Dataset<br /><br />(58,100)|20% Dataset<br /><br />(116,200)|50% Dataset<br /><br />(290,500)|100% Dataset<br /><br />(581,000)|비고|
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|Lin UCB|0.7086<br /><br />(23.66초)|0.7126<br /><br />(49.39초)|0.7165<br /><br />(137.19초)|0.7180<br /><br />(5분 39초)|alpha=0.2|
|Tree Heuristic|0.7154<br /><br />(100.65초)|0.7688<br /><br />(6분 48초)|0.8261<br /><br />(2463.70초)|0.8626<br /><br />(2시간 37분)|3000 trial이<br /><br />지날 때 마다 적합|

## Reference
> [Lin UCB 논문](http://rob.schapire.net/papers/www10.pdf)
> [Tree Heuristic 논문](http://auai.org/uai2017/proceedings/papers/171.pdf)

