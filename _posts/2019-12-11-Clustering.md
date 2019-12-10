---
layout: post
title: Clustering 군집화 기법 설명
author: Youyoung
categories: Machine_Learning
tags: [Machine_Learning, Clustering]
---

## 0. Introduction
본 글은 기본적인 Clustering 알고리즘들의 원리를 소개하고 코드 시연을 통해 실제 적용 방법에 대해 알아보는 글이다.  

Since the task of clustering is subjective, the means that can be used for achieving this goal are plenty. Every methodology follows a different set of rules for defining the ‘similarity’ among data points. In fact, there are more than 100 clustering algorithms known. But few of the algorithms are used popularly, let’s look at them in detail:

Connectivity models: As the name suggests, these models are based on the notion that the data points closer in data space exhibit more similarity to each other than the data points lying farther away. These models can follow two approaches. In the first approach, they start with classifying all data points into separate clusters & then aggregating them as the distance decreases. In the second approach, all data points are classified as a single cluster and then partitioned as the distance increases. Also, the choice of distance function is subjective. These models are very easy to interpret but lacks scalability for handling big datasets. Examples of these models are hierarchical clustering algorithm and its variants.

Centroid models: These are iterative clustering algorithms in which the notion of similarity is derived by the closeness of a data point to the centroid of the clusters. K-Means clustering algorithm is a popular algorithm that falls into this category. In these models, the no. of clusters required at the end have to be mentioned beforehand, which makes it important to have prior knowledge of the dataset. These models run iteratively to find the local optima.

Distribution models: These clustering models are based on the notion of how probable is it that all data points in the cluster belong to the same distribution (For example: Normal, Gaussian). These models often suffer from overfitting. A popular example of these models is Expectation-maximization algorithm which uses multivariate normal distributions.

Density Models: These models search the data space for areas of varied density of data points in the data space. It isolates various different density regions and assign the data points within these regions in the same cluster. Popular examples of density models are DBSCAN and OPTICS.


---
## 1. K-Means 알고리즘  



---
## 2. Mean-shift 알고리즘  



---
## 3. Gaussian Mixture Model 알고리즘  



---
## 4. DBSCAN 알고리즘  



<center><img src="/public/img/Machine_Learning/2019-12-11-Clustering/01.png" width="80%"></center>  

---
## Reference  
> 파이썬 머신러닝 완벽 가이드, 권철민, 위키북스
