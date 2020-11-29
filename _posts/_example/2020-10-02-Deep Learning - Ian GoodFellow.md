---
layout: post
title: Deep Learning(Ian Goodfellow) 책 정리 - 01. Introduction
author: YouWon
categories: Deep Learning(Ian Goodfellow)
tags: [wandb, usage]
---

이번 글에서는 Ian Goodfellow 등이 쓴 [Deep Learning](https://www.amazon.com/Deep-Learning-NONE-Ian-Goodfellow-ebook/dp/B01MRVFGX4)을 정리한다.

양이 너무 많으니... 핵심적인 부분만 적는다.

---

# Chapter 1. Introduction

발명가들은 생각할 수 있는 기계를 만드는 것을 오랫동안 꿈꿔왔다. 이는 적어도 고대 그리스부터 시작되었다.

프로그래밍 가능한 컴퓨터가 처음 생각되었을 때, 사람들은 이러한 기계들이 '지능적'일 수 있는지 궁금해했다.  
오늘날, **artificial intelligent(AI)**는 많은 실용적인 환용과 연구 주제로서 번영하는 분야이다. 

초기의 성공적인 결과는 AI는 상대적으로 아주 좁은 분야에서, 그리고 '지능'이 그다지 필요하지 않은 분야에서 이루어졌다. 예로, IBM의 **Deep Blue**는 1997년 체스 챔피언을 이겼다. 그러나 체스의 세계는 64개의 위치와 움직임이 엄격하게 제한된 32개의 말만이 존재하는 아주 간단한 세계이다.  
컴퓨터는 가장 뛰어난 체스 플레이어를 이겼지만, 겨우 최근에야 물체나 언어를 인식하기 시작했다. 

큼퓨터는 논리 추론 규칙을 사용하여 언어를 자동적으로 추론할 수 있다. 이는 **Knowledge Base**라 불리는 AI의 접근법이다. 그렇지만 이 방법으로는 크게 성공한 적이 없었다.  Cyc, CycL, 등.

AI 시스템은 미가공 데이터(raw data)로부터 패턴을 추출하여 그것만의 지식을 구축할 수 있는 능력이 필요하다. 이러한 능력을 바로 **Machine Learning**이라 부른다. 간단한 ML 알고리즘은 **Logistic Regression**은 제왕절개를 추천할지 결정할 수 있다. 또 다른 간단한 알고리즘은 **Naive Bayes**는 적절한 메일과 스팸메일을 구별할 수 있다.

ML 알고리즘은 주어진 데이터의 표현 형식(representation)에 크게 영향을 받는다. 예로 AI는 환자인지 아닌지 직접 결정할 수 없다. 환자의 표현에 포함된 정보의 각 조각을 **Feature**라 부른다. Logistic regression은 여러 feature들이 어떻게 다양한 결과와 연관되는지를 학습한다.

이러한 표현에 대한 의존성은 컴퓨터과학과 일상생활에서도 일반적인 현상이다. 사람들은 쉽게 아라비아 숫자 연산을 수행하지만, 로마숫자로 연산하기는 쉽지 않다. 즉, 표현의 선택은 ML 알고리즘의 성능에 지대한 영향을 미친다.

그렇지만, 많은 task에서, 어떤 feature가 추출되어야 하는지 아는 것은 상당히 어렵다. 이에 대한 한 가지 해결책은 ML이 단지 표현에서 결과로 매핑하는 것만 아니라 표현 자체를 학습하게 하는 것이다. 이를 **Representation Learning**이라 한다. 이러한 방법은 AI 시스템이 새로운 task에 빠르게 적용될 수 있도록 한다.

Representation Learning의 대표적인 예는 **AutoEncoder**이다. AutoEncoder는 입력 데이터를 다른 표현으로 바꿔주는 **encoder**와 이 표현을 다시 원래 형식으로 바꾸는 **decoder**의 조합으로 이루어져 있다. AutoEncoder는 입력 데이터의 정보를 최대한 많이 저장하도록 학습하며, 다른 종류의 AutoEncoder는 다른 종류의 속성을 얻을 수 있게 한다.

Feature나 알고리즘을 설계할 때, 목표는 관측된 데이터를 설명하는 **factors of variation**을 분리하는 것이다. 실생활에서 AI application의 어려운 점은 factors of variation이 우리가 관측할 수 있는 데이터의 모든 조각에 영향을 줄 수 있다는 것이다. 대부분의 application은 이 factors of variation을 *disentangle*하고 우리가 관심 없는 것을 무시하는 것을 요구한다.

**Deep Learning**은 이러한 표현학습의 핵심 문제를 '표현'을 더 간단한 '표현'으로 변환함으로써 해결한다. Deep Learning은 컴퓨터가 복잡한 문제를 더 간단한 문제로 만들 수 있게 해 준다.   
Deep Learning의 한 예시는 feedforward deep network, 또는 **MultiLayer Perceptron**이다. Multilayer Perceptron은 단지 어떤 입력값들을 출력값으로 바꾸는 수학적 함수이다.  이 함수는 간단한 여러 함수를 조합하여 만들어진다.  레이어를 쌓을수록 더 복잡한 문제를 다룰 수 있게 된다.











































<center><img src="/public/img/2020-06-10-wandb-usage/0.png" width="80%"></center>  



---
