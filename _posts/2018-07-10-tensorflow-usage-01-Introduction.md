---
layout: post
title: TensorFlow 사용법 - 01. 소개 및 설치
author: YouWon
categories: References
tags: [TensorFlow]
---

## 텐서플로(TensorFlow)란?

텐서플로(TensorFlow)는 원애 머신러닝(ML)과 심층 신경망(Deep Neural Network) 연구를 수행하는 구글 브레인 팀에서 개발되었다.  
텐서플로는 Tensor(텐서, 텐서플로의 기본 자료구조. 우선 다차원 배열이라고 생각하면 편하다)를 Data Flow Graph에 따라 수치 연산을 하는 라이브러리이기 때문에 그런 이름이 붙었다.

### 연동 라이브러리

#### [텐서보드](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
모델(알고리즘)이 어떻게 돌아가고 있는이 모니터링/디스플레이해주는 모듈이다.  
알고리즘이 잘 돌아가는지 아닌지를 볼 수 있기에 중요한 모듈이다.

실행하는 것은 명령창에서 `tensorboard --logdir=<추적 파일 directory>`를 입력하면 된다.  
그리고 크롬 브라우저에서 <http://localhost:6006>으로 접속하면 로그를 확인할 수 있다.

#### [텐서플로 서빙](https://www.tensorflow.org/serving/)
텐서플로의 머신러닝 모델을 운영환경으로 배포하는 것을 도와준다. 

일반적으로 작업 파이프라인은 다음과 같다.
1. 데이터를 입수한다.
2. 1의 데이터를 바탕으로 학습을 시킨다. 이 결과 모델이 생성된다.
3. (텐서플로 서빙 등을 통해) 모델을 운영환경으로 배포한다.
4. 배포된 모델을 gRPC(구글이 공개한 RPC 프레임워크)를 통해 클라이언트가 무언가 정보를 얻는다.
5. 클라이언트의 피드백 등으로 데이터가 더 쌓이면, 1~2를 무한 반복한다.
6. 모델의 성능이 향상되면 재배포한다.

---

## 설치

파이썬(3.6 또는 2.7)과 pip은 설치되어 있다고 가정한다. conda를 쓰고 싶다면 써도 된다.  
본인이 프로젝트를 여러 개 할 것이라면 가상환경을 만드는 것을 추천한다(virtualenv 또는 conda)

### CPU 버전 설치

우선 본인이 gpu가 없거나 사용할 계획이 없다면, cpu 버전 설치는 매우 간단하다.

윈도우라면 명령창(cmd)를 관리자 권한으로 실행한다. Mac/Linux라면 (sudo권한으로) 터미널을 실행한다.  
다음 명령들을 차례대로 실행한다. ::뒤의 문구는 주석이므로 입력할 필요 없다.

> virtualenv tensorflow_cpu :: tensorflow_cpu란 이름의 가상환경을 만든다.
> cd tensorflow_cpu/Scripts
> activate
> (tensorflow_cpu) pip install tensorflow :: 가상환경 (tensorflow_cpu) 안에서 tensorflow를 설치

가상환경에서 나오고 싶다면 `deactivate`를 입력하면 된다. 

별다른 문제가 없다면 설치가 완료된다. 만약 에러 메시지가 보인다면 구글링을 하면 웬만해선 해결이 된다.

### GPU 버전 설치

우선 본인의 GPU(그래픽카드)가 텐서플로 사양을 맞추는지 확인해 보자.  
일반적으로 Compute Compatability가 3.0 혹은 3.5 이상이면 다 돌아간다고 보면 된다.

<https://developer.nvidia.com/cuda-gpus>

그리고 그래픽 드라이버가 일정 버전 이상인지 확인하도록 한다. 390 이상이면 문제 없을 것이다.

<http://www.nvidia.co.kr/Download/index.aspx?lang=kr>



## 첫 예제 코드

본인이 원하는 곳에 test.py 파일을 하나 만든다. 그리고 그 안에 다음과 같이 입력해 본다.

```python
import tensorflow as tf

a = tf.placeholder('float')
b = tf.placeholder('float')

y = tf.multiply(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a:3, b:3}))
```

그리고 실행해 본다. 명령창이라면 `python test.py`를 입력하면 된다.

> C:\tensorflow_cpu\Scripts\python.exe C:/tensorflow_cpu/test.py
> 2018-07-10 18:05:50.108245: IT:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] 
> Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
> 9.0
> 
> Process finished with exit code 0

위와 같이 나오면 정상이다.  
CPU 버전을 설치했다면 중간에 `Your CPU ... AVX2` 라는 문구가 경고처럼 뜨는데, CPU 버전이라면 뜨는 문제이므로 별 신경쓰지 않아도 된다.  
저 메시지를 끌 수는 있지만, 다른 경고 메시지까지 없어져 버리므로 추천하지 않는다.

이제 위 코드의 의미를 살펴보면,
1. import는 무엇인지 알 것이다.
2. a와 b를 float형 placeholder로 정의했다. 이것은 a와 b가 각각 float타입의 데이터를 담을 그릇이라는 뜻이다. 하지만 아직 어떤 값이 담길지는 정해지지 않은 상태이다.
3. y는 짐작하다시피 a와 b의 값을 곱한 값으로 정의된다. 하지만 마찬가지로 무슨 값이 곱해져서 들어갈지 정해지지 않았다. 단지 곱셈을 수행할 것이란 사실만을 정의했다.
4. 텐서플로는 코드를 실행할 때 세션(Session)이라는 곳에서 수행해야 한다. 세션의 종류는 여러 개가 있지만, 나중에 설명하도록 하겠다. 지금은 텐서플로의 '실행 코드'는 무조건 세션 안에서 해야 한다는 사실만 기억하자.
5. 이제 세션 안에서 'y'를 실행을 시킨다(sess.run). 그리고 feed_dict라는 옵션이 있다.
    1. sess라는 세션 안에서 y를 실행한다는 의미이다. 조금 전 y는 a와 b의 곱셈으로 정의된다고 하였다.
    2. 그런데 y를 계산하려면 a와 b를 알아야 한다고 했는데, 조금 전 a, b를 선언할 때 무슨 값이 들어갈지 정하지 않았다.
    3. 이것 때문에 feed_dict가 있는 것이다. feed_dict는 파이썬의 딕셔너리 형식인데, 잘 보면 a와 b에 각각 특정 값을 집어넣고 있음을 알 수 있다.
    4. 즉 a와 b에는 선언 시점(placeholder로 쓴 부분)이 아닌 실행 시점(sess.run)에 값이 들어가는 것이다.
    5. 그리고 feed_dict로 전달받는 a와 b의 값을 이용하여, 곱셈으로 정의된 y를 계산한다. 
        1. 물론 이 때 y의 계산에 필요한 a, b가 모두 feed_dict 옵션에 들어가 있어야 한다. 넣지 않으면 에러를 뱉을 것이다.
    6. 계산된 y값을 print()로 출력한다. 9.0이 출력되었다.

세션을 만들고 실행하는 것이 좀 비직관적이지만, 이것이 텐서플로의 구조이다. 이는 나름대로의 장점이 있다.
1. 우선 계산 그래프를 만든다(전체 알고리즘을 기술한다).
2. 그 다음 세션을 생성하고 연산을 실행하는 형태이다.

즉 텐서플로는 '선언(definition)'과 '실행(run)'이 따로 이루어진다. 사실 대부분의 다른 머신러닝 라이브러리는 선언과 대입, 실행이 동시에 이루어지긴 한다.

위의 코드에서는 곱셈만 썼지만, 기본 연산으로 다음과 같은 것들이 있다.

```python
print('add:', sess.run(tf.add(a,b), feed_dict={a:3, b:4}))
print('subtract:', sess.run(tf.subtract(a,b), feed_dict={a:3, b:4}))
print('multiply:', sess.run(tf.multiply(a,b), feed_dict={a:3, b:4}))
print('divide:', sess.run(tf.divide(a,b), feed_dict={a:3, b:4}))
print('mod:', sess.run(tf.mod(a,b), feed_dict={a:3, b:4}))
print('abs:', sess.run(tf.abs(a), feed_dict={a:3}))
print('negative:', sess.run(tf.negative(a), feed_dict={a:3}))
print('sign:', sess.run(tf.sign(a), feed_dict={a:3}))
print('square:', sess.run(tf.square(a), feed_dict={a:3}))
print('round:', sess.run(tf.round(a), feed_dict={a:3.5}))
print('sqrt:', sess.run(tf.sqrt(a), feed_dict={a:3}))
print('pow:', sess.run(tf.pow(a,b), feed_dict={a:3, b:4}))
print('exp:', sess.run(tf.exp(a), feed_dict={a:3}))
print('log:', sess.run(tf.log(a), feed_dict={a:3}))
print('log1p:', sess.run(tf.log1p(a), feed_dict={a:3}))
print('maximum:', sess.run(tf.maximum(a,b), feed_dict={a:3, b:4}))
print('minimum:', sess.run(tf.minimum(a,b), feed_dict={a:3, b:4}))
print('cos:', sess.run(tf.cos(a), feed_dict={a:3}))
print('sinh:', sess.run(tf.sinh(a), feed_dict={a:3})) # hyperbolic 함수
```

이외에도 함수는 많지만 간단한 함수들 위주로 적어 보았다.

IPython 같은 대화형 환경을 사용하는 경우 코드의 일부분만 수행하면서 그래프 구조를 만들고 싶을 때가 있다. 이때는 tf.interactiveSession() 세션을 사용하지만, 이것도 나중에 설명하도록 하겠다.

알아두어야 할 점은, 연산과 데이터에 대한 모든 정보는 그래프 구조(수학 계산을 표현함) 안에 저장된다는 것이다.  
그래프의 노드는 수학 연산(앞에서 설명한 multiply 등)을, 엣지는 노드 사이의 관계를 표현하며 동시에 텐서플로의 기본 자료구조인 텐서를 전달한다.

텐서플로는 그래프 구조로 표현된 정보를 이용해서 트랜잭션 간 의존성을 인식하고 노드에 입력 데이터로 들어올 텐서가 준비되면 cpu/gpu에 비동기적/병렬적으로 연산을 할당한다.




<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)
