---
layout: post
title: 케라스 튜토리얼(1)
author: Youyoung
categories: Keras
tags: [usage, Keras]
---

## 케라스 Basic.

1. 케라스의 모델은 다음과 같이 정의한다.
'''{.python}
model = Sequential([
  Dense(input_dim=2, units=1), Activation=('sigmoid')
])
'''

여기서 Dense는 layer를 생성하며
input_dim은 입력 차원을, units는 출력 차원을 정의한다.

한 번에 모델을 구성하지 않고 계속해서 추가하는 방법도 있다.

```{.python}
model = Sequential()
model.add(Dense(input_dim=2, units=3))
model.add(Activation('relu'))
```

---

2. 다음 단계에서는 Loss Function, Optimizer, Accuracy Metrics를 정의하고 학습시킨다.
```{.python}
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, epochs=500, batch_size=20)
```

---
3. 정확도는 아래와 같이 파악할 수 있다.
> 첫 번째가 Loss Function 값, 두 번째가 예측 정확도이다.

```{.python}
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
```


---

