---
layout: post
title: 케라스 튜토리얼(1)
author: Youyoung
categories: Keras
tags: [usage, Keras]
---

## 케라스 Basic

#### [1] 케라스의 모델 정의 방법은 크게 2가지가 있다.
먼저 아래와 같이 정해진 형식으로 심플한 모델을 만들 수 있다.  
이 Sequential 모델에는 리스트 형식으로 복수의 layer를 전달하면 된다.  
(물론 하나만 전달할 때에는 리스트로 전달할 필요가 없다.)  
```python
model = Sequential([
  Dense(input_shape=(2,2), units=1), Activation=('sigmoid', batch_size=16)
])
```

>여기서 Dense는 layer를 생성하며  
>input_shape은 입력 차원을, units는 출력 차원을 정의한다.  
>input_shape은 튜플 혹은 정수이다.  
>여기서 배치 사이즈를 정하면 모든 인풋을 (16, 2,2)로 하라고 알아 듣는다.  
>한 번에 모델을 구성하지 않고 계속해서 추가하는 방법도 있다.  


```python
model = Sequential()
model.add(Dense(input_dim=2, units=3))
model.add(Activation('relu'))
```

Customized된 모델을 만들고 싶을 때에는 Model API를 이용한다.
```python
def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2/BatchNormalization/Activation/Maxpooling2D
    X = 여러 블록 후 AveragePoooling/Flatten
    X = Dense(classes, activation='softmax',
                      name='fc' + str(classes),
                      kernel_initializer=glorot_uniform(seed=0))(X)
    # 위와 같이 마지막 X는 텐서이다.

    # create model
    model = Model(inputs=X_input, outputs=X, name='Resnet50'
    return model
```

>위와 같이 먼저 Input함수를 통해 input_shape의 shape을 가진 Input 텐서를 만든다.  
>그 Input 텐서를 집어넣은 후 여러 과정으 거쳐 함수를 정의하고  
>마지막으로 return할 model을 대상으로 inputs/outputs를 정의하면 된다.  
>이렇게 하면 원하는 모델의 구조를 생성할 수 있다.  

#### [2] 다음 단계에서는 Loss Function, Optimizer, Accuracy Metrics를 정의하고 학습시킨다.
>이후에는 마찬가지로 Compiling과 Fitting을 진행하면 된다.

```python
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20, batch_size=16)
```

##### Optimizer 종류는 다음과 같다.
기본 값으로 정의하려면 SGD 대신 'sgd'로 입력하면 된다.  
각각의 Optimizer의 정의는 알아서 찾아보길 바란다.  
[케라스 optimizer][https://keras.io/optimizers/]

```python
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
```
**SGD**(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  
**RMSprop**(lr=0.001, rho=0.9, epsilon=None, decay=0.0)  
**Adagrad**(lr=0.01, epsilon=None, decay=0.0)  
**Adadelta**(lr=1.0, rho=0.95, epsilon=None, decay=0.0)  
**Adam**(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
**Adamax**(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  

##### 만약 optimizer에 옵션을 주고 싶다면 이를 미리 생성하여야 한다.
```python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, epochs=500, batch_size=20)
```

##### Loss 종류는 다음과 같다.
mean_squared_error  
mean_absolute_error  
mean_absolute_percentage_erro  
mean_squared_logarithmic_error  
categorical_crossentropy  
sparse_categorical_crossentrop  
binary_crossentropy  
kullback_leibler_divergence  
poisson  
cosine_proximity  

##### 정확도를 나타내는 metrics의 종류는 다음과 같다.
binary_accuracy
categorical_accuracy
sparse_categorical_accuracy
top_k_categorical_accuracy

##### Custom Accuracy는 다음과 같이 이용한다.
```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

#### [3] Evaluation은 아래와 같이 진행한다.
> 첫 번째가 Loss Function 값, 두 번째가 예측 정확도이다.

```python
preds = model.evaluate(X_test, Y_test)
print(preds)
```

#### [4] 케라스 Layer 기본 규칙
>케라스의 Layer 종류에 대해 자세히 설명하기 전에 기본 규칙을 설명할 것이다.

**layer.get_weights()**: np.array들을 담은 리스트로 layer의 weight을 반환한다.  
**layer.set_weights(weights)**: 이번엔 반대로 그 weight을 설정한다.  
**layer.get_config()**: layer의 configuration을 담은 딕셔너리를 반환한다. 아래와 같이 표현 가능하다.

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

layer가 single node를 가지고 있으면 (shared layer가 아니면) 다음을 물어볼 수 있다.  
layer.input  
layer.output  
layer.input_shape  
layer.output_shape  

layer가 여러 개의 node를 갖고 있으면 아래의 메서드를 쓸 수 있다.
layer.get_input_at(node_index)  
layer.get_output_at(node_index)  
layer.get_input_shape_at(node_index)  
layer.get_output_shape_at(node_index)  

#### [5] 간단한 Convolutional Model 만들기
아래와 같은 데이터셋이 있다고 해보자.  
여기서 input_shape은 (32,32,3)이 될 것이다. 이는 32X32의 Color Image shape과 일치한다. (# channel = 3)  
```python
x_train = np.random.normal(loc=20, scale=3, size=3072000).reshape((1000,32,32,3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.normal(loc=20, scale=3, size=307200).reshape((100,32,32,3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
```

간단한 합성곱 신경망 모델을 다음과 같이 만들 수 있다.  
```python
def convmodel(input_shape=(32,32,3), classes=10):
     X_input_tensor = Input(input_shape)
     X = ZeroPadding2D((3,3))(X_input_tensor)
     X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
                kernel_initializer=glorot_uniform(seed=0), name='conv1')(X)
     X = BatchNormalization(axis=3, name='bn_conv1')(X)
     X = Activation('relu')(X)
     X = MaxPooling2D((3, 3), strides=(2, 2))(X)
 
     X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid',
                kernel_initializer=glorot_uniform(seed=0), name='conv2')(X)
     X = BatchNormalization(axis=3, name='bn_conv2')(X)
     X = Activation('relu')(X)
     X = MaxPooling2D((3, 3), strides=(2, 2))(X)
 
     X = Flatten()(X)
     X = Dense(units=classes, activation='softmax',
               name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
 
     model = Model(inputs=X_input_tensor, outputs=X, name='convmodel')
     return model
```

함수 옵션으로 정의한 input_shape을 이용하여 첫 줄에서 X_input_tensor를 만들고  
계속해서 이를 이용하여 마지막 output X까지 정의하는 방식으로 구성된다.  

```python
model = convmodel(input_shape=(32,32,3), classes=10)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, epochs=5)
score = model.evaluate(x_test, y_test, batch_size=50)
```

#### [6] 케라스 Layer
##### (1) Core Layer

```python
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Reshape
```

```python
Dense(units=32, input_shape=(16,), activation=None, kernel_initializer='glorot_uniform')
```

```python
Activation(activation='relu')
```

```python
Dropout(rate=0.5, noise_shape=None, seed=None)
```
여기서 rate은 drop시킬 비율을 나타냄. 텐서플로의 keep_prob과 반대

```python
Flatten(data_format=None)

# example
model = Sequential()
model.add(Dense(input_shape=(32,32,3), outputs=16))

# 현재 model.output_shape = (None, 32,32,16)
model.add(Flatten())
# model.output_shape = (None, 32*32*16)
```

```python
Reshape(target_shape)
```

##### (2) Convolutional Layer
```python
Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='valid',
       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```


##### (3) Pooling Layer
```python
MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')

AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'
```


---

