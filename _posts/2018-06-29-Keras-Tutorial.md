---
layout: post
title: 케라스 튜토리얼
author: Youyoung
categories: Keras
tags: [usage, Keras]
---

### 케라스 Basic

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

#### [7] Channel 설정
```python
from keras import backend as K
K.set_image_data_format('channels_first')
```

#### [8] 케라스 Backend 조작하기
> 케라스 백엔드는 저수준의 AI 엔진을 사용할 때 활용한다.  
> 현재 tensorflow, theano, CNTK 엔진을 지원하지만,  
> 여기서는 tensorflow를 기준으로 설명한다.  

텐서, variable을 만드는 기본적인 방법이다.
```python
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense

# Placeholder
input_tensor = K.placeholder(shape=(None, None, 3), ndim=3, dtype='float32')

# Variable: name에 공백이 있으면 안된다.
# 안의 값을 보고 싶으면 K.eval
var0 = K.eye(size=3)
var1 = K.zeros(shape=(3, 3))
var2 = K.ones(shape=(3, 3))
var3 = K.variable(value=np.random.random((224, 224, 3)),
                  dtype='float32', name='example_var', constraint=None)
var4 = K.constant(value=np.zeros((2, 2)), dtype='int32', shape=(2, 2), name='ex_constant')

var5 = K.variable(value=np.array([[2,2], [4,3]]), dtype='float32')
K.eval(var5)


# 따라하기
arr = np.array([[1,2], [3,4]])
var6 = K.ones_like(arr, dtype='float32')
var7 = K.zeros_like(arr, dtype='int32')
var8 = K.identity(arr, name='identity')


# 텐서 조작: 랜덤 초기화 = Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(64, 64, 3), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(64, 64, 3), mean=0, scale=1) # Gaussian distribution

# Tensor Arithmetic
a = b * K.abs(c)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
a = K.concatenate([b, c], axis=0)
```

저수준의 주요 백엔드 함수들이다.
```python
# 백엔드 함수
# backend check
K.backend()


#1 엡실론 설정
K.epsilon()
K.set_epsilon(1e-05)
K.epsilon()


#2 기본 float 타입 설정
K.floatx()
K.set_floatx('float16')
K.floatx()


#3 채널 순서 정하기
K.image_data_format()
K.set_image_data_format('channels_last')
K.image_data_format()


#4 Clear session
# 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
K.clear_session()


#5 learning_phase
# train time과 test time에 있어 다른 behavior를 적용하는 keras function에 대해
# 0 = test, 1 = train을 가리키는 bool tensor를 인풋으로 제공한다.
K.learning_phase()

# Sets the learning phase to a fixed value.
K.set_learning_phase(1)


#6 is_tensor, is_placeholder
# 타겟이 케라스 layer 혹은 Input에서 반환된 텐서가 맞는지 True, False 반환
keras_placeholder = K.placeholder(shape=(2, 4, 5))
K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.

keras_input = Input(shape=[10])
K.is_keras_tensor(keras_input) # An Input is a Keras tensor.

keras_layer_output = Dense(units=10)(keras_input)
K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.


#7 ndim, dtype: Returns the number of axes, dtype in a tensor
x = K.random_normal_variable(shape=(16, 16, 3), name='x', dtype='float32', mean=0, scale=1)
K.ndim(x)
K.dtype(x)


#8 파라미터 수 세기
var9 = K.random_normal_variable(shape=(4,4,4), dtype='float32', mean=0, scale=1)
K.count_params(var9)


#9 다른 dtype으로 바꾸기
var10 = K.cast(var9, dtype='float64')
print(var10)


#10 update, update_add, update_sub, moving_average_update
old = K.random_normal_variable(shape=(2, 2), dtype='float32', mean=0, scale=1)
new = K.random_uniform_variable(shape=(2, 2), dtype='float32', low=0, high=1)
K.update(old, new)


#11 dot, transpose
K.dot(old, new)
K.transpose(old)


#12 gather: Retrieves the elements of indices in the reference tensor
var11 = K.variable(value=np.array([[1,2], [3,4]]))
what = K.gather(reference=var11, indices=0)
K.eval(what)


#13 max, min, sum, prod, mean, var, std -- 모두 같은 argument
result = K.max(var11, axis=0, keepdims=False)
K.eval(result)


#14 cumsum, cumprod, argmax, argmin -- 같은 argument
result = K.cumprod(var11, axis=0)
K.eval(result)

K.eval(K.argmax(result, axis=0))


#15 수학 계산: square, sqrt, log, exp round, sign equal, not_equal,
# greater, greater_equal, less, less_equal, maximum, minimum


#16 Batch Normalization
# output = (x - mean) / sqrt(var + epsilon) * gamma + beta
x_normed = K.batch_normalization(x, mean=0, var=1, beta, gamma, axis=-1, epsilon=0.001)

# Computes mean and std for batch then apply batch_normalization on batch.
(y_normed, mean, variance) = K.normalize_batch_in_training(x, gamma, beta, reduction_axes=-1, epsilon=0.001)


#17 concatenate, reshape, permute_dimensions
K.concatenate([var3, var4], axis=-1)

var12 = K.variable(np.array(np.arange(0,24,1).reshape(2,3,4)))
print(K.eval(var12).shape)
K.permute_dimensions(var12, pattern=[2,1,0])


#18 resize image: 중간 argument는 양의 정수
img_tensor = K.random_normal_variable(shape=(1, 100, 60, 3), mean=0, scale=1)
K.resize_images(img_tensor, 10, 10, data_format='channels_last')


#19 flatten, expand_dims, squeeze
var13 = K.variable(value=np.ones((224, 224, 3)))
var14 = K.expand_dims(var13, axis=0)
var14
K.flatten(var13)

var14 = K.variable(value=np.ones((224, 224, 1)))
K.squeeze(var14, axis=-1)


#20 spatial_2d_padding
K.spatial_2d_padding(var14, padding=((3, 3), (3, 3)))


#21 기타
# stack: Stacks a list of rank R tensors into a rank R+1 tensor
K.stack(x, axis=0)

# one_hot: Computes the one-hot representation of an integer tensor
K.one_hot(indices, num_classes)

# slice
K.slice(x, start, size)

# get_value, batch_get_value(returns: a list of np arr)
K.get_value(var12)


#22 gradients
# loss=scalar tensor to minimize
# variables=list of varialbes or placholder
# 아래는 style transfer 예시
# 여기서 loss는 스칼라 값이었고, combination_image는 placeholder 였음
K.gradients(loss=loss, variables=combination_image)


#23 함수
K.relu(x='tensor or variable', alpha=0.0, max_value=None)
K.softmax(x, axis=-1)
K.categorical_crossentropy(target='output과 같은 shape의 텐서',
                           output='결과', from_logits=False, axis=-1)

K.dropout(x='tensor', level='fraction of the entries in the tensor that will be set to 0',
          noise_shape=None, seed=None)
K.l2_normalize(x='tensor', axis=None)
K.in_top_k(predictions='a tensor of (batch_size, classes)', targets='1D tensor of length batch size',
           k='# of top elements to consider')


#24 CNN
K.conv2d(x='tensor', kernel='kernel_tensor', strides=(1, 1), padding='valid', data_format='channels_last')
K.pool2d(x, pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_last', pool_mode='max')


#25 returns a tensor with ~ distributions
# random_normal, random_uniform, truncated_normal
var15 = K.random_normal(shape=(3, 3), mean=0.0, stddev=1.0, dtype='float32', seed=None)
K.get_value(var15)
```






