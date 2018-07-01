---
layout: post
title: Residual Neural Network with CNN
author: Youyoung
categories: Deep_Learning
tags: [Keras, CNN, Resnet]
---

## Resnet
> 이 글에서는 Vanishing Gradients를 해결하는 하나의 방안인 **Residual Block**을 사용한
Residual Neural Network를 구성하는 방법에 대해 설명하겠다.
본 글은 아래 참조문헌과
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)
Andrew Ng의 Deep Learning Specialization 강좌의 Programmin Assignment 코드를 대부분 이용한 것임을 밝힌다.

#### [1] 원리 설명
![원리1](C:/Users/YY/Desktop/File/14. Gitblog/image/Res01.jpg)

본 Resnet은 Identity Block과 Convolutional Block을 사용하는데 그 구조는 아래와 같다.
![원리2](C:/Users/YY/Desktop/File/14. Gitblog/image/Res02.jpg)
![원리3](C:/Users/YY/Desktop/File/14. Gitblog/image/Res03.jpg)
![원리4](C:/Users/YY/Desktop/File/14. Gitblog/image/Res04.jpg)
![원리5](C:/Users/YY/Desktop/File/14. Gitblog/image/Res05.jpg)
![원리6](C:/Users/YY/Desktop/File/14. Gitblog/image/Res06.jpg)
![원리7](C:/Users/YY/Desktop/File/14. Gitblog/image/Res07.jpg)


#### [2] 데이터셋 로딩
사용하는 패키지, 모듈은 다음과 같다.
```python
# Setting
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import h5py; import numpy as np; import matplotlib.pyplot as plt
import scipy.misc

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from keras.utils import plot_model
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
```


본 글에서 사용하는 데이터셋은 .h5 파일 형식으로 되어있는데 이 hdf파일은 다음과 같이 로드하면 된다.
위에서처럼 h5py 패키지를 설치 후 import 해주어야 한다.
```python
file = h5py.File('path1/file.h5', 'r+')
test = h5py.File('path2/file.h5', 'r+')
print([n for n in file.keys()])
X_train_orig = list(file['train_set_x'])X_test_orig = list(test['test_set_x'])
Y_train_orig = list(file['train_set_y']); Y_test_orig = list(test['test_set_y'])

Y_train = np.eye(6)[np.array(Y_train_orig).astype(int)]
Y_test = np.eye(6)[np.array(Y_test_orig).astype(int)]

print("X_train shape: ", X_train_orig[0].shape)
print("X_test shape: ", X_test_orig[0].shape)
print("Y_train shape: ", Y_train[0].shape)
print("Y_train shape: ", Y_test[0].shape)
```

---
#### [3] 코드

```python
# Normalize image vectors
X_train = np.array(X_train_orig)/255
X_test = np.array(X_test_orig)/255
```

```python
# Identity Block
def identity_block(X, f, filters, stage, block):
    """
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path: filter shape
    filters -- python list of integers, the number of filters in the CONV layers
    stage -- integer, name the layers, depending on their position in the network
    block -- string/character, name the layers, depending on their position in the network

    Returns: X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value.
    X_shortcut = X

    # glorot_uniform = Xavier uniform, BatchNormalization axis=3 means normalizing channels

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns: X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # Setting
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), name=conv_name_base + '2b',
               padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL
    -> (CONVBLOCK, IDBLOCK*2) -> (CONVBLOCK, IDBLOCK*3) -> (CONVBLOCK, IDBLOCK*5)
    -> (CONVBLOCK, IDBLOCK*2) -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, 라벨 수

    Returns: model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # AVGPOOL: don't use padding in pooling layer
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',
              name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model
```

```python
# 학습
model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20, batch_size=32)

# Test the result
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```

현재 이 모델의 경우 *epochs=20* 정도로 하면 90%를 넘는 정확도를 보이는 것으로 확인되었다.

다른 이미지로 확인을 해보고 싶다면 아래 함수를 이용하면 된다.
```python
# 다른 이미지로 테스트
def img_test(filename='i01.jpg'):
    img_path = 'C:/Users/YY/Documents/Winter Data/NN/Resnet_color_hand_sign/real test/' + str(filename)
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x/255
    my_image = scipy.misc.imread(img_path)  # Use imageio.imread
    result = np.argmax(model.predict(x))
    return result, my_image

result, my_image = img_test('i05.jpg')
print(result)

imshow(my_image)
plt.show()
```

---
## [4] 모델 시각화
모델을 시각화 하고 싶다면 아래와 같은 코드를 이용하면 된다.
현재 py파일이 있는 디렉토리에 png 파일이 저장될 것이다.

```python
# 모델 시각화
model.summary()
plot_model(model, to_file='resnet.png', show_shapes=True, show_layer_names=True)
```

다음은 본 모델 구조의 최하단부를 나타낸다.
![모델구조](C:/Users/YY/Desktop/File/14. Gitblog/image/Res08.jpg)

