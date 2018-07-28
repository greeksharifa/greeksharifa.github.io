---
layout: post
title: Text_Generation
author: Youyoung
categories: Deep_Learning
tags: [Keras, LSTM, Text, Generation]
---

## Text Generation with LSTM

> 본 포스트는 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py  
의 코드를 전부를 이용한 것이며, 코드의 해석과 사용 방법을 설명하는 데에  
주안점을 둔 것임을 밝힌다.

### Setting
```python
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from keras.models import load_model
```
참고로 새로운 텍스트를 Feed할 때에는 Corpus의 크기가 적어도 10만 문자는 되어야 하며  
100만에 달하는 것이 가장 이상적이라고 본문에 적혀있다.  

```python
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

print('corpus length:', len(text))
corpus length: 600893
```
영어 버전의 니체의 글을 이용할 것인데, 이 txt파일엔 약 60만개의 문자가 담겨 있다.

```python
# 사용된 문자 수: 57개임
chars = sorted(list(set(text)))
print('total chars:', len(chars))
total chars: 57

# 각각의 문자에 대해 위치 인자를 부여함: 0~56
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```
영어 알파벳 26개를 포함하여 이 글에는 총 57개의 문자가 사용되었는데,  
이를 하나의 리스트로 만들어 놓은 것이 chars이다. set(text)를 통해 중복을 제거하였다.  

아래 **char_indices**와 **indices_char**은 방금 만든 chars의 element와 그 위치인자를  
딕셔너리의 형태로 정리한 것이다. 이는 이후에 Word Matrix에 대해 **one-hot인코딩**을  
할 때 편리하게 사용된다. 아래는 그 일 부를 나열한 것이다.
```python
{'\n': 0, ' ': 1, '!': 2, '"': 3}
```

### Preprocessing
```python
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('nb sequences:', len(sentences))
nb sequences: 200285
print(sentences[0:2])
['preface\n\n\nsupposing that truth is a woma',
'face\n\n\nsupposing that truth is a woman--']
print(next_chars[0:5])
['n', 'w', 't', 'h', '?']
```

이제부터는 LSTM의 Input형식에 맞게 데이터를 정제하는 작업이다.  
60만개의 문자를 단 한 번에 feed하는 것은 Sequential 데이터에 있어서는 아무 의미가  
없기 때문에 여기서는 40개씩 분리를 해줄 것이다. 이를 한 문장이라고 생각하면 편하다.  
(**maxlen=40**)  
sentences와 nex_chars란 리스트를 채워나갈 것인데,  
for loop를 보면, range(0, len(text)- maxlen, step)이라 되어 있다.  
이를 숫자로 풀어 보면, range(0, 60만-40, 3)이다.  
i가 0부터 시작하므로,  
text[0:40]을 sentences에 넣어주고, text[40]을 next_chars에 넣어준다.  
다음 반복 때에는 text[3: 43]을 sentences에 넣어주고, text[43]을 next_chars에 넣어준다.  
이렇게 형성된 sentences의 길이는 60만을 3으로 나눈 20만이다.  

```python
# Vectorization
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
print(x.shape)
(200285, 40, 57)
print(y.shape)
(200285, 57)
```
이제 feed할 Word Matrix를 만들 때가 되었다.  
np.zeros를 통해 initialize를 시켜주는데, 그 shape은 위에 보는 것과 같이  
200285개의 example을 두고, x의 경우 행은 40, 열은 57이다.  
40은 maxlen을, 57은 chars의 길이를 의미한다.  

```python
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 예시: x[3]을 보라
print(x[3])
[[ True False False ... False False False]
 [False False False ... False False False]
 [False False False ... False False False]
 ...
 [False False False ... False False False]
 [False  True False ... False False False]
 [False False False ... False False False]]
```
위에서 만든 x, y 넘파이 배열은 현재 0, 즉 False로 채워져 있다.  
sentences 리스트에 담겨 있는 각 문장에서 등장하는 '문자'의 위치에 1을 배정한다.  
(True로 바꿔준다.)  
즉 20285, 40, 57의 shape을 갖고 있는 x에서 (1, 40, 57)은 한 문장 내에서 등장하는  
40개의 문자를 57개의 총 문자를 기준으로 **one-hot 인코딩**을 한 셈이다.  
y의 경우는 문장 단위가 아니라 각 문자 단위로 인코딩을 해준다.

### Building Model
```python
# Build the model: a single LSTM
model = Sequential()
model.add(LSTM(units=128, input_shape=(maxlen, len(chars))))
model.add(Dense(units=len(chars)))  # 최종 아웃풋은 길이 57의 벡터
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

### Sampling Function
```python
def sample(preds, temperature=1.0):
    """
    :param preds: 확률 값을 담은 np.array
    :param temperature: exp 승수의 분모
    :return:
    """
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
```

이 사용자 함수는 처음 보기엔 구조가 복잡해보인다.  
다음 함수를 설명한 후에 그 구조를 설명하도록 하겠다.  

```python
def on_epoch_end(epoch, logs):
    """
    LambdaCallback의 인자에 전달할 것임
    각 epoch 끝에 발동하는 함수임. Generated Text를 반환한다.
    """
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    # 60만개의 전체 text에서 랜덤하게 번호를 하나 뽑아 start_index를 설정
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        # 빈 문자열을 만들고
        generated = ''
        # maxlen의 길이를 가진 text를 start_index부터 추출하여 sentence에 집어 넣는다.
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            # x_pred: Word Matrix
            # x_pred.shape: (1, 40, 57) 가로는 행은 문자열 길이,
            # 열은 57개의 기본 문자 종류
            x_pred = np.zeros((1, maxlen, len(chars)))

            # 위와 마찬가지로 발견된 문자에 대해 one-hot 인코딩을 해준다.
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            # x_pred를 Input으로 넣고 model이 이를 predict한 것을 preds라 한다.
            # 왜 [0]으로 인덱싱 했는지는 모르겠다. 안해도 똑같은 넘파이 배열이다.
            # preds.shape: (57, 1)
            # 이 preds는 다음에 생성될 문자를 결정하는 확률값을 담은 배열이다.
            preds = model.predict(x_pred, verbose=0)[0]

            # 이 preds를 바탕으로 sample함수를 돌려 next_index와 next_char을 얻는다.
            # 이 next_index는 0 ~ len(char)-1 중 하나의 숫자를 반환함
            # 이 next_char은 위 next_index에 해당하는 문자이다.
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            # 이렇게 얻은 next_char은 generated의 다음 문자로 채택되어 추가된다.
            # 1개의 문자가 늘어나는 셈
            # 문장 역시 늘어나야 할 것이다.
            generated += next_char
            # sentence의 길이는 계속해서 maxlen으로 유지하고 제일 앞에 한 글자를 빼고
            # 뒤에 하나를 추가한다.
            # loop를 돌면서 sentence는 계속해서 뒤로 하나씩 밀리게 된다.
            # (context: 기준이 변경되는 것)
            # 이렇게 400개의 문자를 추가하고 나서 위로 올라가서
            # 총 diversity의 갯수(4개)만큼 반환한다.
            # 총 4개의 글이 반환되는 셈이다.
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
```
이 함수는 주석으로 대부분의 설명을 대체하겠다.  
결과적으로 이 함수는 LambdaCallback의 on_epoch_end의 인자로 그대로 전달된다.  
일단 위의 sample함수에 대해 설명을 하자면,  
리턴하는 값이 **np.argmax(probas)**인데, 여기서 마지막 preds는  

$$preds = \frac{( e^{preds} )}{\sum e^{preds}} $$
$$ = \frac{e^{\frac{log(preds)}{temperature}}}{\sum e^{\frac{log(preds)}{temperature}}} $$

softmax함수에서 그 확률을 구하는 과정이라고 생각하면 된다.  
그리고 이 preds는 np.random.multinomial(1, preds, 1)에 인자로 삽입되는데,  
이 preds가 이후 on_epoch_end함수에서 (57, 1)의 shape으로 만들어져 형성되기 때문에,  
np.random.multinomial 메서드는 57개의 확률 값을 기준으로 다항 분포 추출을 하게 된다.  
return 값으로는 그 중 가장 큰 index를 반환하게 된다.  
즉, 다음에 올 문자로 가장 높은 문자를 확률 값에 근거하여 생성하는 것이다.  

```python
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
```

Callback은 트레이닝 과정에서 given stages에 적용될 수 있다.  
트레이닝 중 내부의 state나 statistics를 확인할 수 있는 것이다.  
사용을 위해서는 Sequential이나 Model 클래스의 fit 메서드에 list of callbacks을  
전달해야 한다. 그러면 그 callback 방법은 트레이닝의 각 단계에서 called될 것이다.

### Results
모델은 다음과 같이 fit한다.
```python
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
```
그러면 자동으로 학습이 됨과 동시에 각 epoch마다 text를 생성할 것이다.

다음은 모델 저장과 로드 코드이다.
```python
niet.save('Path/Nietzsche.h5')
niet = load_model('Path/Nietzsche.h5')
```
