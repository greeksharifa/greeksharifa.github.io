---
layout: post
title: example-title
author: Youyoung
categories: Deep_Learning
tags: [Keras, Detection, CNN]
---

### Faster RCNN
> 이 포스트는 RCNN > Fast RCNN > Faster RCNN으로 이어지는 Image Detection의 발전 양상을  
> 그 원리 및 코드와 함께 풀이하는 목적으로 작성되었다.

~~그렇지만 아직 완성하지 못하였다. 기다릴 필요가 있다~~

#### 구현 코드 다운로드 받기
다운받는 법을 알아보자.

아래 깃헙에서 다운로드 받자
[깃헙](https://github.com/inspace4u/keras-frcnn)

그러고 나서 이 전체 파일을 원하는 위치에 복사 붙여넣기 하자.  
예) C:\Users\YY\Documents\Deep_Learning\CNN_model

그리고 내부에 들어가면, **keras_frcnn**폴더가 있는데 이 사용자 패키지를  
/path/Lib/site-packages에 고이 담아둔다.  

만약 pre-trained된 모델을 이어서 사용하고 싶다면,  
아래와 같이 train_frcnn.py나 test_frcnn.py와 같은 위치에  
**config.pickle**과 **model_frcnn.hdf5**파일을 넣어두면 된다.
<center><img src="/public/img/Deep_Learning/2018-08-03-FasterRCNN/01.PNG" width="90%"></center>

그리고 나서 test_frcnn.py의 가장 마지막 줄을 수정해보자.
```python
cv2.imwrite('C:/Users/YY/path/output/{}.png'.format(idx),img)
```
이제 저 경로로 output파일이 생성될 것이다.

Input이미지를 담을 폴더를 생성하고,  
예) C:\Users\YY\Documents\test_data  

이제 명령프롬프트를 켜서  
```C
python test_frcnn.py -p C:\Users\YY\Documents\test_data
```
와 같이 코드를 쳐주면, output파일이 생성될 것이다.

