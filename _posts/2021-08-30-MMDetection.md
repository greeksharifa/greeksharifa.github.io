---
layout: post
title: MMDetection 사용법
author: YouWon
categories: References
tags: [Linux, Ubuntu, open-mmlab, usage]
---

이 글에서는 [MMDetection](https://github.com/open-mmlab/mmdetection)를 사용하는 방법을 정리한다.

- [Documentation](https://mmdetection.readthedocs.io/)
- [Github](https://github.com/open-mmlab/mmdetection)

- [Colab Tutorial](https://colab.research.google.com/github/ZwwWayne/mmdetection/blob/update-colab/demo/MMDet_Tutorial.ipynb#scrollTo=Wuwxw1oZRtVZ)

---


## 기본 설명

[OpenMMLab](https://openmmlab.com/)에서는 매우 많은 최신 모델을 Open Source Projects로 구현하여 공개하고 있다.  
2021.08.30 기준 11개의 Officially Endorsed Projects와 6개의 Experimental Projects를 공개하고 있다.

11개 프로젝트의 목록을 아래에 적어 놓았다. 예를 들어 어떤 이미지를 detect하는 모델을 찾고 싶으면 `MMDetection`에서 찾으면 된다. 대부분 따로 설명하지 않아도 무엇을 하는 모델인지 알 것이다.

- MMCV: Computer Vision
- MMDetection
- MMAction2
- MMClassification
- MMSegmentation
- MMDetection3D
- MMEditing: Image and Video Editing
- MMPose: Pose estimation
- MMTracking
- MMOCR
- MMGeneration

각각의 repository는 수십 개의 모델을 포함한다. 예를 들어 [MMDetection](https://github.com/open-mmlab/mmdetection)은 

Supported backbones:

- ResNet (CVPR'2016)
- ResNeXt (CVPR'2017)
- VGG (ICLR'2015)
- HRNet (CVPR'2019)
- RegNet (CVPR'2020)
- Res2Net (TPAMI'2020)
- ResNeSt (ArXiv'2020)

Supported methods:

- RPN (NeurIPS'2015)
- Fast R-CNN (ICCV'2015)
- Faster R-CNN (NeurIPS'2015)
- Mask R-CNN (ICCV'2017)
- Cascade R-CNN (CVPR'2018)
- Cascade Mask R-CNN (CVPR'2018)
- SSD (ECCV'2016)
- RetinaNet (ICCV'2017)
- GHM (AAAI'2019)
- Mask Scoring R-CNN (CVPR'2019)
- Double-Head R-CNN (CVPR'2020)
- Hybrid Task Cascade (CVPR'2019)
- Libra R-CNN (CVPR'2019)
- Guided Anchoring (CVPR'2019)
- FCOS (ICCV'2019)
- RepPoints (ICCV'2019)
- Foveabox (TIP'2020)
- FreeAnchor (NeurIPS'2019)
- NAS-FPN (CVPR'2019)
- ATSS (CVPR'2020)
- FSAF (CVPR'2019)
- PAFPN (CVPR'2018)
- Dynamic R-CNN (ECCV'2020)
- PointRend (CVPR'2020)
- CARAFE (ICCV'2019)
- DCNv2 (CVPR'2019)
- Group Normalization (ECCV'2018)
- Weight Standardization (ArXiv'2019)
- OHEM (CVPR'2016)
- Soft-NMS (ICCV'2017)
- Generalized Attention (ICCV'2019)
- GCNet (ICCVW'2019)
- Mixed Precision (FP16) Training (ArXiv'2017)
- InstaBoost (ICCV'2019)
- GRoIE (ICPR'2020)
- DetectoRS (ArXix'2020)
- Generalized Focal Loss (NeurIPS'2020)
- CornerNet (ECCV'2018)
- Side-Aware Boundary Localization (ECCV'2020)
- YOLOv3 (ArXiv'2018)
- PAA (ECCV'2020)
- YOLACT (ICCV'2019)
- CentripetalNet (CVPR'2020)
- VFNet (ArXix'2020)
- DETR (ECCV'2020)
- Deformable DETR (ICLR'2021)
- CascadeRPN (NeurIPS'2019)
- SCNet (AAAI'2021)
- AutoAssign (ArXix'2020)
- YOLOF (CVPR'2021)
- Seasaw Loss (CVPR'2021)
- CenterNet (CVPR'2019)
- YOLOX (ArXix'2021)

를 포함한다.(많다)

이 글에서는 MMDetection 중 Faster-RCNN 모델을 다루는 법만 설명한다. 나머지도 비슷한 흐름을 따라갈 듯 하다.

---

## 설치

### Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- MMCV

공식 문서에서 Docker를 통해 사용하는 방법도 안내하고 있는데, 현재 Docker의 환경은 다음과 같다.

- Linux
- Python 3.7.7
- PyTorch 1.6.0
- TorchVision 0.7.0
- CUDA 10.1 V10.1.243
- mmcv-full 1.3.5

> 자신이 사용하는 환경이 복잡하다면 얌전히 Docker를 쓰는 편이 낫다..

설치 방법은 

- [Github Installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
- [Official Docs](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

에 나와 있으니 참고하자. Docker 쓰면 별다른 에러 없이 바로 구동 가능하다.

Docker 설치방법 안내에 나와 있지만, `data/` 디렉토리는 자신이 사용하는 환경에서 데이터를 모아놓는 디렉토리에 연결해놓으면 좋다.

그리고 설치 방법에 나와 있는 것처럼 repository를 다운받아 놓자.(이미 되어있으면 상관없다)

```bash
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
```

---

## 간단 실행

### High-level APIs for inference

- [공식 문서](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#high-level-apis-for-inference)
- [ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb)

우선 `checkpoints` 디렉토리를 만들고 다음 모델 파일을 받자.

- [faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)

현재 worktree는 다음과 같다. 

- 참고: 공식 문서에는 config 파일을 따로 받아야 할 것처럼 써 놨지만 repository에 다 포함되어 있다. 

```
mmdetection
├── checkpoints
|   ├── faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
├── configs
│   ├── faster_rcnn
│   │   ├── faster_rcnn_r50_fpn_1x_coco.py
│   │   ├── ...
├── data
├── demo
|   ├── demo.jpg
|   ├── demo.mp4
|   ├── ...
├── mmdet
├── tools
│   ├── test.py
│   ├── ...
├── tutorial_1.py
├── ...
```

`tutorial_1.py` 파일을 만들고 다음 코드를 붙여넣자. 

- 참고: 공식 문서 코드에서는 파일명이 test.jpg 처럼 자신이 직접 집어넣어야 하는 파일들로 되어 있지만, 이건 어차피 튜토리얼이니까 기본 제공되는 demo 폴더 내의 이미지와 비디오 파일을 쓰자.


```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='demo/demo_result.jpg')

# test a video and show the results
video = mmcv.VideoReader('demo/demo.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

`python tutorial_1.py`로 실행해보면 GUI가 지원되는 환경이면 결과가 뜰 것이고, 아니면 `demo/demo_result.jpg`를 열어보면 된다.



<center><img src="/public/img/2021-08-30-MMDetection/demo_result.jpg" width="70%" alt="demo_result.jpg"></center>


### Demos(Image, Webcam, Video)

이미지 한 장에 대해서 테스트하는 경우를 가져왔다. 나머지 경우는 [공식 문서](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#demos)에서 보면 된다. 

```bash
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}]
```

예시:

```bash
python demo/image_demo.py demo/demo.jpg \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --device cuda:0
```

---

### Test existing models on standard datasets

- [공식 문서](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#test-existing-models-on-standard-datasets)

COCO, Pascal VOC, CityScapes(, DeepFashion, Ivis, Wider-Face) 등의 표준 데이터셋의 경우 바로 테스트를 수행해 볼 수 있다. 데이터를 다운받고 압축을 풀어 아래와 같은 형태로 두면 된다.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

COCO-stuff 데이터셋 등 다른 일부 데이터셋 혹은 추가 실행 옵션의 경우 공식 문서를 참조하면 된다.

가장 기본이 되는 1개의 GPU로 결과를 확인하는 코드는 아래와 같다. 아무 키나 누르면 다음 이미지를 보여준다.

```bash
python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --show
```

생성된 이미지를 보여주지 않고 저장하는 코드는 아래와 같다. 그냥 output directory를 지정하기만 하면 된다.

- 참고: 공식 문서에는 config 파일 이름이 조금 잘못된 것 같다.

```bash
python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --show-dir faster_rcnn_r50_fpn_1x_results
```

COCO 말고 다른 데이터셋을 쓰거나, Multi-GPU 등을 사용하는 경우는 공식 문서를 참조하면 된다.

저장된 5000개의 이미지(COCO test, faster_rcnn) 중 하나를 가져와 보았다.

<center><img src="/public/img/2021-08-30-MMDetection/000000000139.jpg" width="70%" alt="faster_rcnn_result.jpg"></center>

---

### Test without Ground Truth Annotations

- [공식 문서](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#test-without-ground-truth-annotations)

#### 데이터셋 형식 변환

COCO 형식으로 변환하는 것을 기본으로 하는 것 같다.

다른 형식의 데이터셋은 다음 코드를 통해 COCO 형식으로 바꿀 수 있다.

```bash
python tools/dataset_converters/images2coco.py \
    ${IMG_PATH} \
    ${CLASSES} \
    ${OUT} \
    [--exclude-extensions]
```

파일 형식에 따라 다음 파일로 대체하면 된다.

- VOC: [pascal_voc.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/pascal_voc.py)
- CityScapes: [cityscapes.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/cityscapes.py)
- Others: [images2coco.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/images2coco.py)

변환을 완료했으면 다음 코드를 통해 GT annotation 없이 테스트를 해볼 수 있다.

```bash
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --format-only \
    --options ${JSONFILE_PREFIX} \
    [--show]
```

--- 

### Train predefined models on standard datasets

- [공식 문서](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#train-predefined-models-on-standard-datasets)

데이터셋 등은 위에서 설명한 대로 준비해 놓자.

학습하는 코드는 다음과 같다.

```bash
# single GPU
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
# Multiple GPUs
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]

# 예시
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py 
    --work-dir work_dir_tutorial_2

bash ./tools/dist_train.sh  
    configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py 
    2
```

log 파일과 checkpoint는 `work_dir/` 또는 `--work-dir`로 지정한 디렉토리에 생성된다.

config 파일만 지정하면 알아서 학습이 진행된다. 실행 환경 정보, 모델, optimizer, 평가방법 등 config 정보 등이 출력되며 학습이 시작된다.  
기본적으로 evaluation을 매 epoch마다 진행하는데, 이는 [추가 옵션](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#training-on-a-single-gpu)으로 바꿀 수 있다.

- 1개의 Titan X로는 5일 7시간, 2개로는  3일 8시간 정도 소요된다고 나온다.


---

이외에도 여러 개의 job을 동시에 돌리는 법, Slurm으로 돌리는 방법 등이 공식 홈페이지에 있으니 쓸 생각이 있으면 참조하면 된다.

---

