---
layout: post
title: MMDetection 사용법 2(Tutorial)
author: YouWon
categories: References
tags: [Linux, Ubuntu, open-mmlab, usage]
---

이 글에서는 [MMDetection](https://github.com/open-mmlab/mmdetection)를 사용하는 방법을 정리한다.

- [Documentation](https://mmdetection.readthedocs.io/)
- [Github](https://github.com/open-mmlab/mmdetection)

- [Colab Tutorial](https://colab.research.google.com/github/ZwwWayne/mmdetection/blob/update-colab/demo/MMDet_Tutorial.ipynb#scrollTo=Wuwxw1oZRtVZ)


[이전 글](https://greeksharifa.github.io/references/2021/08/30/MMDetection/)에서는 설치 및 [Quick Run](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html) 부분을 다루었으니 참고하면 좋다.

---


---

## Tutorial 1: Learn about Configs

- [공식 문서](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)



---



```
mmdetection
├── checkpoints
|   ├── faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
├── configs
│   ├── faster_rcnn
│   │   ├── faster_rcnn_r50_fpn_1x_coco.py
│   │   ├── ...
```

<center><img src="/public/img/2021-08-30-MMDetection/000000000139.jpg" width="70%" alt="faster_rcnn_result.jpg"></center>
