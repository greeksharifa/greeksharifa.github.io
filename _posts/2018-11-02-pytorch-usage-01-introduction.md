---
layout: post
title: PyTorch 사용법 - 01. 소개 및 설치
author: YouWon
categories: PyTorch
tags: [PyTorch, TensorFlow, usage]
---

---

[PyTorch 사용법 - 00. References](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)  
**[PyTorch 사용법 - 01. 소개 및 설치](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-01-introduction/)**  
[PyTorch 사용법 - 02. Linear Regression Model](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)  
[PyTorch 사용법 - 03. How to Use PyTorch](https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/)  

---

*2021.07.12 updated*

## 간단한 소개

PyTorch는 유연성과 속도를 모두 갖춘 딥러닝 연구 플랫폼이다. GPU 사용이 가능하기 때문에 속도가 상당히 빠르다.  
또 입문 난이도가 높지 않은 편이고 코드가 간결하다.  

- [공식 홈페이지](https://pytorch.org/)
- [Documentatino](https://pytorch.org/docs/stable/index.html)

---

## 설치 방법

[여기](https://pytorch.org/)를 참조한다. 자신에게 맞는 OS, package manager, Python 버전, CUDA 버전 등을 선택하면 그에 맞는 명령어 집합이 나온다. 이를 명령창에 실행하면 설치가 진행된다.  
torchvision을 설치할 경우에 무슨 라이브러리가 없다면서 에러 메시지가 뜨긴 하는데, 사용하는 데 별 문제는 없을 것이다. 만약 자신이 그 부분을 꼭 써야 한다면 에러를 해결하고 넘어가자.

설치를 완료했으면, 명령창에 다음과 같이 입력해보자. Anadonda를 플랫폼으로 사용한다면 conda 설정은 직접 해 주어야 한다.

`python`

```python
# 이 부분은 Python Interpreter에서 입력함.
import torch  
x = torch.randn(3,5)  
print(x)
```

결과가 대략 다음과 같이 나오면 설치가 완료되었다. 숫자가 다른 것은 랜덤이니 신경 쓰지 말자.

![01_run_pytorch.PNG](/public/img/PyTorch/2018-11-02-pytorch-usage-01-Introduction/01_run_pytorch.PNG)

---

## GPU 사용을 위한 설치

GPU 사용을 위한 필수 절차는 다음과 같다. 

Ubuntu의 경우 [여기](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)를 참조해도 된다.

1. **호환성 체크**
    1. 컴퓨터에 있는 GPU의 **compute capability** 확인
        - [여기](https://developer.nvidia.com/cuda-gpus)에서 확인
    2. compute capability에 맞는 CUDA SDK 버전 확인
        - [여기](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)에서 확인
    3. CUDA, nvidia-driver 호환 확인
        - [여기](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)에서 확인
        - CUDA toolkit 호환성 확인은 [여기](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)에서
    4. Pytorch와 CUDA의 호환성 확인
        - 설치하고자 하는 PyTorch(또는 Tensorflow)가 지원하는 최신 CUDA 버전이 있다. 이보다 상위 버전의 CUDA를 설치하면 PyTorch 코드가 제대로 돌아가지 않는다.
        - [Pytorch 홈페이지](https://pytorch.org/)에서 정해주는 CUDA 버전을 설치하는 쪽이 편하다. 2020.02.13 기준 최신 버전은 10.1이다.
    5. CUDA에 맞는 cuDNN 버전 확인
        - [여기](https://developer.nvidia.com/rdp/cudnn-archive)에서 확인할 수 있다.
2. **이전 버전의 CUDA 제거**
    1. CUDA를 여러 개 쓸 수도 있지만, 이전 버전의 CUDA를 제거해 주면 좋다. 
        1. Windows의 경우 NVIDIA 관련 프로그램 제거를 해 주면 된다.
        2. Ubuntu의 경우 살짝 까다로운데, 터미널에 다음 코드를 입력한다.
            ```bash
            sudo apt-get purge nvidia*
            sudo apt-get autoremove
            sudo apt-get autoclean
            sudo rm -rf /usr/local/cuda*
            ```
        3. 혹시 오류가 뜨면 아래 **7. 오류 해결법**을 참조하자.
    2. 예전엔 어땠는지 잘 모르겠지만 최근 CUDA 설치 시 그에 맞는 nvidia-driver가 같이 설치된다. 따로 특정 버전의 driver를 요구하는 것이 아니라면 그대로 설치하자.
3. Nvidia Driver 설치
    1. Windows의 경우 Geforce Experience 혹은 [Nvidia](https://www.nvidia.co.kr/Download/index.aspx?lang=kr)에서 적절한 버전의 Driver를 설치한다.
    2. Ubuntu의 경우 다음 코드를 입력해 본다. 
        ```bash
        # 가능 드라이버 확인 
        sudo apt search nvidia-driver 
        # 특정 드라이버 설치 
        sudo apt-get install nvidia-driver-455
        ```
4. **CUDA 설치**
    1. Windows
        1. [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)에서 원하는 CUDA를 다운받는다. 운영체제와 버전 등을 체크하고, 가능하면 Installer Type은 network가 아닌 local로 받는다. 인터넷으로 설치하면서 받는 것이 아닌 한번에 설치파일을 받는 식이다.
            - 같은 버전인데 update가 추가된 버전이 있다. 보통은 이것까지 추가로 설치해 주는 쪽이 좋다. base installer를 먼저 설치한 뒤에 추가로 설치해 주도록 하자.
        2. 설치 파일로 CUDA를 설치한다. 설치 시에는 다른 프로그램을 설치하거나 제거하는 중이면 실행이 되지 않으니 주의하자.
        3. cuda visual studio integration 관련해서 설치 실패가 뜨는 경우가 많은데, 이 부분이 필요한 코드를 실행할 일이 있다면 이 단계에서 다시 설치해 주는 것이 좋다. Visual Studio를 설치하면 해결이 되는 경우가 많다.
        4.  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin`을 등록하자. 버전에 따른 경로를 유의하자.
            - 실행이 잘 안 되는 경우 상위 또는 하위 폴더 몇 개를 추가 등록하면 되는 경우도 있다.
    
    2. Ubuntu 18.04(CUDA 11.0 기준)
        1. 역시 [CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)에 접속한다. Linux 버전에 따라서 알맞게 선택한다. Ubuntu 18.04를 선택한다면, 일반적으로 `Linux - x86_64 - Ubuntu - 18.04`를 따른다.
        2. 다음으로 Installer Type이 있는데, runfile의 경우는 `.exe` 파일처럼 실행이 가능하고, `deb(local)`은 터미널에 코드를 몇 줄 입력하면 되는 방식이다.
        3. runfile을 선택하면 다음 비슷한 코드를 실행하라고 안내가 뜬다.
            ```bash
            wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
            chmod +x cuda_11.0.2_450.51.05_linux.run
            sudo sh cuda_11.0.2_450.51.05_linux.run
            ```
        4. 아래 줄까지 실행하면 안내 페이지가 뜬다. 
            1. 드라이버가 이전에 설치된 게 있다고 뜨는데, 미리 제거해 두는 것이 편하긴 하다. 그러나 제거하지 않아도 될 때도 있다. 엔터키를 누르면 X 표시가 토글된다.
            2. 다음으로 계약 동의를 위해 accept를 입력하고 엔터키를 누른다.
            3. 그냥 기본으로 두고 Install을 해도 된다. 그러나 Driver 설치 단계에서 오류가 나면(설치 실패시 로그를 확인하라고 뜬다), Driver을 엔터키를 눌러 체크 해제한다. CUDA symbolic link를 대체하고 싶지 않다면 역시 symbolic link 부분을 체크 해제한다.
            4. 정상적으로 설치가 된다면 다음과 비슷한 것을 볼 수 있다.
                ```
                ===========
                = Summary =
                ===========

                Driver:   Not Selected
                Toolkit:  Installed in /usr/local/cuda-11.0/
                Samples:  Installed in /root/, but missing recommended libraries

                Please make sure that
                -   PATH includes /usr/local/cuda-11.0/bin
                -   LD_LIBRARY_PATH includes /usr/local/cuda-11.0/lib64, or, 
                add /usr/local/cuda-11.0/lib64 to /etc/ld.so.conf and run ldconfig as root
                ```
            5. `sudo vim /etc/bash.bashrc`을 실행한 다음, 파일의 가장 아래쪽에 다음 코드를 추가하자. 버전에 따른 경로를 유의하자.
                ```
                export PATH=/usr/local/cuda-11.0/bin:$PATH
                export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
                export LD_LIBRARY_PATH=/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
                ```
            6. 다음 코드를 실행하여 변경 사항을 적용한다.
                ```bash
                source /etc/bash.bashrc
                ```
        5. deb(local)을 선택하면 터미널에 코드 몇 줄을 입력하면 된다. 이전에 CUDA 설치가 꼬인 것이 아니라면 보통은 에러 없이 설치된다. 버전에 따라 경로가 달라지므로 유의하자.
            ```bash
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
            sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
            sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
            sudo apt-get update
            sudo apt-get -y install cuda
            ```
5. **cuDNN 설치**
    1. Windows
        1. 우선 [cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)에서 사용하고자 하는 CUDA에 맞는 버전(`cuDNN Library for Windows (x86)`)을 찾아 다운받는다. login이 필요하다.
        2. 압축을 풀어 CUDA 설치 폴더(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0`)에 붙여넣기 하면 된다. 폴더 경로는 설치한 CUDA 버전에 따라 달라진다.
    2. Ubuntu 18.04(cudnn 8.0.3 기준)
        1. 비슷하게 [여기](https://developer.nvidia.com/rdp/cudnn-archive)에서 `cuDNN Library for Linux (x86_64)`을 받는다. 이유는 잘 모르겠으나 wget으로 잘 받아지지 않는 경우가 있으니 브라우저로 접속하여 다운로드하자.
        2. 받고 나서 `tar xvf cudnn-11.0-linux-x64-v8.0.3.33.tgz`으로 압축을 해제한다.
        3. 생성된 CUDA 폴더로 이동하여 파일들을 복사한다. 
            ```bash
            cd cuda
            sudo cp include/cudnn* /usr/local/cuda-11.0/include
            sudo cp lib64/libcudnn* /usr/local/cuda-11.0/lib64/
            sudo chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*
            ```
        4. 설치되었는지 확인하자.
            ```bash
            cat /usr/local/cuda-11.0/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
            ```
            결과:
            ```
            R -A 2
            #define CUDNN_MAJOR 8
            #define CUDNN_MINOR 0
            #define CUDNN_PATCHLEVEL 3
            --
            #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

            #endif /* CUDNN_VERSION_H */
            ```
            8.2.0임을 확인할 수 있다.
6. **설치 확인**
    1. NVCC를 통해 CUDA 설치를 확인해보자.
        ```bash
        nvcc -V
        dpkg - l | grep CUDA 
        dpkg - l | grep cuDNN
        ```
        만약 nvcc가 없으면 다음을 입력하자.
        ```bash
        sudo apt install nvidia-cuda-toolkit
        ```
    1. 다음 코드를 python을 실행하여 입력해보고 `True`가 뜨면 성공한 것이다.
        ```python
        import torch
        torch.cuda.is_available()
        ```
7. **에러 해결법**
    1. `E: sub process /usr/bin/dpkg returned an error code (1)`의 에러가 뜬다면 다음을 터미널에 입력한다.
        ```bash
        sudo rm /var/lib/dpkg/info/*
        sudo dpkg --configure -a
        sudo apt update -y
        ```
    2. `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.` 에러가 뜨면 nvidia-driver를 재설치해야 한다.
        ```bash
        # 설치된 driver 확인
        sudo apt --installed list | grep nvidia-driver
        # 문제 있는 driver 삭제(보통 전부 삭제)
        sudo apt remove nvidia-driver-<version>
        sudo apt autoremove
        # 재설치 & 재부팅
        sudo apt-get install nvidia-driver-<원래 driver 버전>
        sudo reboot now
        ```
    3. nvidia-smi를 쓰려고 하는데 `VNML: Driver/library version mismatch`라는 에러가 날 때가 있다. 그런 경우 `lsmod | grep nvidia`를 터미널에 입력하고 nvidia kernel을 unload하면 된다. 오른쪽 column에 무언가 있다면 unload하면 된다.
        ```bash
        sudo rmmod nvidia_drm
        sudo rmmod nvidia_modeset
        sudo rmmod nvidia_uvm
        sudo rmmod nvidia
        ```
        위의 작업 도중 `rmmod:ERROR: Module nvidia is in use`라는 에러가 뜨면 nvidia 관련 process를 종료시켜주자.
        ```bash
        sudo lsof /dev/nvidia*
        sudo kill -9 <process_id>
        ```
        다시 `lsmod | grep nvidia`를 하고 아무 것도 안 뜬다면 완료된 것이다.
        


---

## 참고: Ubuntu Python 설치

python 3.7 이후 버전은 (그 이전 버전도 있을 수 있다) `apt-get` 설치를 지원한다.


```bash
sudo apt update
sudo apt install python3.7 
sudo apt install python3.8
sudo apt install python3.9
# sudo apt-get install python3 python3-pip python3-dev python3-setuptools
```

Python이 여러 개 설치되어 터미널에 `python`을 입력했을 때 원하는 버전이 나오지 않는다면 `vim ~/.bashrc`로 파일을 열고 맨 아래에 다음과 비슷하게 추가하자.

```bash
# 명확하게 하길 원한다면 경로를 직접 지정하는 것이 편하다. 
# 최신 버전의 경우 '/usr/bin/python*` 또는 `/usr/local/bin/python*` 경로에 존재한다.
alias python='/usr/bin/python3.9'

# python 2 대신 3을 사용하고 싶은 경우
alias python=python3
alias pip=pip3
```


`:wq`를 입력하여 저장하고 나온 뒤 터미널에 `source ~/.bashrc`를 입력하여 변경사항을 적용하자.


Python 버전의 우선순위를 쓰고 싶다면 Python을 먼저 선택지에 추가해야 한다.

```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
# 마지막 숫자는 우선순위로 클수록 높은 우선권을 갖는다.
```

우선순위를 변경하고자 하는 경우 다음 명령을 입력하여 숫자를 누르고 엔터키를 누른다.

```bash
sudo update-alternatives --config python3
```

다시 auto mode로 돌아가려면 `sudo update-alternatives --auto python3`을 입력한다.

### 재설치 방법

```bash
sudo python3 -m pip uninstall pip 
sudo apt-get install python3-pip --reinstall
```


### 오류 해결법

**ModuleNotFoundError: No module named 'pip._internal' 오류**

다음 두 가지를 시도해본다.

```bash
# 재설치
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --force-reinstall
```

```bash
# pip upgrade
python -m pip install --user --upgrade pip
```

**pip3 on python3.9 fails on 'HTMLParser' object has no attribute 'unescape' 오류**

```bash
pip3 install --upgrade setuptools
# 안 되면 다음을 시도한다.
pip3 install --upgrade pip
pip3 install --upgrade distlib
```

---

## PyTorch Project 구조

프로젝트의 구조는 코딩하는 사람 마음대로이긴 하나, 기본적으로는 다음과 같은 구조를 따른다.

1. Set HyperParameter and Run
2. Load Data
3. Define and Load Models
3-1. Define util functions
4. Set Loss function(creterion) and Optimizer
5. Train Model
6. Visualize and save results

PyTorch는 각 단계에서 다음의 장점을 갖는다.
1. PyTorch가 아닌 Python의 특징인데, 여러분은 많은 Machine Learning 코드를 보면서 `python train.py --epochs 50 --batch-size 16` 등 많은 옵션을 설정할 수 있는 것을 보았을 것이다. Python의 `argparse` 패키지는 이것을 가능하게 해 준다. 
2. 데이터 로드 시 `DataLoader`라는 클래스를 제공한다. `DataLoader`를 통해 데이터를 불러오면, 이 안에서 데이터 처리에 대한 거의 모든 것을 쉽게 수행할 수 있다. 
    - 이를테면 Data Augmentation 같은 것도 전부 제공된다.
    - 여러 종류의 Data Transformation이 지원된다.
3. 일반적인 모델을 불러올 때는 다른 Deep Learning Framework도 대체로 간결하지만, PyTorch는 `torchvision`이라는 패키지에서 따로 pretrain까지 된 모델들을 제공하므로 다른 곳에서 모델을 다운로드할 필요 없이 이를 바로 쓸 수 있다.
2-1. 많은 프로그래머들이 `utils.py`에 유틸리티 함수(예를 들면 [YOLO](https://greeksharifa.github.io/paper_review/2018/10/26/YOLOv2/)에서 [IoU](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)를 구하는 함수)를 따로 빼내어 여러 가지를 한번에 정의한다. 프로젝트에서 부가적인 부분은 따로 관리하는 것이 가독성이 좋다.
4. 이 부분은 다른 Deep Learning Framework와 비슷하다.
5. Tensorflow와는 달리 Session을 설정할 필요가 없다.
6. 이 부분도 역시 비슷하다.

---

[다음 글](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/)에서는 Linear Regression Model을 예로 들어서 간단한 프로젝트의 구조를 설명하도록 하겠다.

---

## References

- [Reference](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-00-references/)

PyTorch에서 자주 사용되는 함수들을 정리한 글이다. 
