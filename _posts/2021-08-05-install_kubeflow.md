---
layout: post
title: WSL2 Ubuntu 환경에서 Kubeflow 설치하기
author: Youyoung
categories: [MLOps]
tags: [MLOps, Kubernetes, Minikube, Kubeflow]
---

본 글은 Windows10 운영체제를 메인으로 사용하는 필자가 WSL2를 활용하여 Kubeflow를 설치하는 과정을 기록한 것이다.  


## 1. WSL2 및 Ubuntu 설치  
**Kubeflow**를 설치하기 위해서는 Kubeflow 전용 커맨드 라인 툴인 **kfctl** 설치가 필요한데, 안타깝게도 본 도구의 경우 Windows 환경은 지원하지 않는다. 따라서 Linux용 Windows 하위 시스템인 **WSL2** 설치가 필요하다. Microsoft 홈페이지에 친절한 [WSL 설치 글](https://docs.microsoft.com/ko-kr/windows/wsl/install-win10)이 존재하기에 이 과정을 수행하면 된다. 필자의 경우 Windows 참가자 프로그램에 가입하고 **단순화된 설치** 과정을 수행하였다.  

Ubuntu 18.04 LTS 를 설치하면 아래와 같은 터미널을 확인할 수 있다.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/01.PNG" width="100%"></center>  

설치한 Ubuntu에 WSL Version2가 적용되도록 해주자.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/02.PNG" width="70%"></center>  

기본 설정이 Ubuntu가 되도록 바꿔주어야 한다. [이 페이지](https://docs.docker.com/docker-for-windows/wsl/)를 참고하면 된다.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/03.PNG" width="50%"></center>  


----
## 2. Docker Desktop 및 Minikube 설치  
먼저 [Docker 다운로드 페이지](https://www.docker.com/products/docker-desktop)에서 **Docker Desktop**을 설치해주자. 다만 Docker Desktop에서 Kubernetes를 직접 실행하지는 않을 것이다.  

Setting-General에서 `Use the WSL 2 based engine (Windows Home can only run the WSL 2 backend)`가 체크되어 있는지 확인하자. 이 항목에 체크를 해야 앞서 설치한 WSL2를 사용하게 된다.  

이제 **Minikube**를 설치할 차례이다. 이전 글에서는 Windows 환경에서 바로 설치하는 방법에 대해 기술하였는데, 본 글에서는 (사실상 동일한 과정이나) Ubuntu에서 설치하는 과정에 대해 설명할 것이다.  

일단 업데이트부터 하자.  
```
sudo apt-get update -y  
```

본인의 컴퓨터 환경에 맞게 아래 커맨드를 입력해주면 된다.  

```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64  
sudo install minikube-linux-amd64 /usr/local/bin/minikube  
minikube version  
```

이제 설치는 완료되었다. Minikube를 구동해보자.  

```
minikube start --cpus 4 --memory 8096 --kubernetes-version=v1.21.2  
```

기본 설정은 `cpus 2, memory 8096`이다. 이 역시 본인의 환경에 맞게 설정하면 된다.  


----
## 3. kfctl 설치 및 배포 템플릿 설정  
**kfctl**은 Kubeflow 컴포넌트를 배포/관리하기 위한 커맨드라인 도구이다. 이 도구를 먼저 설치해야만 Kubeflow 설치를 진행할 수 있다. [kfctl 릴리즈 정보](https://github.com/kubeflow/kfctl/releases)를 참고하여 진행해보자.  

```
wget https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_darwin.tar.gz  
tar -xvf kfctl_v1.2.0-0-gbc038f9_darwin.tar.gz  
```

압축을 풀어주면 kfctl이 생성된 것을 확인할 수 있다. 이제 Kubeflow 배포 스크립트가 설치될 디렉토리와 배포 템플릿을 설정할 차례이다.  

```
export PATH=$PATH:$(pwd)  
export KF_NAME='yy-kubeflow'  
export BASE_DIR=/Users/Youyoung/kubeflow  
export KF_DIR=${BASE_DIR}/${KF_NAME}  
export CONFIG_FILE=${KF_DIR}/kfctl_k8s_istio.v1.2.0.yaml  
export CONFIG_URL="https://raw.githubusercontent.com/kubeflow/manifests/master/distributions/kfdef/kfctl_k8s_istio.v1.2.0.yaml"  

sudo mkdir -p ${KF_DIR}  
cd ${KF_DIR}  
```

물론 위에서 yy-kubeflow, Youyoung 등은 본인이 원하는 이름으로 바꿔서 설정해야 한다.  


----
## 4. Kubeflow 설치  
이제 Kubeflow 설치를 수행해보자.  

```
sudo kfctl build -V -f ${CONFIG_URL}  
sudo kfctl apply -V -f ${CONFIG_FILE}  
```

만약 아래와 같은 에러 메시지를 마주했다면, 필요한 컴포넌트들이 다 준비되지 않았다는 뜻이므로 기다리면 된다.  
```
WARN[] Encountered error applying application cert-manager: (kubeflow.error)
```

1번째 Reference인 책에 따르면, 모든 컴포넌트가 업로드 되기까지 10분 정도 걸릴 수 있다고 한다. 경험 상 상황에 따라 20~30분까지 걸리기도 한다. 실제로 그 이전에 들어가보면 몇몇 Pod에서 Error 메시지가 나타나는 것을 볼 수 있는데, 시간이 좀 지나면 정상적인 Running 상태로 바뀐다. `Ctrl+C`를 눌러서 터미널로 돌아간다.  


----
## 5. Kubeflow 접속  
### 5.1. Kubeflow Dashboard  
대시보드는 istio-system의 istio-ingressgateway 서비스를 통해 접속 가능하다.  

```
kubectl get services -n istio-system  
```

확인해보면 istio-ingressgateway 서비스는 31380 NodPort로 80포트에 매핑되어 있다. 먼저 Port를 열어주자.  

```
export NAMESPACE=istio-system  
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80  
```

이제 주소 창에 `http://127.0.0.1:8080`을 입력해주면 드디어 **Kubeflow Dashboard**에 접속할 수 있다.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/12.PNG" width="60%"></center>  

다음 화면에서 본인이 원하는 **Namespace**를 지정해주고 접속하면 된다.  


### 5.2. Minikube 중지  
```
minikube stop  
```

컴퓨터를 종료하면 자동으로 중지되긴 하지만 평소에 정상적으로 종료하는 것이 좋다. 완전히 삭제하고 싶으면 `minikube delete`을 입력하면 된다.  


### 5.3. WSL2 리소스 제한  
WSL2을 처음 설치해서 사용하면, 이 프로그램이 컴퓨터의 모든 가용 메모리를 잡아 먹는 것을 확인할 수 있다. [WSL 이슈 링크](https://github.com/microsoft/WSL/issues/4166)에서 그 해결책을 찾았다.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/15.PNG" width="80%"></center>  

본인의 사용자 폴더 디렉토리로 접속한 후 다음과 같이 파일을 생성하자.  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/13.PNG" width="50%"></center>  

물론 위 6GB는 예시이고, 원하는 만큼 해주면 된다. (필자는 12GB를 선택함)  

<center><img src="/public/img/mlops/2021-08-05-install_kubeflow/14.PNG" width="70%"></center>  

파일 생성 후 재시작하면 위와 같이 정상적으로 메모리를 사용하는 것을 알 수 있다. (이전에는 24GB에서 96%를 사용함)  


----
## References  
1. 쿠버네티스에서 머신러닝이 처음이라면! 쿠브플로우!, 이명환 저
2. [블로그](https://sidepun.ch/entry/Kubeflow-%EC%84%A4%EC%B9%98-WSL2-Ubuntu-Docker-Desktop)  
3. [블로그](https://teamsmiley.github.io/2020/06/23/k9s/)  
4. [블로그](https://lsjsj92.tistory.com/580)  


