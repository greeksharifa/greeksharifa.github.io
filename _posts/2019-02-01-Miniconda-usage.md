---
layout: post
title: Miniconda(Anaconda) 사용법(conda 설치 및 사용법)
author: YouWon
categories: References
tags: [Miniconda, usage]
---

Anaconda는 Continuum Analytics라는 곳에서 만든 파이썬 배포판으로 수백 개의 파이썬 패키지를 포함하는 환경을 구성한다. Anaconda로는 virtualenv와 같은 여러 개의 가상환경을 만들어 각각의 환경을 따로 관리할 수 있다.  
그 중 Miniconda는 이것저것 설치할 것이 많은 Anaconda에서 패키지를 다르게 설치할 여러 환경들을 관리한다는 최소한의 기능만 가진 부분만 포함하는 mini 버전이다. 따라서 이 글에서는 Miniconda를 설치하여 가상환경을 관리하는 법을 알아보겠다.

---

## 설치

<script data-ad-client="ca-pub-9951774327887666" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

[Anaconda](https://www.anaconda.com/distribution/#download-section)를 설치하거나, [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치한다. 설치하고 싶을 운영체제와 버전에 맞는 것을 골라 설치한다. 설치 방법은 공식 홈페이지에 따로 설명되어 있다.

![01_install](/public/img/conda/2019-02-01-Miniconda-usage/01.PNG)
Not recommended라고 되어 있는 옵션이지만 체크하면 PATH에 등록하지 않아도 된다(이건 환경마다 조금 다르다).
![02_install](/public/img/conda/2019-02-01-Miniconda-usage/02.PNG)

설치 후 다음 명령을 명령창(cmd / 터미널)에 입력해본다.
```python
conda list
```

만약 다음과 같이 오류가 뜬다면 conda가 System PATH에 등록되지 않은 것이므로 등록을 해 준다.

![03_install](/public/img/conda/2019-02-01-Miniconda-usage/03.PNG)
![04_install](/public/img/conda/2019-02-01-Miniconda-usage/04.PNG)

윈도우10, Miniconda3인 경우 `C:\ProgramData\Miniconda3\Scripts`를 PATH에 등록해 준다.

설치 패키지 목록은 다를 것이지만 다음과 같이 뜬다. `conda list`는 현재 환경(기본 환경의 이름은 base이다)에서 설치된 패키지 목록을 나타내는 명령이다.

![05_conda_list](/public/img/conda/2019-02-01-Miniconda-usage/05.PNG)

---

## 가상환경 목록 확인, 생성 및 삭제

다음을 명령창에 입력한다.
```python
conda env list
# 또는,
conda info --envs
```

현재 활성화된 가상환경 옆에는 * 가 추가된다.

![06_env_list](/public/img/conda/2019-02-01-Miniconda-usage/06.PNG)

처음 설치했을 때는 기본값인 base 하나만 있을 것이다. 

다음 명령을 통해 새 가상환경을 하나 생성한다.
```python
# -n 옵션은 --name과 같은 것으로, 가상환경 이름을 myenv로 지정한다.
conda create -n myenv
# python=3.6 옵션은 가상환경 생성 시 파이썬 버전을 지정한다.
# 지정하지 않으면 conda에 기본 포함된 파이썬 버전으로 생성된다.
conda create -n condatorch python=3.6

# 특정 패키지 버전을 지정하면서, 그리고 패키지를 설치하면서 생성하는 것도 가능하다.
conda create -n myenv python=3.4 scipy=0.15.0 astroid babel

# 가상환경 생성 시 이것저것 깔리는 것이 싫다면 다음 옵션을 주면 된다.
conda create --no-default-packages -n myenv python

# 새 가상환경을 만들 때 특정 가상환경 안에 설치된 패키지 전부를 설치하면서 생성할 수 있다.
# base 가상환경에 있는 패키지를 전부 설치하면서 생성한다면, 
conda create -n myenv --clone base

# environment.yml 파일이 있다면 다음과 같이 생성할 수 있다.
# 생성 방법은 이후에 설명한다.
conda env create -f environment.yml
```

계속 진행하겠냐는 물음이 보이면 `y`를 입력한다.

![07_env_create](/public/img/conda/2019-02-01-Miniconda-usage/07.PNG)

다시 `conda env list`로 목록을 확인해보면 지정한 이름으로 가상환경이 생성되었음을 확인할 수 있다.

![08_env_create](/public/img/conda/2019-02-01-Miniconda-usage/08.PNG)

위 그림에서 activate condatorch, deactivate 등의 명령이 쓰여 있는 것을 확인할 수 있는데, 이는 특정 가상환경을 활성화 또는 비활성화할때 사용하는 명령이다(가상환경이 무엇에 쓰는 것인지 알면 무슨 말뜻인지 알 수 있을 것이다). 이는 다음 절에서 설명한다.

가상환경 삭제는 다음 명령을 통해 수행할 수 있다.
```python
# 생성할 때와는 다르게 env를 앞에 적어주어야 한다.
# 생성 시에는 env를 앞에 적으면 실행이 되지 않는다.
# remove 앞에 env를 써 주지 않으면 가상환경 삭제가 아닌 패키지 삭제가 이루어진다.
# conda env remove -n <environment_name>
conda env remove -n condatorch
# 다음도 가능하다.
conda remove --name myenv --all
```

![09_env_remove](/public/img/conda/2019-02-01-Miniconda-usage/09.PNG)


### Requirements.txt로 가상환경 생성하기

아래 명령들은 가독성을 위해 두 줄로 펼쳐 놓았다.

Windows 환경이라면 명령창에 다음과 같이 쓰는 것이 가능하다.
```
FOR /F "delims=~" %f in (requirements.txt) 
DO conda install --yes "%f" || pip install "%f"
```

Unix 환경이라면 다음과 같이 쓸 수 있다.
```
while read requirement; do conda install --yes $requirement; 
done < requirements.txt 2>error.log
```
conda로는 설치가 안 되고 pip으로는 설치가 되는 패키지가 있다면 다음과 같이 쓸 수 있다.
```
while read requirement; do conda install --yes $requirement 
|| pip install $requirement; done < requirements.txt 2>error.log
```

다음을 참조하였다: [github 글](https://gist.github.com/luiscape/19d2d73a8c7b59411a2fb73a697f5ed4), [stackoverflow 글](https://stackoverflow.com/questions/35802939/install-only-available-packages-using-conda-install-yes-file-requirements-t)

---

## 가상환경 활성화, 비활성화

가상환경 활성화는 위에서도 설명했듯 다음과 같이 쓰면 된다.
```python
activate <environment_name>
activate condatorch
```
Unix 등의 환경에서는 `activate`가 아닌 `source activate`를 써야 한다.

그러면 명령창의 맨 앞에 (condatorch)와 같이 활성화된 가상환경 이름이 뜬다.

비활성화는 다음 명령으로 할 수 있다.
```python
deactivate
# 설치한 버전에 따라 deactivate는 deprecated되었다는 경고를 볼 수도 있다. 이 경우 conda deactivate이다.
```

![10_activate](/public/img/conda/2019-02-01-Miniconda-usage/10.PNG)

위 그림이 잘 이해가 되지 않는다면, `activate`를 여러 번 쓰지 않을 것을 권장한다.

---

## 가상환경 안에 패키지 설치

버전에 따라 조금씩 다른 경우도 있으나, 최신 버전(2019-02-01 기준)의 Miniconda3에서는 pip, whl, conda를 통한 설치 모두 현재 활성화된(없다면 base 또는 컴퓨터에 깔려 있는 다른 버전의 파이썬에) 가상환경에만 설치된다. 따라서 각 환경 간 거의 완전한 분리가 가능하다.

패키지 설치는 다음과 같다. `pip`과 거의 비슷하다.
```python
conda install seaborn
# 여러 개를 동시에 설치할 경우 comma 없이 그냥 나열한다.
conda install numpy pandas
```

설치된 패키지 목록을 보고 싶으면 다음을 입력한다.
```python
conda list
```

참고로 conda 환경에서도 pip 등을 통한 설치가 가능하다.


### environment.yml 파일 생성 및 가상환경 생성

설치된 패키지 목록을 `.yml` 파일로 저장하는 명령이다.  
`pip freeze > requirements.txt`와 같은 역할이다.
```python
conda env export > environment.yml
```

만들어진 파일은 다음과 비슷하게 생겼다.
```
name: condatorch
channels:
  - pytorch
  - defaults
dependencies:
  - blas=1.0=mkl
  - certifi=2018.11.29=py36_0
  ...
  - zstd=1.3.7=h508b16e_0
  - pip:
    - cycler==0.10.0
    ...
    - six==1.12.0
prefix: C:\ProgramData\Miniconda3\envs\condatorch
```

만들어진 파일로 가상환경을 생성하는 방법은 위에서도 설명했지만 다음과 같다.
```python
# 이 경우에는 env를 앞에 써 주어야 한다.
# -f는 --file을 의미한다.
conda env create -f environment.yml -n myenv
```


### 패키지 업데이트

특정 환경 안의 특정 패키지를 업데이트하려면 다음과 같이 하면 된다.

```python
conda update -n <environment_name> spacy
```

특정 환경 안의 모든 패키지를 업데이트하려면 다음과 같이 하면 된다.

```python
conda update -n <environment_name> --all
# 현재 환경 업데이트
conda update --all
```




---

## Conda 버전 확인 및 update

명령창에서 Conda의 버전을 확인하는 방법은 다음과 같다.
```
conda -V
conda --version
```

Conda 자체를 업데이트하는 방법은 다음과 같다.

```
conda update conda
conda update anaconda
```

---

## References

[공식 홈페이지](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)에서 더 자세한 사용법을 찾아볼 수 있다.
