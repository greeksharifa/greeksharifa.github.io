---
layout: post
title: Python 프로젝트 생성하기
author: Youyoung
categories: [파이썬]
tags: [파이썬]
---

이번 글에서는 새로운 파이썬 프로젝트를 진행하기 위한 환경을 쉽고 빠르게 구축하는 방법에 대해 서술해보겠습니다.  
본 글은 윈도우를 기준으로 설명합니다.  


---
# Python 프로젝트 생성하기  
## Poetry  
`poetry`는 의존성 관리와 빌드를 책임지는 라이브러리입니다.  

`poetry`를 설치한 후, 아래 명령어를 입력합니다.  

```bash
# 새로운 프로젝트를 만들 경우
poetry new my-project

# 기존 프로젝트를 활용할 경우
poetry init
```

이제 `pyproject.toml` 파일이 생성되었을 것입니다.

새로 가상 환경을 만든다고 할 때, 프로젝트의 root directory 아래에 virtualenv 가 있는 것이 편합니다. 만약 새로 만드는 것이 싫다면 아래와 같이 설정합니다.  

```bash
poetry config virtualenvs.create false # 기본 값은 true
```

| [이 링크](https://python-poetry.org/docs/configuration/#virtualenvscreate)를 참조하셔도 좋습니다.  

프로젝트 내부에 `.venv` 폴더를 생성하는 옵션은 아래와 같습니다.

```bash
poetry config virtualenvs.in-project true # 기본 값은 None
```

의존성은 위 파일의 `[tool.poetry.dependencies]`와 `[tool.poetry.dev-dependencies]`에서 관리하고 있습니다. `add` 서브 커맨드를 통해 의존성을 추가할 수 있습니다.  

```bash
poetry add numpy
```

이 때 `poetry.lock` 파일이 생성됩니다.  

다음 명령어를 실행하면 전체적으로 업데이트가 가능합니다.  

```bash
poetry update
```

현재 프로젝트의 `pyproject.toml` 파일을 읽어 의존성 패키지를 실행하고 싶을 때는 아래 명령어를 실행합니다.  

```bash
poetry install
```

설치된 패키지 목록은 `show`를 통해 알아 볼 수 있습니다.  

```bash
# 설치된 모든 패키지
poetry show

# 개발환경용은 제외
poetry show --no-dev

# 세부 내용
poetry show django

# 최신 버전
poetry show --latest (-l)

# 업데이트가 필요한 패키지
poetry show --outdate (-o)

# 의존성 트리
poetry show --tree
```

가상 환경에 대한 정보는 아래 명령어르 확인할 수 있습니다.  

```bash
# 가상 환경 정보 확인
poetry env info

# 가상환경 리스트 확인
poetry env list
```

## Github Action  
**Github Action**은 github에서 제공하는 CI/CD 도구이며, 소프트웨어 개발의 workflow를 자동화해줍니다.  

**Github Action**에는 반드시 알아야 할 개념들이 있습니다. [이 문서](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions)를 확인하는 것이 가장 정확합니다.  

가장 기본적인 설명은 이러합니다.  

| PR 생성과 같이 repository에 특정 `event`가 발생하면 트리거되는 Github Action `workflow`를 구성할 수 있습니다. `workflow`는 순차적 혹은 병렬적으로 동작하는 1개 이상의 `job`을 갖고 있습니다. 각각의 `job`은 할당된 가상 머신 `runner` 내부 혹은 container 내부에서 동작하며 `action` 혹은 script를 실행하도록 되어 있습니다.  

**workflow**  
- 1개 이상의 job을 실행시키는 자동화된 프로세스  
- yaml 파일로 정의함
- repository 내에 .github/workflows 디렉토리 안에서 정의됨  
- 하나의 repository는 복수의 workflow를 정의할 수 있음  

**events**  
- workflow run을 트리거하는 특정 행동  

**jobs**  
- 동일한 runner 내에서 실행되는 여러 step
- 각 step은 shell script이거나 동작할 action

**action**  
- workflow의 가장 작은 블록
- 재사용이 가능한 component

**runner**
- github action runner 어플리케이션이 설치된 머신
- workflow가 실행될 인스턴스

이제 실제로 `workflow`를 작성해 보겠습니다. 앞서 소개한 [여기](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions#create-an-example-workflow)를 참조해주세요.  




<center><img src="/public/img/2022-07-31-python-project/01.PNG" width="60%"></center>  


## References  
- [참고 블로그](https://blog.gyus.me/2020/introduce-poetry/)  
- [참고 블로그](https://velog.io/@ggong/Github-Action%EC%97%90-%EB%8C%80%ED%95%9C-%EC%86%8C%EA%B0%9C%EC%99%80-%EC%82%AC%EC%9A%A9%EB%B2%95)  

