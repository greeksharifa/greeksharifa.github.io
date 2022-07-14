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

<center><img src="/public/img/2022-07-31-python-project/01.PNG" width="60%"></center>  

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

`action` 개념에 대해 추가 설명이 필요하다면 [이 블로그](https://www.daleseo.com/github-actions-basics/)를 참고하셔도 좋겠습니다.  

이제 실제로 `workflow`를 작성해 보겠습니다. 앞서 소개한 [여기](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions#create-an-example-workflow)를 참고해주세요. workflow 구문에 대해 자세히 알고 싶다면 [여기](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)를 참고하면 됩니다.  

workflow는 yaml 파일에 기반하여 구성됩니다. `name`은 workflow의 이름을 의미하며, 이 `name`이 repository의 action 페이지에 표시될 것입니다. `on`은 workflow가 작동하도록 트리거하는 `event`를 정의합니다. 대표적으로 push, pull_request 등을 생각해 볼 수 있을 텐데, 모든 조건에 대해 알고 싶다면 [여기](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows)를 확인해 주세요.  

특정 `event`의 경우 `filter`가 필요한 경우가 있습니다. 예를 들어 push가 어떤 branch에 발생했는지에 따라 트리거하고 싶을 수도 있고 아닐 수도 있습니다. 공식 문서의 설명은 [여기](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpull_requestpull_request_targetbranchesbranches-ignore)에 있습니다.  

지금까지 설명한 내용의 예시는 아래와 같습니다.  

```yaml
name: CI

on:
  pull_request:
    branches: [main]
```
 
 
이제 `job`을 정의해 보겠습니다. 복수의 `job`을 정의할 경우 기본적으로 병렬 동작하게 됩니다. 따라서 순차적으로 실행되길 원한다면 반드시 `jobs.<job_id>.needs` 키워드를 통해 의존 관계를 정의해야 합니다. 각 `job`은 `runs-on`으로 명시된 runner environment에서 실행됩니다. environment에 대해서는 [여기](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#choosing-github-hosted-runners)를 확인하면 됩니다.  

아래 예시에서 my_first_job과 my_second_job이 `job_id`에 해당한다는 점을 알아두세요. My first job과 My second job은 `jobs.<job_id>.name` (name) 이며 github UI에 표시됩니다.  

```yaml
jobs:
  my_first_job:
    name: My first job
  my_second_job:
    name: My second job
```

`needs`를 통해 반드시 이전 `job`이 성공적으로 끝나야만 다음 `job`이 실행되도록 정의할 수 있습니다. 물론 조건을 추가해서 꼭 성공하지 않더라도 실행되도록 작성할 수도 있습니다.  

```yaml
jobs:
  job1:
  job2:
    needs: job1
  job3:
    needs: [job1, job2]
```

앞서 `job` 내에는 연속된 task로 구성된 `steps`가 존재한다고 설명했습니다. `jobs.<job_id>.steps`는 명령어를 실행하거나 setup task를 수행하거나 특정한 action을 실행할 수 있습니다.  

`steps` 아래에 `if` 조건을 추가하게 되면 반드시 이전의 조건이 만족되어야 연속적으로 실행되도록 만들 수 있습니다. context를 이용한다면 아래 예시를 보면 됩니다.  

```yaml
steps:
 - name: My first step
   if: ${{ github.event_name == 'pull_request' && github.event.action == 'unassigned' }}
   run: echo This event is a pull request that had an assignee removed.
```

status check function을 이용해 보겠습니다. My backup step이라는 task는 오직 이전 task가 실패해야만 실행될 것입니다.  

```yaml
steps:
  - name: My first step
    uses: octo-org/action-name@main
  - name: My backup step
    if: ${{ failure() }}
    uses: actions/heroku@1.0.0
```

`steps`의 구성 요소를 좀 더 살펴보겠습니다. `jobs.<job_id>.steps[*].name`은 github UI에 표시될 name을 의미합니다.  

`jobs.<job_id>.steps[*].uses`는 어떠한 action을 실행할지를 의미합니다. 이는 같은 repository나 public repository 혹은 published docker container image에서 정의될 수 있습니다.  

다양한 예시가 존재합니다. versioned action, public action, public action in a subdirectory, same repository 내의 action 및 docker hub action 등을 이용할 수 있습니다. 이에 대한 공식 문서 설명은 [여기](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstepsuses)를 확인해 주세요.  

여러 action 중 가장 많이 사용되는 action은 `checkout`인데, repository로부터 코드를 다운로드 받기 위해 사용됩니다. 이 `checkout`을 github action의 입장에서 바라보면 github의 repository에 올려 둔 코드를 CI 서버로 내려받은 후 특정 branch로 전환하는 작업으로 이해할 수 있다고 합니다. (참고 블로그 인용) 실제로 이 action을 직접 수행하려면 번거로운 작업들이 선행되어야 하지만, github action은 이를 편하게 묶어서 action으로 제공하고 있습니다.  

workflow yaml 파일에서 `steps.uses` 키워드에 사용하고자 하는 action의 위치를 {소유자}/{저장소명}@참조자 형태로 명시해야 한다고 합니다. 예시는 아래와 같습니다.  

```yaml
steps:
  - name: Checkout
  - uses: actions/checkout@v3
```

내부적으로는 git init/config/fetch/checkout/log 등의 명령어를 수행한다고 합니다. 가장 최신 버전의 checkout action에 대해서는 [이 repository](https://github.com/actions/checkout)에서 확인할 수 있습니다.  

python으로 개발을 하는 사용자라면 파이썬을 설치하는 `setup-python` 또한 자주 사용하게 될 것입니다. repo는 [여기](https://github.com/actions/setup-python)입니다. 이 action을 통해 CI 서버에 python을 설치할 수 있으며 특정 버전이 필요할 경우 아래와 같이 작성하면 됩니다.  

```yaml
steps:
    - name: Set up Python
        uses: actions/setup-python@v3
        with:
            python-version: 3.9
```



`jobs.<job_id>.steps[*].run`은 운영체제의 shell을 이용하여 command-line 프로그램을 실행시킨다. 아래와 같이 single-line command를 입력할 수도 있고,  

```yaml
- name: Install Dependencies
  run: npm install
```

multi-line command로 입력할 수도 있습니다.  
```yaml
- name: Clean install dependencies and build
  run: |
    npm ci
    npm run build
```

예시는 [여기](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstepsshell)를 참고해 주세요. 파이썬 스크립트 예시 하나는 기록해 둡니다.  

```yaml
steps:
  - name: Display the path
    run: |
      import os
      print(os.environ['PATH'])
    shell: python
```

`jobs.<job_id>.steps[*].with`는 action에 의해 정의된 input 파라미터들의 map을 의밓바니다. 각 input 파라미터는 key/valu 쌍으로 이루어져있으며, input 파라미터들은 환경 변수들의 집합에 해당합니다.  

그 외에도 `steps` 아래에는 `args`, `entrypoint`, `env` 등 여러 속성을 정의할 수 있습니다.  




## References  
- [Github Docs](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstepsshell)  
- [참고 블로그](https://blog.gyus.me/2020/introduce-poetry/)  
- [참고 블로그](https://www.daleseo.com/github-actions-checkout/)  
