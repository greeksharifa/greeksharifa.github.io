---
layout: post
title: GitHub 사용법 - 01. 소개
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---

## Git이란?

Git은 버전 관리 시스템으로, 파일의 변경 내용을 계속 추적하도록 개발된 것이다.
즉 Git은 분산 버전 관리 시스템으로, 모든 사람이 프로젝트의 현재와 과거 모두의 전체 history를 갖고 있는 것이다.

## GitHub이란?

GitHub은 Git repository를 업로드하는 웹사이트이다. 여러분이 알고 있는 그 [깃헙](https://github.com) 맞다.

Git을 사용하는 이유까지 설명하지는 않도록 하겠다.

---

## Git에서 사용되는 개념

### Repository 저장소

저장소는 당연히, 프로젝트 파일을 모아둔 곳이다. 하나의 root directory와 비슷한 개념이다.

저장소에는 크게 세 가지 종류가 있다고 생각해도 무방하다. 이 중 두 개는 거의 비슷한데, 소유자가 다를 뿐이다.

1. 나의 remote repository
2. 다른 사람의 remote repository
3. 나의 local repository

<small>4. 다른 사람의 local repository</small>

물론 4번은 여러분이 신경쓸 부분은 아니다. 따라서 세 가지만 생각하면 된다.

각 repository 사이에서 상호작용하는 과정은 다음과 같은 것들이 있다. 다른 사람이 하는 것은 생각하지 않도록 하자.

1. `다른 사람의 remote repository`(2)를 `나의 remote repository`(1)로 가져오는 것(fork)
2. `나의 remote repository`(1)을 `나의 local repository`(3)으로 가져오는 것(clone)
3. `나의 local repository`(3)의 변경사항을 `나의 remote repository`(1)에 반영하는 것(push)
4. Fork로 가져온 프로젝트인 `나의 local repository`(3)의 변경사항을 `나의 remote repository`(1)에 반영시킨 후(push),
   다른 사람의 remote repository(2)에 반영하는 것(pull request). 이를 GitHub 프로젝트에 기여했다고 한다.

### Init, Clone

프로젝트를 시작하는 과정은 다음과 같다.

1. 먼저 local에서 directory를 하나 생성한다. 이름은 프로젝트 이름으로 한다.
2. 생성한 directory에서 `git init` 명령을 입력한다.
3. 브라우저에서 git repository를 하나 생성한다.
4. 브라우저에 보이는 안내를 따르면 된다. `git remote add origin ...` 명령을 입력한다.
5. 다음 [Add, Commit, Push](###Add-Commit-Push)과정을 따르면 된다.

이미 일부 혹은 전체가 만들어져 있는 프로젝트를 local에 받아와서 하고 싶을 때가 있다. 이는 다음 과정을 따른다.

1. `git clone ...` 명령으로 remote repository를 `나의 local repository`(3)으로 받아온다.
2. 끝. 이제 개발을 시작하면 된다.

### Add, Commit, Push

여러분이 혼자서 간단한 프로젝트를 진행하게 된다면 가장 많이 쓰게 되는 명령들이다.

앞에서 말한 `repository 상호작용 과정 중 3번`을 가장 많이 하게 되는데, 이는 총 4단계로 이루어진다. 물론 경우에 따라 다른 과정이 추가될 수도 있다.

1. 파일을 프로젝트 목적에 맞게 수정하고 저장한다. 즉 개발 과정이다.
2. Add: 이제 파일을 cache에 잠시 올려 놓는다.
   이 과정이 필요한 이유는, 하나의 commit(한번에 반영할 수정사항)에 여러분이 원하는 파일만 반영할 수 있도록 하기 위함이다.
   만약에 반강제적으로 모든 파일이 반영되어야만 한다면, commit이 제 역할을 하지 못하게 될 수 있다.
3. Commit: 이제 원하는 만큼의 수정사항을 하나의 commit으로 묶는다.
4. Push: 이제 commit을 진짜로 `나의 remote repository`에 반영하는 과정이다.

Commit에는 단지 수정사항을 정리하는 것 외에 해주어야 하는 것이 두 가지 더 있다.

- Commit message: commit할 때 같이 작성한다.
  Commit message란 이 commit이 어떤 수정사항을 담고 있는지를 알려주는 것이다. 자세히 쓸수록 좋다.
- Tag: 여러분이 생각하는 그 태그 맞다. 블로그의 글에 달려 있는 태그랑 같은 기능을 한다. 포스팅 대신 commit을 참조하는 것이 다를 뿐이다.

### Branch 브랜치

새로운 기능을 개발하거나 테스트를 할 때 사용하는 독립적인 commit history이다. 나무에서 메인 줄기가 아닌 옆으로 빠져나온 나뭇가지를 생각하면 된다.

Branch가 필요한 이유는 무엇인가? 혼자서 간단한 것을 할 때라면 사실 branch를 새로 만들 것도 없이 그냥 진행해도 별 문제는 없다.
하지만 프로젝트의 규모가 커지거나 여러 사람이 협업해야 한다면 branch는 필수이다. 모두가 master branch(메인 줄기)를 직접 수정하려고 들면 큰일난다.

모든 Git 프로젝트는 기본적으로 `master` branch를 갖는다. `master` branch가 나무의 메인 줄기로서, 검증이 끝난 프로젝트의 결과물이라 할 수 있다.

branch 간 상호작용은 꽤 종류가 많지만, 여기서는 몇 가지만 간략히 소개한다.

1. branch에서 새 branch를 생성하는 것(`git branch ...`)
2. 어떤 branch에서 다른 특정 branch로 옮겨가는 것(checkout)
3. 검증이 끝난 branch를, 그 branch를 생성한 주 branch에 합치는 것(merge)

새로운 기능을 개발하여 추가하는 과정은 대개 위의 세 과정을 따른다. 물론 2번과 3번 사이에 개발 과정이 있을 것이다.

이렇게 새로운 기능을 개발하는 branch를 feature(topic) branch라 부른다.
또 구버전 소프트웨어 지원 등의 이유로 별도의 branch가 필요한 경우 이를 release branch라 부른다.

### Issue

- 기능에 대해 논의하거나
- 버그 수정사항을 알리거나
- todoList로 활용한다.

협업할 때는 당연히 필요하고, 혼자 할 때도 bugList와 todoList로 쓰면 유용하다.


---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
