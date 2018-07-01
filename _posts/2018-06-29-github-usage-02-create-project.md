---
layout: post
title: GitHub 사용법 - 02. 프로젝트와 repository 생성
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---

프로젝트는 다음과 같은 과정을 거치면 만들 수 있다.

- [Remote Repository 생성](#rreate-repository-생성)
- [Local Repository 생성](#local-repository-생성)
- [프로젝트 수정](#프로젝트-수정)
  - [파일 생성하고 수정하기](#파일-생성하고-수정하기)
  - [git add](#git-add)
  - [git commit](#git-commit)
- [Remote & Local Repository 연결](#remote--local-repository-연결)
- [](#)

***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 당부한다. 눈으로만 보면 100% 잊어버린다.***

## Remote Repository 생성

우선 <https://github.com/>에 접속하여 로그인한다(회원가입은 되어 있어야 한다). 그러면 다음과 비슷한 화면이 보인다.

![01_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/01_new_repository.PNG)

**New repository**를 클릭한 후, 프로젝트 이름을 **git_tutorial**으로 입력한다. 원하는 이름으로 해도 상관없다.
또 **Description**을 성심성의껏 잘 작성한다.

그리고 **Initialize this repository with a README** 체크박스에 체크한다.
체크하지 않고 만든다면 git repository를 처음 만든 이후 local repository와 연결하는 방법을 알려주는 안내 글이 뜬다. 이를 읽어 봐도 괜찮다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/02_create_a_new_repository.PNG)

마지막으로 **Create repository**를 누른다.

그러면 이제 여러분의 GitHub 계정에 **git_tutorial**이란 이름의 remote repository가 생성된 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/03_created_remote_repository.PNG)



## Local Repository 생성

그리고 이제 local에서 directory를 하나 생성한다. Directory 이름은 프로젝트와 같은 이름으로 하고, 생성하는 위치는 본인 마음대로 하면 된다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/04_local_directory.PNG)

다음으로 명령창(터미널 또는 cmd 창)을 하나 띄운다. `cd '경로명'/git_tutorial`으로 하면 된다. 예시는 절대 경로이지만, 상대 경로로 해도 무방하다.
필자는 윈도우 환경(cmd)에서 진행하였다.

> 예) `cd C:\Users\Sharifa-D\WebstormProjects\git_tutorial`

그리고 다음 명령들을 수행해 본다. 만약에 git이 유효한 명령이 아니라는 error가 뜬다면, 본인의 운영체제이 맞는 git을 설치해야 한다.
윈도우의 경우 [여기](https://git-scm.com/download/win)에서 다운받아 설치하면 된다(64bit 기준).

```
git init
git status
```

똑같은 과정을 거쳤다면, 다음 그림과 같은 화면이 나올 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/05_cmd.PNG)

여기까지 했다면, local repository를 성공적으로 생성한 것이다.
오류가 뜬다면, 복잡해 보이는 에러 메시지를 그대로 복붙하여 [구글링](https://google.com/)하면 된다.
여러분이 겪는 문제는 다른 사람들도 겪어 본 문제라는 것을 해결법과 함께 알 수 있다.



## 프로젝트 수정

### 파일 생성하고 수정하기

이제 여러분은 1) remote repository와 2) local repository를 모두 생성했다. 그러면 이제 두 repo(repository)를 연결하는 부분이 필요할 것이다.

연결은 어렵지 않다. 하지만 그 전에 파일을 조금 작성해보자. 빈 프로젝트 갖고 뭘 하기엔 심심하지 않은가?

`first.py`란 파일을 하나 생성한다. 그리고 다음과 같은 내용을 작성해보자.
```python
print("Hello, git!") # instead of "Hello, World!"
```

심심하다면 `python first.py`를 명령창에 입력해 보자.

그리고 `git status`를 다시 입력한다. 조금 전과는 다른 화면을 볼 수 있다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/06_python_first.py.PNG)

여기서 중요한 정보를 몇 개 찾을 수 있다. 앞으로 git을 사용할 때에는 무슨 메시지가 뜨는지 (전부) 살펴보는 것이 굉장히 중요하다.
뭔가 잘못되었는지 아닌지를 볼 수 있기 때문이다.

위에서부터 하나씩 살펴보면 아래 표와 같다.

Message | Descrption
-------- |--------
On branch master | 지금 작업하는 [branch](https://greeksharifa.github.io/github/2018/06/29/github-usage-01-introduction/#branch-%EB%B8%8C%EB%9E%9C%EC%B9%98)가 master라는 의미이다. 본인이 어떤 branch에서 작업 중인지 확인하는 습관을 반드시 가지도록 한다. branch를 잘못 옮긴 줄 모르고 작업을 이어갔다가는 큰일 날 수 있다.
No commits yet   | 아직 생성한 [commit](https://greeksharifa.github.io/github/2018/06/29/github-usage-01-introduction/#add-commit-push)이 없다는 뜻이다. 이후에 commit을 추가하면, 이 부분이 다르게 보일 것이다.
Untracked files:  (use "git add <file>..." to include in what will be committed) | 여러분이 수정하긴 했지만 cache에 올라가지 않은 파일의 목록이다. cache에 올라갔다는 말은 track한다는 말과 같다.
first.py         | 여러분은 아직 `git add` 명령을 사용하지 않았기 때문에 수정/생성/삭제한 유일한 파일인 `first.py`가 tracking되지 않고 있다.
nothing added to commit but untracked files present (use "git add" to track) | tracking하려면 `git add`를 쓰라고 한다. 메시지에는 도움이 되는 내용이 많다.

조금 더 자세히 설명하기 위해, `second.py` 파일을 생성한다.
```python
print("Why don't you answer me, git?")
```

### git add

그리고 조금 전 메시지가 친절히 알려줬던 **git add** 명령을 사용하려고 한다. 명령창에 다음과 같이 입력한다.
```
git add first.py
```

이제 다시 한번 `git status`를 입력하면, 아까보다 메시지가 더 많은 것을 확인할 수 있다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/07_git_status.PNG)

그림을 보면 tracking되고 있는 파일은 초록색, untracked file은 빨간색으로 되어 있음을 알 수 있다.
여러분은 `first.py` 는 `git add`로 추가했기 때문에 초록색으로, `second.py`는 그러지 않았기 때문에 빨간색으로 남아 있음을 확인할 수 있다.

이번에는 다른 명령을 연습해보자. `git add .`을 입력한다. `git status`로 확인해보면?

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/11_git_add_all.PNG)

`.`의 의미는 **모든 파일과 디렉토리**이다.
즉, 여러분은 프로젝트에 존재하는 모든 파일(`first.py`와 `second.py`)를 cache에 추가한 것이다.

옵션으로, `git add`의 다양한 버전을 표로 정리해 두었다.



### 옵션: 같은 파일이 *changes to be committed*와 *Untracked files* 모두에 있는 경우

***이 부분은 옵션이다. git이 아직 잘 이해가 되지 않는다면, 반드시 할 필요는 없다. [여기](#)로 바로 넘어가면 된다.***

`first.py` 파일을 다음과 같이 수정한다.
```python
print("Hello, git!") # instead of "Hello, World!"
print("Don't you hear me, git?")
```

그리고 *다른 명령을 하지 않은 채로* `git status`를 명령창에 입력한다. 그럼 다음과 같을 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/08_both_exists.PNG)

여러분이 한 것을 되짚어 보면 다음과 같다.

1. `first.py`를 생성 및 수정하였다.
2. `first.py`를 `git add` 명령으로 cache에 추가하였다.
3. `first.py`를 또 수정하였다.
4. 다른 명령(`git add`나 `git commit` 등)을 하지 않는 채로 `git status`로 상태를 확인하였다.

이런 과정을 거쳤을 때 여러분은 동일한 파일이 *changes to be committed*와 *Untracked files*에 모두 있는 광경을 볼 수 있는 것이다.

즉, 이는 오류가 아니라,

- 이미 `git add`로 추가한 적이 있으니 *changes to be committed*에 있는 것이고
- 그 이후에 수정한 사항은 cache에 올라가지 않았으니 *Untracked files*에도 있는 것이다.

어렵지 않게 이해할 수 있을 것이다.


### 옵션: `git add 취소`, `git rm --cached <file>`

여러분이 메시지를 꼼꼼히 읽어봤다면, 다음과 같은 문구를 보았을 것이다.

```
(use "git rm --cached <file>..." to unstage)
```

이는 cache에 올라간 파일을 unstage하겠다는 뜻으로, git add를 취소하는 것과 같은 효과를 가진다.
즉 *cached*된 `<file>`을 (cache에서) rm(remove)하겠다는 의미이다.

무슨 일을 하는지 알았으니, `git rm --cached first.py`를 명령창에 입력한다. 그리고 `git status`를 쳐보자.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/09_rm_cached_error.PNG)

에러가 뜬다. 메시지를 의역하면, `first.py`가 실제 파일 내용이랑 git이 인식하는 파일 내용이 달라서 cache에서 제거할 수 없다는 뜻이다.

어차피 여러분은 이 파일을 unstage하는 것이 목적이었으므로, `git add first.py`이후 `git rm --cached first.py`를 입력해주면 그만이다.
`git status`로 상태를 확인해주자.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/10_rm_cached.PNG)

이제 **옵션**을 안 한 상태로 되돌리기 위해, `first.py`에 추가한 내용을 지우고 `git add .`를 입력한다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/11_git_add_all.PNG)

### git commit



## Remote & Local Repository 연결

---

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/.PNG)


## Git 명령어

다음 글에서 원하는 기능을 찾아 볼 수 있다. [GitHub 명령 List](https://google.com/)
