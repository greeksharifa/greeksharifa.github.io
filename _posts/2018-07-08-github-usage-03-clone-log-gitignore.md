---
layout: post
title: GitHub 사용법 - 03. 프로젝트 clone, status check, .gitignore
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/06/29/github-usage-02-create-project/)에서 작업하던 것을 이어서 한다. 저번 글에서는 git_tutorial 디렉토리를 생성하는 것까지 했었다.

---

## Local Directory 생성

이제 git_tutorial 옆에 새로운 디렉토리를 생성한다. 
이름은 자유지만 필자는 `git_tutorial_clone`으로 정했다.

이 상황은, git으로 협력할 때 다른 사람이 프로젝트 repo를 clone하거나, 
여러분이 다른 컴퓨터로 이동하여 작업을 하고 싶을 때를 가정한 것이다.  
그러나 다른 사람을 데려오거나 컴퓨터를 하나 더 사는 것은 부담이 될 테니, 다른 디렉토리에 새로 repo를 만든다.  
한 가지 주의사항으로, 같은 디렉토리 내에서는 할 수 없다. 그래서 새 디렉토리를 만드는 것이다.

![01_git_tutorial_clone](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/01_git_tutorial_clone.PNG)

## git clone 하기

이제 명령창을 켜서, 방금 생성한 디렉토리로 이동한다. 명령어를 잊어버렸을까봐 해서 다시 적어 둔다. 물론 이것은 예시일 뿐이다.

> cd C:\Users\Sharifa-D\WebstormProjects\git_tutorial_clone

그리고 이제 clone을 할 것이다. clone 명령은 간단하다. `git clone \<remote repo 주소\>` 형식이다.  
저번에 생성했던 `git_tutorial` git 주소를 사용하면 된다.

> git clone https://github.com/greeksharifa/git_tutorial.git

![02_git_clone](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/02_git_clone.PNG)

클론은 이게 끝이다. 위 명령을 실행하면 git_tutorial 폴더가 또 만들어질 것이다.

이제 여러분의 디렉토리 구조는 다음과 같을 것이다.

- git_tutorial/
  - .git/
  - first.py
  - second.py
- git_tutorial_clone/
  - git_tutorial/
    - .git/
    - first.py
    - second.py
  
왜 .git/이 있지? 라고 생각할 수 있다.  
.git/ 디렉토리는 윈도우라면 숨김 처리되어 있는 폴더인데, 이 .git/ 디렉토리가 있는 디렉토리만이 local repo로 인정받는다.  
즉, 이 .git/이 있어야만 git repo로서의 역할을 할 수 있는 것이다.  
그리고 git init으로 생성하거나 git clone으로 remote repo를 local repo로 가져온 경우에도 자동으로 .git/ 디렉토리가 존재하게 된다.

그리고 위의 두 개의 first.py와 second.py 각각은 완전히 동일한 파일일 것이다.

이제 두 개의 local repo와 remote repo 사이에 상호작용을 좀 시켜보자.  
[저번 글](https://greeksharifa.github.io/github/2018/06/29/github-usage-02-create-project/)에서 했던
`git status`, `git add`, `git commit`, `git push`, `git pull` 등을 사용해 볼 것이다.

---

## 프로젝트 파일 수정하고 3종 세트 입력하기

아무거나 수정해보자. `git_tutorial_clone/git_tutorial/first.py`를 수정한다. 다음과 같은 내용으로 하자.

> print("Hello, git!") # instead of "Hello, World!"  
> print("Hi, git!!")

그리고 명령창에서 `git status`를 실행한다. 무슨 일을 하는지 잊어버리지는 않았을 것이다.

![03_git_status](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/03_git_status.PNG)

이제 `git add .`와 `git commit -m "Edit first.py"`와 `git push origin master` 3종 세트를 입력하자.

![04_3set](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/04_3set.PNG)

---

## 옵션: 3종 세트 간편 입력(윈도우 기준)

우선 수정사항을 만들기 위해 `git_tutorial_clone/git_tutorial/first.py` 끝에 빈 줄을 하나 추가하자.

3종 세트를 한 번에 입력하는 것은 배치 파일을 만들면 쉽게 할 수 있다. 
여러분의 git local repo 안에 `push.bat`이란 이름의 파일을 하나 만들자.  
명령창에서 `copy con push.bat`이라 입력한 후 입력을 시작해도 된다.  
파일 내용은 다음과 같이 한다.

```
git add .
git status

set str=
set /p str=enter commit message :

git commit -m "%str%"
git push
```

그리고 명령창에서 `push.bat`이라고 입력하여 방금 만든 배치 파일을 실행시켜보자.  
그러면 enter commit message: 앞에서 멈춰 있을 것이다. 원하는 커밋 메시지를(enter를 입력하지 않고) 짧게 입력해보자.  
그러면 3종 세트가 하는 모든 것이 완료된다.

![05_push_bat](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/05_push_bat.PNG)

---

## local repo 상태 확인하고 git pull로 local repo 업데이트하기

이제 명령창에서 다음 명령을 입력한다. `git_tutorial` 디렉토리로 이동하기 위함이다(`git_tutorial_clone` 내부의 것이 아니다).
 
> cd ../../git_tutorial/

여기서 `first.py`를 확인해 보면, `print("Hi, git!!")` 문장이 없는 것을 확인할 수 있다.  
즉, local repo는 자동으로 업데이트되지 않는다.

이제 새로운 명령어를 몇 개 배워 볼 것이다. 명령창에 다음을 입력한다.

> git log

이 명령은 현재 local repo의 commit history를 보여준다. 아마 다음과 같을 것이다.

![06_git_log](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/06_git_log.PNG)

이제 local 말고 remote repo가 궁금할 것이다(`git_tutorial_clone`에서 커밋을 한두 개 날렸으니까).  
그럴 때는 다음 명령을 사용한다.

> git log HEAD..origin/master

![07_git_log](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/07_git_log.PNG)

- 우선 `git log` 명령은 commit log를 보여준다.
- `..`은 Double Dot으로, 한쪽에는 있고 다른 쪽에는 없는 커밋이 무엇인지 Git에게 물어보는 것이다.
- 한쪽은 `HEAD`이다. 다른 한쪽은 `origin/master`이다.
- HEAD는, 현 local repo의 현재 상태를 의미한다. HEAD에 대한 자세한 설명은 나중에 다룰 것이다.
- 이 명령은 `HEAD`에는 없고 `origin/master`에는 있는 커밋이 무엇인지 보여준다.
  - 즉 local repo에는 없고 remote repo에는 있는 커밋을 볼 수 있다.
- HEAD는 생략할 수 있다. 즉, `git log ..origin/master`로도 같은 효과를 얻는다.
- 순서를 바꾸면 remote repo에는 없고 HEAD에는 있는 커밋을 보여주게 된다. 현 시점에서는 아무것도 표시되지 않을 것이다. 
- `git log` 명령을 쓸 때 간략하게 보고 싶으면 `--oneline` 옵션을 추가한다.

> git log ..origin/master --oneline
 
이렇게 확인하고 나면 local repo와 remote repo가 어떤 status를 갖고 있는지 확인할 수 있다. 즉, 

- local repo status
  - 97f92d3 (HEAD -> master) Initial commit for git_tutorial
- remote repo status
  - 9401817 (origin/master) 3set simple commit
  - f569352 Edit first.py
  - Initial commit for git_tutorial

이 부분까지 자세히 알 필요는 없지만, 그래도 있으니 간략한 설명을 적어둔다. 

- 앞의 16진수 숫자는 커밋의 고유 번호이다(해시값). 유일한 값이므로, 혹시 커밋 메시지를 너무 간결히 작성해서 커밋이 비슷해 경우 이 해시값으로 구분할 수 있다.
- (HEAD -> master)은 HEAD(현재 local repo 상태)에서 master(remote repo의 master 브랜치)로 push했다는 것을 의미한다.
- (origin/master)는 remote repo의 현재 상태를 의미한다.


그러고 `git status`로 상태를 한번 확인해 본다. 깨끗하다는 메시지를 볼 수 있을 것이다.

이제 `git pull origin master`(혹은 `git pull`)을 입력할 차례다. 

![08_git_pull](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/08_git_pull.PNG)

이제 local repo가 최신으로 업데이트되었다.

---

## .gitignore 추가

이제 `.gitignore` 파일을 추가해 보자.  
`.gitignore` 파일은 remote repo에 올리지 않을 파일이나 디렉토리를 지정하는 파일이다.  
즉, 여러분이 `git add *`를 아무리 시도해도, `.gitignore`파일에 명시된 파일 혹은 디렉토리는 절대 staging area에 올라가지 않는다.

올라가지 않겠지만 파일을 하나 추가하자. `third.py`로 하면 좋을 것 같다.

```python
# third.py
print('This file is useless!')
```

다음으로는 `.gitignore` 파일은 만들 차례다.  
윈도우에선 `.`으로 시작하는 파일을 그냥은 만들어 주지 않기 때문에, 명령창에서 조금 전처럼 `copy con .gitignore`이라고 입력한다.  
그러면 빈 줄에서 커서가 깜빡거리는데, 이때 다음을 입력한다. 

> third.py

엔터 한번 쳐 준 다음에, `Ctrl + C`를 누른다. 그럼 파일 입력을 종료하고 다시 터미널로 빠져나온다.

![09_create_gitignore](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/09_create_gitignore.PNG)

이제 `git status`를 입력해보자. 

![10_git_status](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/10_git_status.PNG)

상태창에 `third.py`는 없고 `.gitignore` 파일만 있다. 
- `.gitignore` 파일이야 방금 추가했으니 목록에 뜨는 것이 맞다.
- `third.py`는 `.gitignore` 파일에 지정되어 staging area에 올라갈 수 없다. 즉, `git add` 명령을 써도 staging area에 올라가지 않는다.

이제 3종 세트를 입력하거나, 옵션에서 했던 `push.bat` 파일을 실행시킨다. 여기서는 3종 세트를 입력하겠다.

![11_3set](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/11_3set.PNG)

이제 브라우저에서 remote repo의 상태를 확인해 보자. 

![12_remote_repo](/public/img/GitHub/2018_07_08_github_usage_03_clone_log_gitignore/12_remote_repo.PNG)

`third.py`는 존재하지 않는다. `.gitignore` 파일이 `third.py`를 훌륭하게 무시해 주었다.

`.gitignore`파일은 이럴 때 많이 쓴다.

- (용량이 큰) 데이터 파일. git repo는 기본적으로는 아주 큰 파일은 repo에 올려주지 않는다. 총 git repo 용량 제한도 있다(기업의 경우는 잘 모르겠다). 따라서 데이터 파일은 제외하는 경우가 많다.
- 프로젝트 설정 파일. 열려 있는 파일이라 오류가 날 수도 있고, IDE가 수시로 바꾸는 경우가 많으므로 제외할 때가 많다.
- dependency 파일. 누구나 웹에서 다운받아 설치할 수 있는 용량 큰 파일을 굳이 git repo에 넣지 않는다.
  - 대신 따로 dependency 목록을 만들어 관리한다. 

이제 git의 프로젝트에 대한 설명은 대략 다 끝났다. [다음 글](https://greeksharifa.github.io/github/2018/08/07/github-usage-04-branch-basic/)에서는 branch에 대해서 알아본다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
