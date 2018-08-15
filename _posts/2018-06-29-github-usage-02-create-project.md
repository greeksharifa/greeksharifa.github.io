---
layout: post
title: GitHub 사용법 - 02. 프로젝트와 repository 생성
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

---

## Remote Repository 생성

우선 <https://github.com/>에 접속하여 로그인한다(회원가입은 되어 있어야 한다). 그러면 다음과 비슷한 화면이 보인다.

![01_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/01_new_repository.PNG)

**New repository**를 클릭한 후, 프로젝트 이름을 **git_tutorial**으로 입력한다. 원하는 이름으로 해도 상관없다.  
또 **Description**을 성심성의껏 잘 작성한다.

그리고 **Initialize this repository with a README** 체크박스에 체크한다.  
체크하지 않고 만든다면 git repository를 처음 만든 이후 local repository와 연결하는 방법을 알려주는 안내 글이 뜬다. 이를 읽어 봐도 괜찮다.

지금은 체크하지 않는 것이 간단하므로 체크하지 않겠다. 그림에 체크되어 있는 것은 무시하자.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/02_create_a_new_repository.PNG)

마지막으로 **Create repository**를 누른다.

그러면 이제 여러분의 GitHub 계정에 **git_tutorial**이란 이름의 remote repository가 생성될 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/03_created_remote_repository.PNG)



---

## Local Repository 생성

그리고 이제 local에서 directory를 하나 생성한다. Directory 이름은 프로젝트와 같은 이름으로 하고, 생성하는 위치는 본인 마음대로 하면 된다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/04_local_directory.PNG)

다음으로 명령창(터미널 또는 cmd 창)을 하나 띄운다. `cd '경로명'/git_tutorial`으로 하면 된다. 예시는 절대 경로이지만, 상대 경로로 해도 무방하다.  
필자는 윈도우 환경(cmd)에서 진행하였다.

> cd C:\Users\Sharifa-D\WebstormProjects\git_tutorial

그리고 다음 명령들을 수행해 본다. 만약에 git이 유효한 명령이 아니라는 error가 뜬다면, 본인의 운영체제이 맞는 git을 설치해야 한다.  
윈도우의 경우 [여기](https://git-scm.com/download/win)에서 다운받아 설치하면 된다(64bit 기준).

> git init  
> git status

똑같은 과정을 거쳤다면, 다음 그림과 같은 화면이 나올 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/05_cmd.PNG)

여기까지 했다면, local repository를 성공적으로 생성한 것이다.
오류가 뜬다면, 복잡해 보이는 에러 메시지를 그대로 복붙하여 [구글링](https://google.com/)하면 된다.  
여러분이 겪는 문제는 다른 사람들도 겪어 본 문제라는 것을 해결법과 함께 알 수 있다.


---


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

위에서부터 하나씩 살펴보면 아래와 같다.

- On branch master
  - 지금 작업하는 [branch](https://greeksharifa.github.io/github/2018/06/29/github-usage-01-introduction/#branch-%EB%B8%8C%EB%9E%9C%EC%B9%98)가 master라는 의미이다. 본인이 어떤 branch에서 작업 중인지 확인하는 습관을 반드시 가지도록 한다. branch를 잘못 옮긴 줄 모르고 작업을 이어갔다가는 큰일 날 수 있다.
- No commits yet
  - 아직 생성한 [commit](https://greeksharifa.github.io/github/2018/06/29/github-usage-01-introduction/#add-commit-push)이 없다는 뜻이다. 이후에 commit을 추가하면, 이 부분이 다르게 보일 것이다.
- Untracked files:  (use "git add <file>..." to include in what will be committed)
  - 여러분이 수정하긴 했지만 staging area에 올라가지 않은 파일의 목록이다. staging area에 올라갔다는 말은 track한다는 말과 같다.
- first.py
  - 여러분은 아직 `git add` 명령을 사용하지 않았기 때문에 수정/생성/삭제한 유일한 파일인 `first.py`가 tracking되지 않고 있다.
- nothing added to commit but untracked files present (use "git add" to track)
  - tracking하려면 `git add`를 쓰라고 한다. 메시지에는 도움이 되는 내용이 많다.

조금 더 자세히 설명하기 위해, `second.py` 파일을 생성한다.
```python
print("Why don't you answer me, git?")
```

### git add

그리고 조금 전 메시지가 친절히 알려줬던 **git add** 명령을 사용하려고 한다. 명령창에 다음과 같이 입력한다.

> git add first.py

이제 다시 한번 `git status`를 입력하면, 아까보다 메시지가 더 많은 것을 확인할 수 있다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/07_git_status.PNG)

그림을 보면 tracking되고 있는 파일은 초록색, untracked file은 빨간색으로 되어 있음을 알 수 있다.
여러분은 `first.py` 는 `git add`로 추가했기 때문에 초록색으로, `second.py`는 그러지 않았기 때문에 빨간색으로 남아 있음을 확인할 수 있다.

이번에는 다른 명령을 연습해보자. `git add .`을 입력한다. `git status`로 확인해보면?

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/11_git_add_all.PNG)

`.`의 의미는 **모든 파일과 디렉토리**이다.
즉, 여러분은 프로젝트에 존재하는 모든 파일(`first.py`와 `second.py`)를 staging area에 추가한 것이다.

옵션으로, `git add`의 다양한 버전을 표로 정리해 두었다.

| 명령어 | Description
| -------- | --------
| git add first.py | first.py 파일 하나를 staging area에 추가한다.
| git add my_directory/  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  | my_directory라는 이름의 디렉토리와 그 디렉토리 안의 모든 파일과 디렉토리를 staging area에 추가한다.
| git add . | 현재 폴더의 모든 파일과 디렉토리, 하위 디렉토리에 든 전부를 staging area에 추가한다. 규모가 큰 프로젝트라면 써서는 안 된다.
| git add -p [\<파일\>] | 파일의 일부를 staging하기
| git add -i | Git 대화 모드를 사용하여 파일 추가하기
| git add -u [\<경로\>] | 수정되고 추적되는 파일의 변경 사항 staging하기 

### 옵션: 같은 파일이 *changes to be committed*와 *Untracked files* 모두에 있는 경우

***이 부분은 옵션이다. git이 아직 잘 이해가 되지 않는다면, 반드시 할 필요는 없다. 
[여기](https://greeksharifa.github.io/github/2018/06/29/github-usage-02-create-project/#git-commit)로 바로 넘어가면 된다.***

`first.py` 파일을 다음과 같이 수정한다.
```python
print("Hello, git!") # instead of "Hello, World!"
print("Don't you hear me, git?")
```

그리고 *다른 명령을 하지 않은 채로* `git status`를 명령창에 입력한다. 그럼 다음과 같을 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/08_both_exists.PNG)

여러분이 한 것을 되짚어 보면 다음과 같다.

1. `first.py`를 생성 및 수정하였다.
2. `first.py`를 `git add` 명령으로 staging area에 추가하였다.
3. `first.py`를 또 수정하였다.
4. 다른 명령(`git add`나 `git commit` 등)을 하지 않는 채로 `git status`로 상태를 확인하였다.

이런 과정을 거쳤을 때 여러분은 동일한 파일이 *changes to be committed*와 *Untracked files*에 모두 있는 광경을 볼 수 있는 것이다.

즉, 이는 오류가 아니라,

- 이미 `git add`로 추가한 적이 있으니 *changes to be committed*에 있는 것이고
- 그 이후에 수정한 사항은 staging area에 올라가지 않았으니 *Untracked files*에도 있는 것이다.

어렵지 않게 이해할 수 있을 것이다.


### 옵션: git add 취소, git rm --cached \<file\>

여러분이 메시지를 꼼꼼히 읽어봤다면, 다음과 같은 문구를 보았을 것이다.

> (use "git rm --cached <file>..." to unstage)

이는 staging area에 올라간 파일을 unstage하겠다는 뜻으로, git add를 취소하는 것과 같은 효과를 가진다.
즉 *cached*된 `<file>`을 (staging area에서) rm(remove)하겠다는 의미이다.

무슨 일을 하는지 알았으니, `git rm --cached first.py`를 명령창에 입력한다. 그리고 `git status`를 쳐보자.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/09_rm_cached_error.PNG)

에러가 뜬다. 메시지를 의역하면, `first.py`가 실제 파일 내용이랑 git이 인식하는 파일 내용이 달라서 staging area에서 제거할 수 없다는 뜻이다.

어차피 여러분은 이 파일을 unstage하는 것이 목적이었으므로, `git add first.py`이후 `git rm --cached first.py`를 입력해주면 그만이다.
`git status`로 상태를 확인해주자.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/10_rm_cached.PNG)

이제 **옵션**을 안 한 상태로 되돌리기 위해, `first.py`에 추가한 내용을 지우고 `git add .`를 입력한다.


### git commit

현재 다음과 같은 상태일 것이다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/11_git_add_all.PNG)

이제 commit을 할 차례이다. 커밋이란 수정사항들을 하나로 묶는 것이라 보면 된다.   
실제 프로젝트에서 하나의 커밋이란 하나의 기능이라 보면 된다. 하니의 기능이란 새 기능일 수도 있고, 버그 수정일 수도 있고, 단순 개선 사항일 수 있다.

커밋은 여러 종류가 있지만, 가장 간단한 버전은 다음과 같다.  
***주의: 아직 명령창에 적지 않는다.*** 

> git commit -m "commit-message"

명령어를 입력할 때 `-`에 알파벳을 붙여 쓰는 경우가 있다. 이는 옵션을 주겠다는 의미이다.  
`-m` 옵션을 주면서 `"`로 묶은 메시지를 전달하면, **commit-message라는 description으로 커밋을 하나 만든다**라는 의미가 된다.

#### 옵션: 좋은 commit message 작성법

왜 명령창에 적지 말라고 했냐면, 이 방식으로 하는 것은 큰 프로젝트를 다룰 때 굉장히 안 좋은 습관이다. 
고작 한 문장짜리로 커밋의 모든 내용을 설명할 수 있겠는가? 만약에 커밋 내용이 다음과 같은 일을 한다고 하자.

> 이 커밋은 #101번과 #104번 이슈를 해결하기 위해 작성된 것이다. 이 문제에 관한 자세한 내용은 #203번과 #223번을 참조하라.  
> 해당 문제를 해결하기 위해 다음과 같은 방법을 사용하였다. 블라블라  
> 문제는 해결되었지만, 아직 다음과 같은 (사소한) 문제가 남아 있다. #401번을 참조하라.

한 문장에 실수 없이 적을 수 있겠는가?  
자신이 있다면 말리진 않겠지만, 별로 좋은 습관이 아니란 것은 명확하다.

좋은 commit message 작성법에 관한 내용은 [여기](https://item4.github.io/2016-11-01/How-to-Write-a-Git-Commit-Message/)를 참조하라.

#### 다시 commit하기

아무튼 다음과 같이 입력한다.

> git commit

그럼 뭔가 화려한 편집 창이 보인다. vi 편집기라고 보면 된다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/12_git_commit.PNG)

**i**를 누른다. insert를 한다는 뜻이다.  
그리고 commit message를 최대한 상세히 입력한다.

다 입력했으면, **ESC**를 누른다. 그리고, **:wq**를 입력한 후 **Enter**를 누른다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/14_git_commit_complete.PNG)

vi 편집기에서는, 입력 모드와 명령 모드가 있다.  
입력 모드는 일반적인 텍스트 편집기와 같다.  
명령 모드는 옵션을 주고 여러 조작을 할 수 있다. 자세한 설명은 [Vim 사용법](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)를 참조하면 된다. 
여기서는 **w**는 저장이고 **q**는 quit을 의미한다는 것만 알아도 된다.

이제 commit에 대한 간단한 설명이 끝났다.


---


## Remote & Local Repository 연결

이제 연결을 할 차례이다. 근데 할 것이 한 가지 더 남았다. 사용자 등록 과정이다.

등록 과정은 두 가지 방법이 있다. 전역 설정을 하느냐, repo 별로 설정을 하느냐이다.

- 전역 설정
  - `git config --global user.name "Your name"`
    - 여러분의 github 아이디를 적으면 된다.
  - `git config --global user.email "Your email"`
    - 여러분의 github 이메일 계정을 적으면 된다.
- repo별 설정
  - `git config user.name "Your name"`
    - --global 옵션이 없다. 여러분의 github 아이디를 적으면 된다.
  - `git config  user.email "Your email"`
    - 역시 --global 옵션이 없다. 여러분의 github 이메일 계정을 적으면 된다.
    
전역 설정과 repo별 설정의 차이를 굳이 설명할 필요는 없을 것이다.

일단은 global 설정부터 시작하자. Your name/email은 여러분 스스로 입력하길 바란다.

> git config --global user.name "Your name"
> git config --global user.email "Your email"

추가로 해야만 하는 것은 없지만, 등록된 사용자를 확인하는 방법도 알아야 하지 않겠는가?

| 명령어 | Description
| -------- | --------
| git config --global --list | 전역 설정 정보 조회
| git config --list | repo별 설정 정보 조회

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/15_git_config.PNG)

이제 진짜로 연결하는 과정이다. 딱 한 문장으로 끝난다.

> git remote add \<remote repo 이름\> \<repo url\>  
> ex) git remote add origin https://github.com/greeksharifa/git_tutorial.git

url은 여기서 확인할 수 있다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/17_origin.PNG)

일반적으로 remote repo의 이름은 origin으로 둔다.

## 옵션: git 설정하기

다음 명령은 Git의 출력결과 색상을 활성화하는 명령이다.

> git config \-\-global color.ui "auto"

앞으로 git push를 할 때마다 git 비밀번호를 입력해야 하는데, 이것이 매우 귀찮다면 비밀번호를 저장해 둘 수 있다.

> git config \-\-global credential.helper store

단 보안상의 이슈가 걱정된다면, store 대신 cache 옵션을 주어 15분간 저장되게 할 수 있다. 15분 대신 다른 시간을 지정하려면 다음과 갈이 한다.

> git config \-\-global credential.helper cache --timeout 86400



## git pull, git push

이제 정말로 연결한 remote repo에 local repo의 수정사항을 올릴 때가 되었다. push 명령은 어렵지 않다.

> git push origin master

한동안 origin이라는 이름만 쓰고 **master** branch만 이용할 계획이라면, 위의 명령에 `-u` 옵션을 붙인다. 
즉, `git push -u origin master`라고 입력하면 된다.   
그러면 앞으로 `git push`만 입력해도 origin master에 push가 이루어진다.

그런데 이때 그냥 push를 하면 error가 뜰 수 있다. 이는 remote repo를 만들 때 README.md 파일을 생성했기 때문이다.   
[GitHub Instruction](https://greeksharifa.github.io/github/2018/06/29/github-usage-01-introduction/)에서 간략히 설명한 대로, 
`git pull origin master`(혹은 그냥 `git pull`)으로 remote repo의 변경사항을 local repo로 받아온다.

그리고 현재의 master branch와 remote repo의 branch를 연결하기 위해, 다음과 같이 입력한다.  
여기서 upstream branch란 remote repo의 branch를 가리키는 말이라 봐도 좋다.

> git push --set-upstream origin master

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/18_git_push.PNG)

이제 브라우저에서 git_tutorial repo를 확인해 본다.

![02_create_a_new_repository](/public/img/GitHub/2018_06_29_github_usage_02_create_project/19_complete.PNG)

끝났다!


---

[다음 글](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/)에서는 프로젝트 clone, status, .gitignore에 대해서 알아본다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
