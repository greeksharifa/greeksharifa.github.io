---
layout: post
title: GitHub 사용법 - 09. Overall
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---

[저번 글](https://greeksharifa.github.io/github/2018/08/19/github-usage-08-conflict/)에서는 Conflict에 대해서 알아보았다.  
이번 글에서는, 전체 Git 명령어들의 사용법을 살펴본다.

---

명령어에 일반적으로 적용되는 규칙:

- 이 글에서 `<blabla>`와 같은 token은 여러분이 알아서 적절한 텍스트로 대체하면 된다.
- 각 명령에는 여러 종류의 옵션이 있다. ex) `git log`의 경우 `--oneline`, `-<number>`, `-p` 등의 옵션이 있다.
- 각 옵션은 많은 경우 축약형이 존재한다. 일반형은 `-`가 2개 있으며, 축약형은 `-`가 1개이며 보통 첫 일반형의 첫 글자만 따온다. ex) `--patch` = `-p`. 축약형과 일반형은 효과가 같다.
- 각 옵션의 순서는 상관없다. 명령의 필수 인자와 옵션의 순서를 바꾸어도 상관없다.
- 각 명령에 대한 자세한 설명은 `git help <command-name>`으로 확인할 수 있다.
- ticket branch는 parent branch로부터 생성되어, 어떤 특정 기능을 추가하고자 만든 실험적 branch라 생각하면 된다.

---

## Working tree(작업트리) 생성

### git init

빈 디렉토리나, 기존의 프로젝트를 **git 저장소**(=**git repository**)로 변환하고 싶다면 이 문단을 보면 된다.  

일반적인 디렉토리(=git 저장소가 아닌 디렉토리)를 git working tree로 만드는 방법은 다음과 같다. **명령창**(cmd / terminal)에서 다음을 입력한다.

```vim
git init

# 결과 예시
Initialized empty Git repository in blabla/sample_directory/.git/
```

그러면 해당 디렉토리에는 `.git` 이라는 이름의 숨김처리된 디렉토리가 생성된다. 이 디렉토리 안에 든 것은 수동으로 건드리지 않도록 한다.

참고) `git init` 명령만으로는 인터넷(=**원격 저장소** = **remote repository**)에 그 어떤 연결도 되어 있지 않다. [여기](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#git-repository-%EC%97%B0%EA%B2%B0)를 참조한다.

### git clone 

인터넷에서 이미 만들어져 있는 작업트리를 본인의 컴퓨터(=**로컬**)로 가져오고 싶을 때에는 해당 git repository의 `https://github.com/blabla.git` 주소를 복사한 뒤 다음과 같은 명령어를 입력한다.

```vim
git clone <git-address>

# 명령어 예시 
git clone https://github.com/greeksharifa/git_tutorial.git

# 결과 예시
Cloning into 'git_tutorial'...
remote: Enumerating objects: 56, done.
remote: Total 56 (delta 0), reused 0 (delta 0), pack-reused 56
Unpacking objects: 100% (56/56), done.
```
그러면 현재 폴더에 해당 프로젝트 이름의 하위 디렉토리가 생성된다. 이 하위 디렉토리에는 인터넷에 올라와 있는 모든 내용물을 그대로 가져온다(`.git` 디렉토리 포함).  
단, 다른 branch의 내용물을 가져오지는 않는다. 다른 branch까지 가져오려면 [추가 작업]()이 필요하다.

---

## Git Repository 연결

이 과정은 `git clone`으로 원격저장소의 로컬 사본을 생성한 경우에는 필요 없다.

먼저 [github](https://github.com/) 등에서 원격 저장소(remote repository)를 생성한다.

로컬 저장소를 원격저장소에 연결하는 방법은 다음과 같다.

```vim
git remote add <remote-name> <git address>

# 명령어 예시
git remote add origin https://github.com/greeksharifa/git_tutorial.git
```

`<remote-name>`은 원격 저장소에 대한 일종의 별명인데, 보통은 `origin`을 쓴다. 큰 프로젝트라면 여러 개를 쓸 수도 있다.

이것만으로는 완전히 연결되지는 않았다. [upstream 연결](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#upstream-%EC%97%B0%EA%B2%B0)을 지정하는 `git push -u` 명령을 사용해야 수정사항이 원격 저장소에 반영된다.

### 연결된 원격 저장소 확인

```vim
git remote --verbose
git remote -v

# 결과 예시
origin  https://github.com/greeksharifa/git_tutorial.git (fetch)
origin  https://github.com/greeksharifa/git_tutorial.git (push)
```
---

## Git 준비 영역(index)에 파일 추가

로컬 저장소의 수정사항이 반영되는 과정은 총 3단계를 거쳐 이루어진다.

1. `git add` 명령을 통해 준비 영역에 변경된 파일을 추가하는 과정(stage라 부른다)
2. `git commit` 명령을 통해 여러 변경점을 하나의 commit으로 묶는 과정
3. `git push` 명령을 통해 로컬 commit 내용을 원격 저장소에 올려 변경사항을 반영하는 과정

이 중 `git add` 명령은 첫 단계인, **준비 영역**에 파일을 추가하는 것이다.

```vim
git add <filename1> [<filename2>, ...]
git add <directory-name>
git add *
git add --all
git add .

# 명령어 예시
git add third.py fourth.py
git add temp_dir/*
```

`*`은 와일드카드로 그냥 쓰면 변경점이 있는 모든 파일을 준비 영역에 추가한다(`git add *`). 특정 directory 뒤에 쓰면 해당 directory의 모든 파일을, `*.py`와 같이 쓰면 확장자가 `.py`인 모든 파일이 준비 영역에 올라가게 된다.  
`git add .`을 현재 directory(`.`)의 모든 파일을 추가하는 명령으로 `git add --all`과 효과가 같다. 

`git add` 명령을 실행하고 이미 준비 영역에 올라간 파일을 또 수정한 뒤 [`git status`](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#git-status) 명령을 실행하면 같은 파일이 **Changes to be committed** 분류와 **Changes not staged for commit** 분류에 동시에 들어가 있을 수 있다. 딱히 오류는 아니고 해당 파일을 다음 commit에 반영할 계획이면 한번 더 `git add`를 실행시켜주자.

### 한 파일 내 수정사항의 일부만 준비 영역에 추가

예를 들어 `fourth.py`를 다음과 같이 변경한다고 하자.

```python
# 변경 전
print('hello')

print(1)

print('bye')

#변경 후
print('hello')
print('git')

print('bye')
print('20000')
```
이 중 `print('bye'); print('20000')`을 제외한 나머지 변경사항만을 준비 영역에 추가하고 싶다고 하자. 그러면 `git add <filename>` 명령에 다음과 같이 `--patch` 옵션을 붙인다. 

```diff
git add --patch fourth.py
git add fourth.py -p

# 결과 예시
diff --git a/fourth.py b/fourth.py
index 13cc618..4c8cfb6 100644
--- a/fourth.py
+++ b/fourth.py
@@ -1,5 +1,5 @@
 print('hello')
+print('git')

-print(1)
-
-print('bye')
\ No newline at end of file
+print('bye')
+print('20000')
\ No newline at end of file
stage this hunk [y,n,q,a,d,s,e,?]? 
```

그러면 수정된 코드 덩이(hunk)마다 선택할지를 물어본다. 인접한 초록색(+) 덩이 또는 인접한 빨간색 덩이(-)가 하나의 코드 덩이가 된다.

각 옵션에 대한 설명은 다음과 같다. `?`를 입력해도 도움말을 볼 수 있다.

| Option | Description |
| -------- | -------- | 
y | stage this hunk
n | do not stage this hunk
q | quit; do not stage this hunk or any of the remaining ones
a | stage this hunk and all later hunks in the file
d | do not stage this hunk or any of the later hunks in the file
s | split the current hunk into smaller hunks
e | manually edit the current hunk
? | print help

여기서는 `y`, `y`, `n`을 차례로 입력하면 원하는 대로 추가/추가하지 않을 수 있다. (영어 원문을 보면 알 수 있듯이 (stage) = (준비 영역에 추가하다)와 같은 의미라고 보면 된다.)

`-p` 옵션으로는 인접한 추가/삭제 줄들이 전부 하나의 덩이로 묶이기 때문에, 이를 더 세부적으로 하고 싶다면 위 옵션에서 `e`를 선택하면 된다. 

`git add -p` 명령을 통해 준비 영역에 파일의 일부 변경사항만 추가하고 나면 같은 파일이 **Changes to be committed** 분류와 **Changes not staged for commit** 분류에 동시에 들어가게 된다.

---

## Commit하기

준비 영역에 올라간 파일들의 변경사항을 하나로 묶는 작업이라 보면 된다. Git에서는 이 commit(커밋)이 변경사항 적용의 기본 단위가 된다.

### git commit [-m "message"] [--amend]

기본적으로, commit은 다음 명령어로 수행할 수 있다.

```vim
git commit

# 결과 예시:
All text in first line will be showed at --oneline

Maximum length is 50 characters.
Below, is for detailed message.

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# On branch master
# Your branch is up to date with 'origin/master'.
#
# Changes to be committed:
#       modified:   .gitignore
#       new file:   third.py
#
~
~
```

`git commit`을 입력하면 vim 에디터가 열리면서 commit 메시지 편집을 할 수 있다. 방법은:

- `i`를 누른다. insert의 약자이다.
- 이후 메시지를 마음대로 수정할 수 있다. 이 때 규칙이 있는데,
    - 첫 번째 줄은 log를 볼 때 `--oneline` 옵션에서 나타나는 대표 commit 메시지이다. 기본값으로, 50자 이상은 무시된다.
    - 그 아래 줄에 쓴 텍스트는 해당 commit의 자세한 메시지를 포함한다.
    - 맨 앞에 `#`이 있는 줄은 주석 처리되어 commit 메시지에 포함되지 않는다.
- 편집을 마쳤으면 다음을 순서대로 누른다. `ESC`, `:wq`, `Enter`.
    - `ESC`는 vim 에디터에서 명령 모드로 들어가가, `:wq`는 저장 및 종료 모드 입력을 뜻한다. 잘 모르겠으면 그냥 따라하라.
- 맨 밑에 있는 물결 표시(`~`)는 파일의 끝이라는 뜻이다. 빈 줄도 아니다.


commit의 자세한 메시지를 작성하기 귀찮다면(*별로 좋은 습관은 아니다.*), 간단한 메시지만 작성할 수 있다:

```vim
git commit -m "<message>"

# 명령 예시:
git commit -m "hotfix for typr error"
```

물론 이미 작성한 commit 메시지를 변경할 수 있다. 

```vim
git commit --amend
```

그러면 vim 에디터에서 수정할 수 있다.


---

## 수정사항을 원격저장소에 반영하기: git push

### upstream 연결

`git remote add` 명령으로 원격저장소를 연결했으면 `git push <git-address>` 명령으로 로컬 저장소의 commit을 원격 저장소에 반영할 수 있다. 즉, 최종 반영이 되는 것이다.

```vim
git push <git-address>
git push https://github.com/greeksharifa/gitgitgit.git

# 결과 예시
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 200 bytes | 200.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/greeksharifa/gitgitgit.git
 * [new branch]      master -> master
```

그러나 매번 git address를 인자로 주어가며 변경사항을 저장하는 것은 매우 귀찮으니, 다음 명령을 통해 upstream 연결을 지정할 수 있다. 이는 `git remote add` 명령을 통해 원격 저장소의 이름을 이미 지정한 경우의 얘기이다.

```vim
git push --set-upstream <remote-name> <branch-name>
git push -u <remote-name> <branch-name>

# 명령어 예시
git push --set-upstream origin master
git push -u origin master

# 결과 예시
Everything up-to-date
Branch 'master' set up to track remote branch 'master' from 'origin'.
```

`git push --set-upstream <remote-name> <branch-name>` 명령은 `<branch-name>` branch의 upstream을 원격 저장소 `<remote-name>`로 지정하는 것으로, 앞으로 `git push`나 `git pull` 명령 등을 수행할 때 `<branch name>`과 `<remote name>`을 지정할 필요가 없도록 지정하는 역할을 한다. 즉, 앞으로는 commit을 원격 저장소에 반영할 때 `git push`만 입력하면 된다.

위와 같은 방법으로 지정하지 않은 branch나 원격 저장소에 push하고자 하는 경우, `git push <remote-name> <branch-name>`을 사용한다. 

```vim
# 명령어 예시
git push origin ticket-branch
```

### upstream 삭제

더 이상 필요 없는 원격 branch를 삭제할 때는 다음 명령을 사용한다.

```vim
git push --delete <remote-name> <remote-branch-name>

# 명령어 예시
git push --delete origin ticket-branch
git push -d origin ticket-branch
```

---

## Git Directory 상태 확인

### git status

현재 git 저장소의 상태를 확인하고 싶다면 다음 명령어를 입력한다.

```vim
git status

# 결과 예시 1:
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean

# 결과 예시 2:

On branch master
Your branch is up to date with 'origin/master'.

Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        modified:   first.py

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   .gitignore
        deleted:    second.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        third.py
```

`git status`로는 로컬 git 저장소에 변경점이 생긴 파일을 크게 세 종류로 나누어 보여준다.

1. **Changes to be committed**
    - Tracking되는 파일이며, 준비 영역(stage)에 이름이 올라가 있는 파일들. 이 단계에 있는 파일들만이 commit 명령을 내릴 시 다음 commit에 포함된다. (그래서 to be commited이다)
    - 마지막 commit 이후 `git add` 명령으로 준비 영역에 추가가 된 파일들.
2. **Changes not staged for commit:**
    - Tracking되는 파일이지만, 다음 commit을 위한 준비 영역에 이름이 올라가 있지 않은 파일들. 
    - 마지막 commit 이후 `git add` 명령의 대상이 된 적 없는 파일들.
3. **Untracked files:**
    - Tracking이 안 되는 파일들. 
    - 생성 이후 한 번도 `git add` 명령의 대상이 된 적 없는 파일들.

위와 같이 준비 영역 또는 tracked 목록에 올라왔는지가 1차 분류이고, 2차 분류는 해당 파일이 처음 생성되었는지(ex. `third.py`), 변경되었는지(modified), 삭제되었는지(deleted)로 나눈다.

---

## 특정 파일/디렉토리 무시하기: .gitignore

프로젝트의 최상위 디렉토리에 `.gitignore`라는 이름을 갖는 파일을 생성한다. 윈도우에서는 `copy con .gitignore`라 입력한 뒤, 내용을 다 입력하고, `Ctrl + C`를 누르면 파일이 저장되면서 생성된다. 


`.gitignore` 파일을 열었으면 안에 원하는 대로 파일명이나 디렉토리 이름 등을 입력한다. 그러면 앞으로 해당 프로젝트에서는 `git add` 명령으로 준비 영역에 해당 종류의 파일 등이 추가되지 않는다.

예시는 다음과 같다.

```vim
dum_file.py             # `dum_file.py`라는 이름의 파일을 무시한다.
*.zip                   # 확장자가 `.zip`인 모든 파일을 무시한다.
data/                   # data/ 디렉토리 전체를 무시한다.
!data/regression.csv    # data/ 디렉토리는 무시되지만, data/regression.csv 파일은 무시되지 않는다. 
                        # 이 경우는 data/ 이전 라인에 작성하면 적용되지 않는다.
**/*.json               # 모든 디렉토리의 *.json 파일을 무시한다.
```

`.gitignore` 파일을 저장하고 나면 앞으로는 해당 파일들은 tracking되지 않는다. 즉, 준비 영역에 추가될 수 없다.  
그러나 이미 tracking되고 있는 파일들은 영향을 받지 않는다. 따라서 [`git rm --cached`]() 명령을 통해 tracking 목록에서 제거해야 한다.

### 전체 프로젝트에 .gitignore 적용하기

특정 프로젝트가 아닌 모든 프로젝트 전체에 적용하고 싶으면 다음 명령을 입력한다.

```vim
git config --global core.excludesfile <.gitignore-file-path>

# 명령 예시
git config --global core.excludesfile ~/.gitignore
git config --global core.excludesfile C:\.gitignore
```

그러면 해당 위치에 `.gitignore` 파일이 생성되고, 이는 모든 프로젝트에 적용된다. 일반적으로 `git config --global` 명령을 통해 설정하는 것은 특정 프로젝트가 아닌 해당 로컬에서 작업하는 모든 프로젝트에 영향을 준다. [여기]()를 참고하라.

---

## History 검토

### git log: 현재 존재하는 commit 검토

저장소 commit 메시지의 모든 history를 역순으로 보여준다. 즉, 가장 마지막에 한 commit이 가장 먼저 보여진다.

```vim
git log

# 결과 예시
commit da446019230a010bf333db9d60529e30bfa3d4e3 (HEAD -> master, origin/master, origin/HEAD)
Merge: 4a521c5 2eae048
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Sun Aug 19 20:59:24 2018 +0900

    Merge branch '3rd-branch'

commit 2eae048f725c1d843cad359d655c193d9fd632b4
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Sun Aug 19 20:29:48 2018 +0900

    Unwanted commit from 2nd-branch

...
:
```

이때 commit의 수가 많으면 다음 명령을 기다리는 커서가 깜빡인다. 여기서 space bar를 누르면 다음 commit들을 계속해서 보여주고, 끝에 다다르면(저장소의 최초 commit에 도달하면) `(END)`가 표시된다.  
끝에 도달했거나 이전 commit들을 더 볼 필요가 없다면, `q`를 누르면 log 보기를 중단한다(quit).

#### git log 옵션: --patch(-p), --max-count(-\<number\>), --oneline(--pretty=oneline)

각 commit의 diff 결과(commit의 세부 변경사항, 변경된 파일의 변경된 부분들을 보여줌)를 보고 싶으면 다음을 입력한다.

```diff
git log --patch

# 결과 예시
commit 2eae048f725c1d843cad359d655c193d9fd632b4
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Sun Aug 19 20:29:48 2018 +0900

    Unwanted commit from 2nd-branch

diff --git a/first.py b/first.py
index 2d61b9f..c73f054 100644
--- a/first.py
+++ b/first.py
@@ -9,3 +9,5 @@ print("This is the 1st sentence written in 3rd-branch.")
 print('2nd')

 print('test git add .')
+
+print("Unwanted sentence in 2nd-branch")
```

가장 최근의 commit들 3개만 보고 싶다면 다음과 같이 입력한다.
```vim
git log -3
```

commit의 대표 메시지와 같은 핵심 내용만 보고자 한다면 다음과 같이 입력한다.
```vim
git log --oneline

# 결과 예시
da44601 (HEAD -> master, origin/master, origin/HEAD) Merge branch '3rd-branch'
2eae048 Unwanted commit from 2nd-branch
4a521c5 Desired commit from 2nd-branch
```

참고로, 다음과 같이 입력하면 commit의 고유 id의 전체가 출력된다.
```vim 
git log --pretty=oneline

# 결과 예시
da446019230a010bf333db9d60529e30bfa3d4e3 (HEAD -> master, origin/master, origin/HEAD) Merge branch '3rd-branch'
2eae048f725c1d843cad359d655c193d9fd632b4 Unwanted commit from 2nd-branch
4a521c56a6c2e50ffa379a7f2737b5e90e9e6df3 Desired commit from 2nd-branch
```

옵션들은 중복이 가능하다.
```vim
git log --oneline -5
```

### git reflog: commit과 commit의 변화 과정 전체를 검토

```vim
git reflog

# 결과 예시:
87ab51e (HEAD -> master, tag: specific_tag) HEAD@{0}: commit: All text in first line will be showed at --onel
ine
da44601 (origin/master, origin/HEAD) HEAD@{1}: clone: from https://github.com/greeksharifa/git_tutorial.git
```

위와 같이 `HEAD@{0}`: commit과 `HEAD@{1}`: clone 이라는 변화를 볼 수 있다. `git reflog`는 commit 뿐 아니라 commit이 삭제되었는지, 재배치했는지, clone이나 rebase 같은 변화가 있었는지 등등 git에서 일어난 모든 변화를 기록한다. 

---

## HEAD: branch의 tip

HEAD는 현 branch history의 가장 끝을 의미한다. 여기서 끝은 가장 최신 commit 쪽의 끝이다(시작점을 가리키지 않는다).  
다른 의미로는 checkout된 commit, 또는 현재 작업중인 commit이다.

예를 들어, `HEAD@{0}`은 1번째 최신 commit(즉, 가장 최신 commit)을 의미한다. index는 많은 프로그래밍 언어가 그렇듯 0부터 시작한다. 비슷하게, `HEAD@{1}`은 2번째 최신 commit을 의미한다.

`HEAD^`는 HEAD의 직전, 즉 가장 최신 commit을 가리킨다.

범위를 나타낼 땐 `~`를 사용한다. 예를 들어, `HEAD~3`은 가장 최신 commit(1번째)부터 3번째 commit까지를 가리킨다.

`HEAD~2^`는 `HEAD^`(가장 최신, 즉 1번째 commit)보다 2번 더 이전 commit까지 간 것이고, 범위(`~`)를 나타내므로 1~3번째 commit을 가리킨다. 헷갈리니까 3개의 commit을 다루고 싶으면 그냥 `HEAD~3`을 쓰자.


---

## Tag 붙이기

태그는 특정한 commit을 찾아내기 위해 사용된다. 즐겨찾기와 같은 개념이기 때문에, 여러 commit에 동일한 태그를 붙이지 않도록 한다.

우선 태그를 붙이고 싶은 commit을 찾자.

```vim
# 명령어 예시 1
git log --oneline -3

# 결과 예시 1
87ab51e (HEAD -> master) All text in first line will be showed at --oneline
da44601 (origin/master, origin/HEAD) Merge branch '3rd-branch'
2eae048 Unwanted commit from 2nd-branch

# 명령어 예시 2
git log 87ab51e --max-count=1
git show 87ab51e

# 결과 예시 2
commit 87ab51eecef1a526cb504846ddcaed0459f685c8 (HEAD -> master)
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Thu May 28 14:49:13 2020 +0900

    All text in first line will be showed at --oneline

    Maximum length is 50 characters.
    Below, is for detailed message.
```

### git tag

이제 태그를 commit에 붙여보자.

```vim
git tag <tag-name> 87ab51e

# 명령어 예시
git tag specific_tag 87ab51e
```

지금까지 붙인 태그 목록을 보려면 다음 명령을 입력한다.
```vim
git tag

# 결과 예시
specific_tag
```

해당 태그가 추가된 commit을 보려면 [여기](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#git-show-tag-name)를 참조한다.

---

## 특정 commit 보기

### git show

commit id를 사용해서 특정 commit을 보고자 하면 다음과 같이 쓴다.

```vim
git log 87ab51e --max-count=1
git show 87ab51e

# 결과 예시
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Thu May 28 14:49:13 2020 +0900

    All text in first line will be showed at --oneline

    Maximum length is 50 characters.
    Below, is for detailed message.
```

#### git show \<tag-name\>

```diff
git show <tag-name>

# 명령어 예시
git show specific_tag

# 결과 예시
commit 87ab51eecef1a526cb504846ddcaed0459f685c8 (HEAD -> master, tag: specific_tag)
Author: greeksharifa <greeksharifa@gmail.com>
Date:   Thu May 28 14:49:13 2020 +0900

    All text in first line will be showed at --oneline

    Maximum length is 50 characters.
    Below, is for detailed message.

diff --git a/.gitignore b/.gitignore
index 8d16a4b..6ec8ec8 100644
--- a/.gitignore
+++ b/.gitignore
@@ -1,3 +1,2 @@
-third.py
 .idea/
 *dummy*
diff --git a/third.py b/third.py
new file mode 100644
index 0000000..0360dad
--- /dev/null
+++ b/third.py
@@ -0,0 +1 @@
+print('hello 3!')
```
---


## Git Branch

### branch 목록 보기

로컬 branch 목록을 보려면 다음을 입력한다.

```vim
git branch
git branch --list
git branch -l

# 결과 예시
* master
```

branch 목록을 보여주는 모든 명령에서, 현재 branch(작업 중인 branch)는 맨 앞에 asterisk(`*`)가 붙는다.

모든 branch 목록 보기:

```vim
git branch --all
git branch -a

# 결과 예시
* master
  remotes/origin/2nd-branch
  remotes/origin/3rd-branch
  remotes/origin/HEAD -> origin/master
  remotes/origin/master
```

`remotes/`가 붙은 것은 원격 branch라는 뜻이며, branch의 이름에는 `remotes/`가 포함되지 않는다.

원격 branch 목록 보기:

```vim
git branch --remotes
git branch -r

# 결과 예시
  origin/2nd-branch
  origin/3rd-branch
  origin/HEAD -> origin/master
  origin/master
```

### 원격 branch 목록 업데이트

로컬 저장소와 원격 저장소는 실시간 동기화가 이루어지는 것이 아니기 때문에(일부 git 명령을 내릴 때에만 통신이 이루어짐), 원격 branch 목록은 자동으로 최신으로 유지되지 않는다. 목록을 새로 확인하려면 다음을 입력한다.

```vim
git fetch
```

별다른 변경점이 없으면 아무 것도 표시되지 않는다.

---

### branch 전환

단순히 branch 간 전환을 하고 싶으면 다음 명령어를 입력한다.

```vim
git checkout <branch-name>

# 명령어 예시
git checkout master

# 결과 예시
Switched to branch 'master'
M       .gitignore
D       second.py
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)
```

전환을 수행하면, 
- 변경된 파일의 목록과
- 현재 로컬 브랜치가 연결되어 있는 원격 브랜치 사이에 얼마만큼의 commit 차이가 있는지

도 알려준다.

로컬에 새 branch를 생성하되, 그 내용을 원격 저장소에 있는 어떤 branch의 내용으로 하고자 하면 다음 명령을 사용한다.

```vim
git checkout --track -b <local-branch-name> <remote-branch-name>

# 명령어 예시
git checkout --track -b 2nd-branch origin/2nd-branch

# 결과 예시
Switched to a new branch '2nd-branch'
M       .gitignore
D       second.py
Branch '2nd-branch' set up to track remote branch '2nd-branch' from 'origin'.
```

출력에서는 `2nd-branch`라는 이름의 새 branch로 전환하였고, 파일의 현재 수정 사항을 간략히 보여주며, 로컬 branch `2nd-branch`가 `origin`의 원격 branch `2nd-branch`를 추적하게 되었음을 알려준다.  
즉 원격 branch의 로컬 사본이 생성되었음을 알 수 있다.

### 새 branch 생성

```vim
git branch <new-branch-name>

# 명령어 예시
git branch fourth-branch
```

위 명령은 branch를 생성만 한다. 생성한 브랜치에서 작업을 시작하려면 checkout 과정을 거쳐야 한다.

### branch 생성과 같이 checkout하기

```vim
git checkout -b <new-branch-name> <parent-branch-name>

# 명령어 예시
git checkout -b fourth-branch master

# 결과 예시
Switched to a new branch 'fourth-branch'
```

새로운 branch는 생성 시점에서 parent branch와 같은 history(commit 기록들)을 갖는다.

### branch 병합

`git merge <branch-name>`를 사용한다. `<branch-name>` branch의 수정 사항들(commit)을 **현재 branch**로 가져와 병합한다. 

```vim
git merge <branch-name>

# 명령어 예시
git merge ticket-branch

# 결과 예시
Updating 96c99dc..94d511c
Fast-forward
 .gitignore | 2 +-
 fourth.py  | 5 +++++
 second.py  | 9 ---------
 third.py   | 0
 4 files changed, 6 insertions(+), 10 deletions(-)
 create mode 100644 fourth.py
 delete mode 100644 second.py
 create mode 100644 third.py
```

이와 같은 방법을 history fast-forward라 한다(히스토리 빨리 감기).

병합할 때 ticket branch의 모든 commit들을 하나의 commit으로 합쳐서 parent branch에 병합하고자 할 때는 `--squash` 옵션을 사용한다.

```vim
# 현재 branch가 parent branch일 때
git merge ticket-branch --squash
```

`--squash` 옵션은 애초에 branch를 분리하지 말았어야 할 상황에서 쓰면 된다. 즉, 병합 후 parent branch 입장에서는 그냥 하나의 commit이 반영된 것과 같은 효과를 갖는다.

위와 같이 처리했을 때는 ticket branch가 더 이상 필요 없으니 삭제하도록 하자.

### branch 삭제

```vim
git branch --delete <branch-name>
git branch -d <branch-name>

# 명령어 예시
git branch --delete ticket-branch

# 결과 예시
Deleted branch fourth-branch (was 94d511c).
```

branch 삭제는 해당 branch의 수정사항들이 다른 branch에 병합되어서, 더 이상 필요없음이 확실할 때에만 문제없이 실행된다.  
아직 수정사항이 남아 있음에도 그냥 해당 branch 자체를 폐기처분하고 싶으면 `--delete` 대신 `-D` 옵션을 사용한다.

이미 원격 저장소에 올라간 branch를 삭제하려면 [여기](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#upstream-%EC%82%AD%EC%A0%9C)를 참조한다.

---


## 작업 취소하기

먼저 가능한 작업 취소 명령들을 살펴보자.

| 원하는 것 | 명령어 |
| -------- | -------- | 
특정 파일의 수정사항 되돌리기 | `git checkout -- <filename>`
모든 수정사항을 되돌리기 | `git reset --hard`
준비 영역의 모든 수정사항을 삭제 | `git reset --hard <commit>`
여러 commit 통합 | `git reset <commit>`
이전 commit들을 수정 또는 통합, 혹은 분리 | `git rebase --interactive <commit>`
untracked 파일을 포함해 모든 수정사항을 되돌리기 | `git clean -fd`
이전 commit을 삭제하되 history는 그대로 두기 | `git revert <commit>`

아래는 [Git for Teams](https://www.amazon.com/Git-Teams-User-Centered-Efficient-Workflows/dp/1491911182)라는 책에서 가져온 flowchart이다. 뭔가 잘못되었을 때 사용해보도록 하자.

<center><img src="/public/img/2020-05-27-github-usage-09-overall/01.png" width="100%"></center>  

여러 명이 협업하는 프로젝트에서 이미 원격 저장소에 잘못된 수정사항이 올라갔을 때, 이를 강제로 되돌리는 것은 금물이다. '잘못된 수정사항을 삭제하는' 새로운 commit을 만들어 반영시키는 쪽이 훨씬 낫다.

물론 branch를 잘 만들고, pull request 시스템을 적극 활용해서 그러한 일이 일어나지 않도록 하는 것이 최선이다. 


---

### 특정 파일의 수정사항 되돌리기: checkout, reset

특정 파일을 지워 버렸거나 수정을 잘못했다고 하자. 이 때에는 다음 전제조건이 있다.

> 수정사항을 commit하지 않았을 때

commit하지 않았다면, 다음 두 가지 경우가 있다. `git status`를 입력하면 친절히 알려준다.

```vim
git status

#결과 예시
On branch master

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   third.py

no changes added to commit (use "git add" and/or "git commit -a")
```

마지막 줄에서 아직 commit된 것이 없다는 것을 확인해야 한다. 

1. 수정사항을 준비 영역에 올리지 않았을 때(`git add`를 안 수행했을 때)
    - `git checkout -- <filename>`
    - 그러면 파일이 원래대로 복구된다. 
2. 수정사항을 stage했을 때(`git add`를 수행했을 때)
    - 그러면 위 결과 예시처럼 `no changes added to commit ...`이라는 메시지가 없다. 다음 두 명령을 입력한다.
    - `git reset HEAD <filename>`
    - `git checkout -- <filename>`
    을 입력한다.
    - 그러면 가장 최신(HEAD) commit에 저장되어 있는 파일의 원래 상태가 복구된다. commit하지 않았을 때 사용할 수 있는 이유가 이것이다.
    - 아니면 명령어 두 개를 합친 다음 명령을 써도 된다.
    - `git reset --hard HEAD -- <filename>`

`git reset <filename>`은 `git add <filename>`의 역방향이라고 보면 된다. 물론 `git reset <commit> <filename>`은 파일을 여러 commit 이전으로 되돌릴 수 있기 때문에 상황에 따라서는 다른 작업일 수 있다.

비슷하게, `git reset -p <filename>`은 `git add -p <filename>`의 역 작업이다.

`git reset`의 옵션은 여러 개가 있다. 

- `git reset [-q | -p] [--] <paths>`: `<paths>`는 `<filename>`을 포함한다. 즉, filename 뿐만 아니라 디렉토리 등도 가능하다. 이 명령의 효과는 `git add [-p]`의 역 작업이다.
- `git reset [--soft | --mixed [-N] | --hard | --merge | --keep] -[q] [<commit>]`
    - `--hard`: `<commit>` 이후 발생한 모든 수정사항과 준비 영역의 수정사항이 폐기된다. 
    - `--soft`는 파일의 수정사항이 남아 있으며, 수정된 파일들이 모두 **Changes to be committed** 상태가 된다.
    - `--mixed`는 파일의 수정사항은 남아 있으나 준비 영역의 수정사항은 폐기된다. mixed가 기본 옵션이다.
    - `--merge`는 준비 영역의 수정사항은 폐기하고 `<commit>`과 `HEAD` 사이 수정된 파일들을 업데이트하지만 수정된 파일들은 stage되지 않는다.
    - `--keep`은 `--merge`와 비슷하나 `<commit>`때와 `HEAD` 때가 다른 파일에 일부 변화가 있는 경우에는 `reset` 과정이 중단된다.

모든 파일의 수정사항 되돌리기:

```vim
git reset --hard HEAD
```

---

### branch 병합 취소하기

먼저 다음 [flowchart](https://www.amazon.com/Git-Teams-User-Centered-Efficient-Workflows/dp/1491911182)를 살펴보자.

<center><img src="/public/img/2020-05-27-github-usage-09-overall/02.png" width="100%"></center>  

바로 직전에 한 병합(merge)를 취소하려면 다음 명령어를 입력한다.

```vim
git reset --merge ORIG_HEAD
```

병합 후 추가한 commit이 있으면 해당 지점의 commit을 지정해야 한다.

```vim
git reset <commit>
```

어디인지 잘 모르겠으면 [reflog](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#git-reflog-commit%EA%B3%BC-commit%EC%9D%98-%EB%B3%80%ED%99%94-%EA%B3%BC%EC%A0%95-%EC%A0%84%EC%B2%B4%EB%A5%BC-%EA%B2%80%ED%86%A0)를 사용해보자.


---

### 커밋 합치기: git reset \<commit\>

기본적으로, `git reset`은 branch tip을 `<commit>`으로 옮기는 과정이다. 그래서, `git reset <option> HEAD`는 마지막 commit의 상태로 준비 영역 또는 파일 내용을 되돌리는(reset) 작업이다.  
또한, 바로 위에서 살펴봤듯이, `git reset`은 기본 옵션이 `--mixed`이며, 이는 옵션을 따로 명시하지 않으면 `git reset`은 파일의 수정사항은 그대로 둔 채 준비 영역에는 추가된 수정사항이 없는 상태로 만든다.  

그래서 특정 이전 commit을 지정하여 `git reset <commit>`을 수행하면 해당 `<commit>`부터 `HEAD`까지의 파일의 수정사항은 작업트리(=프로젝트 디렉토리 전체)에 그대로 남아 있지만, 준비 영역에는 아무런 변화도 기록되어 있지 않다.  
먼저 어떤 커밋들을 합칠지 `git log --oneline`으로 확인해보자.

```
# 결과 예시
c8c731b (HEAD -> master, origin/master, origin/HEAD) doong commit
87ab51e (tag: specific_tag) All text in first line will be showed at --oneline
da44601 Merge branch '3rd-branch'
2eae048 Unwanted commit from 2nd-branch
4a521c5 Desired commit from 2nd-branch
```

이제 가장 최신 2개의 commit을 합치고 싶으면, 현재 branch의 HEAD를 `c8c731b`에서 `da44601`로 옮기면 된다. 

```vim
git reset da44601
```
그러면 직전 2개의 commit의 수정사항이 파일에는 그대로 남아 있지만, 준비 영역이나 commit 내역에선 사라진다. 이제 stage, commit, push 3단계를 수행하면 최종적으로 commit 2개가 1개로 합쳐진다.

`<commit>` id를 지정하는 것이 헷갈린다면 `git reset HEAD~2`로 실행하자. 이는 [여기](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#head-branch%EC%9D%98-tip)에서 볼 수 있듯이 범위로 2개의 commit을 포함한다.

---


### git rebase

rebase는 일반적으로 history rearrange의 역할을 한다. 즉, 여러 commit들의 순서를 재배치하는 작업이라 할 수 있다. 혹은 parent branch의 수정사항을 가져오면서 자신의 commit은 그 이후에 추가된 것처럼 하는, 마치 분기된 시점을 뒤로 미룬 듯한 작업을 수행할 수도 있다.

그러나 rebase와 같은 기존 작업을 취소 또는 변경하는 명령은 일반적으로 충돌(conflict)이 일어나는 경우가 많다. 충돌이 발생하면 git은 작업을 일시 중지하고 사용자에게 충돌을 처리하라고 한다.

#### master branch의 commit을 topic branch로 가져오기

다음과 같은 상황을 가정하자. 각 알파벳은 하나의 commit이며, 각 이름은 branch의 이름을 나타낸다.  
아래 각 예시는 `git help`에 나오는 도움말을 이용하였다.

```
          A---B---C topic
         /
    D---E---F---G master
```

commit F, G를 topic branch에 반영(포함)시키려 한다면,

```
                  A'--B'--C' topic
                 /
    D---E---F---G master
```

commit A'와 A는 프로젝트에 동일한 수정사항을 적용시키지만, 16진수로 된 commit의 고유 id(`da44601` 같은)는 다르다. 즉, 엄밀히는 다른 commit이다.

commit을 재배열하는 명령어는 다음과 같다. 현재 branch는 topic이라 가정한다.

```vim
git rebase master
git rebase master topic
```

commit A, B, C가 F, G와 코드 상으로 동일한 파일 또는 다른 일부분을 수정하지 않았다면, 이 rebase 작업은 자동으로 완료된다.  


만약 topic branch에 이미 master branch로부터 가져온 commit이 일부 존재하면, 이 commit들은 새로 배치되지 않는다.

```
          A---B---C topic
         /
    D---E---A'---F master
```
에서
```
                   B'---C' topic
                  /
    D---E---A'---F master
```
로 바뀐다.

#### branch의 parent 바꾸기: --onto

topic을 next가 아닌 master에서 분기된 것처럼 바꾸고자 한다. 즉,

```
    o---A---B---o---C  master
         \
          D---o---o---o---E  next
                           \
                            o---o---o  topic
```

이걸 아래와 같이 바꿔보자.

```
    o---A---B---o---C  master
        |            \
        |             o'--o'--o'  topic
         \
          D---o---o---o---E  next
```

topic branch의 history에는 이제 commit D~E 대신 commit A~B가 포함되어 있다.

이는 다음과 같은 명령어로 수행할 수 있다:

```
git rebase --onto master next topic
```

다른 예시는:

```
                            H---I---J topicB
                           /
                  E---F---G  topicA
                 /
    A---B---C---D  master
```

```vim
git rebase --onto master topicA topicB
```

```
                 H'--I'--J'  topicB
                /
                | E---F---G  topicA
                |/
    A---B---C---D  master
```

#### 특정 범위의 commit들 제거하기

```
    E---F---G---H---I---J  topic
```

topic branch의 5번째 최신 commit부터, 3번째 최신 commit **직전**까지 commit을 topic branch에서 폐기하고 싶다고 하자. 그러면 다음 명령어로 사용 가능하다.

```vim
git rebase --onto <branch-name>~<start-number> <branch-name>~<end-number> <branch-name>

# 명령어 예시
git rebase --onto topic~5 topic~3 topic
```

```
    E---H'---I'---J'  topic
```

여기서 5(번째 최신 commit, F)은 삭제되고, 3(번째 최신 commit, H)은 삭제되지 않음을 주의하라. rebase가 되기 때문에 commit의 고유 id는 바뀐다(H -> H')

#### 충돌 시 해결법

일반적으로 rebase에서 수정하는 2개 이상의 commit이 같은 파일을 수정하면 충돌이 발생한다.

보통은 다음 과정을 거치면 해결된다.

- 충돌이 일어난 파일에 적절한 조취를 취한다. 파일을 남기거나/삭제하거나, 또는 파일 일부분에서 남길 부분을 찾는다. 코드 중 다음과 비슷해 보이는 부분이 있을 것이다. 적절히 지워서 해결하자.

```
ㅤ<<<<<<<< HEAD
ㅤ<current-code>
ㅤ========
ㅤ<incoming-code>
ㅤ>>>>>>>> da446019230a010bf333db9d60529e30bfa3d4e3
```

- `git add <conflict-resolved-filename>`
- `git rebase --continue`

그냥 다 모르겠고(?) rebase 작업을 취소하고자 하면 다음을 입력한다.

```
git rebased --abort
```

#### rebase로 commit 합치거나 수정하기

다음과 같은 history가 있다고 하자.

```vim
c3eace0 (HEAD -> master, origin/master, origin/HEAD) git checkout, reset, rebase
f6c56ef what igt
bd80626 github hem
b7801a2 github overall
608a518 highlighter theme change
```

여러 개의 commit들을 합치거나, commit message를 수정하거나 하는 작업은 모두 rebase로 가능하다.  
실행하면, vim 에디터가 열릴 것이다(ubuntu의 경우 nano일 수 있다). vim을 쓰는 방법은 [여기](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/#git-commit--m-message-amend)를 참고한다.
 
rebase하는 부분에서는 다른 git command들과는 달리 수정할 commit 중 가장 오래된 commit이 가장 위에 온다.

```vim
git rebase --interactive <commit>
git rebase -i <commit>

# 명령 예시
git rebase -interactive 608a518
git rebase -i HEAD~4

# 결과 예시

pick c3eace0 (HEAD -> master, origin/master, origin/HEAD) git checkout, reset, rebase
pick f6c56ef what igt
pick bd80626 github hem
pick b7801a2 github overall
# Rebase 608a518..c3eace0 onto 608a518
#
# Commands:
# p, pick = use commit
# r, reword = use commit, but edit the commit message
# e, edit = use commit, but stop for amending
# s, squash = use commit, but meld into previous commit
# f, fixup = like "squash", but discard this commit's log message
# x, exec = run command (the rest of the line) using shell
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```

설명을 잘 살펴보면 다음을 알 수 있다:

- `pick` = `p`는 수정 사항과 commit을 그대로 둔다. 각 commit의 맨 앞에는 기본적으로 `pick`으로 설정되어 있다. 이 상태에서 아무 것도 안 하고 나간다면 이번 `rebase`는 아무 효과도 없다. 
- `reword` = `r`은 `pick`과 거의 같지만 commit message를 수정할 수 있다. commit message를 수정하고 앞의 `pick`을 `reword`나 `r`로 바꾸면 commit의 메시지를 수정할 수 있다. 가장 최신의 commit에 `r`을 붙였다면 `git commit --amend`와 효과가 같다.
- `edit` = `e`는 해당 commit을 수정할 수 있다. reset 등의 작업이 가능하다.
- `squash` = `s`는 해당 commit이 바로 이전 commit에 흡수되며, commit message 또한 합쳐져서 하나로 된다. 합친 메시지들이 존재하는 에디터가 다시 열린다.
- `fixup` = `f`는 `squash`와 비슷하지만, 해당 commit의 message는 삭제된다.
- `exec` = `x`는 commit들 아래 줄에 명령어를 추가하여 실행하게 할 수 있다. 

수정한 예시는 다음과 같다. 약어를 써도 되고 안 써도 된다.

```vim
pick c3eace0 (HEAD -> master, origin/master, origin/HEAD) git checkout, reset, rebase
f f6c56ef what igt
f bd80626 github hem
fixup b7801a2 github overall
...(아래 주석은 지워도 되고 안 지워도 된다. 어차피 commit에서는 무시되는 도움말이다)
```

#### 하나의 commit을 2개로 분리하기

가장 최신 commit이라면 `git reset HEAD~1`을 사용하여 직전 commit 상태로 되돌린 뒤 stage-commit을 2번 수행하면 되고, 그 이전 commit이라면 rebase에서 해당 commit을 `edit`으로 두고 같은 과정을 반복하면 된다.

```vim
# 명령어 예시
git rebase HEAD~4
# pick -> edit
git add -p <filename>
git commit -m <1st-commit-message>
git add -p <filename1> <filename2>
git commit -m "2nd-commit-message>
git rebase --continue
```



---


git cherry-pick -x