---
layout: post
title: GitHub 사용법 - 09. Overall
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/19/github-usage-08-conflict/)에서는 Conflict에 대해서 알아보았다.  
이번 글에서는, 전체 Git 명령어들의 사용법을 살펴본다.

---

명령어에 일반적으로 적용되는 규칙:

- 각 명령에는 여러 종류의 옵션이 있다. ex) `git log`의 경우 `--oneline`, `-<number>`, `-p` 등의 옵션이 있다.
- 각 옵션은 많은 경우 축약형이 존재한다. 일반형은 `-`가 2개 있으며, 축약형은 `-`가 1개이며 보통 첫 일반형의 첫 글자만 따온다. ex) `--patch` = `-p`. 축약형과 일반형은 효과가 같다.
- 각 옵션의 순서는 상관없다. 명령의 필수 인자와 옵션의 순서를 바꾸어도 상관없다.
- 각 명령에 대한 자세한 설명은 `git help <command-name>`으로 확인할 수 있다.

---

## Git Directory 생성

### git init

빈 디렉토리나, 기존의 프로젝트를 **git 저장소**(=**git repository**)로 변환하고 싶다면 이 문단을 보면 된다.  

일반적인 디렉토리(=git 저장소가 아닌 디렉토리)를 git 디렉토리로 만드는 방법은 다음과 같다. **명령창**(cmd / terminal)에서 다음을 입력한다.

<pre><code class="git">git init

# 결과 예시
Initialized empty Git repository in blabla/sample_directory/.git/ </code></pre>

그러면 해당 디렉토리에는 `.git` 이라는 이름의 숨김처리된 디렉토리가 생성된다. 이 디렉토리 안에 든 것은 수동으로 건드리지 않도록 한다.

참고) `git init` 명령만으로는 인터넷(=**원격 저장소** = **remote repository**)에 그 어떤 연결도 되어 있지 않다. [여기]()를 참조한다. 되고

### git clone 

인터넷에서 이미 만들어져 있는 git 디렉토리를 본인의 컴퓨터(=**로컬**)로 가져오고 싶을 때에는 해당 git repository의 `https://github.com/blabla.git` 주소를 복사한 뒤 다음과 같은 명령어를 입력한다.

<pre><code class="git">git clone <git-address>

# 명령어 예시 
git clone https://github.com/greeksharifa/git_tutorial.git

# 결과 예시
Cloning into 'git_tutorial'...
remote: Enumerating objects: 56, done.
remote: Total 56 (delta 0), reused 0 (delta 0), pack-reused 56
Unpacking objects: 100% (56/56), done.   </code></pre>

그러면 현재 폴더에 해당 프로젝트 이름의 하위 디렉토리가 생성된다. 이 하위 디렉토리에는 인터넷에 올라와 있는 모든 내용물을 그대로 가져온다(`.git` 디렉토리 포함).  
단, 다른 branch의 내용물을 가져오지는 않는다. 다른 branch까지 가져오려면 [추가 작업]()이 필요하다.

---

## Git Repository 연결

로컬 저장소를 원격(remote) 저장소에 연결하는 방법은 다음과 같다.

git add remote origin


---

## Git stage에 파일 추가

로컬 저장소의 수정사항이 반영되는 과정은 총 3단계를 거쳐 이루어진다.

1. `git add` 명령을 통해 stage에 변경된 파일을 추가하는 과정
2. `git commit` 명령을 통해 여러 변경점을 하나의 commit으로 묶는 과정
3. `git push` 명령을 통해 로컬 commit 내용을 원격 저장소에 올려 변경사항을 반영하는 과정

이 중 `git add` 명령은 첫 단계인, **stage**에 파일을 추가하는 것이다.

<pre><code class="git">git add <filename1> [<filename2>, ...]
git add <directory-name>
git add *
git add --all
git add .

# 명령어 예시
git add third.py fourth.py
git add temp_dir/*  </code></pre>

`*`은 와일드카드로 그냥 쓰면 변경점이 있는 모든 파일을 stage에 추가한다(`git add *`). 특정 directory 뒤에 쓰면 해당 directory의 모든 파일을, `*.py`와 같이 쓰면 확장자가 `.py`인 모든 파일이 stage에 올라가게 된다.  
`git add .`을 현재 directory(`.`)의 모든 파일을 추가하는 명령으로 `git add --all`과 효과가 같다. 

`git add` 명령을 실행하고 이미 stage에 올라간 파일을 또 수정한 뒤 [`git status`]() 명령을 실행하면 같은 파일이 Changes to be committed 분류와 Changes not staged for commit 분류에 동시에 들어가 있을 수 있다. 딱히 오류는 아니고 해당 파일을 다음 commit에 반영할 계획이면 한번 더 `git add`를 실행시켜주자.

### 한 파일 내 수정사항의 일부만 stage에 추가

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
이 중 `print('bye'); print('20000')`을 제외한 나머지 변경사항만을 stage에 추가하고 싶다고 하자. 그러면 `git add <filename>` 명령에 다음과 같이 `--patch` 옵션을 붙인다. 

<pre><code class="git">git add --patch fourth.py
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
Stage this hunk [y,n,q,a,d,s,e,?]? </code></pre>

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

여기서는 `y`, `y`, `n`을 차례로 입력하면 원하는 대로 stage할 수 있다. (영어 원문을 보면 알 수 있듯이 stage하다 = stage에 추가하다와 같은 의미라고 보면 된다.)

`-p` 옵션으로는 인접한 추가/삭제 줄들이 전부 하나의 덩이로 묶이기 때문에, 이를 더 세부적으로 하고 싶다면 위 옵션에서 `e`를 선택하면 된다. 

`git add -p` 명령을 통해 stage에 파일의 일부 변경사항만 추가하고 나면 


---

## Git Directory 상태 확인

### git status

현재 git 저장소의 상태를 확인하고 싶다면 다음 명령어를 입력한다.

<pre><code class="git">git status

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

        third.py </code></pre>

`git status`로는 로컬 git 저장소에 변경점이 생긴 파일을 크게 세 종류로 나누어 보여준다.

**1. Changes to be committed**
    - Tracking되는 파일이며, stage(스테이지)에 이름이 올라가 있는 파일들. 이 단계에 있는 파일들만이 commit 명령을 내릴 시 다음 commit에 포함된다. (그래서 to be commited이다)
    - 마지막 commit 이후 `git add` 명령으로 stage에 추가가 된 파일들.
**2. Changes not staged for commit:**
    - Tracking되는 파일이지만, 다음 commit을 위한 stage에 이름이 올라가 있지 않은 파일들. 
    - 마지막 commit 이후 `git add` 명령의 대상이 된 적 없는 파일들.
**3. Untracked files:**
    - Tracking이 안 되는 파일들. 
    - 생성 이후 한 번도 `git add` 명령의 대상이 된 적 없는 파일들.

위와 같이 stage 또는 tracked 목록에 올라왔는지가 1차 분류이고, 2차 분류는 해당 파일이 처음 생성되었는지(ex. `third.py`), 변경되었는지(modified), 삭제되었는지(deleted)로 나눈다.

---

## History 검토

### git log

저장소 commit 메시지의 모든 history를 역순으로 보여준다. 즉, 가장 마지막에 한 commit이 가장 먼저 보여진다.

<pre><code class="git">git log

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
: </code></pre>

이때 commit의 수가 많으면 다음 명령을 기다리는 커서가 깜빡인다. 여기서 space bar를 누르면 다음 commit들을 계속해서 보여주고, 끝에 다다르면(저장소의 최초 commit에 도달하면) `(END)`가 표시된다.  
끝에 도달했거나 이전 commit들을 더 볼 필요가 없다면, `q`를 누르면 log 보기를 중단한다(quit).

#### git log 옵션: --patch(-p), -\<number\>, --oneline(--pretty=oneline)

각 commit의 diff 결과(commit의 세부 변경사항, 변경된 파일의 변경된 부분들을 보여줌)를 보고 싶으면 다음을 입력한다.

<pre><code class="git">git log --patch

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
+print("Unwanted sentence in 2nd-branch") </code></pre>

가장 최근의 commit들 3개만 보고 싶다면 다음과 같이 입력한다.
<pre><code class="git">git log -3 </code></pre>

commit의 대표 메시지와 같은 핵심 내용만 보고자 한다면 다음과 같이 입력한다.
<pre><code class="git">git log --oneline

# 결과 예시
da44601 (HEAD -> master, origin/master, origin/HEAD) Merge branch '3rd-branch'
2eae048 Unwanted commit from 2nd-branch
4a521c5 Desired commit from 2nd-branch </code></pre>

참고로, 다음과 같이 입력하면 commit의 고유 id의 전체가 출력된다.
<pre><code class="git">git log --pretty=oneline

# 결과 예시
da446019230a010bf333db9d60529e30bfa3d4e3 (HEAD -> master, origin/master, origin/HEAD) Merge branch '3rd-branch'
2eae048f725c1d843cad359d655c193d9fd632b4 Unwanted commit from 2nd-branch
4a521c56a6c2e50ffa379a7f2737b5e90e9e6df3 Desired commit from 2nd-branch </code></pre>

옵션들은 중복이 가능하다.
<pre><code class="git">git log --oneline -5 </code></pre>


---


## Git Branch

### branch 목록 보기

로컬 branch 목록을 보려면 다음을 입력한다.

<pre><code class="git">git branch
git branch --list
git branch -l

# 결과 예시
* master </code></pre>

branch 목록을 보여주는 모든 명령에서, 현재 branch(작업 중인 branch)는 맨 앞에 asterisk(`*`)가 붙는다.

모든 branch 목록 보기:

<pre><code class="git">git branch --all
git branch -a

# 결과 예시
* master
  remotes/origin/2nd-branch
  remotes/origin/3rd-branch
  remotes/origin/HEAD -> origin/master
  remotes/origin/master </code></pre>

`remotes/`가 붙은 것은 원격 branch라는 뜻이며, branch의 이름에는 `remotes/`가 포함되지 않는다.

원격 branch 목록 보기:

<pre><code class="git">git branch --remotes
git branch -r

# 결과 예시
  origin/2nd-branch
  origin/3rd-branch
  origin/HEAD -> origin/master
  origin/master </code></pre>

### 원격 branch 목록 업데이트

로컬 저장소와 원격 저장소는 실시간 동기화가 이루어지는 것이 아니기 때문에(일부 git 명령을 내릴 때에만 통신이 이루어짐), 원격 branch 목록은 자동으로 최신으로 유지되지 않는다. 목록을 새로 확인하려면 다음을 입력한다.

<pre><code class="git">git fetch </code></pre>

별다른 변경점이 없으면 아무 것도 표시되지 않는다.

---

### branch 전환

단순히 branch 간 전환을 하고 싶으면 다음 명령어를 입력한다.

<pre><code class="git">git checkout <branch-name>

# 명령어 예시
git checkout master

# 결과 예시
Switched to branch 'master'
M       .gitignore
D       second.py
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits) </code></pre>

전환을 수행하면, 
- 변경된 파일의 목록과
- 현재 로컬 브랜치가 연결되어 있는 원격 브랜치 사이에 얼마만큼의 commit 차이가 있는지

도 알려준다.

로컬에 새 branch를 생성하되, 그 내용을 원격 저장소에 있는 어떤 branch의 내용으로 하고자 하면 다음 명령을 사용한다.

<pre><code class="git">git checkout --track -b <local-branch-name> <remote-branch-name>

# 명령어 예시
git checkout --track -b 2nd-branch origin/2nd-branch

# 결과 예시
Switched to a new branch '2nd-branch'
M       .gitignore
D       second.py
Branch '2nd-branch' set up to track remote branch '2nd-branch' from 'origin'. </code></pre>

출력에서는 `2nd-branch`라는 이름의 새 branch로 전환하였고, 파일의 현재 수정 사항을 간략히 보여주며, 로컬 branch `2nd-branch`가 `origin`의 원격 branch `2nd-branch`를 추적하게 되었음을 알려준다.  
즉 원격 branch의 로컬 사본이 생성되었음을 알 수 있다.

### 새 branch 생성

<pre><code class="git">git branch <new-branch-name>

# 명령어 예시
git branch fourth-branch </code></pre>

위 명령은 branch를 생성만 한다. 생성한 브랜치에서 작업을 시작하려면 checkout 과정을 거쳐야 한다.

### branch 생성과 같이 checkout하기

<pre><code class="git">git checkout -b <new-branch-name> <parent-branch-name>

# 명령어 예시
git checkout -b fourth-branch master

# 결과 예시
Switched to a new branch 'fourth-branch' </code></pre>



새로운 branch는 부모 브랜치와



git push --set-upstream origin fourth-branch
