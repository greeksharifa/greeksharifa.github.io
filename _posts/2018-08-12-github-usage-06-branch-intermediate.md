---
layout: post
title: GitHub 사용법 - 06. branch 관리
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/11/github-usage-05-branch-basic/)에서 작업하던 것을 이어서 한다. 저번 글에서는 두 사람이 각각 브랜치를 만들어 자신의 브랜치에서 작업하는 상황을 가정하여 진행했었다.

---

## Branch naming

Branch 관리는 사실 하기 나름이지만, [다음의 방법](https://nvie.com/posts/a-successful-git-branching-model/)이 괜찮기에 추천한다.

1. 기본적으로 **master** 브랜치에서 진행한다. 이건 아주 기본적인 사항이다. master branch는 곧 배포할 코드를 보관한다.
2. 개발의 중심이 되는 브랜치는 **develop** 브랜치에서 진행한다. develop 브랜치에서는 지금까지 merge된 코드들이 오류 없이 안정적으로 동작하는지를 검사한다. 물론 develop 브랜치에 merge되기 전에도 다른 팀원들이 검사하는 것이 일반적이지만, 모든 것을 다 판별할 수는 없기 때문에 안정성을 검사하는 것이다.
3. **feature** 브랜치는 새로운 기능을 추가할 때 사용된다. 기능을 새로이 개발해야 할 때 local에서 만든 다음, 개발이 끝나면 develop 브랜치로 merge된다. 일반적으로 local에서 진행되고 remote repo에 push되지 않으나, 여러 명이 한 기능을 동시에 개발하는 경우에는 push한다.
4. **release** 브랜치는 실제 동작 환경과 유사한 곳에서 테스트를 한번 더 하는 브랜치인데, 서버 환경이거나 아주 큰 프로젝트같은 것이 아니라면 굳이 필요하지는 않다. develop 브랜치로부터 생성되고, 안정성 검사가 끝나면 master branch로 version number와 함께 merge된다. 일반적으로 release 관련 작업은 주기적(1주 등)으로 이루어진다.
5. **hotfix** 브랜치는 배포 이후에 발견된 버그를 빠르게 고쳐 패치하기 위한 브랜치이다(hotfix 패치는 어디선가 많이 봤을 것이다). 대개 master 혹은 develop 브랜치에서 생성되며 버그를 고친 이후에는 둘 모두에 각각 merge된다.

위의 경우는 큰 프로젝트 혹은 서버의 경우이고, 규모가 작은 프로젝트라면 master 브랜치와 개인 브랜치(feature 브랜치와 비슷), 그리고 필요하다면 develop 브랜치 정도까지만 있어도 괜찮다.

---

## Branch update

저번 글에서 master, 2nd-branch, 3rd-branch까지 만들어서 작업을 했었다. 그런데 이 모든 내용은 local repo에 절대 자동으로 업데이트되지 않는다.

git_tutorial_clone 폴더로 이동한다. [여기](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/#local-directory-%EC%83%9D%EC%84%B1)에서 했었는데, 아마 기억하고 있을 것이다.

그리고 브랜치 상태를 확인해 보자.

> git status  
> git branch -a  

![01_clone](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/01_clone.PNG)

업데이트를 전혀 안 했기 때문에, `git pull`을 입력해 보자.

![02_pull](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/02_pull.PNG)

remote repo는 업데이트되었다. 만약 잘 되지 않았다면 `git fetch`를 추가로 입력하라. 이 명령은 저번 글에서 설명하였다.  
그러나 local branch 목록은 전혀 업데이트되지 않았다. 


업데이트하기 전에, 우선 브라우저로 가서 2nd-branch로 이동한다. 그리고 `second.py`를 수정한다.

![03_browser](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/03_browser.PNG)

마지막 줄에 다음을 추가한다. 이렇게 브라우저에서도 파일을 수정할 수 있다.  
추천되는 방법은 아니지만, readme.md를 수정하는 용도로는 쓸 만하다.

```python
print("Directly modified sentence")
```

![04_browser](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/04_browser.PNG)

그리고 **Commit changes** 버튼을 누른다.

다시 git_tutorial_clone/git_tutorial 폴더로 돌아오자.

local branch 목록은 전혀 업데이트되지 않았다. 이는 git은 자동으로 모든 것을 복사해오지 않기 때문이다. 

remote branch를 local branch로 가져오는 방법은 다음과 같다.

첫번째 방법:

> git checkout -t origin/2nd-branch  

`-t` 옵션은 `--track`과 같다.  
만약 이전 버전의 git을 쓰고 있다면, `-b` 옵션을 추가로 붙인다.

두번째 방법:

> git checkout 2nd-branch

그리고 다음을 입력한다. 이는 checkout 하기 전 master 브랜치에서 진행해도 상관없다.
> git pull  
> type second.py

`type`은 윈도우 명령창에서 파일의 내용을 출력하는 명령어이다. Linux 환경에서는 `cat`이다.

![05_local](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/05_local.PNG)

위의 방법은 branch를 직접 가져오는 것이다. git pull을 하면 파일 내용이 업데이트된다.

그런데 remote branch를 가져오는 다른 방법이 있다. *detached HEAD* 상태로 가져오는 방법인데, 
1. 이는 branch의 내용을 갖고 오면서 직접 수정도 할 수 있지만, 
2. commit이나 push를 할 수는 없고, 
3. 다른 branch로 checkout하면 사라진다.

![06_detached](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/06_detached.PNG)

---

지금까지 remote branch를 가져오는 법을 알아보았다. [다음 글](https://greeksharifa.github.io/github/2018/08/15/github-usage-07-diff-add-commit-gitignore-intermediate/)에서는 `git diff`, `git add`, `git commit`, `.gitignore`의 더 자세한 사용 방법을 알아본다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
