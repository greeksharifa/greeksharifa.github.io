---
layout: post
title: GitHub 사용법 - 04. branch 기본
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

---

## Branch

[저번 글](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/)에서 작업하던 것을 이어서 한다. 저번 글에서는 master branch란 말을 잠깐 언급했었다.

여러분은 별다른 설정을 하지 않았지만 master branch에서 계속 작업을 하고 있었다.  
**master branch**란 모든 repository의 기본 혹은 메인이라고 보면 된다. 일반적으로 repo의 모든 것은 master branch를 중심으로 행해진다.  

큰 프로젝트든 개인 프로젝트이든 최종 결과물은 master branch에 있기 마련이며, master branch로부터 파생된 다른 branch들로부터 수정 사항을 만든 다음 master에 병합하는 과정을 거치게 된다.  

여러 사람이 협업할 경우 각자 따로 브랜치를 쓰게 되며, 각 브랜치에서는 새로운 기능을 개발하거나 버그 수정이 이루어진다. 물론 완료되면 master branch에 병합되게 된다.

위의 설명이 정석적인 git repo의 운영방법이고, master branch에는 일반적으로 직접 수정을 가하지 않는다. 따라서 별다른 일이 없다면 본 글에서부터는 master branch에 직접 commit을 날리지 않고, branch를 만든 다음 병합하는 과정을 거칠 것이다.

잠깐 브라우저를 켜서 브랜치 부분을 클릭해 보자.

![01_init_screen](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/01_init_screen.PNG)

그러면 'Default'로 표시되어 있는 master branch 하나가 보일 것이다.

![02_master](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/02_master.PNG)

이제 명령어 입력을 시작하자.

---

### branch 목록 보기

목록을 보는 옵션은 여러 가지가 있다. `git branch` 혹은 `git branch --list`를 입력해 보자.

> git branch

![03_list](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/03_list.PNG)

위 명령은 local repo에 저장되어 있는 branch들의 리스트를 보여 준다. 다른 branch를 만들지 않았기 때문에 master 하나밖에 보이지 않을 것이다.

이제 여러분은 다음과 같은 형태의 트리를 갖고 있다.  
그렇다. branch는 tree의 것이다.

<center><img src="/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/04_branch.png" width="50%"></center>

<br>

조금 더 자세하게 그리기 위해, `git log --oneline`을 명령창에 입력한다. [무엇을 하는 것](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/#local-repo-%EC%83%81%ED%83%9C-%ED%99%95%EC%9D%B8%ED%95%98%EA%B3%A0-git-pull%EB%A1%9C-local-repo-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8%ED%95%98%EA%B8%B0)인지 잊어버리지는 않았을 것이다.

![05_log](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/05_log.PNG)

이제 트리를 다시 그려보자.

![06_branch](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/06_branch.png)

앞으로 위와 비슷한 그림이 자주 나올 텐데, 각각의 의미를 정확히 알고 있는 것이 앞으로 git을 이해하는 데 큰 도움이 될 것이다.

1. 저번 글에서 간략히 언급했는데, 앞의 복잡한 숫자는 16진수 숫자로서 각 커밋의 고유한 코드이다. 유일한 값이므로 어떤 커밋인지 분간할 때 도움이 될 것이다.


![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)




이제 git의 프로젝트에 대한 설명은 대략 다 끝났다. [다음 글](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)에서는 branch에 대해서 알아본다.

---

## Git 명령어

다음 글에서 원하는 기능을 찾아 볼 수 있다. [GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)
