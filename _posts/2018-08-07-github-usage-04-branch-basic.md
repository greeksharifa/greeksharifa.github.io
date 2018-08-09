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

[저번 글](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/)에서 작업하던 것을 이어서 한다. 그때 master branch란 말을 잠깐 언급했었다.

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
    1. 여러분이 커밋 메시지를 간략하게 작성할수록 16진수 코드를 봐야 하는 상황이 많아진다.
2. 16진수 코드 다음에는 여러분이 입력한 커밋 메시지가 나온다.
3. (HEAD -> master, origin/master) 메시지는 이전 글에서 설명했다. 사실 1, 2번도 전부 설명했다.

그런데 origin이 무엇인지 궁금하지 않은가?

---

### 옵션: origin

저번 글에서는 일반적으로 remote repo의 이름은 origin으로 둔다고 하였다. 그러나 이는 사용자 마음이다.

dummy repo를 하나 만들자.

![08_dummy](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/08_dummy.PNG)

dummy local repo도 만들어 올려보자. 이때 origin 대신 dummy_origin으로 입력해 보자.

![10_dummy](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/10_dummy.PNG)

그리고 파일을 수정하고 추가한 뒤 git push origin master를 입력하면 오류가 뜬다.  
dummy_origin으로 입력하면 잘 되는 것을 확인할 수 있다.  
즉 origin은 add나 commit 명령처럼 정해져 있는 것이 아니라 사용자가 마음대로 정하는 것이지만, 별다른 이유가 없다면 origin으로 두는 것이 괜찮다.

![12_dummy](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/12_dummy.PNG)

참고로 `git status -s`는 `git status`보다 간략한 버전이다.

이제 dummy는 dummy니까 갖다 버리면 된다. Settings 탭의 맨 아래로 가보면 다음과 같은 부분이 있다.

![14_dummy](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/14_dummy.PNG)

삭제하면 된다.

![15_dummy](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/15_dummy.PNG)

---

이제 dummy repo는 잊어버리고, 원래 하던 것으로 돌아오자.

local repo 말고 remote repo의 브랜치를 알고 싶다면 다음 중 하나를 입력한다.

> git branch -r  
> git branch --remote

local이랑 remote 전부 보고 싶으면 다음을 입력한다.

> git branch -a  
> git branch --all
    

![07_branch_op](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/07_branch_op.PNG)

`-r` 옵션과 `-a` 옵션의 remote repo 표기가 조금 다르다.  
`-a` 옵션은 local repo와 remote repo를 구분하기 위해 'remotes/'를 remote repo 앞에 붙인다.  
`-r` 옵션은 remote repo만 보여주기 때문에 'remotes/' 표시가 필요 없다.

이제 local repo와 remote repo에 무엇이 있는지 알았으니, 브랜치를 새로 만들어 보자.  
원래는 서브 브랜치를 새로 생성할 메인 브랜치(master일 필요는 없다)로 이동하는 과정이 우선되어야 하지만, 지금은 master branch 하나뿐이므로 그럴 이유가 없다.

> git branch 1st-branch  
> git branch -a

![16_create_branch](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/16_create_branch.PNG)

현재 있는 브랜치 앞에는 '*'이 있다. 여러분의 컴퓨터 환경에 따라 다를 수는 있으나, Windows cmd 기준으로는 현재 있는 브랜치는 초록색, remote repo의 브랜치는 빨간색으로 표시된다.

이제 새로운 브랜치로 이동해 보자.

> git checkout 1st-branch

![17_checkout](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/17_checkout.PNG)

`first.py`에 다음을 추가한다. 이제부터 여러분은 `first.py`를 수정한 다음, 커밋을 만들고, master branch에 병합하는 과정을 거칠 것이다.  
앞서 설명했듯 서브 브랜치를 만들어서 그곳에서 작업한 후 master branch에 병합하는 것이 정석적인 방법이다.

```python
print("This is the 1st sentence written in 1st-branch.")
```

그리고 3종 세트를 입력한다. 싫으면 [옵션](https://greeksharifa.github.io/github/2018/07/08/github-usage-03-clone-log-gitignore/#%EC%98%B5%EC%85%98-3%EC%A2%85-%EC%84%B8%ED%8A%B8-%EA%B0%84%ED%8E%B8-%EC%9E%85%EB%A0%A5%EC%9C%88%EB%8F%84%EC%9A%B0-%EA%B8%B0%EC%A4%80)에서 다루었던 `push.bat`을 입력해도 상관없다.

![18_1st_commit](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/18_1st_commit.PNG)

사실 그냥은 안 된다. remote repo 기준에서 여러분이 로컬에 만든 1st-branch라는 브랜치는 전혀 알 수 없는 것(정확히는 local repo에 upstream branch가 없는 것이다)이며, 연결 작업이 필요하다.  
다행히 upstream branch(local branch와 연결할 remote branch)를 설정하는 방법을 명령창에서 친절히 알려 준다.

> git push --set-upstream origin 1st-branch

remote repo 이름이 origin이고 current branch의 이름이 1st-branch이기 때문에 저렇게 입력해주면 된다.

![19_upstream](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/19_upstream.PNG)

브라우저를 확인해보자.

![20_push](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/20_push.PNG)

브랜치가 2개가 되었음을 확인할 수 있다.

이제 다음과 같은 상태가 되었다.

![21_log](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/21_log.PNG)

![22_branch](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/22_branch.png)

참고로 1st-branch의 커밋들의 16진수 코드는 master branch의 같은 커밋의 16진수 코드와 똑같다.  
master branch로부터 생성했으니 당연한 말이다.

방금 수정했기 때문에, `first.py`의 현재 상태는 다음과 같을 것이다.

```python
print("Hello, git!") # instead of "Hello, World!"
print("Hi, git!!")

print("This is the 1st sentence written in 1st-branch.")
```

그럼 이제 master branch로 다시 돌아가 본다. 

> git checkout master

그리고 `first.py`를 다시 확인해 보라.

```python
print("Hello, git!") # instead of "Hello, World!"
print("Hi, git!!")
```

마지막 문장이 사라졌을 것이다. 편집기를 사용하고 있었다면 '다시 로드'를 클릭하라.

여러분은 1st-branch에서 작업했을 뿐이다. master branch로부터 서브 브랜치를 생성하는 순간부터, 어떤 새로운 상호작용을 하기 전까지 1st-branch는 master branch와는 '거의' 독립적인 공간에 가깝다. 따라서 branch를 checkout하는 순간 1st-branch에서 수정했던 사항이 보이지 않게 되는 것이다.

물론 진짜로 사라진 것은 아니다. 다시 1st-branch로 checkout하면 내용이 돌아올 것이다.  
아무튼 다시 master branch로 이동하자.

글의 윗부분에서 프로젝트 진행 과정을 설명하면서 다음과 비슷한 말을 했었다.

1. 서브 브랜치를 만들어 작업한다.
2. 메인 브랜치(master branch)로 이동한다.
3. 서브 브랜치의 내용을 메인 브랜치에 병합한다.

위 과정 중 여러분은 1, 2번을 완료했다. 이제 3번을 하기만 하면 된다.  
현재 브랜치가 master branch인지 꼭 확인한 후 다음을 진행해야 한다.

> git merge 1st-branch

![23_merge](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/23_merge.PNG)

그리고 `first.py`를 확인하면 1st-branch에서 추가했던 문장이 들어 있는 것을 확인할 수 있다.  
또한 커밋의 16진수 코드도 1st-branch의 것과 같음을 확인할 수 있다.

![24_log](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/24_log.PNG)










![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_07_github_usage_04_branch-basic/.PNG)




이제 git의 프로젝트에 대한 설명은 대략 다 끝났다. [다음 글](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)에서는 branch에 대해서 알아본다.

---

## Git 명령어

다음 글에서 원하는 기능을 찾아 볼 수 있다. [GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)
