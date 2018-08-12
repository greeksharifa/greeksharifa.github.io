---
layout: post
title: GitHub 사용법 - 05. branch 기본
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

---

## Branch 생성, checkout, 강제 삭제

[저번 글](https://greeksharifa.github.io/github/2018/08/07/github-usage-04-branch-basic/)에서 작업하던 것을 이어서 한다. 저번 글에서 1st-branch를 만들고 master branch에 merge한 뒤 삭제하는 작업을 진행했었다.

현재 작업 트리의 상황은 다음과 같다. 1st-branch가 삭제되어 있다.

![01](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/01.png)

이제 브랜치 2개를 더 만들어 보자. 다음 명령을 입력한다.

> git branch 2nd-branch  
> git checkout -b 3rd-branch

첫 번째 명령은 다 알 것이라고 생각하고, 두 번째 명령을 설명하겠다.

`git checkout`은 원래 다른 브랜치로 이동할 때 쓰는데, `-b` 옵션을 주면 해당 브랜치를 만들면서 checkout하는 효과를 갖는다. 즉, 여러분은 현재 3rd-branch 브랜치에 위치해 있다.

![02_create](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/02_create.PNG)

물론 위와 같이 이미 있는 이름으로 만드려고 하면 이미 존재한다고 오류를 뱉어낸다.

이미 완전히 삭제해버린 브랜치 이름(1st-branch)로 만드는 것은 오류를 뱉어내지 않는다.  
그러나 뭔가 확실하지 않은 부분이 있다면 똑같은 이름으로 재생성하는 것은 별로 추천되지 않는다.

하지만 예시를 보여주기 위해 만들어 보겠다.

> git branch 1st-branch  
> git checkout 1st-branch

그리고 `first.py` 파일을 생성하고 마지막 줄에 다음을 입력한다. 곧 삭제할 것이기 때문에 아무거나 입력해도 괜찮다.

```python
print('Dummy!')
```

다음으로 여러분은 이 수정 사항이 쓸모 있는 것이라고 생각하고 다음 명령까지 진행했다고 하자.

> git add first.py  
> git commit -m "dummy!"

커밋 메시지가 너무 단순하다고 생각된다면, 다음 명령을 입력하라.

> git commit --amend

그러면 (아마도 vim이라고 하는) 편집기가 뜬다. 이 편집기의 사용법은 나중에 좀 더 자세히 다룰 텐데, 지금은 다음 과정만 따른다.

1. 처음 편집기에 들어왔을 때는 '명령 모드'이다. 여기서 `i`를 누르면 `INSERT` 모드(명령창의 맨 아래에 뜬다)가 되어, 커밋 메시지를 수정할 수 있다.
2. 다음으로 커밋 메시지를 바꾼다. `Dummy commit!`
3. 그리고 다음 키들을 차례로 입력한다. 하나도 틀려서는 안 된다.
    1. `ESC`
    2. `:`   (콜론)
    3. `wq`  (영문자 두 개이다)
    4. `Enter`

그러면 작성했던 커밋 메시지가 수정된다.

![03_1st_branch](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/03_1st_branch.PNG)

![04_dummy_commit](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/04_dummy_commit.PNG)


여기까지는 문제가 없다. 그러나 추천되지 않는 행동이기에, 바로 삭제하려고 한다.  
여러분도 프로젝트를 진행하다 보면 브랜치를 만들고 작업을 하다가 필요가 없어져서 삭제하려고 하는 상황이 올 것이다.

2nd-branch로 checkout을 하고 삭제 명령을 입력한다.

> git checkout 2nd-branch  
> git branch -d 1st-branch

그러나 에러가 뜬다. 변경된 작업사항을 push하거나 merge하지 않았기 때문에, 수정사항을 잃어버릴 수도 있다고 경고하는 것이다.  
물론 여러분은 이것이 쓸모없는 브랜치인 것을 알기 때문에, 강제로라도 삭제하면 된다.

친절하게도 강제 삭제 명령을 git에서 알려 주었다.

> git branch -D 1st-branch

![05_D](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/05_D.PNG)

이제 2nd-branch에서 다시 시작하자.

---

### Non fast-forward 병합

이번엔 두 개의 서브 브랜치(참고로 티켓 브랜치라고도 부른다)에서 모두 master branch에 변경사항을 만들어 볼 것이다.  
두 명의 사람이 각각 자신의 브랜치를 만들어서 master에 적용하려는 상황을 생각하라.  
참고: 같은 컴퓨터에서는 명령창을 두 개 놓고 다른 브랜치에서 작업하는 것은 불가능하다.

우선 2nd-branch에서, `second.py` 파일의 끝에 다음을 추가한다.

```python
print("This is the 1st sentence written in 2nd-branch.")
```

그리고 다음 명령들을 차례로 입력한다. 이 부분은 fast-forward 병합이다.  
참고: 명령창에서 second.py를 입력할 때 's' 정도만 타이핑한 후 `Tab` 키를 누르면 파일(또는 디렉토리) 이름이 자동완성된다.

> git add second.py  
> git commit -m "This is the 1st commit written in 2nd-branch"  
> git checkout master  
> git merge 2nd-branch  

파일 수정 시 빈 줄을 몇 개나 넣었느냐에 따라서 수정 사항(1 file changed, 3 insertions(+), 1 deletion(-)) 이 달라 보일 수 있으나, 여기선 별 상관 없다.

![06_fast](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/06_fast.PNG)

그리고 이제 커밋 코드를 확인해 보자.

![07_log](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/07_log.PNG)

16진수 코드가 같다. fast-forward 방식으로 병합되었기 때문임은 설명했다.

이제 3rd-branch로 이동한다.  
그리고 3rd-branch에서는 `first.py`에 다음을 추가한다.

```python
print("This is the 1st sentence written in 3rd-branch.")
```

그리고 다음 명령들을 입력한다. 조금 전과 매우 유사하다.

> git add first.py  
> git commit -m "This is the 1st commit written in 3rd-branch"  
> git checkout master  
> git merge 3rd-branch  

세 브랜치의 로그를 확인해 보면 위와는 조금 차이가 있다.

![08_log](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/08_log.PNG)

작업 트리를 그려보면 다음과 같다. master branch로 checkout했을 때를 기준으로 하였다(HEAD).  
또한 fast-forward와 non fast-forward를 구분하여 그렸다.

![09](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/09.png)

명령창에서 그래프를 그릴 수도 있다. 다음 둘 중 하나를 입력한다.

> git log --graph  
> git log --graph --oneline

![10_graph](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/10_graph.PNG)

위의 그림들에서는 설명할 것이 많다. 

1. 


![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)

![](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/.PNG)


---

이제 git의 프로젝트에 대한 설명은 대략 다 끝났다. [다음 글](https://greeksharifa.github.io/github/2018/08/11/github-usage-05-branch-basic/)에서는 branch에 대해서 알아본다.

---

## Git 명령어

다음 글에서 원하는 기능을 찾아 볼 수 있다. [GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)
