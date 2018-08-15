---
layout: post
title: GitHub 사용법 - 05. branch 기본 2
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/07/github-usage-04-branch-basic/)에서 작업하던 것을 이어서 한다. 저번 글에서 1st-branch를 만들고 master에 merge한 뒤 삭제하는 작업을 진행했었다.

---

## Branch 생성, checkout, 강제 삭제

현재 작업 트리의 상황은 다음과 같다. 1st-branch가 삭제되어 있다.

![01](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/01.png)

이제 브랜치 2개를 더 만들어 보자. 다음 명령을 입력한다.  
2nd-branch는 여러분이 사용하는 브랜치이고, 3rd-branch는 여러분 말고 다른 팀원이 사용하는 branch라 생각하자.  
물론 사람을 데려오긴 어렵기 때문에, 여러분이 직접 두 개를 다 만들도록 한다.

> git branch 2nd-branch  
> git checkout -b 3rd-branch

첫 번째 명령은 다 알 것이라고 생각하고, 두 번째 명령은 조금 다르다.  
`git checkout`은 원래 다른 브랜치로 이동할 때 쓰는데, `-b` 옵션을 주면 해당 브랜치를 만들면서 checkout하는 효과를 갖는다. 즉, 여러분은 현재 3rd-branch 브랜치에 위치해 있다.

![02_create](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/02_create.PNG)

물론 위와 같이 이미 있는 이름으로 만드려고 하면 이미 존재한다고 오류를 뱉어낸다.

이미 완전히 삭제해버린 브랜치 이름(1st-branch)로 만드는 것은 오류를 뱉어내지 않는다.  
그러나 뭔가 확실하지 않은 부분이 있다면 똑같은 이름으로 재생성하는 것은 별로 추천되지 않는다.

### 옵션: Branch 재생성, 강제 삭제

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

## Non fast-forward 병합

이번엔 두 개의 서브 브랜치(참고로 티켓 브랜치라고도 부른다)에서 모두 master에 변경사항을 만들어 볼 것이다.  
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
참고: 마지막 두 줄은 `git merge master 2nd-branch`로 가능하다. 그러나 해당 브랜치에서 추가 작업이 필요한 경우 별로 편리한 명령은 아니다.

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

작업 트리를 그려보면 다음과 같다. master로 checkout했을 때를 기준으로 하였다(HEAD).  
또한 fast-forward와 non fast-forward를 구분하여 그렸다.

![09](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/09.png)

명령창에서 그래프를 그릴 수도 있다. 다음 둘 중 하나를 입력한다.

> git log --graph  
> git log --graph --oneline

![10_graph](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/10_graph.PNG)

위의 그림들에서는 설명할 것이 많다. 

1. 2nd-branch merge 시: master에 새로운 수정사항이 없는 상태에서 다른 서브 브랜치의 커밋을 merge했을 때, fast-forward 방식으로 merge된다. 작업 트리에서 보듯이 master는 해당 브랜치의 커밋을 그대로 가져올 뿐이다.
2. 3rd-branch merge 시: master에 수정사항이 있는 경우(2nd-branch로부터의 커밋 1개), non fast-forward 방식으로 merge된다. 이 때는 master가 3rd-branch의 커밋을 그대로 가져오기는 하지만, merge commit(90ce4f2)이라는 새로운 커밋이 생성된다. 

보충 설명은 다음과 같다.

1. 빨간색 글씨로 (origin/master)이라 되어 있는 부분이 있다. 이는 remote repo의 HEAD가 현재 local repo의 어떤 커밋까지 적용되어 있는지를 의미한다. 여러분은 'This is 1st commit written in 1st-branch'까지만 remote repo에 push했기 때문에, 현재 remote repo는 그 위치에 멈춰 있다. 이후에 push를 하면 위치가 바뀔 것이다.
2. master의 로그를 보면,
    1. (origin/master) 이후에 (2nd-branch)가 있다. 이는 2nd-branch로부터 (merge하여) 가져온 커밋임을 의미한다. 바로 다음의 (3rd-branch)도 마찬가지이다.
    2. (HEAD -> master)는 이전에 잠깐 설명했었는데, 현재 위치(HEAD)를 나타내는 동시에 현재 브랜치의 이름을 나타낸다.
3. 로그를 출력한 그림의 맨 윗부분을 보면, "Your branch is ahead of 'origin/master' by 3 commits"라는 문구가 있다. 이는 조금 전 설명과도 일치하는 부분인데, 여러분의 local repo의 master가 remote repo의 master보다 3개의 커밋만큼 수정사항이 더 많다는 뜻이다. 3개의 커밋은 각각 (origin/master) 커밋 이후의 커밋들 3개이다.
    1. (2nd-branch) This is the 1st commit written in 2nd-branch
    2. (3rd-branch) This is the 1st commit written in 3rd-branch
    3. (HEAD -> master) Merge branch '3rd-branch'
4. 그럼 왜 2nd-branch와 3rd-branch로 checkout했을 때는 그런 메시지가 없는지 알 수 있을 것이다. 
    1. 2nd-branch와 3rd-branch는 push한 적이 전혀 없다. 즉, remote repo에는 해당 브랜치들에 대한 정보가 전혀 없다. 브라우저를 켜서 확인해보라.

이제 local repo의 master의 변경사항을 remote repo에 업데이트하자. 

> git status  
> git push  

![11_push](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/11_push.PNG)

master에서는 파일을 직접 수정하지 않고 merge만 했기 때문에, `git add`나 `git commit` 명령이 따로 필요 없다.

![12_browser](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/12_browser.PNG)

---

## Ticket branch 업데이트

티켓 브랜치(서브 브랜치)의 로그는 master의 로그보다 조금씩 적었다. 그 이유는 master에는 모든 수정사항이 적용되었지만, 2nd-branch와 3rd-branch는 각각 상대 서브 브랜치의 수정사항이 적용되어있지 않기 때문이다.

물론 2nd-branch와 3rd-branch는 각각 서로 부모 자식 간의 관계가 아니기 때문에, 서로 직접 상호작용할 이유는 없다.  
상호작용은 직접 연관된 master와만 하면 된다.

master에 적용된 수정사항(자신의 것이 아닌, 팀원이 만든 커밋을 가져온다)을 가져오고 싶으면 다음을 수행하라.  
바로 push까지 수행한다.

> git checkout 2nd-branch  
> git rebase master  
> git push --set-upstream origin 2nd-branch  

![13_2nd](/public/img/GitHub/2018_08_11_github_usage_05_branch-basic/13_2nd.PNG)

위의 그림에서 "Fast-forwarded 2nd-branch to master."라는 문구를 볼 수 있다. 2nd-branch에서 더 이상의 수정사항이 발생하지 않은 상태에서 master의 커밋을 가져왔기 때문에 fast-forward 조건이 성립하고, 이를 확인할 수 있다.

다음 명령들을 수행하여 master의 로그와 비교하면 2nd-branch의 로그와 커밋 코드가 완전히 같아졌음을 볼 수 있다.

> git checkout master  
> git log --oneline

팀원의 브랜치도 업데이트해 주자.

> git checkout 3rd-branch  
> git merge master  
> git push --set-upstream origin 3rd-branch  

이 경우에는 merge나 rebase나 별다른 차이가 없다.  

이제 브랜치의 기본적인 내용은 끝났다.

---

## Merge vs Rebase

브랜치 간 병합을 수행할 수 있는 명령은 merge와 rebase가 있는데, 간략히 차이를 적어 보면 다음과 같다.

Merge: `git merge <branch>`로 사용. 해당 브랜치의 변경사항을 현재 브랜치로 (그대로) 가지고 온다. 이때 가져온 커밋들은 현재 HEAD의 위쪽에 그대로 쌓인다. 경우에 따라 Merge commit이 남을 수 있다.  
Rebase: 이름 그대로 base를 바꾸는 것이다. base가 되는 커밋의 위치를 바꾼다.

base를 예시를 들어 설명하면, 
1. 3rd-branch는 master에서 생성되었다. 이때 생성 시점은 master가 "6df3479 This is 1st commit written in 1st-branch" 커밋까지 포함한 상태였다. 여기서 3rd-branch의 base는 바로 이 커밋이다.
2. `git rebase master` 명령을 실행하면, master의 tip(HEAD) 으로 base를 재설정한다. 그러면 master의 커밋들로부터 시작한 셈이 되므로, master의 커밋들을 가져온 결과와 같다. 

Merge와 rebase는 사용법이 조금 다르다. 이는 나중에 설명하도록 하겠다.  
사람에 따라서는 별 구분 없이 사용하기도 한다.

---

이제 git의 branch에 대한 기본적인 설명을 알아보았으니, [다음 글](https://greeksharifa.github.io/github/2018/08/12/github-usage-06-branch-intermediate/)에서는 브랜치를 관리하는 법에 대해서 알아본다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
