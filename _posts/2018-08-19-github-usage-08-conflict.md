---
layout: post
title: GitHub 사용법 - 08. Conflict
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/15/github-usage-07-diff-add-commit-gitignore-intermediate/)에서 작업하던 것을 이어서 한다. 저번 글에서는 `diff`, `add`, `commit`, `.gitignore`에 대해서 알아보았다.

---

## Conflict

Conflict는 이름 그대로 충돌인데, 다음의 경우일 때 conflict가 생긴다.

> 같은 파일의 같은 부분을 동시에 두 곳(이상)에서 수정했을 때

이런 경우는 보통 여러 사람의 분업이 명확하게 이루어지지 않아 코드의 같은 부분을 수정할 때 일어난다.  
물론 1인 팀에서도 코드 관리를 잘못하여, 혹은 여러 컴퓨터에서 작업하게 될 때 이러한 실수가 일어나기도 한다.

Conflict의 발생 및 해결 순서는 다음과 같다.

1. 동시에 같은 파일의 같은 부분을 수정하고, merge 혹은 push를 할 때 일어난다. 이는 같은 파일을 수정했다 하더라도 명확히 다른 부분이 수정되었다면 git이 알아서 병합 과정을 처리해준다는 뜻이다. 
    - 충돌이 일어났다면, git은 병합 과정을 진행하지 않고 충돌 상태를 그대로 둔다. 알아서 처리하는 대신 사용자가 충돌을 살펴보고 원하는 코드만 남길 때까지 기다린다.
2. 사용자가 편집기에서 코드를 원하는 부분만 남긴다. 충돌이 일어난 부분은 git이 명확하게 표시를 해 준다. 
    - 표시를 한다는 것은, 실제로 코드 파일을 git이 수정한다는 뜻이다. 물론 알아서 충돌을 해결한다는 뜻이 아니라, "여기 충돌 생겼어"하고 강력하게 표시를 한다는 뜻이다. 만약 코드 테스트를 한다면 틀림없이 이 부분에서 syntax error가 뜬다.
3. 사용자가 직접 수정을 끝냈으면, commit을 한 다음 merge 혹은 push 작업을 완료한다.
    - 이때 따로 새로운 commit이 생기는 대신 원래 있어야 할 merge commit만 생성된다.

하나씩 살펴보자.

---

### 1. Conflict 발생시키기

일부러는 절대 해서는 안 되지만, 예시를 보여주어야 하기 때문에 고의로 conflict를 발생시켜 보겠다.

일단은 git_tutorial repo의 3rd-branch로 이동한 다음, `git rebase master` 명령을 실행한다.  
2nd-branch부터 시작한 수정사항이 반영되어있지 않기 때문이다.

> git checkout 3rd-branch  
> git rebase master

`second.py`의 마지막에 다음을 추가한다.

git_tutorial repo의 2nd-branch로 이동한 다음, `first.py`의 마지막에 다음을 추가한다.

> git checkout 2nd-branch

```python
print("Desired sentence in 2nd-branch")
```

commit을 한 뒤, master branch에서 2nd-branch의 내용을 merge한다. 

> git add first.py  
> git commit -m "Desired commit from 2nd-branch"  
> git checkout master  
> git merge 2nd-branch  

그리고 3rd-branch로 이동하여 비슷하게 반복한다. 수정하는 파일은 당연히 `first.py`이다.

> git checkout 3rd-branch

```python
print("Unwanted sentence in 3nd-branch")
```

> git add first.py  
> git commit -m "Unwanted commit from 2nd-branch"  
> git checkout master  
> git merge 3rd-branch  

![01_conflict](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/01_conflict.PNG)

예상대로 conflict가 뜬다.


---

### 2. Conflict 해결하기

이제 편집기로 가서 `first.py`를 살펴보라. 메모장 코딩을 하는 것이 아니라면, 에러 표시줄이 여럿 보일 것이다. 

![02_file](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/02_file.PNG)

파일을 살펴보면 확실히 어느 부분에서 conflict가 일어났는지 바로 확인이 가능하다.  
참고로 필자와 빈 줄의 개수가 달라도 별 상관은 없다.  

git이 수정해놓은 부분을 보면 다음과 갈은 구조로 되어 있다.  

```
<<<<<<< HEAD
(현재 브랜치의 HEAD의 내용)
=======
(merge한 브랜치의 내용)
>>>>>>> (merge한 브랜치 내용)
```

여기서 각 브랜치의 내용 중 사용자가 원하는 부분만 남기고 모두 지우면 된다. 한쪽 브랜치의 내용만 남길 수도 있고, 양쪽 모두의 내용의 일부 혹은 전체를 남길 수도 있다.

수정을 마쳤으면 필요 없는 부분인 `<<<<<<< HEAD`, `=======`, `>>>>>>> <branch>` 등은 모두 제거하면 된다.

![04_resolve](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/04_resolve.PNG)

예상대로 남길 부분은 "Desired sentence"이므로 이 문장만 남기고 나머지 부분은 모두 삭제하면 된다.

IDE에 따라서는 다음과 같이 표시될 수도 있다. 이때는 조금 더 편하게 진행할 수 있다.  
아래 예시는 Visual Studio Code의 경우이다.

![03_vscode](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/03_vscode.PNG)

수정하기 전 '변경 사항 비교'를 누르면 어떤 부분이 다른지를 양쪽에 나누어 보여준다.  
내용을 확인한 뒤 '현재 변경 사항 수락'을 누르면 원하는 부분만 남겨지고 나머지는 알아서 삭제될 것이다. 물론 다른 부분을 남겨도 상관없다.

---

### 3. commit(merge) & push하기

그리고 수정한 파일을 `git add` 명령으로 추가한 뒤 commit한다. 정상적으로 처리되었는지 보기 위해 로그도 한번 출력해 보자.

> git add first.py  
> git commit  

![05_commit](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/05_commit.PNG)

> **ESC 입력 후 :wq**  
> git log --oneline

![06_merge](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/06_merge.PNG)

이러면 conflict가 해결된 것이다. remote repo에 push하자.

> git push

그리고 3rd-branch로 이동하여 rebase를 한다.

> git checkout 3rd-branch  
> git rebase master


---

### 4. Conflict가 발생하지 않는 경우

조금 전 처음에 했던 것처럼 2nd-branch로 이동해 업데이트한다.

> git checkout 2nd-branch  
> git rebase master

이번엔 `first.py` 파일 끝에 다음을 추가한다.

```python
print("This is the 2nd sentence written in 2nd-branch.")
```

그리고 `second.py`의 내용은 다음 문장 빼고 모두 지운다.
```python
print("This is the 1st sentence written in 2nd-branch.")
```

다시 비슷한 과정을 반복한다.

> git add *.py  
> git commit -m "No-collision commit from 2nd-branch"  
> git checkout master  
> git merge 2nd-branch  

> git checkout 3rd-branch

다음으로는 3rd-branch로 이동하여, `first.py` 파일의 내용을 다음 문장 빼고는 모두 지운다.  
지우기 전에, `print("This is the 1st sentence written in 2nd-branch.")` 문장은 없어야 정상이다. 있다면, checkout을 제대로 했는지 살펴보라.

```python
print("Desired sentence in 2nd-branch")
```

> git add first.py  
> git commit -m "No-collision commit from 3rd-branch"  
> git checkout master  
> git merge 3rd-branch  

![07_no_conflict](/public/img/GitHub/2018_08_19_github_usage_08_Conflict/07_no_conflict.PNG)

문제없이 잘 병합된 것을 확인할 수 있다. 다른 파일, 혹은 같은 파일을 수정했더라도 수정한 부분이 다르면 conflict가 일어나지 않는다.  
위 예시의 경우 `first.py`를 2nd-branch에서는 파일의 끝 부분을, 3rd-branch에서는 파일의 시작 부분을 수정했기 때문에 문제가 일어나지 않았다.

---

### 5. 이유없이 conflict가 생기는 것 같은 경우

사실 이유가 없는 경우는 없지만, 간혹 두 branch 간 차이가 전혀 없어 보이고 파일 수정까지 끝마쳤는데도 conflict가 계속해서 발생하는 경우가 있다.

다른 원인일 수도 있지만, 정말로 아무 차이도 없어 보인다면 운영체제의 line-feed 문자의 차이로 인한 문제일 수 있다.  
즉 Windows는 `'\r\n'`을, Linux나 Mac은 `'\n'`을 개행 문자로 사용하기 때문인데, 이 차이를 제대로 인식하지 못해 실패하는 경우가 있으니 참고하면 되겠다.

[해결법](https://stackoverflow.com/questions/861995/is-it-possible-for-git-merge-to-ignore-line-ending-differences/12194759#12194759)은 다음과 갈다.

> git config merge.renormalize true

그리고 merge를 시도하면 된다.


---

[다음 글](https://greeksharifa.github.io/github/2020/05/27/github-usage-09-overall/)에서는 Git 전체 명령어 사용법을 다룬다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
