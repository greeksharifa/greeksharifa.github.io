---
layout: post
title: GitHub 사용법 - 07. add, commit, .gitignore 중급
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/12/github-usage-06-branch-intermediate/)에서 작업하던 것을 이어서 한다. 저번 글에서는 다른 local repo의 branch update까지 살펴보았다.

---

## git add: 중급

다시 git_tutorial_clone 디렉토리 밖으로 빠져 나와서, 원래 git_tutorial repository로 돌아가자. 그리고 업데이트를 한다.

> cd ../../git_tutorial  
> git pull  

[여기](https://greeksharifa.github.io/github/2018/06/29/github-usage-02-create-project/#git-add)에서 `git add` 명령의 다양한 옵션을 설명했었다.  
페이지를 옮겨다니기 귀찮을 것이므로 다시 한번 가져왔다.

| 명령어| Description
| -------- | --------
| git add first.py | first.py 파일 하나를 cache에 추가한다.
| git add my_directory/  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | my_directory라는 이름의 디렉토리와 그 디렉토리 안의 모든 파일과 디렉토리를 cache에 추가한다.
| git add . | 현재 폴더의 모든 파일과 디렉토리, 하위 디렉토리에 든 전부를 cache에 추가한다. 규모가 큰 프로젝트라면 써서는 안 된다.
| git add -p [\<파일\>] | 파일의 일부를 staging하기
| git add -i | Git 대화 모드를 사용하여 파일 추가하기
| git add -u [\<경로\>] | 수정되고 추적되는 파일의 변경 사항 staging하기 

사실은 위의 것 말고도 조금 다른 방법이 있다. 바로 와일드카드이다.

파일을 추가할 때 `.py` 파일을 전부 추가하고 싶다고 하자. 그러면 다음과 같이 쓸 수 있다.

> git add first.py  
> git add second.py  
> ...

그러나 이는 귀찮을 뿐더러 빠트리는 경우도 얼마든지 있을 수 있다. 이럴 땐 `*` 를 사용한다.

> git add *.py

이를 사용하면 `.py`로 끝나는 모든 파일이 stage에 추가된다.

표에서 위쪽 세 종류의 명령은 어려운 부분이 아니므로, 다른 옵션을 설명하겠다.

### git add -p [\<파일\>]

2nd-branch로 이동한다. master에는 직접 수정을 가하지 않는다.

> git checkout 2nd-branch

그리고 second.py를 수정한다. 최종 결과물은 다음과 같다.

```python
print('1st')
print("Why don't you answer me, git?")

print('2nd')
print("This is the 1st sentence written in 2nd-branch.")

print('3rd')
print('4th')

```

참고로 모든 파일의 마지막 줄에는 빈 줄은 추가해 두는 것이 commit log를 볼 때 편하다. 이유는 마지막에 빈 줄만 추가하고 staging시켜 보면, 마지막 줄의 내용을 삭제한 후 마지막 줄의 내용 그리고 빈 줄을 추가한 것처럼 나오기 때문이다.

![07_diff](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/07_diff.PNG)

별로 깔끔하지 않기 때문에 빈 줄을 추가하라. commit log 볼 때뿐만 아니라 나중에 편집할 때에도 조금 더 편하다.  
IDE에 따라서는 빈 줄이 없으면 경고를 띄워 주기도 한다.

그리고 다음 명령을 입력한다. 지금은 `second.py`만 수정했기 때문에 해당 파일만 추가한다.  
조금 위의 표에서 봤듯이 `-p` 옵션은 파일의 일부만을 staging(stage/cache에 올리는 것)하는 과정이다.  
`-p`는 `--patch`의 단축 옵션이다.

> git add -p second.py

그러면 다음과 같이 뜬다.

![08_patch](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/08_patch.PNG)

- 초록색 줄은 추가된 줄, 빨간색 줄은 삭제된 줄이다.
- 파일의 일부분만을 추가하는데, 모든 한 줄마다 따로 추가하는 것이 아니라 hunk라는 덩어리로 한 번에 stage에 추가할지 말지를 결정한다. 만약에 git이 나눠준 hunk가 너무 크다면, `s`를 입력하여 더 잘게 쪼갠다. 위의 경우는 너무 크기 때문에, 잘게 쪼갤 것이다.
- 만약에 무슨 옵션이 있는지 궁금하다면 `?`를 입력하라. 도움말이 표시된다.

명령 | 설명
-------- | --------
y | stage this hunk
n | do not stage this hunk
q | quit; do not stage this hunk or any of the remaining ones
a | stage this hunk and all later hunks in the file
d | do not stage this hunk or any of the later hunks in the file
g | select a hunk to go to
/ | search for a hunk matching the given regex
j | leave this hunk undecided, see next undecided hunk
J | leave this hunk undecided, see next hunk
k | leave this hunk undecided, see previous undecided hunk
K | leave this hunk undecided, see previous hunk
s | split the current hunk into smaller hunks
e | manually edit the current hunk
? | print help

`s`, `y`, `n`, `y`를 차례대로 입력한다. `s`를 입력하면 3개의 hunk로 분리되었다고 알려 준다.

![09_hunk](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/09_hunk.PNG)

참고:
1. untracked files는 `-p`를 할 때 나오지 않는다. 새로 추가된 파일이라면 먼저 stage에 올린 후, 수정한 파일만을 `p` 옵션으로 처리하라.
2. 디버깅을 위해 넣어 놓은 print문 등을 제거하고 push할 때 유용하다.
3. 파일명으로 추가하는 대신 `*`나 `.`를 쓰는 것도 가능하다.

---

### git add 








![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

![](/public/img/GitHub/2018_08_12_github_usage_06_branch-intermediate/.PNG)

---

이제 git의 branch에 대한 기본적인 설명을 알아보았으니, [다음 글](https://greeksharifa.github.io/github/2018/08/11/github-usage-05-branch-basic/)에서는 branch에 대해서 알아본다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
