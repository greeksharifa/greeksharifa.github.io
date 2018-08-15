---
layout: post
title: GitHub 사용법 - 00. Command List
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---

명령어 | 설명
-------- | --------
git init | local repo를 생성한다.
git status \[\-s] | 현재 local branch의 git 상태를 확인한다(수정 파일, cache, commit 등) `-s` 옵션은 간략히 표시한다.
 | 
git add \<file \| directory\> | stage에 파일이나 디렉토리 혹은 전체(*)를 올린다.
git add \-p\|\-i\|\-u | 각각 부분/대화형/수정 사항을 stage에 추가
git rm \-\-cached \<file \| directory\> | 파일이나 디렉토리를 cache에서 제거한다.
git commit \[-m "commit message"\ | 수정사항들을 하나의 커밋으로 묶고 커밋 메시지를 작성한다.
git commit \-v | git diff 명령을 포함하는 커밋 메시지 편집창을 열어 커밋한다.
git diff \[HEAD] | 마지막 커밋과 현재 수정사항 사이의 차이를 보여준다.
git diff \[\<branch1\>] \<branch2\> | 다른 브랜치와의 차이를 보여준다.
git diff \[\<commit1\>] \<commit1\> | 다른 커밋과의 차이를 보여준다.
 | 
git tag \<tag\> \[\-a \[\-m] ]  | 마지막 커밋에 태그를 붙인다. `-a` 옵션은 Annotated 태그를 가리킨다. `-m` 옵션은 메시지를 작성한다.
git tag \<tag\> \<commit\> | 지정한 코드에 해당하는 커밋에 태그를 붙인다.
git tag | 태그 목록을 보여준다.
git show \<tag\> | 해당 태그에 대한 자세한 설명을 보여준다.
 | 
git remote add origin \<remote repo 주소\> | local branch를 remote branch와 연결시킨다.
git clone \<remote repo 주소\> | remote repo의 파일 복제본을 local로 가져온다. local repo가 생성된다.
 | 
git log \[\-\-oneline] | 현재 브랜치의 commit log를 표시한다. `--oneline` 옵션은 한줄로 간략히 표시한다.
git log origin/master..\[HEAD] | remote repo에는 없고 HEAD에는 있는 커밋을 표시한다. 
git log \-\-graph | 현재 브랜치의 commit log를 그래프 형태로 보여준다.
 | 
git branch \[\-\-list \| \-r \-a] | local/remote/전체 repo의 branch 목록 조회
git checkout \[\-b] \<branch\> | 선택한 branch로 이동. `-b` 옵션은 브랜치를 생성하면서 이동
git checkout -t \[\-b] \<origin\/branch\> | 선택한 remote branch의 파일을 다운로드하면서 checkout
git branch \-d\[\-D] \<branch\> | 선택한 local branch 삭제, `-D` 옵션은 강제 삭제
git push -d origin \<branch\> | remote branch 삭제
git fetch | remote branch 목록 업데이트
 | 
git merge \<branch\> | 현재 브랜치에서 해당 브랜치의 수정사항을 가져온다.
git merge \<branch1\> \<branch2\> | branch2의 변경사항을 branch1로 가져온다.
git rebase \<branch\> | 현재 브랜치의 base를 해당 브랜치의 tip으로 설정하여, 해당 브랜치의 변경사항을 가져온다.

---

기타 명령어 | 설명 
-------- | --------
git help \[\<command\>] <br> git \[\<command\>] \-\-help <br> man git-\[\<commmand\>]| 명령어에 대한 도움말을 볼 수 있다.
git config \[\-\-global] user.name "Your name" | local repo의 git id(name)을 Your name으로 설정한다. \-\-global 옵션을 주면 모든 local repo에 적용한다.
git config \[\-\-global] user.name "Your name" | local repo의 git email을 Your email으로 설정한다. \-\-global 옵션을 주면 모든 local repo에 적용한다.
git config \[\-\-list \| \<config settings\> \ | config 세팅 상태를 볼 수 있다. \-\-list 옵션은 config 세팅 전부를 보여준다. 
git config \-\-global color.ui "auto" | Git의 출력결과 색상을 활성화
git config \-\-global credential.helper store | 비밀번호 저장
git config \-\-global credential.helper cache --timeout "seconds" | seconds 초 동안 비밀번호 저장



