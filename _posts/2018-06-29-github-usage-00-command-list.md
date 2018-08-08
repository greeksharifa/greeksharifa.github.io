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
git status | 현재 local repo의 git 상태를 확인한다(수정 파일, cache, commit 등)
git status \-s | 간략하게 local repo의 상태를 확인한다.
git add \<file \| directory\> | cache에 파일이나 디렉토리 혹은 전체(*)를 올린다.
git rm \-\-cached \<file \| directory\> | 파일이나 디렉토리를 cache에서 제거한다.
git commit \[-m "commit message"\] | 수정사항들을 하나의 커밋으로 묶고 커밋 메시지를 작성한다.
git remote add origin \<remote repo 주소\> | local repo를 remote repo와 연결시킨다.
git clone \<remote repo 주소\> | remote repo의 파일 복제본을 local로 가져온다. local repo가 생성된다.
 | 
git log \[\-\-oneline\] | 현재 local repo의 commit log를 표시한다. \-\-oneline 옵션은 한줄로 간략히 표시한다.
git log origin/master..\[HEAD\] | remote repo에는 없고 HEAD에는 있는 커밋을 표시한다. 

 
---

기타 명령어 | 설명 
-------- | --------
git help \[\<command\>\] <br> git \[\<command\>\] \-\-help <br> man git-\[\<commmand\>\]| 명령어에 대한 도움말을 볼 수 있다.
git config \[\-\-global\] user.name "Your name" | local repo의 git id(name)을 Your name으로 설정한다. \-\-global 옵션을 주면 모든 local repo에 적용한다.
git config \[\-\-global\] user.name "Your name" | local repo의 git email을 Your email으로 설정한다. \-\-global 옵션을 주면 모든 local repo에 적용한다.
git config \[\-\-list \| \<config settings\> \] | config 세팅 상태를 볼 수 있다. \-\-list 옵션은 config 세팅 전부를 보여준다. 
git config \-\-global color.ui "auto" | Git의 출력결과 색상을 활성화
git config \-\-global credential.helper store | 비밀번호 저장
git config \-\-global credential.helper cache --timeout "seconds" | seconds 초 동안 비밀번호 저장
git branch \[\-\-list \| \-r \-a] | local/remote/전체 repo의 branch 목록 조회


