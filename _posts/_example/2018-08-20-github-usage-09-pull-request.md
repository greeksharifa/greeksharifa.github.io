---
layout: post
title: GitHub 사용법 - 09. Fork, Pull Requests
author: YouWon
categories: GitHub
tags: [GitHub, usage]
---


***주의: 이 글을 읽는 여러분이, 만약 git을 많이 써 봐서 익숙한 것이 아니라면, 반드시 손으로 직접 따라 칠 것을 권한다. 눈으로만 보면 100% 잊어버린다.***

[저번 글](https://greeksharifa.github.io/github/2018/08/19/github-usage-08-conflict/)에서는 Conflict에 대해서 알아보았다.  
이번 글에서는, 새로운 프로젝트를 시작하겠다. 저번 글에서까지의 장난스러운 커밋들 대신 의미 있는 프로그램을 만들면서, 협업에 대한 개념과 그 방법을 익히도록 한다.  
물론 이 튜토리얼 역시 복잡한 프로젝트는 아니니 안심해도 된다.

---

## 새 repository 생성

**규칙 01**: 새로운 프로젝트는 원래 있던 repo에 생성하지 말 것

그런 식으로 하나의 repo에 코드를 마구마구 쌓아가다 보면 어느 순간부터는 무슨 코드가 들어 있는지도 모르게 될 것이다.  
진짜로 reference 용도로 코드를 적어두는 용도가 아니라면, 프로젝트는 새 repo부터 시작하자.

[여기](https://greeksharifa.github.io/github/2018/06/29/github-usage-02-create-project/)를 참조하여 새 repo를 생성한다. readme는 아직 작성하지 않도록 한다.

새 프로젝트는 계산기이다. repo 이름은 `calculator`로 하겠다.  
이 계산기는, 수식을 입력받으면 그 결과를 계산해 준다. 수식에 오류가 있으면 재입력을 요구한다.

```
계산기 사용법:
1) 수식을 입력하세요.
2) 종료하려면 quit 또는 Ctrl + z 를 입력합니다.

수식을 입력하세요: (3*6) -4) +2
수식에 오류가 있습니다. 다시 입력하세요.
수식을 입력하거라: (1+5)** 2-1
답: 35
수식을 입력하세요: quit

계산기를 종료합니다.
Bye
```

계산기는 다음과 같은 방식으로 개발할 것이다.
1. 인터페이스 구현: 입출력 부분을 맡는다. 기본 모듈이 될 것이다.
2. 수식 오류 판별: 수식에 오류가 있는지를 판별하는 모듈이다.
3. 수식 정리: 불필요한 공백을 제거하고, 후위 표기법으로 변환한다. 메인 모듈이다.
4. 후위 표기법으로 된 수식을 계산하여 기본 모듈에 전달하는 모듈을 개발한다.

파이썬으로 코딩을 할 것인데, 파이썬 구문이 이해가 가지 않는다면 코드만 복붙하고 git에 대한 내용만 이해해도 충분하다.  
후위 표기법에 대한 사전 지식이 없다면 [이곳](http://home.zany.kr:9003/board/bView.asp?bCode=10&aCode=1777&cBlock=0&cPageNo=1&sType=0&sString=)을 참조한다.

여기서 여러분은 브랜치 전략으로 두 가지를 쓸 수 있다.
1. 메인 repo를 하나 만들고, 모든 팀원들은 각자 fork를 하여 자신의 GitHub 계정으로 가져온다.
    1. 이 fork된 repo는 마음대로 바꿔도 **pull request**를 하기 전까지는 메인 repo에 영향을 미치지 않으므로, 상대적으로 안전하다.
    2. 자신의 repo에서 마음껏 수정을 한 뒤 잘 정리하여 pull request란 것을 메인 repo에 날리면, 팀원들이 잘 리뷰한 뒤 merge하기에 충분하다고 생각되면 merge를 진행한다.
2. 메인 repo를 하나 만들고, 모든 팀원들은 같은 repo에 자신의 브랜치를 하나씩 만든다.
    1. 자신의 브랜치에서도 새로운 브랜치를 만들어 마음껏 작업할 수 있다. 새 브랜치를 push할 경우 메인 repo에 모두 올라간다.
    2. 같은 repo라도 master branch에 pull request를 날릴 수 있다. 그러면 다른 팀원들이 잘 검수한 뒤 문제없음을 나타내는 댓글을 단 뒤 merge를 할지 아니면 수정을 더 하라고 댓글을 달지 결정할 수 있다.

1번 방법이 더 좋기는 하지만 프로젝트가 지금처럼 상당히 작거나, 너무 귀찮거나, 단지 git 실습을 위한 것이라면 2번 방법을 써도 무방하다.  
이 글에서는 2번 방법을 사용하도록 하겠다. 귀찮아서를 제외한 두 이유에 해당하기 때문이다.

시작하기 전에, 다른 사람의 repo를 fork하는 방법을 간단히 알아보도록 하겠다.

### Fork

fork는 포크로 다른 사람의 repo를 푹 찍어와서 자신의 계정에 갖다놓는 것이라 할 수 있다. 그런데 이 포크는 마법의 포크여서 repo를 찍을 때 복사가 된다. 즉 포크는 훔쳐오는 것이 아니라 복제해오는 것이다.  
그리고 위에서 설명했듯 복제해온 repo를 마음껏 수정해도 원래 repo에는 영향을 미치지 않는다. 또 그 사람의 repo에 무언가 기여하고 싶다면 코드를 잘 짜서 pull request를 보내면, 해당 사람이 잘 검토해서 merge를 해줄 수도 있다.  
그리고 실제로 이렇게 진행되는 것이 open-source 프로젝트이다.

포크를 떠오는 방법은 간단하다. [다른 사람의 repo](https://github.com/ocasoyy/Deep_Learning)를 적당히 찾아서 들어가자.

그리고 Fork 버튼을 누른다. fork를 떠올 때는 보통 바로 옆의 Star를 눌러주는 것이 예의라고 한다.

![01_fork](/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/01_fork.PNG)

다음 화면을 보면서 잠시 기다리면... 

<center><img src="/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/02_forking.PNG" width="50%"></center>

fork된 repo를 볼 수 있다. 여기서 여러분이 마음껏 작업하면 된다. 심심하면 pull requests를 날려보라.

<center><img src="/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/03_forked.PNG" width="50%"></center>

---

이제 calculator repo로 돌아와서, master branch부터 시작하자.

---

## README.md 작성

다음과 같이 작성한다.

```markdown
### Calculator(계산기)

이 프로젝트는 git tutorial을 위해 만들어진 것입니다.

팀원이 여러 명인 것을 전제하나, 튜토리얼이기 때문에 한 사람이 여러 명인 척 하면서 프로젝트를 진행하게 될 것입니다.

조금은 *건방진* 계산기일 수 있습니다.

#### 계산기 사용법

1) 수식을 입력하세요.
2) 종료하려면 quit 또는 Ctrl + z 를 입력합니다.

```

3종 세트를 입력한다. 잊지 않았을 것이다.  
원래는 마스터 브랜치에 직접 수정을 가하지 않는 것이 원칙이지만, 처음에 프로젝트 설정할 때는 코드가 꼬일 일이 없으므로 너무 딱딱하게 적용할 필요는 없다.  
물론 나중에 README.md를 수정하게 된다면 직접 수정은 좋은 생각은 아니다.

> git add  
> git commit -m "Write README.md"  
> git push

![04_create_project](/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/04_create_project.PNG)

그러면 브라우저에 보이는 화면이 다음과 같이 변한다. `README.md` 파일은 프로젝트 개요를 정리해두는 파일이자, 패키지 같은 것이라면 사용법을 설명해주는 설명서이다. 









---




![](/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/.PNG)

![](/public/img/GitHub/2018_08_20_github_usage_09_fork_pull_request/.PNG)

---

[다음 글](https://greeksharifa.github.io/references/2018/07/13/it-will-update-soon/)에서는 github에 대한 총 정리를 다룬다.

---

## Git 명령어

[GitHub 사용법 - 00. Command List](https://greeksharifa.github.io/github/2018/06/29/github-usage-00-command-list/)에서 원하는 명령어를 찾아 볼 수 있다.
