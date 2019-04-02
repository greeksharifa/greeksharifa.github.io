---
layout: post
title: PyCharm 사용법
author: YouWon
categories: References
tags: [usage]
---

PyCharm은 Jetbrains 사에서 제작 및 배포하는 **유료** 프로그램이다.

---

## 설치

PyCharm 홈페이지에서 설치 파일을 다운받는다. 
[Windows](https://www.jetbrains.com/pycharm/download/#section=windows), [Mac](https://www.jetbrains.com/pycharm/download/#section=mac), [Linux](https://www.jetbrains.com/pycharm/download/#section=linux)

유료 버전을 구매했거나 학생 인증이 가능하다면, Professional 버전을 다운받도록 한다.

### Settings

설치 시 다음 창을 볼 수 있다. 해당 컴퓨터에 설치한 적이 있으면 설정 파일 위치를 지정하고, 아니면 만다.


<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/01.PNG" width="70%"></center>

필자는 Darcula로 지정했고, 왼쪽 아래의 `Skip Remaining and Set Defaults` 버튼을 누른다. 본인이 추가 설정하고 싶은 부분이 있으면 이후 설정에서 마음대로 바꾸면 된다.

![02_install](/public/img/PyCharm/2019-02-07-PyCharm-usage/02.PNG)

설정을 완료하면 아래와 같은 화면을 볼 수 있다. 오른쪽 아래의 `Configure` > `Settings` 를 클릭한다.

![03_install](/public/img/PyCharm/2019-02-07-PyCharm-usage/03.PNG)

정확히는 `Settings for New Projects`라는 대화창을 볼 수 있다. 이는 새 프로젝트를 만들 때 적용되는 **기본 설정**이다. 새로운 설정을 만들고 싶다면 `Default` 설정을 복제(Duplicate)한 뒤 새 설정에서 바꾸도록 한다.

![04_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/04.PNG)

설정에서 `Appearance & Behavior` > `Appearance`에서, `Theme`를 `Darcula` 또는 다른 것으로 지정할 수 있다. 아래의 `Use Custom Font`는 메뉴 등의 폰트를 해당 폰트로 지정할 수 있다.  
참고로, 코드의 폰트는 `Editor` > `Font`에서 지정한다. 이 두 가지 역시 구분하도록 한다. 기본값을 `Monospaced`이다.

![05_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/05.PNG)

`Keymap`에서는 단축키를 지정할 수 있다. PyCharm의 기본 단축키는 타 프로그램과 좀 다른 부분이 많아 필자는 일부를 바꿨다.  
변경하고 싶은 단축키를 찾아서 더블클릭 또는 우클릭하면 기존에 지정되어 있는 단축키를 삭제하고 새 단축키를 지정할 수 있다. 이때 겹친다면 기존 단축키를 남겨둘지 제거할지 선택할 수 있다.

![06_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/06.PNG)

추천하는 변경할 단축키는 다음과 같다. 

Menu | 변경 전 | 변경 후
-------- | -------- | --------
Execute selection in console | Alt + Shift + E | Ctrl + Enter
Edit > Find > Replace | Ctrl + H | Ctrl + R
Refactor > Rename | Shift + F6 | F2
Other > Terminal | | Alt + T
Other > Python Console | | Alt + 8
Other > SciView | | Alt + 0
Show in Explorer | | Ctrl + Alt + Shift + E

필자의 경우 나머지 설정은 그대로 두는 편이나, `Ctrl + Enter`로 바꿀 때는 다른 곳에 할당된 것을 지운다(Already assigned 경고창에서 Leave 대신 Remove를 선택). 안 그러면 선택한 부분이 Python Console(대화형)에서 실행되지 않는다.

![07_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/07.PNG)

위 그림에서 기본 Python Interpreter 파일(python.exe)를 설정한다. 새 프로젝트를 생성 시 Configure Python Interpreter라는 경고가 보이면서 코드 실행이 안 되면 인터프리터가 설정되지 않은 것이다. 컴퓨터에 설치된 파이썬 파일을 찾아 설정하자.

![08_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/08.PNG)

`Show All...`을 클릭하면 처음에는 빈 창이 보인다. `+`를 눌러서 원하는 환경을 추가한다. 기존의 것을 추가하거나, 새로운 가상환경(virtualenv 또는 conda)를 즉석에서 생성 가능하다.  
이렇게 만든 가상환경은 해당 프로젝트에서만 쓰거나(기본 설정), 아래쪽의 `Make available to all projects`를 체크하여 다른 프로젝트에서도 해당 인터프리터를 택할 수 있도록 정할 수도 있다.

![09_settings](/public/img/PyCharm/2019-02-07-PyCharm-usage/09.PNG)

PyCharm에서 코드 실행을 대화형으로 하면 Python Console에 자꾸 `Special Variables`라는 창이 뜨는 것을 볼 수 있다. 보통 쓸 일이 없는데 기본으로 표시되는 것이므로, `Build, Execution, Deployment` > `Console`에서 `Show console variable by default` 체크를 해제한다.

해당 설정을 마쳤으면 첫 화면에서 `Create New Project`를 클릭한다.

![10_new_projects](/public/img/PyCharm/2019-02-07-PyCharm-usage/10.PNG)

프로젝트 이름은 기본적으로 Untitled 이므로 바꿔주고, 아래쪽의 Project Interpreter를 설정해 둔다. 미리 설정했다면 목록이 보일 것이고, 아니라면 새로 생성하거나 `python.exe` 위치를 찾아 지정해준다.

### Sync Settings

시작 화면에서 `Configure` > `Settings Repository...`, 또는 프로젝트 생성 후 `File` > `Settings Repository...` 를 클릭하면 지금까지 설정한 설정들을 git repository에 저장할 수 있다. git을 알고 있다면, Merge, Overwrite Local, Overwrite Remote의 뜻을 알 것이라 믿는다. git repository에 저장하면 컴퓨터를 옮겨도 동일한 설정을 쉽게 지정할 수 있다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/11.PNG" width="80%"></center>

이를 지정하려면 Personal Access Token이 필요하다. [여기](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)를 참조한다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/12.PNG" width="70%"></center>

등록이 완료되면 Merge, Overwrite Local(git에 저장된 내용을 local로 덮어씀), Overwrite Remote(현재 local 설정을 인터넷에 덮어씀) 중 하나를 선택해 설정을 동기화할 수 있다.

여기까지 초기 설정이 끝났다(원하는 부분만 진행해도 좋다). 이제 PyCharm 프로젝트 화면을 살펴보도록 하자.

---

## Project 창(`Alt + 1`)

처음 프로젝트를 열면 다음과 같은 화면이 보일 것이다. (Show tips at startup은 무시한다)

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/13.PNG" width="100%"></center>

맨 왼쪽에는 프로젝트 창이 있다. 맨 왼쪽 빨간 박스로 표시한 곳을 클릭하면 프로젝트 창을 접었다 폈다 할 수 있다. 단축키를 눌러도 된다(Alt + 1).  

필자는 현재 untitled라는 이름으로 프로젝트를 생성했기 때문에, 루트 폴더는 현재 untitled이다. 주황 박스를 오른쪽 클릭하면 꽤 많은 옵션이 있다. 참고로 프로젝트 내 모든 디렉토리 또는 파일에 오른쪽 클릭하여 기능을 쓸 수 있다. 디렉토리를 우클릭했을 때와 파일을 우클릭했을 때 옵션이 조금 다르다.
<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/14.PNG" width="70%"></center>

각 옵션을 대략 설명하면,
- New: File, Directory, Python File(`.py`), Jupyter Notebook(`.ipynb`) 등을 생성한다. 자주 쓰는 기능이지만 안타깝게도 단축키는 설정이 어려운 것 같다.
- Cut, Copy, Paste 등은 설명하지 않겠다.
- Copy Parh, Copy Relative Path: 각각 해당 디렉토리 또는 파일의 절대/상대 경로를 복사한다. 이미지나 데이터 파일 등의 경로를 써야 할 때 유용하게 쓸 수 있다. 단, 사용 환경에 따라 디렉토리 구분자가 `/`, `\`, `//` 등으로 달라지는 경우가 있으니 주의.
- Refactor: 해당 디렉토리 또는 파일의 이름을 변경한다. 이때 이 파일명을 사용하는 코드(file open 등)이 있으면 그 코드를 자동으로 수정하게 할 수 있다.
- Find Usages: 해당 파일을 참조하는 코드를 살펴볼 수 있다. Refactor와 같이 사용하면 좋다.
- Show in Explorer: 해당 디렉토리나 파일이 있는 디렉토리를 탐색기나 Finder 등에서 열 수 있다.
- Mark Directory as: 디렉토리의 속성을 설정한다. 세부 옵션이 4개 있다.
    - Sources Root: 프로젝트에서 코드의 최상위 폴더를 지정한다. 코드를 짜다 보면 프로젝트 루트 폴더에 직속된 파일이 아닌 경우 패키지나 파일 reference를 찾지 못하는 경우가 많은데, 그럴 때는 해당 코드를 포함하는 파일 바로 상위의 디렉토리를 Sources Root로 설정하면 빨간 줄이 사라지는 것을 볼 수 있다.
    - Excluded: PyCharm 색인(Index)에서 제외시킨다. PyCharm은 Find Usages와 같은 기능을 지원하기 위해 프로젝트 내 모든 파일과 코드에 대해 indexing을 수행하는데(목차를 생성하는 거랑 비슷함), 프로젝트 크기가 크면 굳이 필요 없는 수많은 파일들까지 indexing해야 한다. 이는 PyCharm 성능 저하와 함께 색인 파일의 크기가 매우 커지므로(임시 파일까지 포함하여 수 GB까지 되기도 함) 너무 많으면 적당히 제외시키도록 하자.
    - Resource Root: 말 그대로 Resource Root로 지정한다. 
    - Template Folder: 템플릿이 있는 폴더에 지정하면 된다. Pure Python을 쓸 때에는 별 의미 없다.
- Add to Favorites: Favorites창에 해당 디렉토리나 파일을 추가한다. 즐겨찾기 기능이랑 같다. 프로젝트 창 아래에서 창을 찾을 수 있고, `Alt + 2` 단축키로 토글할 수 있다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/14_1.PNG" width="60%"></center>

### 새 파일 생성

이제 우클릭 > New > Python File로 새 파이썬 파일을 하나 생성하자. (현재 프로젝트 이름은 `PythonTutorial`이다)

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/15.PNG" width="100%"></center>

안타깝게도 새 Python 파일 생성을 위한 단축키는 지정할 수 없는 듯하다.

---

## 코드 실행(전체 또는 선택)

그리고 원하는 파일명을 입력한다. 필자는 `tutorial`이라고 입력하겠다. 그러면 파일명은 `tutorial.py`가 될 것이다.

이제 코딩할 수 있는 창이 열렸으니 코드를 입력하자.

```python
print('Hello Pycharm!')
```

코드를 작성했으면 실행을 해 보아야 하지 않겠는가? 실행하는 방법은 여러 가지가 있다.

- 실행하고 싶은 코드 라인에 커서를 놓거나 실행할 만큼 드래그를 한 다음 [위](https://greeksharifa.github.io/references/2019/02/07/PyCharm-usage/#settings)에서 단축키를 바꿨다면 `Ctrl + Enter`, 바꾸지 않았다면 `Alt + Shift + E`를 누른다. 그러면 `Python Console`이라는 창이 아래쪽에 열리면서 실행한 코드와 실행 결과가 나타난다. 역시 단축키를 설정했다면 `Alt + 8`로 열 수 있다. PyCharm Default settings에는 단축키가 할당되어 있지 않다.
    - 이것은 정확히는 Interpreter라고 부르는 대화형 파이썬 창에서 실행시키는 것이다. 명령창(cmd, terminal)에서 `python`을 실행시키고 코드를 입력하는 것과 같은 형태이다. [Jupyter notebook](https://greeksharifa.github.io/references/2019/01/26/Jupyter-usage/)과도 비슷하다.
    - 장점은 명령창에서 바로 입력하는 경우 오타가 나면 다시 입력해야 하는데 편집기에 코드를 써 놓고 필요한 만큼만 `Ctrl + Enter`로 실행시키는 이 방식은 코드 수정과 재사용이 훨씬 편하다는 것이다.
    - 콘솔에 문제가 있거나 해서 현재 실행창을 재시작하고 싶으면 `Python Console` 왼쪽 `Rerun` 버튼(화살표)을 누르거나 `Ctrl + F5`를 입력한다.
    - 참고로 PyCharm 아래쪽/왼쪽/오른쪽에 있는 창들 중에서 옆의 숫자는 단축키를 간략하게 나타낸 것이다. 예를 들어 필자는 좀 전 설정에서 `Python Console` 창의 단축키를 `Alt + 8`로 설정해 놨는데, 그래서 옆에 `8` 이라는 숫자가 표시된다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/16.PNG" width="100%"></center>

- Run > Run...을 누르면 실행시키고 싶은 파일 목록이 나타난다. 이 중 원하는 파일(현재는 `tutorial`)을 선택하면 `Terminal`이라는 창에서 ***해당 파일의 전체 코드***가 실행된다.
    - 다시 실행할 때는 Run > Run을 선택하면 마지막으로 실행한 파일이 전체 실행된다. 
    - 아래 그림의 `Terminal` 창 왼쪽의 `ReRun` 버튼을 눌러도 마지막으로 실행한 파일이 다시 실행된다. 단축키는 `Ctrl + F5`이다.
    - PyCharm 오른쪽 위에서도 실행할 파일을 선택 후 실행시킬 수 있다. 

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/17.PNG" width="100%"></center>

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/18.PNG" width="100%"></center>

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/19.PNG" width="80%"></center>

- PyCharm 아래쪽의 `Terminal` 창을 클릭하거나 `Alt + T` 단축키(바꾼 것이다)로 `Terminal` 창을 열어서 `python tutorial.py`를 입력한다. 
    - 그렇다. Python 파일 실행 방법과 똑같다. 이 `Terminal` 창은 명령창(cmd 또는 터미널)과 똑같다.
    - 대략 `tu` 정도까지만 입력하고 `Tab` 키를 누르면 파일명이 자동완성된다. 
    - 이 방법도 역시 해당 파일에 들어있는 모든 코드를 전체 실행시킨다.
    - 터미널 창 답게 여러 개의 세션을 열어 놓을 수 있다. 기본적으로 `Local`이라는 이름의 탭이 생성되며, 오른쪽의 `+` 버튼을 클릭하라.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/20.PNG" width="80%"></center>

- `Project` 창에서도 해당 파일을 `우클릭 > Run (파일명)`을 클릭하면 해당 파일의 코드 전체가 실행된다.
- 편집 창에서도 파일명 탭을 `우클릭 > Run (파일명)`해도 된다. 실행 방법은 많다.

---

## 편집 창(코드 편집기)

코드를 편집하는 부분에도 여러 기능들이 숨어 있다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/27.PNG" width="100%"></center>

위 그림의 오른쪽 부분을 보자. 경고인 듯한 느낌표와 함께 여러 색깔의 줄이 있다. 현재 커서는 9번째 라인의 `example` 변수에 위치해 있다.

- 먼저 왼쪽에는 줄 번호(line number)라는 것을 다들 알 수 있을 것이다.
    - 하지만 이 단축키는 모르는 사람이 많다. `Ctrl + G`를 누르면 원하는 라인으로 이동할 수 있다. 줄의 어느 부분으로 이동할지도 `line:column` 형식으로 정할 수 있다. 줄 번호만 지정하고 싶으면 그냥 숫자만 입력하라.
- 빨간 화살표가 가리키고 있는 경고 표시는 현재 이 파일에 **syntax error**가 있다는 뜻이다. 메인 화면에도 해당 부분에는 빨간 줄이 그어진다(`printf`). 그리고 오른쪽에도 빨간색 bar가 생긴다. 
    - 이 bar들은 현재 파일에서의 상대적 위치를 뜻한다. 즉, 예를 들어 맨 아래에 있는 오류 코드가 화면에 안 보이더라도 bar는 제일 아래쪽 근처에 표시된다.
- 커서가 위치한 곳이 변수나 함수 등이라면 해당 파일의 모든 부분에서 같은 이름을 가진 변수(또는 함수)에는 옅은 초록색 배경색이 칠해진다. 그리고 해당 변수(함수)가 선언된 곳에는 옅은 주황색으로 배경색이 칠해진다(이 색깔은 `Settings`에서 바꿀 수 있다). 어디서 사용되고 있는지 쉽게 알 수 있다. 그리고 그림에서 오른쪽에도 주황색 또는 초록색 짧은 bar가 생긴 것을 볼 수 있다.
    - 옅어서 잘 안보인다면 색깔을 바꾸거나 아니면 Find and Replace(`Ctrl + H`)로 찾으면 더 선명하게 표시되기는 하는데, 해당 이름을 포함한 다른 변수 등도 같이 선택된다는 문제가 있다. 적당히 선택하자.
- 특별히 ***TODO*** 주석문은 일반 회색 주석과는 다르게 연두색으로 눈에 띄게 칠해진다. 또한 오른쪽에 파란색 bar가 생긴다. 이 주석은 참고로 `TODO` 창(`Alt + 6`)에서도 확인 가능하다. 못다한 코딩이 있을 때 쓸 수 있는 좋은 습관이다.

편집 창의 아무 부분을 우클릭하여 `Local History > Show History`를 클릭하면 해당 파일이 어떻게 수정되어 왔었는지가 저장된다. 잘 안 쓸 수도 있지만 잘못 지운 상태로 코딩을 좀 진행했다거나 하는 상황에서 쓸모 있는 기능이다. 

---

### 빠른 선택, 코드 정리, 편집 등등 단축키

원하는 부분을 빠르게 선택할 수 있는 단축키는 많다. 이를 다 알고 빠르게 할 수 있다면 코딩 속도는 아주 빨라진다.  

- 변수/함수 더블클릭: 해당 변수 이름 선택
- `Ctrl + Z`: 실행 취소(Undo)
- `Ctrl + Shift + Z`: 재실행(Redo)
- `Ctrl + D`(Duplicate): 현재 커서가 있는 한 줄(또는 드래그한 선택 범위)을 복사해 아래에 붙여 넣는다.
- `Ctrl + X` / `Ctrl + C`: 현재 커서가 있는 한 줄(또는 드래그한 선택 범위)을 잘라내기/복사한다. 한 줄도 된다는 것을 기억하라.
- `Ctrl + W`: 현재 선택 범위의 한 단계 위 범위를 전체 선택한다. 무슨 말인지 모르겠다면 직접 해 보면 된다. 범위는 블록이나 괄호 등을 포함한다.
- `Tab`: 현재 커서가 있는 한 줄(또는 드래그한 선택 범위)를 한 단계(오른쪽으로 이동) indent한다.
- `Shift + Tab`: 현재 커서가 있는 한 줄(또는 드래그한 선택 범위)를 반대 방향으로(왼쪽으로 이동) 한 단계 indent한다.
- `Ctrl + A`: 현재 파일의 코드를 전체선택한다.

- `Ctrl + Shift + O`(Import Optimization): 코드 내에 어지럽게 널려 있는 import들을 파일 맨 위로 모아 잘 정리한다.
- `Ctrl + Shift + L`: 코드의 빈 줄, indentation 등을 한 번에 정리한다.

- `Ctrl + 좌클릭`: 해당 변수/함수가 선언된 위치로 화면/커서가 이동한다. 변수가 어떻게 정의됐는지 또는 함수가 어떻게 생겼는지 보기 유용하다.

이외에도 기능은 정말 많다(Toggle Case, Convert indents to space/tab, Copy as Plain Text, Paste without Formatting, ...). 한번 잘 찾아보자.

---

### 찾기(및 바꾸기), (`Ctrl + F | Ctrl + H`)

찾기 및 바꾸기의 기본 단축키는 `Ctrl + R`이다(**R**eplace). 많은 다른 프로그램들은 `Ctrl + H`를 쓰기 때문에 바꾸는 것도 좋다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/28.PNG" width="100%"></center>

여기도 여러 기능들이 있다. `찾기` 설명은 찾기 및 바꾸기의 설명 안에 포함되므로 생략하겠다.  
아래에서 설명할 기능들은 모두 그림에 나온 버튼이나 체크박스 등에 대한 것이다. 

- 왼쪽 검색창에 찾고자 하는(또는 대체될) 문자열 또는 정규식을 입력한다. 아래쪽 창에는 대체할 문자열을 입력한다.
    - 왼쪽 돋보기를 클릭하면 이전에 검색했던 문자열들을 재검색할 수 있다.
- `F3`: 다음 것 찾기
- `Shift + F3`: 이전 것 찾기
- Find All: 전부 다 찾아서 보여준다.
- Select All Occurrences: 매칭되는 결과를 전부 선택한다.
- Show Filter Popup: 찾을 범위를 지정할 수 있다. 전부(Anywhere), 주석에서만(In comments), 문자열에서만(In String Literals), 둘 다에서만, 혹은 제외하고 등의 필터를 설정 가능하다.
- Match Case: 체크하면 대소문자를 구분한다.
- Words: 정확히 단어로 맞아야 할 때(해당 문자열을 포함하는 단어를 제외하는 등) 체크한다.
- Regex: [정규표현식](https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/)을 사용하여 찾는다. 잘 쓸 줄 안다면 아주 좋다.
- 오른쪽에는 몇 개나 매칭되는 문자열을 찾았는지 보여준다(3 matches). 만약 하나도 없으면 문자 입력 창이 빨갛게 되면서 No matches라고 뜬다.
- Replace(`Alt + p`): 현재 선택된 부분을 대체한다..
- Replace all(`Alt + a`): 매칭되는 모든 문자열을 찾아 대체한다.
- Exclude: 해당 매칭된 부분은 대체할 부분에서 제외한다.
- Preserve Case: 대체 시 대소문자 형식을 보존한다.
- In Selection: 파일 전체가 아닌 선택한 부분에서만 찾는다.

### 더 넓은 범위에서 찾기

선택한 파일 말고 더 넓은 범위에서 찾으려면 `Ctrl + Shift + F`를 누르거나 다음 그림을 참고한다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/29.PNG" width="80%"></center>

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/30.PNG" width="80%"></center>

위의 Match Case등은 사용법이 똑같지만, 여기서는 파일뿐 아니라 프로젝트 전체, 모듈, 디렉토리, 또는 특정 범위(scope)에서 찾을 수 있다. `Edit > Find > `안의 다른 선택지들 역시 사용법은 크게 다르지 않으니 참고하자.

다른 (범용) 찾기 단축키로 `Shift + Shift`(Shift 키를 두번 누름)이 있다. 한번 해 보자.

### 변수/함수 등이 사용된 위치 찾기

찾고자 하는 변수/함수를 우클릭하여 `Find Usages`를 클릭하거나 `Alt + F7`을 누른다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/31.PNG" width="100%"></center>

그러면 해당 변수/함수가 어디서 사용되었는지 정보가 전부 나온다. 왼쪽에 있는 많은 버튼들로 적절한 그룹별로 묶거나 하는 등의 작업을 할 수 있다.

### Refactor(이름 재지정)

변수명을 바꾸고 싶어졌을 때가 있다. 무식하게 일일이 바꾸거나, 아니면 `Find and Replace`로 선택적으로 할 수도 있다.

하지만 매우 쉽고 편리한 방법이 있다. 해당 변수를 선택하고 `Shift + F6`을 누른다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/32.PNG" width="100%"></center>

원하는 이름으로 바꾸고 `Refactor`을 누르면 해당 변수만 정확하게 원하는 이름으로 바뀐다. 심지어 import해서 사용한 다른 파일에서도 바뀐다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/33.PNG" width="100%"></center>

*아주 편리하다.*

---

## 창 위치 및 크기 변경

`Python Console` 등의 창은 위치나 크기가 변경 가능하다. 크기는 창의 경계에 마우스 커서를 갖다대는 방식이니 굳이 설명하지 않겠다.  
위치는 탭을 끌어서 이동시키거나 아니면 `우클릭 > Move to > 원하는 곳`을 선택하면 된다.

또 모니터를 2개 이상 쓴다면 `View Mode`에서 해당 설정을 변경할 수 있다. 기본은 PyCharm 내부에 위치 고정된 `Dock Pinned` 모드이다. `Float`이나 `Window`를 선택하면 위치를 자유롭게 이동할 수 있다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/21.PNG" width="80%"></center>

모니터 크기는 충분한데 코드는 위아래로만 길게 보여서 공간이 아까웠다면, PyCharm에서는 굳이 그럴 필요 없다. Vim의 Split View와 비슷한 기능이 있다.

편집 창(메인 화면)의 탭을 우클릭한 다음 `Split Vertically`를 클릭해 보라. `Split Horizontally`도 괜찮다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/23.PNG" width="80%"></center>

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/24.PNG" width="80%"></center>

동일한 파일을 여러 번 열고 다른 부분을 보는 것도 가능하다. 꽤 유용한 기능이다.

---

## Favorites 창(`Alt + 2`)

`Alt + 2`를 눌러 `Favorites` 창을 연다.

말 그대로 즐겨찾기이다. 자주 사용하는 파일을 `Favorites`에 등록할 수 있다. 기본적으로 현재 프로젝트 이름으로 리스트가 하나 생성되어 있다.  
이게 싫거나 새로운 리스트를 추가하고 싶으면 아래 그림의 오른쪽에 보이는 `Add to New Favorites List`를 클릭하라.

그러면 `Favorites` 창에 해당 리스트에 추가한 파일이 등록된다. 이제 프로젝트 창에서 찾을 필요 없이 바로 파일을 열어볼 수 있다.

<center><img src="/public/img/PyCharm/2019-02-07-PyCharm-usage/25.PNG" width="100%"></center>

---

## Run 창(`Alt + 4`)

`Alt + 3`은 기본적으로 할당되어 있지 않다. 추가하고 싶으면 추가하라.

`Run` 창은 조금 전 코드를 실행할 때 본 것이다. 여기서는 왼쪽에 몇 가지 버튼이 있는데, 각각

- Rerun(마지막 실행 파일 재실행)
- Stop(현재 실행 중인 파일 실행 중단)
- Restore Layout(레이아웃 초기화)
- Pin Tab(현재 실행 탭 고정)
- Up/Down to Stack Trace(trace 상에서 상위 또는 하위 단계로 이동)
- Soft-Wrap(토글 키. 활성화 시 출력 내용이 한 줄을 넘기면 아래 줄에 출력됨. 비활성화 시 스크롤해야 나머지 내용이 보인다)
- Scroll to the end(제일 아래쪽으로 스크롤)
- Print(출력 결과를 정말 프린터에서 뽑는 거다)
- Clear All(현재 출력 결과를 모두 지우기)

Soft-wrap 등은 꽤 유용하므로 잘 사용하자.

`Run` 창을 우클릭 시 `Compare with Clipboard` 항목이 있는데, 현재 클립보드에 있는(즉, `Ctrl + C` 등으로 복사한) 내용과 출력 결과를 비교하는 창을 띄운다. 정답 출력 결과를 복사해 놨다면 유용하게 쓸 수 있다.

---

## TODO 창(`Alt + 6`)










---

## References

[공식 홈페이지](https://www.jetbrains.com/pycharm/)에서 더 자세한 사용법을 찾아볼 수 있다.
