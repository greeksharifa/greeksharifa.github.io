---
layout: post
title: Jupyter Notebook 사용법(주피터 노트북 설치 및 사용법)
author: YouWon
categories: References
tags: [Jupyter, Miniconda, usage]
---

Jupyter notebook은 대화형 파이썬 인터프리터(Interpreter)로서 웹 브라우저 환경에서 파이썬 코드를 작성 및 실행할 수 있는 툴이다.  
서버에 Jupyter notebook을 설치하여 포트를 개방한 후 해당 url에 접속하여 원격으로 사용하거나, 로컬 환경에서 브라우저를 띄워 대화형 환경에서 코드를 작성 및 실행할 수 있다.

## 설치 및 실행

### 설치

설치는 두 가지 방법이 있는데, 첫 번째는 [Anaconda](https://www.anaconda.com/distribution/)와 함께 설치하는 방법이 있다. Anaconda를 설치할 때 Jupyter Notebook도 같이 설치하면 된다.  
Anaconda와 같은 역할을 하는 Miniconda 사용법은 [여기](https://greeksharifa.github.io/references/2019/02/01/Miniconda-usage/)를 참조하도록 한다. 

Anadonda를 설치하는 방법 외에 기본적으로 pip은 Jupyter 패키지 설치를 지원한다. 설치 방법은 다른 패키지 설치 방법과 똑같다.

```
pip install jupyter
```
파이썬3과 2를 구분지어야 한다면 pip 대신 pip3를 사용한다.

### 실행 및 종료

<script data-ad-client="ca-pub-9951774327887666" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

```
jupyter notebook
```
![01_jupyter_notebook](/public/img/Jupyter/2019-01-26-Jupyter-usage/01.PNG)

위 명령을 입력하면 자동으로 어떤 html 파일을 열면서 브라우저가 실행된다. 만약 실행되지 않는다면 http://localhost:8888 으로 접속하거나 위 그림의 맨 마지막 줄에 있는 url을 복사하여 브라우저에서 접속한다. 

그러면 위 명령을 실행한 디렉토리 위치(위 그림에서 `jupyter notebook`을 실행한 줄에서 볼 수 있다. 필자의 경우 `C:\JupyterTest`)의 파일들이 브라우저에 보이게 된다.

![02_broswer](/public/img/Jupyter/2019-01-26-Jupyter-usage/02.PNG)

Jupyter의 실행을 종료하려면 명령창에서 `Ctrl + C`를 입력한다.

![03_terminate](/public/img/Jupyter/2019-01-26-Jupyter-usage/03.PNG)

### 고급: 실행 옵션

명령 옵션의 도움말을 표시한다.
```
jupyter notebook --help
```

실행 속도 상승을 위해 MathJax를 무효화할 수 있다. MathJax는 수식 입력을 위해 필요한 JavaScript 라이브러리이다.
```
jupyter notebook --no-mathjax
```

웹 브라우저를 지정하거나 실행시키지 않을 수 있다. 포트 번호 지정도 가능하다.
```
jupyter notebook --browser="safari"
jupyter notebook --no-browser
jupyter notebook --port=8889
```

노트북 실행 시 실행 디렉토리를 지정할 수 있다. 기본값은 현재 밍령창에서의 실행 위치이다.
```
jupyter notebook --notebook-dir=/user/define/directory
```

### 고급: 설정 파일 수정

매번 옵션을 지정해서 실행하기가 귀찮다면, Jupyter Notebook의 기본 설정을 변경하기 위해 다음 명령을 입력한다.
```
jupyter notebook --generate-config
```
그러면 Jupyter가 실행되는 대신 설정 파일이 열린다.  
Linux에서는 기본적으로 `/home/<username>/.jupyter/jupyter_notebook_config.py` 파일로 생성되며, 윈도우에서는 `C:\Users\<username>\.jupyter\jupyter_notebook_config.py`로 생성된다.

설정 파일에서 필요한 옵션을 변경하여 사용하면 된다. 기본적으로 사용하지 않는 옵션은 모두 주석 처리되어 있다. 

기본 설정 파일을 재지정하고 싶으면 다음과 같이 입력한다.
```
jupyter notebook --config=custom_config.py
```

임시로 설정 파일을 변경해서 실행하고 싶다면 일반 옵션 주듯이 하면 된다.
```
jupyter notebook --config custom_config.py
```


Jupyter notebook을 단순히 로컬 환경에서 실행하는 것이 아니라 서버로 띄워 놓고 원격 접속을 하려면, 위 방법으로 허용 포트나 접속 주소 등 설정 파일을 수정해야 한다. 

### 고급: 원격 접속 설정

localhost(127.0.0.1) 말고 다른 컴퓨터에서 (서버로) 원격접속하고 싶을 때가 있다. 그럴 때는 다음 과정을 따른다.

1. 명령창(cmd or terminal)에 python 또는 ipython을 입력하여 대화창을 연다.
    - 다음을 입력한다: 
        ```
        >>> from notebook.auth import passwd
        >>> passwd()
        Enter password: 
        Verity password: 
        'sha1:c5b493745105:0d26dcd6e9cf868d3b49f43d'
        ```
    - 출력으로 나온 암호화된 비밀번호를 기억해 둔다.
    - 참고로 linux에서나 윈도우에서나 passwd() 등으로 비밀번호를 입력할 때에는 명령창에 입력하고 있는 문자가 전혀 표시되지 않는다. 별표(`*`)로도 표시되지 않으니 참고.
    - 대화창을 종료한다.
2. 이제 조금 전에 생성한 `jupyter_notebook_config.py`를 편집기로 연다.
    - 아래처럼 주석처리된 부분을 다음과 같이 바꾼다. 물론 비밀번호는 조금 전 여러분이 생성한 문자열로 입력해야 한다.
        ```
        #c.NotebookApp.password = '' 
        ```
        ```
        c = get_config()
        c.NotebookApp.password = 'sha1:c5b493745105:0d26dcd6e9cf868d3b49f43d'
        ```
    - **필수:** 비슷하게 다음 설정들을 바꿔주어야 한다. 모든 설정을 변경할 때에는 앞의 주석(`#`)을 지우도록 한다.
        - 외부접속 허용: `c.NotebookApp.allow_origin = '*'`
        - IP 설정: `c.NotebookApp.ip = <여러분의 IP>`
    - **옵션:** 다음은 하고 싶으면 하도록 한다. 
        - 작업경로 설정: `c.NotebookApp.notebook_dir = <원하는 경로>`
        - 포트 설정: `c.NotebookApp.port = <원하는 port>`
        - jupyter notebook으로 실행 시 브라우저 실행 여부: `c.NotebookApp.open_browser = False`

이제 외부접속을 할 때는 서버에서 
- jupyter notebook을 실행시킨 다음
- <여러분의 IP>:<원하는 port> 형식을 브라우저의 주소창에 입력하면 된다.
    - 예시: `123.212.321.14:8888`
- 여러분이 설정한 비밀번호를 입력한다. 암호화된 문자열이 아니라 `passwd()` 에서 입력한 비밀번호면 된다.
- 물론 일반 가정집에서는 그냥 ip를 할당할 수 없기 때문에 공유기 설정을 해주거나, 회사 컴퓨터 등이라면 따로 접속 허용하는 절차를 거쳐야 한다. 이 부분은 여기서는 ~pass~
    - 그냥 되는 경우도 있다. 안 되는 경우에만 검색해서 해 보기 바람.

---


## Jupyter의 기본 사용법

### 새 파일 생성 

![04_new](/public/img/Jupyter/2019-01-26-Jupyter-usage/04.PNG)

오른쪽 부분의 `New` 버튼을 클릭하면 Python 3, Text File, Folder, Terminal 등의 옵션이 있다(파이썬 버전에 따라 Python 2가 있을 수 있다). 우선 Python 3을 클릭하여 Python 3 코드를 입력할 수 있는 창을 열도록 한다.

![05_python3](/public/img/Jupyter/2019-01-26-Jupyter-usage/05.PNG)

생성하면 맨 위에 기본적으로 Untitled라는 제목으로 생성이 된다. 파일 탐색기나 Finder 등에서도 `Untitled.ipynb`라는 파일을 확인할 수 있다.

![06_ipynb](/public/img/Jupyter/2019-01-26-Jupyter-usage/06.PNG)

위의 checkpoints 디렉토리는 자동으로 생성된다. Jupyter는 자동저장이 되고(맨 위의 autosaved), 체크포인트를 따로 설정할 수 있다.

제목은 Untitled 부분을 클릭하면 수정할 수 있다. 

### 편집 / 명령 모드

편집 모드에서는 셀의 내용을 편집할 수 있고(셀의 테두리가 초록색), 명령 모드는 편집중이 아닌 상태 또는 셀 자체에 조작을 가하는 상태(셀의 테두리가 파란색)이다.  
명령 모드에서 편집 모드로 들어가려면 `Enter`키를, 반대로는 `Esc` 키를 누르면 된다.

### 셀의 타입

Code 타입, Markdown 타입이 있다.  
Code 타입은 일반 코드를 실행할 수 있는 셀이다. 기본적으로 셀을 생성하면 Code 타입으로 생성된다.  
Markdown 타입은 [Markdown](https://greeksharifa.github.io/references/2018/06/29/markdown-usage/)으로 셀의 내용을 작성할 수 있다. 코드로 실행되지는 않으며, 수식을 작성할 수 있다. 수식은 MathJax에 의해 지원된다. 수식 작성 방법은 [여기](https://greeksharifa.github.io/references/2018/06/29/equation-usage/)를 참고한다.

Markdown 타입으로 변경하면 Markdown 코드를 작성할 수 있다. `Shift + Enter` 키를 누르면 마크다운이 실제 보여지는 방식으로 변경되며, 다시 수정하려면 `Enter` 또는 `더블 클릭`하면 편집 가능하다.

![07_markdown](/public/img/Jupyter/2019-01-26-Jupyter-usage/07.PNG)

### 셀 실행

실행하고 싶은 셀의 아무 곳에나 커서를 위치시킨 후 `Shift + Enter` 키를 누른다.  
실행하면 셀 아래쪽에는 실행 결과가 표시되고, 셀 옆의 'In [ ]'과 'Out [ ]'에 몇 번째로 실행시켰는지를 나타내는 숫자가 표시된다. 여러 번 실행하면 계속 숫자가 올라간다.

![08_run](/public/img/Jupyter/2019-01-26-Jupyter-usage/08.PNG)

### 강제 중단 / 재실행

제목 아래 줄의 탭에 Kernel 탭이 있다. 커널은 IPython 대화창 아래에서 백그라운드 비슷하게 실행되는 일종의 운영체제 같은 개념이다. IPython 대화창을 관리한다고 보면 된다. 

![09_interrupt](/public/img/Jupyter/2019-01-26-Jupyter-usage/09.PNG)

Kernel 탭의 모든 버튼은 코드를 삭제하지는 않는다. 각 버튼의 기능을 설명하면, 

- Interrupt: 실행 중인 코드를 강제 중지한다. 중지하면 위 그림과 같은 에러가 뜨며 실행이 중지된다.
- Restart: 실행 중인 코드가 중지되며 재시작된다. 코드나 실행 결과는 삭제되지 않는다.
- Restart & Clear Output: 코드는 중지되며 실행 결과도 삭제한다. 
- Restart & Run All: 재시작 후 모든 셀의 코드를 위에서부터 순차적으로 한 번씩 실행한다.
- Reconnect: 인터넷 연결이 끊어졌을 때 연결을 재시도한다.
- Shutdown: 커널을 종료한다. 이 버튼을 누르면 실행 결과는 삭제되지 않으나 완전 종료된 상태로 더 이상 메모리를 잡아먹지 않는다.

Shutdown되었거나, 인터넷 연결이 끊어졌거나, 기타 문제가 있으면 아래와 같이 탭 옆에 알림이 표시된다. Shutdown 된 경우 No kernel이라고 뜬다.

![10_shutdowned](/public/img/Jupyter/2019-01-26-Jupyter-usage/10.PNG)

현재 실행중인 커널이 있는지 확인하는 방법은 두 가지다. 첫 번째는 Home 화면에서 `ipynb` 파일의 아이콘이 초록색이면 실행중, 회색이면 중단된 또는 시작되지 않은 상태이다. 여기서는 해당 디렉토리에서 실행중인 것만 확인할 수 있다.

![11_shutdowned](/public/img/Jupyter/2019-01-26-Jupyter-usage/11_shutdowned.PNG)

또 하나는 Home 화면에서 Files 탭 대신 Running 탭을 클릭하면 실행 중인 IPython과 터미널의 목록을 확인할 수 있다. 이 탭에서는 전체 디렉토리에서 실행중인 파일 또는 터미널을 전부 볼 수 있다.

![12_list](/public/img/Jupyter/2019-01-26-Jupyter-usage/12.PNG)

### Text File 생성

New 버튼에서 Text File을 생성하면 `.txt` 파일이나 `.py` 파일 등을 만들 수 있다. 이렇게 만든 파일은 대화 형식으로 실행되지 않고, 터미널에서 실행시켜야 한다. 읽는 것은 IPython 창에서도 가능하다.

### Folder 생성

디렉토리를 생성할 때 사용한다. 폴더랑 같은 것이다.

### 터미널

New 버튼으로 Terminal을 클릭하면, 터미널을 하나 새로 연다. 이것은 윈도우나 맥 등의 명령창(cmd 또는 terminal)과 같다. 여기서 .py 파일을 실행시킬 수 있고, 파일의 목록을 보거나 삭제하는 등의 명령이 모두 가능하다. Running 탭에서 중지시킬 수 있다.

![13_terminal](/public/img/Jupyter/2019-01-26-Jupyter-usage/13.PNG)

### 파일 이름 변경 또는 삭제

파일 맨 왼쪽의 체크박스를 클릭하면 복제, 수정, 삭제 등이 가능하다. 물론 로컬 파일 탐색기에서 수정이나 삭제를 해도 되며, 서버가 연결에 문제가 없으면 바로 반영된다.

![14_name](/public/img/Jupyter/2019-01-26-Jupyter-usage/14.PNG)

### 자동완성

웬만한 IDE에는 다 있는 자동완성 기능이다. 변수나 함수 등을 일부만 입력하고 `Tab` 키를 누르면 된다. 따로 설명할 필요는 없을 듯 하다.

---

## 단축키

단축키 정보는 [Help] - [Keyboard Shortcuts] 또는 명령 모드에서 `H`를 눌러서 표시할 수 있다.

공용 단축키 | 설명
-------- | --------
Shift + Enter | 액티브 셀을 실행하고 아래 셀을 선택한다.
Ctrl + Enter | 액티브 셀을 실행한다.
Alt + Enter | 액티브 셀을 실행하고 아래에 셀을 하나 생성한다.

편집 모드 단축키 | 설명
-------- | --------
Ctrl + Z | Undo 명령이다.
Ctrl + Shift + Z | Redo 명령이다.
Tab | 자동완성 또는 Indent를 추가한다.
Shift + Tab | 툴팁 또는 변수의 상태를 표시한다.
Ctrl + Shift + - | 커서의 위치에서 셀을 잘라 두 개로 만든다.

참고로 명령 모드 단축키 중 콤마(`,`)로 되어 있는 것은 연속해서 누르라는 의미이다. 예로 `D`를 두 번 누르면 액티브 코드 셀을 삭제한다.

명령 모드 단축키 | 설명
-------- | --------
↑, ↓ | 셀 선택
A | 액티브 코드 셀의 위(Above)에 셀을 하나 생성한다.
B | 액티브 코드 셀의 위(Below)에 셀을 하나 생성한다.
Ctrl + S | Notebook 파일을 저장한다.
Shift + L | 줄 번호 표시를 토글한다.
D, D | (D 두번 연속으로 타이핑)액티브 코드 셀을 삭제한다.
Z | 삭제한 셀을 하나 복원한다.
Y | 액티브 코드 셀을 Code 타입(코드를 기술하는 타입)으로 한다. 
M | 액티브 코드 셀을 Markdown 타입으로 한다. 
O, O | 커널을 재시작한다.
P | 명령 팔레트를 연다.
H | 단축키 목록을 표시한다. `Enter` 키로 숨긴다.

---

## Jupyter의 기능

### DocString의 표시

선언한 변수 뒤에 `?`를 붙여서 셀을 실행하는 것으로 해당 변수의 상태를 확인할 수 있다.

약간 다른 방법으로 변수를 타이핑한 후 `Shift + Tab`을 누르면 툴팁이 표시된다.  
툴팁에는 DocString의 일부 내용이 표시된다.

### 이미지 첨부하기

Drag & Drop으로 첨부할 수 있다.

### shell(명령 프롬프트)의 이용

명령창에서 쓰는 명령을 그대로 쓰되, 맨 앞에 `!`를 입력하여 사용 가능하다.

```
!cd Documents
```

### 매직 명령어 이용

맨 앞에 `%`를 붙이고 특정 명령을 수행할 수 있다. 이는 파이썬 문법에는 포함되지 않은, Jupyter notebook만의 기능이다.

![15_magic](/public/img/Jupyter/2019-01-26-Jupyter-usage/15.PNG)

매직 명령어 | 설명
-------- | --------
%pwd | 현재 디렉토리 경로 출력
%time `코드` | `코드`의 실행 시간을 측정하여 표시
%timeit `코드` | `코드`를 여러 번 실행한 결과를 요약하여 표시
%history -l 3 | 최근 3개의 코드 실행 이력 취득
%ls | 윈도우의 dir, Linux의 ls 명령과 같음
%autosave `n` | 자동저장 주기를 설정한다. 초 단위이며, 0이면 무효로 한다.
%matplotlib | 그래프를 그리는 코드 위에 따로 설정한다. `%matplotlib inline`으로 설정하면 코드 셀의 바로 아래에, `%matplotlib tk`로 설정하면 별도 창에 그래프가 출력된다. `%matplotlib notebook`으로 하면 코드 셀 바로 아래에 동적으로 그래프를 조작할 수 있는 그래프가 생성된다.

```python
# 코드 실행 시간 측정
%time sum(range(10000))
# 결과:
# CPU times: user 225 us, sys: 0 ns, total: 225 us
# Wall time: 228 us
# 499950000

# 1000회 반복, 3회 실행
%timeit sum(range(10000))
# 결과: 
# 1000 loops, best of 3: 238 us for loop

# 옵션 지정하기
%timeit -n 2000 -r 5 sum(range(10000))

# 셀 전체의 시간 측정
%%timeit -n 1000 -r 3
s = 0
for i in range(10000):
    s += i
```

