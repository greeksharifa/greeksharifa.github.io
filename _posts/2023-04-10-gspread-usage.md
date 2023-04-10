---
layout: post
title: gspread 사용법(python gspread - google spreadsheet)
author: YouWon
categories: References
tags: [gspread, usage]
---

구글 스프레드시트는 어떤 결과 같은 것을 정리하기에 꽤 괜찮은 도구이다.  
그렇다면 파이썬으로 실험한 결과를 스프레드시트에 자동 기록하게 한다면..?

이 글에서는 python **gspread**를 이용해 자동으로 구글 스프레드시트에 값을 기록하는 방법을 정리한다.

---

## gspread 설치

설치는 꽤 간단하다.

```bash
pip install gspread
```

---

## 권한 및 환경 설정

먼저 프로젝트를 생성한다.

[구글 클라우드](https://console.developers.google.com/)에 로그인을 한 다음에 `프로젝트 만들기`를 누른다.

<center><img src="/public/img/2023-04-10-gspread-usage/01.png" width="80%"></center>

아래 스크린샷처럼 프로젝트 이름을 마음대로 지정한다. 위치를 지정하고 싶으면 지정한 다음(선택), 만들기를 누른다.

<center><img src="/public/img/2023-04-10-gspread-usage/02.png" width="80%"></center>

그러면 프로젝트가 하나 만들어진다. 먼저, 위쪽의 검색 창에서 `Google Drive API`를 검색한다.


<center><img src="/public/img/2023-04-10-gspread-usage/03.png" width="80%"></center>

그리고 사용을 눌러 준다. 마찬가지로, `Google Sheets API`를 검색한 다음 사용을 클릭한다.


<center><img src="/public/img/2023-04-10-gspread-usage/04.png" width="80%"></center>

이제 API 및 서비스 > 사용자 인증 정보(영어라면 `APIs & Services > Credentials`) 페이지로 이동한 다음, `사용자 인증 정보 만들기 > 서비스 계정`(`Create credentials > Service account key`)를 클릭한다. 설명을 보면 로봇 계정이라 되어 있다. python을 쓸 수 있을 것 같지 않은가?

<center><img src="/public/img/2023-04-10-gspread-usage/05.png" width="100%"></center>

서비스 계정 이름을 설정한다. 그러면 자동으로 계정 ID도 만들어지는데, 수정해도 상관없다.  
아래의 계정 설명이나 액세스 권한 부여(2, 3)번은 선택사항으로 패스해도 된다. 제일 아래의 `완료`를 클릭한다.

<center><img src="/public/img/2023-04-10-gspread-usage/06.png" width="100%"></center>

그러면 서비스 계정이 하나 생성된 것을 볼 수 있다. `서비스 계정 관리`(`Manage service accounts`)를 클릭해 준다.

<center><img src="/public/img/2023-04-10-gspread-usage/07.png" width="100%"></center>

아래 그림과 같이 3점 버튼을 누르고 `키 관리`(`Manage Keys`)를 클릭한다.

<center><img src="/public/img/2023-04-10-gspread-usage/08.png" width="100%"></center>

`키 추가` > `새 키 만들기`를 클릭한다. (`ADD KEY > Create new key`)

<center><img src="/public/img/2023-04-10-gspread-usage/09.png" width="100%"></center>

그리고 뜨는 창에서 `JSON`을 그대로 두고 `만들기`(`Create`)를 클릭한다.

<center><img src="/public/img/2023-04-10-gspread-usage/10.png" width="50%"></center>

그러면 json 파일이 하나 다운로드될 것이다. json 파일을 열어보면, 아래랑 비슷하게 생겼다.
```json
{
    "type": "service_account",
    "project_id": "api-project-XXX",
    "private_key_id": "2cd … ba4",
    "private_key": "-----BEGIN PRIVATE KEY-----\nNrDyLw … jINQh/9\n-----END PRIVATE KEY-----\n",
    "client_email": "473000000000-yoursisdifferent@developer.gserviceaccount.com",
    "client_id": "473 … hd.apps.googleusercontent.com",
    ...
}
```

`client_email` key가 보일 것이다. 옆에 value로 자동 생성된 어떤 email 주소가 보일 것이다. 이 이메일 주소를, 이제 python으로 결과를 입력할 스프레드시트에 다른 사용자를 추가하듯이 사용자 추가를 하면 된다.

<center><img src="/public/img/2023-04-10-gspread-usage/11.png" width="100%"></center>

이제, 아까 그 json 파일을 아래 기본 경로에 둔다. 파일 이름도 `service_account.json`로 바꿔준다.

- Linux: `~/.config/gspread/service_account.json`
- Windows: `%APPDATA%\gspread\service_account.json`

이제 테스트를 해 보자. 스프레드시트를 원하는 제목으로 하나 만든다. (필자는 제목을 `gorio_test_spread`으로 했다)  
1번째 행, 1번째 열(`A1` 셀)에 텍스트를 아무거나 입력하고(`gspread test`와 같은), 아래 코드를 python으로 실행시켜보자.

```python
import gspread

gc = gspread.service_account()
sh = gc.open("gorio_test_spread") # 스프레드시트 제목으로 설정할 것

print(sh.sheet1.get('A1'))

# 결과: [['gspread test']]
```


참고로, 기본 권한 파일은 위의 기본 경로에 저장된 `service_account.json`이고, 다른 파일(즉, 다른 권한을 가진 계정 등)을 쓰려면 아래와 같이 쓰면 된다.

```python
import gspread

gc = gspread.service_account(filename='C:/rnsrnbin_gorio_test_spread.json')
sh = gc.open("gorio_test_spread") # 스프레드시트 제목으로 설정할 것

print(sh.sheet1.get('A1'))
```

service account는 credential을 따로 또는 dictionary 형태롤 받을 수 있다.

```python
from google.oauth2.service_account import Credentials

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_file(
    'path/to/the/downloaded/file.json',
    scopes=scopes
)

gc = gspread.authorize(credentials)
```
또는
```python
import gspread

credentials = {
    "type": "service_account",
    "project_id": "api-project-XXX",
    "private_key_id": "2cd … ba4",
    "private_key": "-----BEGIN PRIVATE KEY-----\nNrDyLw … jINQh/9\n-----END PRIVATE KEY-----\n",
    "client_email": "473000000000-yoursisdifferent@developer.gserviceaccount.com",
    "client_id": "473 … hd.apps.googleusercontent.com",
    ...
}

gc = gspread.service_account_from_dict(credentials)

sh = gc.open("Example spreadsheet")

print(sh.sheet1.get('A1'))
```

credential과 관련해 더 자세한 내용은 [여기](https://docs.gspread.org/en/latest/oauth2.html#enable-api-access)를 참조하면 된다. 사실 위의 내용과 거의 같은 내용도 포함한다.

---

## gspread 사용법

### Spreadsheet open, create, share

권한과 환경설정을 모두 마쳤으면, 스프레드시트를 불러올 차례가 되었다.

스프레드시트는 시트의 제목, 또는 url로부터 고유 id를 가져오거나, 그냥 url 전체를 복붙하면 된다.

```python
worksheet = gc.open('gorio_test_spread')
worksheet = gc.open_by_key('1kXGuatJwOmkMAat38UZ7OLdGBjt3HS1AobTGmlOuDns')
worksheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1kXGuatJwOmkMAat38UZ7OLdGBjt3HS1AobTGmlOuDns/edit#gid=0')
```
단, 제목으로 하면 같은 제목을 가진 스프레드시트가 여러 개 있을 경우에 그냥 마지막으로 수정된 시트가 열린다. 가능하면 url이라도 쓰자.

스프레드시트는 다음과 같이 생성할 수 있다.

```python
worksheet = gc.create('A new spreadsheet')
```

공유하는 방법도 어렵지 않다.
```python
worksheet.share('otto@example.com', perm_type='user', role='writer')
```

### Sheet select, create, delete




###

---

## References

- https://docs.gspread.org/en/latest/oauth2.html#enable-api-access