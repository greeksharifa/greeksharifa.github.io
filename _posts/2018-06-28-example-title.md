---
layout: post
title: example-title
author: YouWon
categories: References
tags: [usage]
---

---

이 글에서는 포스팅 방식을 설명한다.

우선 이 `2018-06-28-example-title.md` 파일을 같은 폴더에 복사 붙여넣기 한다. 물론 이름은 다른 것으로 바꿔준다.

이름의 형식은 다음 문단에서 설명한다.

---

### 파일명

파일 제목의 형식은 무조건 `yyyy-mm-dd-example-name-of-title.md` 형식이어야 한다. `-` 대신 공백 문자도 허용되는 것 같다.

---

### 파일의 헤더

title, author, categories, tags를 각각 원하는 대로 바꿔준다.

주의사항은 다음과 같다.

- title 등을 적을 때 콜론 `:` 을 적으면 글이 올라가지 않는 것 같다.
- 태그명은 `_`, `-`, ` `(공백 문자) 외의 특수문자는 쓰지 않는 것을 원칙으로 한다. @ 같은 문자를 쓰면 글이 올라가지 않는다.
- 태그를 여러 개 쓸 때는 `[example_tag1, example_tag2]`와 같은 형식으로 한다.
- 특수 태그로는 다음이 있다.
  - usage: 어떤 툴 등의 사용법을 알려주는 글에 붙이는 태그이다. 이 글에도 붙어 있다.

---

#### 목차는 자동으로 추가된다.

---

글을 다 썼다면, 다음 과정을 따른다.

##### 간략한 버전

terminal에 `push.bat`(pus 치고 탭 눌러도 됨)을 치면 `enter commit message`라는 문구가 보인다. 커밋 메시지를 입력하면, 끝.

만약 에러 메시지가 보인다면 `git pull`을 먼저 하고 시도해 보라.

그래도 문제가 있다면 `git add .`을 포함해서 하나하나 다 하면 된다.

##### 정석 버전

1. 먼저 `git pull ... ` 명령을 통해 remote(원격) repository의 update 내용을 받아온다.
2. `git add .`
3. `git commit -m "commit-message"
4. `git push ...`

위의 ...은 아무것도 입력하지 않아도 될 수도 있다. 그러나 앞으로 branch를 통해 작업을 하거나 force push를 할 경우에는 조금 달라진다.

별다른 문제가 없으면 아무것도 입력하지 않아도 된다.

---

### 코드 입력

```python
def p_print(c):
    print('hi %s!' % c)

p_print('Gorio')
```

### 수식 입력

참조: [stackexchange.com](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

표로 정리한 글: [YW & YY's blog](https://greeksharifa.github.io/references/2018/06/29/equation-usage/)

수식 입력은

inline style: `$ a^2 + b^2 = c^2 $`

$ a^2 + b^2 = c^2 $

display style: `$$ a^2 + b^2 = c^2 $$`

$$ a^2 + b^2 = c^2 $$

로 한다.

---

### 이미지 추가

반드시 /로 시작해야 함.

예시:

<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)

---

### Markdown

다음 글을 참조한다. <https://greeksharifa.github.io/references/2018/06/29/markdown-usage/>

---

### References

다음의 글에 정리한다. <https://greeksharifa.github.io/references/2018/06/29/references/>
