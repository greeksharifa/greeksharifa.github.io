---
layout: post
title: Markdown 사용법
author: YouWon
categories: References
tags: [Markdown, usage]
---

마크다운을 쓸 때는 메모장으로도 되지만 JetBrains의 Webstorm에 Markdown support plugins을 설치하는 것이 도움이 된다.

참조: [simhyejin.github.io](https://simhyejin.github.io/2016/06/30/Markdown-syntax/)

---

Python Code:

```python
# coding=utf-8 
# 둥

import torch
from torch.utils.data import Dataset

MSCOCO_IMGFEAT_ROOT = 1048576
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
}

class VQADataset:
    """
    A VQA data example in json file:
            "answer_type": "other",
    """
    def __init__(self, splits: str):
        self.name = splits
        print('self.name = splits:', self.name, sep='\t')
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/comn_sents_%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)
```


{% highlight python %}
print('what?', end='\t')
{% endhighlight %}

---

위의 글에 잘 설명되어 있지만,
복사해 놓고 쓰기 편하도록 본 글에 정리해 두었다.

참고로 넓은 개행을 하려면 한 줄을 띄우고 작성해야 한다.

좁은 개행은 문장 끝에 공백을 두 개 붙이면 된다.



그러나 여러 줄을 띄워도 효과는 똑같다.
공백 문자                   (스페이스바)            도 마찬가지이다.

들여쓰기는 `&nbsp;`을 사용한다. 하나당 하나의 공백이다.

# 가장 큰 제목


### 적당한 제목

이 글은 보고 쓰기 위해 작성되었다.

`[링크](https://google.com/)`

[링크](https://google.com/)

```
[참조 링크][1]

[1]:  https://greeksharifa.github.io/references/2018/06/29/markdown-usage/ "YW & YY's blog: markdown-usage"
```

[참조 링크][1]

[1]:  https://greeksharifa.github.io/references/2018/06/29/markdown-usage/ "YW & YY's blog: markdown-usage"

빈 줄을 넣는 것을 추천한다.

`url 링크: <https://google.com/>`

url 링크: <https://google.com/>

```
어쩐지 처음으로 돌아가고 싶은가? 내부 링크는 이렇게 사용한다. [가장 큰 제목](#가장-큰-제목)
```

어쩐지 처음으로 돌아가고 싶은가? 내부 링크는 이렇게 사용한다. [가장 큰 제목](#가장-큰-제목)


내부 링크의 #id 규칙은 다음과 같다.

0. 가능하면 괄호는 쓰지 않는다. 제대로 작동하지 않는다.
1. 영문자는 lowercase로, 한글은 그대로 둔다.
2. 문자/숫자/space/하이픈 외의 문자는 모두 제거한다.
3. space는 하이픈으로 대체한다.
4. 만약 제목이 유일한 이름이 아니라면, `-1` `-2` `-3` 등을 붙인다.

```

> 인용하려면 이와 같이 한다.
>> 인용을 안에 또 하고 싶으면 이렇게 한다.
>>> 더 할 수 있다.
```

> 인용하려면 이와 같이 한다.
>> 인용을 안에 또 하고 싶으면 이렇게 한다.
>>> 더 할 수 있다.

```
코드 블럭이다. ```을 써도 되고 ~~~을 써도 된다.
```
~~~
greetings = input()
print('Hello, Markdown!') # greetings는 안썼음
~~~
`조그맣게 쓰고 싶다면 이렇게`

```
*기울여 쓰기*     _기울여 쓰기_

**Bold로 쓰기**     __Bold로 쓰기__

***기울이고 Bold로 쓰기*** ___기울이고 Bold로 쓰기___

~~취소하기~~
```


*기울여 쓰기*     _기울여 쓰기_

**Bold로 쓰기**     __Bold로 쓰기__

***기울이고 Bold로 쓰기*** ___기울이고 Bold로 쓰기___

~~취소하기~~
```

---

수평선. 앞뒤로 빈 줄을 하나씩 넣는 것이 좋다.

***

다시 수평선

___

-------------

* * *

```

---

수평선. 앞뒤로 빈 줄을 하나씩 넣는 것이 좋다.

***

다시 수평선

___

-------------

* * *

```

- [ ] 체크리스트
- [x] 완료 리스트

1. 순서 있는 리스트
2. 순서 있는 리스트
0. 사실 숫자는 순서가 없어도 된다.
-1232. 물론 음수는 안 된다.
11111111. 뭐 그래도 숫자는 맞춰 주는 것이 좋긴 하다.

* 순서를 없애고 싶으면 이렇게
  * 탭을 누르고 치면 이렇게 보인다.
  - 사실 *, -, + 를 섞어서 써도 잘 보인다.
  + 하지만 굳이 그렇게 할 필요는 없다.
       * 탭은 적당히 쳐야 잘 보인다.
                          * 너무 많이 하면 효과가 없다.
                          - 이미 효과가 없다.
```
- [ ] 체크리스트
- [x] 완료 리스트

1. 순서 있는 리스트
2. 순서 있는 리스트
0. 사실 숫자는 순서가 없어도 된다.
-1232. 물론 음수는 안 된다.
11111111. 뭐 그래도 숫자는 맞춰 주는 것이 좋긴 하다.

* 순서를 없애고 싶으면 이렇게
  * 탭을 누르고 치면 이렇게 보인다.
  - 사실 *, -, + 를 섞어서 써도 잘 보인다.
  + 하지만 굳이 그렇게 할 필요는 없다.
       * 탭은 적당히 쳐야 잘 보인다.
                          * 너무 많이 하면 효과가 없다.
                          - 이미 효과가 없다.
```

Header 1 | Header 2
------------------ | --------------------
Content 1 | ---의 개수는 상관없음.
Content 2 | Content 4

| Header 1 | Header 2 | Header 3 |
| :-------- | :----------: | --------: |
| Left | Center | ---의 개수가 달라도 됨. |
```

Header 1 | Header 2
------------------ | --------------------
Content 1 | ---의 개수는 상관없음.
Content 2 | Content 4

| Header 1 | Header 2 | Header 3 |
| :-------- | :----------: | --------: |
| Left | Center | ---의 개수가 달라도 됨. |

```

이미지는 이렇게(링크랑 비슷)

![alt text](/test1.png)
![alt text](image_URL)
![alt text][1]
[1]: /test3.png

예시:

<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)

```
이미지는 이렇게(링크랑 비슷)

![alt text](/test1.png)
![alt text](image_URL)
![alt text][1]
[1]: /test3.png

예시:

<center><img src="/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg" width="50%"></center>

![01_new_repository](/public/img/Andre_Derain_Fishing_Boats_Collioure.jpg)

편하게 하려면,

- /public/img directory 안에 /categories_name/post_file_name directory를 만든다.
- 만든 directory 안에 이미지를 붙여 넣는다.
- WebStorm에 보이는 이미지에서 Ctrl + Shift + Alt + C 를 눌러 상대 경로를 복사한다.
- 그리고 위의 예시의 src 항목에다 붙여넣기 하면 된다. 이때 반드시 `/public/img/`으로 시작해야 한다.
