---
layout: post
title: Python argparse 사용법
author: YouWon
categories: References
tags: [PyTorch, Argparse, usage]
---

이 글에서는 Python 패키지인 argparse에 대해 알아본다. Machine Learning 코드를 볼 때 꽤 자주 볼 수 있을 것이다.

---

# Import

```python
import argparse
```

---

# argparse

<script data-ad-client="ca-pub-9951774327887666" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

```
python train.py --epochs 50 --batch-size 64 --save-dir weights
```
Machine Learning을 포함해서, 위와 같은 실행 옵션은 많은 코드에서 볼 수 있었을 것이다. 학습 과정을 포함하여 대부분은 명령창 또는 콘솔에서 `python 파일명 옵션들...`으로 실행시키기 때문에, argparse에 대한 이해는 필요하다.

***중요:***
- 기본적으로 `argparse` 라이브러리는 명령창(터미널)에서 실행하는 것을 원칙으로 한다. Jupyter notebook이나 (iPython) 대화형 실행 framework에서는 제대로 실행되지 않을 수 있다. 또한 이러한 대화형 framework에서는 코드 상에서 명시적으로 집어 넣는 게 아닌 이상 인자에 값을 바로 줄 수도 없다. 
- 그래도 쓰고 싶다면 `args = parser.parse_args()`를 `args = parser.parse_args(args=[])`로 바꾸고 사용할 수는 있다...하지만 위의 이유로 인해 별 의미는 없을 듯하다.

필자는 이 글에서 위의 명령 중 `--epochs`와 같은 것을 **인자**, `50`과 같은 것을 (같이 준) **값**으로 부르겠다.

argparse는 python에 기본으로 내장되어 있다.
```python
import argparse
import os
```
`import os`는 output directory를 만드는 등의 역할을 위해 필요하다.

argparse를 쓰려면 기본적으로 다음 코드가 필요하다.
```python
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
# argument는 원하는 만큼 추가한다.
parser.add_argument('--print-number', type=int, 
                    help='an integer for printing repeatably')

args = parser.parse_args()

for i in range(args.print_number):
    print('print number {}'.format(i+1))
```
1. 일단 [ArgumentParser](https://docs.python.org/3/library/argparse.html?highlight=argparse#argparse.ArgumentParser)에 원하는 description을 입력하여 parser 객체를 생성한다. description 외에도 usage, default value 등을 지정할 수 있다.
2. 그리고 `add_argument()` method를 통해 원하는 만큼 인자 종류를 추가한다.
3. `parse_args()` method로 명령창에서 주어진 인자를 파싱한다.
4. `args`라는 이름으로 파싱을 성공했다면 `args.parameter` 형태로 주어진 인자 값을 받아 사용할 수 있다.

실행 결과
```
> python argparseTest.py -h
usage: argparseTest.py [-h] [--print-number PRINT_NUMBER]

Argparse Tutorial

optional arguments:
  -h, --help            show this help message and exit
  --print-number PRINT_NUMBER
                        an integer for printing repeatably

> python argparseTest.py --print-number 5
print number 1
print number 2
print number 3
print number 4
print number 5
```

argparse의 인자는 지정할 수 있는 종류가 상당히 많다.
## --help, -h
`--help` 또는 `-h`: 기본으로 내장되어 있는 옵션이다. 이 인자를 넣고 python으로 실행하면 인자 사용법에 대한 도움말이 출력된다.
```
> python argparseTest.py -h
usage: argparseTest.py [-h] [--print-number PRINT_NUMBER]
...
```

## argument 이름 정의
인자의 이름을 지정할 때 여러 이름을 짓는 것이 가능하다. 지정할 때 두 개를 연속해서 나열한다. 보통 1~2개를 지정하는데, `--help`와 `-h`같이 fullname과 약자를 하나씩 지정하는 편이다. 또 `help=`에서 description을 써줄 수 있다.  
참고로 help 메시지는 % formatting을 지원한다.
```python
parser.add_argument('--print-number', '-p', help='an integer for printing repeatably')
```

## type 지정
기본적으로 `parse_args()`가 주어진 인자들을 파싱할 때는 모든 문자를 숫자 등이 아닌 문자열 취급한다. 따라서 데이터 타입을 지정하고 싶으면 `add_argument()`에서 `type=`을 지정해 주어야 한다. default는 말한 대로 `str`이다.
- ex) `parser.add_argument('--print-number', '-p', type=int, ...)`
- type으로 사용 가능한 것은 한 개의 문자열을 받아들여 return 문이 있는 모든 callable 객체이다.
- Common built-in types과 functions이 사용 가능한데, `str`, `int`, `float`, `bool`과 `open` 등이 있다. `list`와 같은 것은 불가능하다. list처럼 쓰고 싶으면 아래쪽에서 설명할 `action=append`를 이용한다.
- `argparse.FileType()` 함수도 `type=`에 사용 가능한데, `mode=`, `bufsize=`, `encoding=`, `errors=` parameter를 취하는 함수로서 다양한 파일을 여러 옵션으로 지정할 수 있다. 예를 들어 `argparse.FileType('w')`는 쓰기 가능한 파일을 만든다. 자세한 것은 [여기](https://docs.python.org/3/library/argparse.html?highlight=argparse#type)를 참조한다.
    

## positional / optional 인자
positional 인자와 optional 인자가 있다. 인자의 이름 앞에 `-`가 붙어 있으면 optional, 아니면 positional 인자로서 필수로 지정해야 한다.  
단, positional 인자도 필수로 넣어야 하게끔 할 수 있다. `add_argument()` 함수에 `required=True`를 집어넣으면 된다. 그러나 C언어에서 `#define true false`같은 짓인 만큼 권장되지 않는다.
```python
# argparseTest.py
# ...
parser.add_argument('--foo', '-f') # optional
parser.add_argument('bar')         # positional
args = parser.parse_args()
print('args.foo:', args.foo)
print('args.bar:', args.bar)
```
```
# optional 인자는 지정하지 않아도 되고, 그럴 경우 기본값이 저장된다.
> python argparseTest.py bar_value
args.foo: None
args.bar: bar_value

# positional 인자는 반드시 값을 정해 주어야 한다.
> python argparseTest.py --foo 1
usage: argparseTest.py [-h] [--foo FOO] bar
argparseTest.py: error: the following arguments are required: bar

# optional 인자 뒤에는 반드시 저장할 값을 지정해야 한다. 
# 이는 `action=store`인 optional 인자에 해당한다. 6번 항목에서 설명하겠다.
> python argparseTest.py bar_value --foo
usage: argparseTest.py [-h] [--foo FOO] bar
argparseTest.py: error: argument --foo/-f: expected one argument

# optional 인자는 `--foo 3`또는 `--foo=3` 두 가지 방식으로 지정할 수 있다.
# positional 인자는 그런 거 없다.
> python argparseTest.py --foo=5 bar=bar_value
args.foo: 5
args.bar: bar_value

# positional 인자가 여러 개라면 순서를 반드시 지켜야 한다.
# optional 인자는 값만 잘 지정한다면 어디에 끼워 넣어도 상관없다.
> python argparseTest.py bar_value --foo 7
args.foo: 7
args.bar: bar_value
```
## default 값 지정
 값을 저장할 때 명시적으로 지정하지 않았을 때 들어가는 기본값을 설정할 수 있다. `add_argument()`에서 `default=` 옵션을 지정한다.
    - `argparse.SUPPRESS`를 적을 경우, 인자를 적지 않았을 때 None이 들어가는 것이 아닌 아예 인자 자체가 생성되지 않는다. 또한 `--help`에도 표시되지 않는다.
```python
parser.add_argument('--foo', '-f', type=int, default=5)
```
```
> python argparseTest.py
args.foo: 5

# 그러나 인자를 적어 놓고 값은 안 주면 에러가 난다. 
# 기본적으로 한 개의 값을 추가로 받아야 하기 때문이다.
# 이걸 바꾸려면 6번이나 7번 항목을 참조한다.
> python argparseTest.py --foo
usage: argparseTest.py [-h] [--foo FOO]
argparseTest.py: error: argument --foo/-f: expected one argument
```
## action의 종류 지정
인자를 정의(`add_argument()`에 의해)할 때 action을 지정할 수 있다. 액션에는 다음과 같은 것들이 있으며, 기본값은 `store`이다.
- `store`: action을 지정하지 않으면 `store`이 된다. 인자 이름 바로 뒤의 값을 해당 인자에 대입(저장)시킨다. 
- `store_const`: `add_argument()`에서 미리 지정되어 있는 `const=`에 해당하는 값이 저장된다. `const=`는 반드시 써 주어야 한다.
- `store_true`, `store_false`: 인자를 적으면(값은 주지 않는다) 해당 인자에 `True`나 `False`가 저장된다.
- `append`: 값을 하나가 아닌 여러 개를 저장하고 싶을 때 쓴다. 인자를 여러 번 호출하면 같이 주는 값이 계속 append된다.
- `append_const`: append와 비슷하지만 사전에 지정한 const 값이 저장된다.
- `count`: 인자를 적은 횟수만큼 값이 올라간다. 보통 `verbose` 옵션에 많이 쓴다.
- `help`: 도움말 메시지를 출력하게 하고 종료하여 코드는 실행시키지 않는다. `--help` 역할을 대신한다.
- `version`: `version` 인자에 사용가능하다. 버전 정보를 출력하고 종료한다.

```
parser.add_argument('--foo', action='store_const', const=10)
> python argparseTest.py --foo
args.foo: 10

# 인자를 적지 않으면 default 값(None)이 저장된다.
parser.add_argument('--foo', action='store_const', const=10)
> python argparseTest.py
args.foo: None

# default 값을 지정하면 당연히 바뀐다.
parser.add_argument('--foo', action='store_const', const=10, default=5)
> python argparseTest.py
args.foo: 5

# store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다.
# store_false의 경우 반대이다.
parser.add_argument('--foo1', action='store_true')
parser.add_argument('--foo2', action='store_true')
parser.add_argument('--foo3', action='store_false')
parser.add_argument('--foo4', action='store_false')
args = parser.parse_args()

print('args.foo1:', args.foo1)
print('args.foo2:', args.foo2)
print('args.foo3:', args.foo3)
print('args.foo4:', args.foo4)
> python argparseTest.py --foo1 --foo4
args.foo: True
args.foo: False
args.foo: True
args.foo: False

# 참고로 한 번만 호출해도 args.foo는 데이터 타입이 list가 된다. 안 하면 None이다.
parser.add_argument('--foo', action='append')
> python argparseTest.py --foo 1 --foo 123 --foo=xyz
args.foo: ['1', '123', 'xyz']
```

## attribute name: -, _ 구분
인자의 이름에는 `-`와 `_`을 쓸 수 있다. 단, python 기본 문법은 변수명에 `-`를 허용하지 않기 때문에, 인자의 이름에 `-`가 들어갔다면 `args.인자`로 접근하려면 `-`를 `_`로 바꿔 주어야 한다.
- `--print-number`의 경우 `args.print_number`로 접근할 수 있다.
- `--print_number`의 경우 `args.print_number`로 동일하다.

## dest: 적용 위치 지정
argument를 지정할 때 store나 action의 저장 또는 적용 위치를 바꿔서 지정할 수 있다. 예를 들어 `--foo`의 `dest=` 옵션을 `--foo-list`로 지정하면, `args.foo_list`에 값이 저장되는 식이다.
```python
parser.add_argument('--foo', action='append', dest='foo_list')
parser.add_argument('--bar', dest='bar_value')
args = parser.parse_args()

print('args.foo_list:', args.foo_list)
print('args.bar_value:', args.bar_value)

try:
    if args.foo is not None:
        print('Hmm?')
except AttributeError as e:
    print('Where are you gone?', e)
```
```
> python argparseTest.py --foo 1 --foo 123 --foo=xyz --bar ABC
args.foo_list: ['1', '123', 'xyz']
args.bar_value: ABC
Where are you gone? 'Namespace' object has no attribute 'foo'
```

## nargs: 값 개수 지정
argparse는 일반적으로 1개의 값을 추가로 받거나, `action=store_true`의 경우는 값을 추가로 받지 않는다. 이를 바꿔 주는 것이 `nargs=` 이다.
- `N`: N개의 값을 읽어들인다.
- `?`: 0개 또는 1개의 값을 읽어들인다. 
    - 인자와 값을 모두 적은 경우 해당 값이 저장된다.
    - 인자만 적은 경우 const 값이 저장된다.
    - 아무것도 적지 않았으면 default 값이 저장된다.
- `*`: 0개 이상의 값을 전부 읽어들인다.
- `+`: 1개 이상의 값을 전부 읽어들인다. 정규표현식의 것과 매우 비슷하다.
- `argparse.REMAINDER`: 남은 값을 개수 상관없이 전부 읽어들인다.

예제는 [원문](https://docs.python.org/3/library/argparse.html?highlight=argparse#nargs)이나 [번역본](https://docs.python.org/ko/3.7/library/argparse.html#nargs)을 참조한다.

## choices: 값 범위 지정
인자와 같이 주어지는 값의 범위를 제한하고 싶으면 `choices=` 옵션을 쓰면 된다. `choices=`에 들어갈 수 있는 정의역은 list 등 iterable 객체이다(`in` 조건검사를 할 수 있으면 된다).
```
parser.add_argument('--foo', choices=range(1, 5))
> python argparseTest.py --foo 5
usage: argparseTest.py [-h] [--foo {1,2,3,4}]
argparseTest.py: error: argument --foo: 
invalid choice: '5' (choose from 1, 2, 3, 4)
```

## metavar: 이름 재지정
 metavar은 `help=`에서 도움말 메시지를 생성할 때 표시되는 이름을 변경할 수 있다(직접 값을 참조하는  `args.foo` 같은 경우 기본 이름 또는 `dest=`에 의해 재지정된 이름을 써야 한다).


---

# References

- [원문](https://docs.python.org/3/library/argparse.html)
- [번역본](https://docs.python.org/ko/3.7/library/argparse.html)