---
layout: post
title: 파이썬 collections, heapq 모듈 설명
author: Youyoung
categories: 파이썬
tags: [References, 파이썬]
---

## 1. collections 모듈  
### 1.1. collections.Counter 객체  
collections 모듈에서 가장 기본을 이루는 class는 `collections.Counter`이다. 이 class에 argument로 반복 가능한 (iterable) 객체를 지정하거나 dictionary와 같은 mapping 객체를 지정하면 Counter 객체를 생성할 수 있다. 예를 들어보면,  

```python
import collections

counter = collections.Counter([1, 2, 3, 2])
# counter = collections.Counter({1: 1, 2: 2, 3: 1})
print(counter)

Counter({1: 1, 2: 2, 3: 1})
```

주석 처리된 line이 바로 후자의 방법에 해당한다. 이렇게 생성된 객체는 수정될 수 있다.  

```python
counter[1] += 1
print(counter)

Counter({1: 2, 2: 2, 3: 1})
```

이 외에도 여러 계산이 가능한데, 아래를 참고하길 바란다.  

|연산자|설명|
|:--------:|:--------:|
|-=|뺀다. 결과가 음수면 그 요소는 삭제된다.|
|&=|좌변의 Counter 객체 요소 중 우변의 Counter 객체 요소에 미포함되어 있는 <br /><br /> key의 요소를 삭제한다. 요소의 값은 둘 중 작은 쪽의 값이 된다.|
|l=|2개의 Counter 객체 전체의 요소로부터 새롭게 Counter 객체를 생성한다. <br /><br /> key가 같으면 두 값 중 큰 쪽의 값이 된다.|

위 누계 연산자에서 =를 빼고 `+, -, &, |` 만 사용할 경우 이항 연산자로 작용한다.  

또한, 이 객체에서 미등록 key를 참조한다 하더라도 KeyError는 발생하지 않는다.  
```python
print(counter[4])

0
```

### 1.2. collections.ChainMap: 사전 통합  
```python
dict1 = {'banana': 1}
dict2 = {'apple': 2}

counter = collections.ChainMap(dict1, dict2)
print(counter['apple'])

2
```

위와 같이 ChainMap 메서드는 여러 사전 객체를 모아 하나로 통합하는 기능을 갖고 있다. 만약 통합한 객체에 변화를 줄 경우, 원래의 사전들에도 그 변경 사항이 반영된다. `clear` 메서드를 사용하면 사전을 삭제할 수 있다.  


### 1.3. collections.defaultdict: 기본 값이 있는 사전  
일반적으로 사전 객체에 미등록된 key를 참조하면 KeyError가 발생한다. `collections.defaultdict`는 이러한 문제를 해결하기에 적합한 객체이다. 

```python
d = {'orange': 10}

def get_default_value():
    return 'default-value'

# 여기서 get_default_value와 같은 callable 객체나 None을 입력할 수 있다.
# None을 입력할 경우 일반 사전과 마찬가지로 KeyError가 발생한다.
e = collections.defaultdict(get_default_value, orange=10)
print(e['ham'])

'default-value'
```

만약 기본 값으로 수치 0이나 빈 사전, 리스트를 반환하고 싶다면 int, dict, list형 객체를 지정하면 된다.  
```python
e = collections.defaultdict(int)
e = collections.defaultdict(dict)
e = collections.defaultdict(list)
```

### 1.4. collections.OrderedDict: 순서가 있는 사전  
**for loop**와 같은 과정 속에서 등록한 순서대로 요소를 추출하고 싶으면 이 class를 이용하면 좋다. 시퀀스를 이용하여 객체를 생성하면 순서대로 등록된 것을 확인할 수 있다.  
```python
mydict = collections.OrderedDict([("orange", 10), ("banana", 20)])
print(mydict)

OrderedDict([('orange', 10), ('banana', 20)])
```

그러나 키워드 인수나 일반 사전으로 초깃값을 등록하면 순서가 무시된다. `OrderedDict` 객체에는 유용한 기능들이 있는데, 아래를 참조하면 좋을 것이다.
```python
mydict = collections.OrderedDict([("orange", 10), ("banana", 20), ("blueberry", 30), ("mango", 40)])

# popitem 에서 last=True로 하면 마지막 요소를 사전에서 삭제하고 반환하고,
# False로 하면 첫 요소에 효과를 적용한다.
mydict.popitem(last=True)

# move_to_end에서 last=True로 하면 지정한 키를 맨 끝으로 이동시키고, False이면 맨 처음으로 이동시킨다.
mydict.move_to_end(key="banana", last=True)

print(mydict)

OrderedDict([('orange', 10), ('blueberry', 30), ('banana', 20)])
```

### 1.5. collections.namedtuple  
데이터를 효율적으로 관리하기에 적합한 class가 바로 namedtuple이다. 속성 이름을 지정하여 가독성을 높이고 튜플을 활용하여 원하는 요소를 쉽게 추출하도록 하게 해준다.   
```python
point = collections.namedtuple("point", "X, Y, Z")
data = point(-2, 6, 3)
print(data.Y)

6
```

---
## 2. heapq 모듈  
데이터를 정렬된 상태로 저장하고, 이를 바탕으로 효율적으로 최솟값을 반환하기 위해서는 이 **heapq** 모듈을 사용하면 매우 편리하다. 사용하기 위해서는 최소 heap을 먼저 생성해야 한다. 빈 리스트를 생성해서 heapq 모듈의 메서드를 호출할 때마다 이를 heap argument의 인자로 투입해야 한다. 


```python
import heapq

heap = []

# heappush(heap, item): heap에 item을 추가함
# 주의점: keyword 인자를 입력하면 Error가 발생함
heaqp.heappush(heap, 2)
heaqp.heappush(heap, 1)

# heappop(heap): heap에서 최솟값을 삭제하고 그 값을 반환함
# 최솟값을 삭제하지 않고 참조하고 싶다면 heap[0]을 쓰자
heapq.heappop(heap)

1
```

이 외에도 여러 메서드를 사용할 수 있다. 만약 어떤 변화하는 시퀀스에서 최솟값을 얻고 싶다고 하자. 아래와 같은 코딩이 가능하다.
```python
heap = [79, 24, 50, 62]

# heapify(heap): heap의 요소를 정렬함
heapq.heapify(heap)

# heappush(heap, item): heap에 item을 추가한 뒤, 최솟값을 삭제하고 그 값을 반환함
heapq.heappushpop(heap, 10)

10

# heapreplace(heap, item): 최솟값을 삭제한 뒤, heap에 item을 추가하고 삭제한 값을 반환함
# 주의점: 추가한 값 아님
heapq.heapreplace(heap, 10)

24
```

---
## Reference  
> 파이썬 라이브러리 레시피, 프리렉
