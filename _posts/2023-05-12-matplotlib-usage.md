---
layout: post
title: matplotlib 사용법(python matplotlib.pyplot 사용법)
author: YouWon
categories: References
tags: [matplotlib, seaborn, usage]
---


이 글에서는 python **matplotlib**의 사용법을 정리한다.


따로 명시하지 않으면 이 글에서의 예제 데이터는 다음으로 설정한다.

```python
data = pd.DataFrame(data={
    'A': [1,4,3,6,5,8,7,9],
    'B': [6,5,7,8,9,9,8,9],
    'C': [8.8,7.7,6.6,5.5,4.4,3.3,2.2,1.1]
})
```


---

## matplotlib 설치

설치는 꽤 간단하다.

```bash
pip install matplotlib
```

import는 관례적으로 다음과 같이 한다.

```python
from matplotlib import pyplot as plt
```


---

## Font 설정

아무 설정을 하지 않으면 한글이 깨지는 경우가 많다.

```bash
RuntimeWarning: Glyph 44256 missing from current font.
  font.set_text(s, 0.0, flags=flags)
```

보통 다음과 같이 설정해주면 된다.

```python
from matplotlib import rcParams
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
```

참고로 설정 가능한 폰트 목록을 확인하고 싶다면 다음 코드를 실행해 보자.
```python
import matplotlib.font_manager
fpaths = matplotlib.font_manager.findSystemFonts()
font_names = []
for i in fpaths:
    f = matplotlib.font_manager.get_font(i)
    font_names.append(f.family_name)

print(font_names[:5])

for fn in font_names:
    if 'malgun' in fn.lower():
        print(fn)
```

---

## Jupyter 전용 기능

다음 magic command를 사용사면 `plt.show()` 함수를 사용하지 않아도 그래프를 보여준다.

```bash
%matplotlib inline
```

```python
plt.scatter(x=[1,2,3], y=[4,5,6])
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/01.png" width="70%"></center>


보통의 환경이라면 `plt.show()` 함수를 사용해야 그래프가 보인다. [PyCharm](https://greeksharifa.github.io/references/2019/02/07/PyCharm-usage/#sciview-%EC%B0%BDalt--0)이라면 `SciView` 창에서 열린다.

---

## 그래프 종류

### 산점도(scatter)

```python
plt.scatter(x=[1,2,3], y=[4,5,6])
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/01.png" width="70%"></center>

### 선 그래프(plot)

```python
plt.plot(data.index, data['B'])
# 하나만 입력하면 기본 index로 그려진다.
plt.plot(data['C']) 
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/10.png" width="70%"></center>


### 막대 그래프(bar)

```python
plt.bar(data.index, data['B'])
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/15.png" width="70%"></center>

### boxplot

```python
x = np.random.normal(50, 5, 100)
plt.boxplot(x)
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/18.png" width="70%"></center>

여러 boxplot을 한 번에 그리려면 리스트에 담아서 전달한다.

```python
x1 = np.random.normal(15, 5, 500)
x2 = np.random.normal(10, 10, 100)
plt.boxplot([x1, x2])
plt.xticks([1, 2], ["x1", "x2"])
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/19.png" width="70%"></center>


### Histogram

구간 개수는 `bins`으로 설정한다.

```python
x = np.random.normal(10, 2, 100)
plt.hist(x, bins=10)
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/20.png" width="70%"></center>

### Heatmap

matshow라는 함수가 있지만 이건 seaborn의 heatmap이 더 편하다.

### pd.DataFrame.plot

- pandas의 dataframe에서 `.plot()` method를 호출하면 바로 그래프를 그릴 수 있다.
  - 종류는 `line, bar, barh, hist, box, scatter` 등이 있다. `barh`는 수평 막대 그래프를 의미한다.



```python
data.plot(kind='line')
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/21.png" width="70%"></center>

좀 더 자세하게 설정할 수도 있다. 이 글에 있는 스타일 설정을 대부분 적용할 수 있다.

```python
data.plot(kind = "bar", y = ["B", "C"], figsize = (10, 6),
    yticks=[0, 5, 10])
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/22.png" width="70%"></center>

단 boxplot과 histogram은 `y`만 설정할 수 있다.

---

## 스타일 설정

### 색상(color)

```python
plt.plot(data['B'], label='B', color='red')
plt.plot(data['C'], label='C', color='green')
plt.legend()
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/11.png" width="70%"></center>

### 선 스타일(linestyle), 두께(linewidth)

- 선 스타일은 `linestyle` parameter를 전달하며 기본값인 `solid`와 `dashed, dotted, dashdot`이 있다.
- 선 두께는 `linewidth`로 설정하고 기본값은 1.5이다.

```python
plt.plot(data['A'], label='A', linestyle='dashed', linewidth=2)
plt.plot(data['B'], label='B', linestyle='dotted', linewidth=3)
plt.plot(data['C'], label='C', linestyle='dashdot', linewidth=4)
plt.legend()
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/12.png" width="70%"></center>

### bar 스타일(width, color, linewidth)

bar의 두께를 설정할 수 있다. 각 데이터마다 다른 값을 주면 각각 다르게 설정할 수도 있다.

```python
plt.bar(data.index, data['A'], width=0.4, color=["red", "blue", "green", "purple", "red", "blue", "green", "purple"], linewidth=2.5)
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/17.png" width="70%"></center>



### 마커(marker)

각 data point마다 marker를 표시할 수 있다.

- 점: `.`, 원: `o`, 사각형: `s`, 별: `*`, 다이아몬드: `D`, `d`
- marker의 사이즈도 `markersize` parameter로 지정할 수 있다.
- 산점도(scatter)의 마커 크기는 `s` parameter로 설정한다. 단 크기가 10배 차이난다.

```python
plt.plot(data['A'], label='A', marker='o', markersize=4)
plt.plot(data['B'], label='B', marker='s', markersize=8)
plt.scatter(data.index, data['C'], label='C', marker='.', s=120, color='red')
plt.legend()
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/13.png" width="70%"></center>


### 투명도(alpha)

데이터가 너무 많은 경우 투명도를 조절하면 좀 더 잘 보이게 할 수 있다.

```python
import numpy as np
x = np.random.normal(0, 1, 16384)
y = np.random.normal(0, 1, 16384)
plt.scatter(x, y, alpha = 0.04, color='purple')
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/14.png" width="70%"></center>


---

## 그래프 전체 설정

### 그래프 크기 설정

```python
plt.figure(figsize=(6, 3))
plt.scatter(x=data.index, y=data['A'])
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/09.png" width="70%"></center>



### 그래프 제목(title)

```python
plt.scatter(x=data.index, y=data['A'])
plt.title("Gorio")
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/02.png" width="70%"></center>


### 범례 설정(legend)


```python
plt.scatter(x=data.index, y=data['A'], label='Gorio')
plt.legend()
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/07.png" width="70%"></center>

legend 위치는 그래프를 가리지 않는 위치에 임의로 생성된다. 위치를 지정하려면 `loc` parameter를 설정한다.

```python
plt.scatter(x=data.index, y=data['A'], label='Gorio')
plt.legend(loc='lower right')
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/08.png" width="70%"></center>

가능한 옵션을 총 9개이다. `left, center, right, upper left, upper center, upper right, lower left, lower center, lower right`

---

## x, y축 설정

### 축 제목(xlabel, ylabel)

```python
plt.scatter(x=data.index, y=data['A'])
plt.xlabel("x axis")
plt.ylabel("y axis")
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/03.png" width="70%"></center>


### 축 범위 설정(xlim, ylim)

```python
plt.scatter(x=data.index, y=data['A'])
plt.xlim(left=0, right=10)
plt.ylim(bottom=0, top=10)
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/04.png" width="70%"></center>

참고로 `left, right, bottom, top` 중 설정하지 않은 값은 데이터 최솟값/최댓값에 맞춰 자동으로 설정된다. 

### 눈금 값 설정(xticks, yticks)

`ticks`에 tick의 위치를 지정하고, `labels`에 원하는 tick 이름을 지정할 수 있다.

```python
plt.scatter(x=data.index, y=data['A'])
plt.xticks(ticks=[0,3,6], labels=['zero', 'three', 'six'])
plt.ylim(bottom=0, top=10)
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/05.png" width="70%"></center>


xticks나 yticks에 값을 1개만 주면 `ticks` parameter가 설정된다.

```python
plt.scatter(x=data.index, y=data['A'])
plt.xticks([0,3,6])
plt.ylim(bottom=0, top=10)
```

<center><img src="/public/img/2023-05-12-matplotlib-usage/06.png" width="70%"></center>


참고로 막대그래프는 기본적으로 막대의 중심 좌표를 기준으로 계산하지만 `align` parameter를 주면 왼쪽 위치로 계산할 수 있다.


```python
plt.bar(data.index, data['B'], align='edge')
plt.xticks([0,3,6], ['zero', 'three', 'six'])
```
<center><img src="/public/img/2023-05-12-matplotlib-usage/16.png" width="70%"></center>



---

## 그래프 저장(savefig)

```python
plt.scatter(x=data.index, y=data['A'])
plt.savefig("gorio.png", dpi=300)
```

dpi는 dot per inch의 약자이다. 높을수록 해상도가 좋아지고 파일 크기도 커진다.



---

## References

- 