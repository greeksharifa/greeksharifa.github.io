---
layout: post
title: 디닉 알고리즘(Dinic's Algorithm)
author: YouWon
categories: [Algorithm & Data Structure]
tags: [Dinic, Network Flow, Maximum Flow]
---

## 참조

분류 | URL
-------- | --------
문제 | [최대 유량](https://www.acmicpc.net/problem/6086)
응용 문제 | [스포일러 1](https://www.acmicpc.net/problem/11495)
[참조 라이브러리](https://greeksharifa.github.io/algorithm%20&%20data%20structure/2018/07/07/algorithm-library/) | [sharifa_header.h](https://github.com/greeksharifa/ps_code/blob/master/library/sharifa_header.h), [bit_library.h](https://github.com/greeksharifa/ps_code/blob/master/library/bit_library.h)
<<<<<<< HEAD
이 글에서 설명하는 라이브러리 | []()
=======
이 글에서 설명하는 라이브러리 | [dinic.h](https://github.com/greeksharifa/ps_code/blob/master/library/dinic.h)
>>>>>>> ea2928b6cf420c33e027cb99efab88b746056954


--- 

## 개요

### 시간복잡도: $ O(V^2 \cdot E) $
### 공간복잡도: $ O(V^2) $ 또는 $ O(V+E) $
- V는 정점(vectex)의 수, E는 간선(edge)의 수이다.

이 글에서는 네트워크 플로우(Network Flow) 분야에서 Maximum Flow를 구하는 알고리즘인 
[디닉 알고리즘](https://en.wikipedia.org/wiki/Dinic%27s_algorithm)에 대해서 설명한다.  

Maximum Flow를 구하는 다른 대표적인 알고리즘으로 
[에드몬드-카프 알고리즘(Edmonds–Karp algorithm)](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm), 
[포드-풀커슨 알고리즘(Ford–Fulkerson algorithm)](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm)이 있지만,
현재 PS에서 구현할 만한 알고리즘 중 가장 빠른 알고리즘은 디닉 알고리즘이다.   
그래서 여러 개 외울 필요 없이 알고리즘 하나만 알아두는 것이 좋을 듯 하다.

## Network Flow(네트워크 플로우)

네트워크 플로우에 대한 기본적인 설명을 조금 적어 두려고 한다.   
예를 하나 들면, Network Flow는 파이프가 복잡하게 연결되어 있고, 각 파이프는 물이 흐를 수 있는 최대 양이 정해져 있고, 
source에서 sink방향으로 물이 흐를 때, 물이 흐를 수 있는 최대 양을 구하는 것이라고 보면 된다.

![01_network_flow](/public/img/Algorithm_and_Data_Structure/2018-07-11-algorithm-dinic/01_network_flow.png)

`출처: wikipedia`

- s는 source를 의미한다. 물이 나오는 원천이라고 생각하면 된다.
- t는 sink를 의미한다. 물이 최종적으로 들어가는 곳이라 생각하면 된다. 모든 물(유량)은 source에서 sink로 흐른다.
- 두 정점을 잇는 간선은 해당 정점 사이에 흐를 수 있는 최대 물(유량)을 의미한다. s에서 1번 정점으로 0/10이라고 적혀 있는데, 여기서 10이 최대 유량이다.
- 잔여유량은 최대유량에서 현재 유량을 뺀 값이다. s에서 1번 정점으로 0/10인 것은 현재 유량이 0이고 따라서 잔여유량은 10이다.

네트워크 플로우의 Maximum Flow는 Minimum Cut과 깊은 연관이 있다.  

---

## 알고리즘

디닉 알고리즘은 크게 두 단계로 이루어진다.

1. BFS를 써서 레벨 그래프(Level Graph)를 생성하는 것
2. DFS를 써서, 레벨 그래프에 기초한 차단 유량(Blocking Flow)의 규칙을 지키면서, 최대 유량을 흘려주는 것

### 레벨 그래프(Level Graph)

레벨 그래프는 각 정점들에 source로부터의 최단 거리를 레벨 값을 할당한 그래프이다.
아래 그림에서 source의 레벨은 0이되고 source와 인접한 1번과 2번 정점의 레벨은 1, 이후는 2... 등이 된다.  
빨간색 숫자로 적혀 있는 것이 레벨이다. 

![02_residual_capacity](/public/img/Algorithm_and_Data_Structure/2018-07-11-algorithm-dinic/02_residual_capacity.png)

레벨 그래프는 BFS로 구현한다.

### 차단 유량(Blocking Flow)

디닉 알고리즘에서는, 유량을 흘려보낼 때 레벨 차이가 딱 1이 나는 정점으로만 유량을 보낼 수 있다.  
즉 바로 위의 그림에서의 간선과 같은 곳으로만 보낼 수 있다. 레벨이 같아도 안 된다.

유량을 흘려보내는 것은 DFS로 구현한다. source에서 시작하여, 차단 유량 규칙을 만족하는 정점으로만 따라가면서 최종적으로 
sink에 도달할 때까지 탐색하는 과정을 반복한다.

위의 레벨 그래프에서는 다음 세 경로를 DFS로 찾을 수 있다.

(s, 1, 3, 4): 유량 4  
(s, 1, 4, t): 유량 6  
(s, 2, 4, t): 유량 4

### 반복

BFS 1번 그리고 DFS를 한번씩 해서는 최대 유량이 찾아지지 않는다. 다만 복잡한 것은 아니고, 위의 과정을 반복해주면 된다.

다시 BFS를 돌려 레벨 그래프를 새로 생성한다.

![03_residual_capacity](/public/img/Algorithm_and_Data_Structure/2018-07-11-algorithm-dinic/03_residual_capacity.png)

위의 레벨 그래프에서는 다음 경로를 DFS로 찾을 수 있다.

(s, 2, 4, 3, t): 유량 5

다시 레벨 그래프를 그리면, 더 이상 sink로 가는 경로가 없음(sink의 레벨이 $\inf$)을 알 수 있다. 알고리즘을 종료한다.

![04_residual_capacity](/public/img/Algorithm_and_Data_Structure/2018-07-11-algorithm-dinic/04_residual_capacity.png)


## 구현

다음과 같다.

```cpp
#pragma once

#include "sharifa_header.h"

#define MAX_V 52
#define S 0     // source
#define T 25    // sink
#define INF 1000000009

struct Edge {   // u -> v
    int v, cap, ref;
    Edge(int v, int cap, int ref) :v(v), cap(cap), ref(ref) {}
};

class Dinic {
    vector<vector<Edge> > edges;    // graph
    // level: 레벨 그래프, next_v: DFS에서 flow 계산 시 역추적에 사용
    vector<int> level, next_v;

public:
    void init() {
        edges.resize(MAX_V);
        level.resize(MAX_V);
        next_v.resize(MAX_V);
    }

    void addEdge(int u, int v, int cap) {
        edges[u].emplace_back(v, cap, (int)edges[v].size());
        edges[v].emplace_back(u, 0, (int)edges[u].size() - 1);
    }

    bool bfs() {
        fill(level.begin(), level.end(), -1);
        //level = vector<int>(MAX_V, -1);

        queue<int> q;
        level[S] = 0;
        q.push(S);

        while (!q.empty()) {
            int u = q.front();   q.pop();
            for (auto edge : edges[u]) {
                int v = edge.v, cap = edge.cap;

                if (level[v] == -1 && cap > 0) {
                    level[v] = level[u] + 1;
                    q.push(v);
                }
            }
        }
        return level[T] != -1;
    }

    void reset_next_v() {
        fill(next_v.begin(), next_v.end(), 0);
        //next_v = vector<int>(MAX_V, 0);
    }

    int dfs(int u, int max_flow) {
        if (u == T)
            return max_flow;

        for (int &i = next_v[u]; i < edges[u].size(); i++) {
            int v = edges[u][i].v, cap = edges[u][i].cap;

            if (level[u] + 1 == level[v] && cap > 0) {
                int flow = dfs(v, min(max_flow, cap));

                if (flow > 0) {
                    edges[u][i].cap -= flow;
                    edges[v][edges[u][i].ref].cap += flow;
                    return flow;
                }
            }
        }
        return 0;
    }
};
```

## 문제 풀이

### BOJ 06086(최대 유량)

문제: [최대 유량](https://www.acmicpc.net/problem/6086)

풀이: [BOJ 06086(최대 유량) 문제 풀이](https://greeksharifa.github.io/ps/2018/07/12/PS-06086/)


### 스포일러 문제

문제: [스포일러 문제](https://www.acmicpc.net/problem/11495)

풀이: [스포일러 풀이]()
