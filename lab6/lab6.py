import numpy as np
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import queue
import time

VERTEX = 100
EDGES = 500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ITERATE = 10

random.seed(0)


def generate_matrix():
    matrix = np.zeros((VERTEX, VERTEX))
    cnt = 0
    for i in range(VERTEX):
        for j in range(i + 1, VERTEX):
            if cnt < EDGES:
                if random.randint(1, EDGES) >= EDGES / 2:
                    continue
                matrix[i][j] = random.randint(1, EDGES)
                matrix[j][i] = matrix[i][j]
                cnt += 1
    return matrix


def build_list(m):
    array = []
    for i in range(VERTEX):
        array.append([])
        for j in range(VERTEX):
            if m[i][j] != 0:
                array[-1].append((j, int(m[i][j])))
    return array


def print_graph(g):
    gr = nx.Graph()
    for i in range(VERTEX):
        gr.add_node(i)
    for i in range(VERTEX):
        for j in g[i]:
            gr.add_edge(i, j[0])
    nx.draw(gr)
    plt.savefig("{}/images/{}.png".format(BASE_DIR, "gr"))
    return gr


def deicstra(g, s):
    d = [1e9 for i in range(VERTEX)]
    u = [False for i in range(VERTEX)]
    d[s] = 0
    for i in range(VERTEX):
        v = -1
        for j in range(VERTEX):
            if not u[j] and (v == -1 or d[j] < d[v]):
                v = j
        if d[v] == 1e9:
            break
        u[v] = True
        for j in range(len(g[v])):
            to = g[v][j][0]
            ln = g[v][j][1]
            if d[v] + ln < d[to]:
                d[to] = d[v] + ln
    for i, j in enumerate(d):
        print(i, '->', j)
    return d


def belman_ford(g, s):
    d = [1e9 for i in range(VERTEX)]
    d[s] = 0
    e = []
    for i in range(VERTEX):
        for (j, c) in g[i]:
            e.append((i, j, c))
    for i in range(VERTEX):
        for j in range(len(e)):
            if d[e[j][0]] < 1e9:
                d[e[j][1]] = min(d[e[j][1]], d[e[j][0]] + e[j][2])
    return d


def measure(method, g, v):
    sm = 0
    for i in range(ITERATE):
        start_time = time.monotonic_ns()
        method(g, v)
        end_time = time.monotonic_ns()
        sm += end_time - start_time
    sm = sm / ITERATE
    return sm


def a_star():
    pass


if __name__ == "__main__":
    m = generate_matrix()
    l = build_list(m)
    deicstra(l, 1)
    """
    r = measure(deicstra, l, 1)
    print("nanos", r)
    r = measure(belman_ford, l, 1)
    print("nanos", r)
    """
