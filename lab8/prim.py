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


def print_graph_min(g, p):
    gr = nx.Graph()
    for i in range(VERTEX):
        gr.add_node(i)
    for to in range(VERTEX):
        fr = p[to]
        if fr == -1:
            continue
        gr.add_edge(to, fr)
    nx.draw(gr)
    plt.savefig("{}/images/{}.png".format(BASE_DIR, "gr_min"))
    plt.clf()
    return gr


def print_graph(g):
    gr = nx.Graph()
    for i in range(VERTEX):
        gr.add_node(i)
    for i in range(VERTEX):
        for j in g[i]:
            gr.add_edge(i, j[0])
    nx.draw(gr)
    plt.savefig("{}/images/{}.png".format(BASE_DIR, "gr"))
    plt.clf()
    return gr


def prim(g, s):
    min_e = [1e9 for i in range(VERTEX)]
    u = [False for i in range(VERTEX)]
    p = [-1 for i in range(VERTEX)]
    min_e[s] = 0
    for i in range(VERTEX):
        v = -1
        for j in range(VERTEX):
            if not u[j] and (v == -1 or min_e[j] < min_e[v]):
                v = j
        if min_e[v] == 1e9:
            break
        u[v] = True
        for j in range(len(g[v])):
            to = g[v][j][0]
            ln = g[v][j][1]
            if ln < min_e[to]:
                min_e[to] = ln
                p[to] = v
    for i, j in enumerate(min_e):
        print(i, '->', j)
    print("====")
    for i, j in enumerate(p):
        print(i, '->', j)
    return min_e, p


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
    print_graph(l)
    _, p = prim(l, 1)
    print_graph_min(l, p)
