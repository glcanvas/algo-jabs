import numpy as np
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import queue

VERTEX = 100
EDGES = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_matrix():
    matrix = np.zeros((VERTEX, VERTEX))
    cnt = 0
    for i in range(VERTEX):
        for j in range(i + 1, VERTEX):
            if cnt < EDGES:
                if i + j % 2 == 0:
                    continue
                matrix[i][j] = 1
                matrix[j][i] = 1
                cnt += 1
    return matrix


def print_matrix(m):
    print("__", end=" ")
    for i in range(VERTEX):
        print(str(i).zfill(2), end=" ")
    print()
    for i in range(VERTEX):
        print(str(i).zfill(2), end=" ")
        for j in range(VERTEX):
            print(str(int(m[i][j])).zfill(2), end=" ")
        print()


def build_list(m):
    array = []
    for i in range(VERTEX):
        array.append([])
        for j in range(VERTEX):
            if m[i][j] == 1:
                array[-1].append(j)
    return array


def print_list(l):
    for i in range(VERTEX):
        print(str(i).zfill(2), end="->")
        for j in l[i]:
            print(str(j).zfill(2), end=" ")
        print()


def print_graph(g):
    gr = nx.Graph()
    for i in range(VERTEX):
        gr.add_node(i)
    for i in range(VERTEX):
        for j in g[i]:
            gr.add_edge(i, j)
    nx.draw(gr)
    plt.savefig("{}/images_5/{}.png".format(BASE_DIR, "gr"))
    return gr


def dfs(v: int, color: int, l: list, color_list: list):
    color_list[v] = color
    for to in l[v]:
        if color_list[to] == 0:
            dfs(to, color, l, color_list)


def dfs_init(l):
    color_list = [0 for i in range(VERTEX)]
    color = 1
    for i in range(VERTEX):
        if color_list[i] == 0:
            dfs(i, color, l, color_list)
            color += 1
    color = color - 1
    print(color)
    return color


def bfs(l: list, v: int):
    dist = [101 for _ in range(VERTEX)]
    q = queue.Queue()
    q.put(v)
    dist[v] = 0
    while not q.empty():
        v = q.get()
        for to in l[v]:
            if dist[to] == 101:
                dist[to] = dist[v] + 1
                q.put(to)
    for i in range(VERTEX):
        print(str(i).zfill(2) + "->" + str(dist[i]).zfill(3))
    return dist


if __name__ == "__main__":
    m = generate_matrix()
    print_matrix(m)
    l = build_list(m)
    print_list(l)
    gr = print_graph(l)
    dfs_init(l)
    bfs(l, 5)
