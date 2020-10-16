"""
THE BEST INDIAN SOURCE CODE:https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import queue
import time

ROW = 10
COLUMN = 10
BLOCKS = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(0)
ITERATE = 1000


class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:

        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:

            for closed_child in closed_list:
                if child == closed_child:
                    continue

            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                    (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)


def gen_data():
    array = np.zeros((ROW, ROW))
    for i in range(BLOCKS):
        p = (random.randint(0, ROW - 1), random.randint(0, ROW - 1))
        array[p[0]][p[1]] = 1
    return array


def print_cells(maze, path):
    for i in range(ROW):
        for j in range(ROW):
            if maze[i][j] == 1:
                plt.scatter([i], [j], color="red")
            else:
                plt.scatter([i], [j], color="blue")

    for (i, j) in path:
        plt.scatter([i], [j], color="green")
    plt.savefig("{}/images/{}.png".format(BASE_DIR, "maze"))
    plt.clf()


def measure(maze):
    sm = 0
    for i in range(ITERATE):
        print(i)
        start_time = time.monotonic_ns()
        a = (random.randint(0, ROW - 1), random.randint(0, ROW - 1))
        b = (random.randint(0, ROW - 1), random.randint(0, ROW - 1))
        while maze[a[0]][a[1]] == 1 or maze[b[0]][b[1]] == 1:
            a = (random.randint(0, ROW - 1), random.randint(0, ROW - 1))
            b = (random.randint(0, ROW - 1), random.randint(0, ROW - 1))
        print(a, b, maze[a[0]][a[1]], maze[b[0]][b[1]])
        astar(maze, a, b)
        end_time = time.monotonic_ns()
        sm += end_time - start_time
    sm = sm / ITERATE
    return sm


def main():
    maze = gen_data()
    print(maze)
    start = (0, 0)
    end = (7, 6)
    path = astar(maze, start, end)
    print(path)
    print_cells(maze, path)
    print("nanos", measure(maze))


if __name__ == '__main__':
    main()
