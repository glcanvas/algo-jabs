import matplotlib.pyplot as plt
import math
import os

MAX_ITERATE_NUMBER = 1_000
EPS = 0.001


def to_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def one_dimensional_method_solver(current_function, a_0: float, b_0: float, eps: float, max_iter_count: int,
                                  left_x_border, right_x_border):
    a = [a_0]
    b = [b_0]
    current_iter = 0

    while abs(b[-1] - a[-1]) > eps and current_iter < max_iter_count:
        x_left = left_x_border(a[-1], b[-1])
        x_right = right_x_border(a[-1], b[-1])
        if current_function(x_left) < current_function(x_right):
            a.append(a[-1])
            b.append(x_right)
        else:
            a.append(x_left)
            b.append(b[-1])
        current_iter += 1
    return a, b, current_iter, \
           a[-1] if current_function(a[-1]) < current_function(b[-1]) else b[-1], \
           min(current_function(a[-1]), current_function(b[-1]))


def dichotomy(f, a_0, b_0):
    delta = EPS / 3
    left_func = lambda a, b: (a + b) / 2 - delta
    right_func = lambda a, b: (a + b) / 2 + delta
    return one_dimensional_method_solver(f, a_0, b_0, EPS, MAX_ITERATE_NUMBER, left_func, right_func)


def golden_search(f, a_0, b_0):
    left_func = lambda a, b: a + (3 - math.sqrt(5)) / 2 * (b - a)
    right_func = lambda a, b: b + (math.sqrt(5) - 3) / 2 * (b - a)
    return one_dimensional_method_solver(f, a_0, b_0, EPS, MAX_ITERATE_NUMBER, left_func, right_func)


def exhaustive_search(f, a_0, b_0):
    iter_counts = int((b_0 - a_0) / EPS + 100)
    delta = (b_0 - a_0) / iter_counts
    a = [a_0]
    min_f = f(a_0)
    root = a_0
    for i in range(iter_counts):
        x = a_0 + delta * i
        f_v = f(x)
        if min_f > f_v:
            min_f = f_v
            root = x
        a.append(x)
    return a, [], iter_counts, root, min_f


def f_1(x):
    return x * x * x


def f_2(x):
    return abs(x - 0.2)


def f_3(x):
    return x * math.sin(1 / x)


def visualize_one_dim_method(func, method, a_0, b_0):
    a_points, b_points, iter_count, root, root_value = method(func, a_0, b_0)
    delta = (b_0 - a_0) / MAX_ITERATE_NUMBER
    x = [a_0 + i * delta for i in range(MAX_ITERATE_NUMBER)]
    y = [func(i) for i in x]

    plt.plot(x, y, label="initial chart")
    plt.scatter([i for i in a_points], [func(i) for i in a_points], label="left points", marker='o', color="green")
    plt.scatter([i for i in b_points], [func(i) for i in b_points], label="right points", marker='o', color="red")
    plt.scatter(root, root_value, label="answer", color="orange")
    plt.legend()

    plt.savefig("{}\\images\\{}-{}.png".format(os.path.dirname(os.path.abspath(__file__)), to_name(func),
                                               to_name(method)))
    print(to_name(method), " = ", iter_count)
    plt.clf()


if __name__ == "__main__":
    print("=" * 20)
    print("f = x ^ 3")
    visualize_one_dim_method(f_1, golden_search, 0, 1)
    visualize_one_dim_method(f_1, exhaustive_search, 0, 1)
    visualize_one_dim_method(f_1, dichotomy, 0, 1)

    print("=" * 20)
    print("f = |x - 0.2|")
    visualize_one_dim_method(f_2, golden_search, 0, 1)
    visualize_one_dim_method(f_2, exhaustive_search, 0, 1)
    visualize_one_dim_method(f_2, dichotomy, 0, 1)

    print("=" * 20)
    print("f = x * sin(1/x)")
    visualize_one_dim_method(f_3, golden_search, 0.01, 1)
    visualize_one_dim_method(f_3, exhaustive_search, 0.01, 1)
    visualize_one_dim_method(f_3, dichotomy, 0.01, 1)
