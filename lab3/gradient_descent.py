import numpy as np
import matplotlib.pyplot as plt
import math
import os
import lab2.direct_methods as dm
from scipy import optimize
EPS = 0.001
OFFSET = 0

def to_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def calculate_lse(data, method, a, b):
    return sum([(method(x, a, b) - y) ** 2 for (x, y) in data])


# plz be carefull with gradient
def approx_func_linear(p, a, b):
    return p * a + b


def approx_func_linear_grad_a(p, a, b):
    return p


def approx_func_linear_grad_b(p, a, b):
    return 1


def approx_func_rational(p, a, b):
    return a / (1 + b * p + EPS)


def approx_func_rational_grad_a(p, a, b):
    return 1 / (1 + b * p + EPS)


def approx_func_rational_grad_b(p, a, b):
    return - a * p / ((1 + b * p + EPS) ** 2)


def build_differencial(data, approx_func, approx_func_grad_a, approx_func_grad_b):
    # just F
    f_a_b = lambda a, b: sum([(approx_func(x_p, a, b) - y_p) ** 2 for (x_p, y_p) in data])
    # F' a
    f_grad_a_a_b = lambda a, b: sum(
        [(approx_func(x_p, a, b) - y_p) * 2 * approx_func_grad_a(x_p, a, b) for (x_p, y_p) in data])
    # F' b
    f_grad_b_a_b = lambda a, b: sum(
        [(approx_func(x_p, a, b) - y_p) * 2 * approx_func_grad_b(x_p, a, b) for (x_p, y_p) in data])

    return f_a_b, f_grad_a_a_b, f_grad_b_a_b


def generate_data():
    a, b = np.random.random(2)
    x = [OFFSET + i / 100 for i in range(100)]
    tow = np.random.normal(size=100)
    y = [a * xx + tt + b for (xx, tt) in zip(x, tow)]
    return list(zip(x, y))


def minimize_lambda(a, b, f_a_b, f_grad_a, f_grad_b):
    # argmin l: F(x + l * grad F(x))
    # argmin l: F(a + l * grad_a(x), b + l * grad_b(x))

    min_func = lambda l: f_a_b(a - l * f_grad_a(a, b), b - l * f_grad_b(a, b))
    # _, _, _, argmim, _ = dm.golden_search(lambda l: f_a_b(a - l * f_grad_a(a, b), b - l * f_grad_b(a, b)), -1, 1)
    #return argmim
    return optimize.golden(min_func, brack=(-1, 1))

def fast_gradient_descent_method(iter_count, a, b, f, f_grad_a, f_grad_b):
    current_iter = 0
    points = [(a, b)]
    ls = [-1]
    while True:
        cur_a, cur_b = points[-1]
        l = minimize_lambda(cur_a, cur_b, f, f_grad_a, f_grad_b)
        next_a = cur_a - l * f_grad_a(cur_a, cur_b)
        next_b = cur_b - l * f_grad_b(cur_a, cur_b)

        ls.append(l)

        points.append((next_a, next_b))
        current_iter += 1
        if abs(f(cur_a, cur_b) - f(next_a, next_b)) < EPS or current_iter > iter_count:
            break
    return points, ls, current_iter


def visualize_least_sq_error(func, data):
    # x, y here is a and b in general
    x = [OFFSET + i / 100 for i in range(200)]
    y = [OFFSET + i / 100 for i in range(200)]
    z = []
    min_value = 1e9
    min_point = (OFFSET, OFFSET)
    for i in x:
        z_ = []
        for j in y:
            i = i
            j = j
            value = sum([(func(a, i, j) - b) ** 2 for (a, b) in data])
            if value < min_value:
                min_value = value
                min_point = (i, j)
            z_.append(value)
        z.append(z_)
    plt.contourf(x, y, z)
    plt.scatter(min_point[1], min_point[0], color="red", label="min point")
    print("trust min value = {}".format(min_value))
    print("trust min point = {}".format(min_point))


def visualize(data, apox, aprox_grad_a, aprox_grad_b):

    f_a_b, f_grad_a_a_b, f_grad_b_a_b = build_differencial(data, apox, aprox_grad_a, aprox_grad_b)
    pts, ls, _ = fast_gradient_descent_method(100, 1, 1, f_a_b, f_grad_a_a_b, f_grad_b_a_b)

    print("method min value = {}".format(calculate_lse(data, apox, pts[-1][0], pts[-1][1])))
    print("method min point = {}".format(pts[-1]))
    print("iters_count = {}".format(len(ls)))

    visualize_least_sq_error(apox, data)


    x, y = list(zip(*pts))
    # just because stupid omerican system
    plt.plot(y, x, color="green", label="method {} moving".format(to_name(apox)))

    plt.legend()
    plt.savefig(
        "{}\\images_3\\fast_grad_{}_{}.png".format(os.path.dirname(os.path.abspath(__file__)), "depth", to_name(apox)))
    plt.clf()


if __name__ == "__main__":
    data = generate_data()
    visualize(data, approx_func_linear, approx_func_linear_grad_a, approx_func_linear_grad_b)
    print("=" * 40)
    visualize(data, approx_func_rational, approx_func_rational_grad_a, approx_func_rational_grad_b)
