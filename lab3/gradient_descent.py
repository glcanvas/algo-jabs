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


def zero(p, a, b):
    return 0


def approx_func_rational(p, a, b):
    return a / (1 + b * p)


def approx_func_rational_grad_a(p, a, b):
    return 1 / (1 + b * p)


def approx_func_rational_grad_a_a(p, a, b):
    return 0


def approx_func_rational_grad_a_b(p, a, b):
    return -p / (1 + b * p)


def approx_func_rational_grad_b(p, a, b):
    return - a * p / ((1 + b * p) ** 2)


def approx_func_rational_grad_b_a(p, a, b):
    return -  p / ((1 + b * p) ** 2)


def approx_func_rational_grad_b_b(p, a, b):
    return 2 * a * p * p / ((1 + b * p) ** 3)


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


def build_differencial_second(data, f, f_grad_a, f_grad_a_a, f_grad_a_b, f_grad_b, f_grad_b_a, f_grad_b_b):
    # F = (f(x) - y) ^ 2
    # F'a = 2 * (f(x) - y) * f'a(x)
    # F'a'b = 2 * (f(x) - y) * f'a'b(x) + 2 * f'b(x) * f'a(x)

    # just F
    f1 = lambda a, b: sum([(f(x_p, a, b) - y_p) ** 2 for (x_p, y_p) in data])
    # F' a
    f1_grad_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a(x_p, a, b) for (x_p, y_p) in data])
    # F'' a a
    f1_grad_a_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a_a(x_p, a, b) + 2 * (f_grad_a(x_p, a, b) ** 2)
         for (x_p, y_p) in data])
    # F'' a b
    f1_grad_a_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_a_b(x_p, a, b) + 2 * f_grad_a(x_p, a, b) * f_grad_b(x_p, a, b)
         for (x_p, y_p) in data])

    # F' b
    f1_grad_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b(x_p, a, b) for (x_p, y_p) in data])
    # F'' b b
    f1_grad_b_b = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b_b(x_p, a, b) + 2 * (f_grad_b(x_p, a, b) ** 2)
         for (x_p, y_p) in data])
    # F'' b a
    f1_grad_b_a = lambda a, b: sum(
        [(f(x_p, a, b) - y_p) * 2 * f_grad_b_a(x_p, a, b) + 2 * f_grad_a(x_p, a, b) * f_grad_b(x_p, a, b)
         for (x_p, y_p) in data])

    return f1, (f1_grad_a, f1_grad_b), ((f1_grad_a_a, f1_grad_a_b), (f1_grad_b_a, f1_grad_b_b))


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
    # return argmim
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


def visualize_data_line(data, func, a, b, type):
    plt.scatter([i for (i, _) in data], [j for (_, j) in data], color="blue", label="initial data")
    x_line = [i / 100 for i in range(100)]
    y_line = [func(i, a, b) for i in x_line]
    plt.plot(x_line, y_line, color="green", label="approximation line: {} {}".format("fast_grad", to_name(func)))
    plt.legend()
    plt.savefig(
        "{}\\images_3\\{}_{}_{}.png".format(os.path.dirname(os.path.abspath(__file__)), type, "line", to_name(func)))
    plt.clf()


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

    visualize_data_line(data, apox, pts[-1][0], pts[-1][1], "fast_grad")


def conjugate_gradient_descent(data, aprox, aprox_grad_a, aprox_grad_b):
    f_a_b, f_grad_a_a_b, f_grad_b_a_b = build_differencial(data, aprox, aprox_grad_a, aprox_grad_b)
    fprime = lambda v: np.asarray((f_grad_a_a_b(v[0], v[1]), f_grad_b_a_b(v[0], v[1])))
    f = lambda v: f_a_b(v[0], v[1])
    x0 = np.asarray((0.1, 0.1))
    res = optimize.fmin_cg(f, x0=x0, fprime=fprime)
    print(res)
    visualize_data_line(data, aprox, res[0], res[1], "sgd")


def newton_method(data, aprox, aprox_grad_a, aprox_grad_a_a, aprox_grad_a_b, aprox_grad_b, aprox_grad_b_a,
                  aprox_grad_b_b):
    func, f_first, f_second = build_differencial_second(data, aprox, aprox_grad_a,
                                                        aprox_grad_a_a, aprox_grad_a_b,
                                                        aprox_grad_b, aprox_grad_a_b,
                                                        aprox_grad_b_b)
    fprime = lambda v: np.asarray((f_first[0](v[0], v[1]), f_first[1](v[0], v[1])))
    fprime2 = lambda v: np.asarray(((f_second[0][0](v[0], v[1]), f_second[0][1](v[0], v[1])),
                                    (f_second[1][0](v[0], v[1]), f_second[1][1](v[0], v[1]))))
    f = lambda v: func(v[0], v[1])

    res = optimize.minimize(f, np.asarray((0.1, 0.2)), jac=fprime, hess=fprime2, method="Newton-CG")
    print(res)
    visualize_data_line(data, aprox, res.x[0], res.x[1], "newton")


def levenberg_marquardt_method(data, aprox):
    #
    f1 = lambda ab: [(aprox(x_p, ab[0], ab[1]) - y_p) ** 2 for (x_p, y_p) in data]

    res = optimize.leastsq(f1, np.asarray([0.1, 0.1]))
    print(calculate_lse(data, aprox, res[0][0], res[0][1]))
    print(res)


if __name__ == "__main__":
    data = generate_data()
    """
    visualize(data, approx_func_linear, approx_func_linear_grad_a, approx_func_linear_grad_b)
    print("=" * 40)
    visualize(data, approx_func_rational, approx_func_rational_grad_a, approx_func_rational_grad_b)
    """
    #print("=" * 40)
    #conjugate_gradient_descent(data, approx_func_linear, approx_func_linear_grad_a, approx_func_linear_grad_b)
    #print("=" * 40)
    conjugate_gradient_descent(data, approx_func_rational, approx_func_rational_grad_a, approx_func_rational_grad_b)
    print("=" * 40)

    # newton_method(data, approx_func_linear, approx_func_linear_grad_a, zero, zero,
    #              approx_func_linear_grad_b, zero, zero)

    # newton_method(data, approx_func_rational, approx_func_rational_grad_a, approx_func_rational_grad_a_a,
    #              approx_func_rational_grad_a_b,
    #              approx_func_rational_grad_b, approx_func_rational_grad_a_b, approx_func_rational_grad_b_b)

    levenberg_marquardt_method(data, approx_func_rational)
