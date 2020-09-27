import numpy as np
import matplotlib.pyplot as plt
import math
import os
import lab2.direct_methods as dm
from scipy import optimize

K = 1_000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCTION = lambda x, a, b, c, d: (a * x + b) / (x ** 2 + c * x + d)


def to_name(method):
    return method.__name__.replace('_', ' ').capitalize()


def calculate_lse(data, method, a, b, c, d):
    return sum([(method(x, a, b, c, d) - y) ** 2 for (x, y) in data])


def generate_data():
    tow = np.random.normal(size=K)
    f_k = lambda k: 1 / (k ** 2 - 3 * k + 2)
    xx = []
    yy = []
    for k in range(0, K):
        x_k = 3 * k / K
        y = tow[k]
        if f_k(x_k) < - 100:
            y += -100
        elif -100 <= f_k(x_k) <= 100:
            y += f_k(x_k)
        else:
            y += 100
        xx.append(x_k)
        yy.append(y)
    return list(zip(xx, yy))


def sabstitude_point(a, b, c, d):
    x = []
    y = []
    for k in range(0, K):
        x_k = 3 * k / K
        x.append(x_k)
        y.append(FUNCTION(x_k, a, b, c, d))
    return x, y


def build_function(data):
    return lambda a, b, c, d: sum([(FUNCTION(x, a, b, c, d) - y) ** 2 for (x, y) in data])


def nelder_meald(data):
    f = build_function(data)
    res = optimize.minimize(lambda x: f(x[0], x[1], x[2], x[3]), np.array((0.1, 0.2, 0.3, 0.4)), method='Nelder-Mead')
    print("Nelder-Mead", res)
    return res.x


def levenberg_marquardt_method(data):
    f1 = lambda ab: [(FUNCTION(x_p, ab[0], ab[1], ab[2], ab[3]) - y_p) ** 2 for (x_p, y_p) in data]

    res = optimize.leastsq(f1, np.asarray([0.1, 0.2, 0.3, 0.4]))
    print("levenberg_marquardt", res)
    return res[0]


def differential_evolution(data):
    f = build_function(data)
    res = optimize.differential_evolution(lambda x: f(x[0], x[1], x[2], x[3]), ((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    print("differential_evolution", res)
    return res.x


def simultaneous_anneal(data):
    f = build_function(data)
    res = optimize.dual_annealing(lambda x: f(x[0], x[1], x[2], x[3]), ((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    print("simultaneous_anneal", res)
    return res.x


def show_data(data, point, name):
    xx = [i for (i, _) in data]
    yy = [i for (_, i) in data]

    plt.scatter(xx, yy, label="initial data", color="black")
    for p, n in zip(point, name):
        xx, yy = sabstitude_point(*p)
        plt.plot(xx, yy, label=n)
    plt.legend()
    plt.savefig("{}\\images_4\\xx.png".format(BASE_DIR))
    plt.clf()


if __name__ == "__main__":
    data = generate_data()
    ptr = nelder_meald(data)
    print("lse", calculate_lse(data, FUNCTION, *ptr))
    print("=" * 60)
    ptr2 = levenberg_marquardt_method(data)
    print("lse", calculate_lse(data, FUNCTION, *ptr2))
    print("=" * 60)
    ptr3 = differential_evolution(data)
    print("lse", calculate_lse(data, FUNCTION, *ptr3))
    print("=" * 60)
    ptr4 = simultaneous_anneal(data)
    print("lse", calculate_lse(data, FUNCTION, *ptr4))

    show_data(data, [ptr, ptr2, ptr3, ptr4],
              ["Nelder-Mead", "levenberg_marquardt", "differential_evolution", "simultaneous_anneal"])
