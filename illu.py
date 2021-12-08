import numpy as np
import matplotlib.pyplot as plt

def formula_x(t):
    return -15 * t ** 4 + 6 * t ** 5 + 10 * t ** 3

def formula_v(t):
    return -60 * t ** 3 + 30 * t ** 4 + 30 * t ** 2

def formula_a(t):
    return -180 * t ** 2 + 120 * t ** 3 + 60 * t

def formula_j(t):
    return -360 * t + 360 * t ** 2 + 60

if __name__ == '__main__':
    T = np.linspace(0, 1, 101)
    X = [formula_x(t) for t in T]
    plt.plot(T, X)
    V = [formula_v(t) for t in T]
    plt.plot(T, V)
    A = [formula_a(t) for t in T]
    plt.plot(T, A)
    # J = [formula_j(t) for t in T]
    # plt.plot(T, J)
    plt.show()
