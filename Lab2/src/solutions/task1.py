import sys
from math import log, sqrt
import matplotlib.pyplot as plt


def f(x):
    return 2**x + x**2 - 2


def f_deriv(x):
    return 2**x * log(2) + 2*x


def f_second_deriv(x):
    return 2**x * (log(2))**2 + 2


def phy(x):
    return sqrt(2 - 2 ** x)


def phy_deriv(x):
    return 1 / (2 * sqrt(2 - 2 ** x)) * (-(2 ** x * log(2)))


def inp():
    if len(sys.argv) == 1:
        eps = 10 ** int(input("Введите точность (степень): "))
    else:
        f = open(sys.argv[1], 'r', encoding="UTF-8")
        old_stdin, sys.stdin = sys.stdin, f
        eps = 10 ** int(sys.stdin.readline().strip("\n\t "))
        sys.stdin = old_stdin
    return eps


def main():
    eps = inp()

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"hspace": 0.3, "wspace": 0.2})
    axs[0][0].set_title("График уравнения")
    axs[1][0].set_title("Сходимость")
    axs[0][1].set_title("phi(x)")
    axs[1][1].set_title("phi derivative")

    left_border, right_border = 0, 1

    x_arr = [left_border + 2 * eps * i for i in range(int((right_border - left_border) / (2 * eps)) + 1)]
    y_arr = [f(x) for x in x_arr]
    axs[0][0].plot(x_arr, y_arr, color='green')

    left_border, right_border = 0.1, 1

    x_arr = [left_border + 2 * eps * i for i in range(int((right_border - left_border) / (2 * eps)) + 1)]
    approximation_x = 0.1

    if f(approximation_x) * f_second_deriv(approximation_x) > 0:
        return

    new_approximation_x = approximation_x - f(approximation_x) / f_deriv(approximation_x)
    newton_eps = [abs(new_approximation_x - approximation_x)]

    while abs(new_approximation_x - approximation_x) > eps:
        approximation_x = new_approximation_x
        new_approximation_x = approximation_x - f(approximation_x) / f_deriv(approximation_x)
        newton_eps.append(abs(new_approximation_x - approximation_x))

    print(f"Ньютон = {new_approximation_x:.{int(-log(eps, 10)) + 2}}")

    left_border, right_border = 0.6, 0.7
    x_arr = [left_border + 2 * eps * i for i in range(int((right_border - left_border) / (2 * eps)) + 1)]

    phy_y = [phy(x) for x in x_arr]
    axs[0][1].plot(x_arr, phy_y, color='green')

    phy_deriv_y = [phy_deriv(x) for x in x_arr]
    axs[1][1].plot(x_arr, phy_deriv_y, color='red')

    q = abs(phy_deriv(right_border))
    approximation_x = 0.6
    new_approximation_x = phy(approximation_x)
    simp_eps = [q / (1 - q) * abs(new_approximation_x - approximation_x)]

    while (q / (1 - q) * abs(new_approximation_x - approximation_x)) > eps:
        approximation_x = new_approximation_x
        new_approximation_x = phy(approximation_x)
        simp_eps.append(q / (1 - q) * abs(new_approximation_x - approximation_x))

    print(f"Итерации = {new_approximation_x:.{int(-log(eps, 10)) + 2}}")
    axs[1][0].semilogy([i for i in range(1, len(simp_eps) + 1)], simp_eps, '-o', color='red', markersize=4)
    axs[1][0].semilogy([i for i in range(1, len(newton_eps) + 1)], newton_eps, '-o', color='green', markersize=4)

    plt.show()


if __name__ == "__main__":
    main()