import numpy as np


def cubic_spline_coefficients(x, y):
    """Вычисление коэффициентов кубического сплайна"""
    n = len(x) - 1
    h = np.diff(x)

    # Матрица для системы уравнений
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # Граничные условия (нулевая кривизна)
    A[0, 0] = 1
    A[n, n] = 1

    # Внутренние точки
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Решение системы для c_i
    c = np.linalg.solve(A, b)

    # Вычисление остальных коэффициентов
    a = y[:-1]
    b_coef = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b_coef, c[:-1], d


def evaluate_spline(x, y, x_point, a, b, c, d):
    """Вычисление значения сплайна в точке"""
    n = len(x) - 1

    # Определение интервала
    for i in range(n):
        if x[i] <= x_point <= x[i + 1]:
            dx = x_point - x[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return None


x_spline = np.array([0.0, 1.7, 3.4, 5.1, 6.8])
y_spline = np.array([0.0, 1.3038, 1.8439, 2.2583, 2.6077])
x_star_spline = 3.0

a, b, c, d = cubic_spline_coefficients(x_spline, y_spline)
spline_value = evaluate_spline(x_spline, y_spline, x_star_spline, a, b, c, d)

print(f"Узлы: x = {x_spline}")
print(f"Значения: y = {y_spline}")
print(f"X* = {x_star_spline}")
print(f"Значение сплайна в X*: {spline_value:.6f}")

# Вывод коэффициентов сплайна
print("\nКоэффициенты сплайна:")
for i in range(len(a)):
    print(f"Интервал [{x_spline[i]}, {x_spline[i + 1]}]:")
    print(f"  a_{i + 1} = {a[i]:.6f}, b_{i + 1} = {b[i]:.6f}, c_{i + 1} = {c[i]:.6f}, d_{i + 1} = {d[i]:.6f}")
