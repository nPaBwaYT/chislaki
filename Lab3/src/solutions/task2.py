import numpy as np
import matplotlib.pyplot as plt

import Lab1.src.solutions.task3_SI as lq

# 3.13, 3.14
def cubic_spline_coefficients(x, y):
    n = len(x) - 1
    # 3.13
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
    c = lq.main(A, B)

    # Вычисление остальных коэффициентов
    # 3.14
    a = y[:-1]
    b_coef = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        # 3.14
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        # 3.14
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b_coef, c[:-1], d

# 3.11
def evaluate_spline(x, y, x_point, a, b, c, d):
    n = len(x) - 1

    # Определение интервала
    for i in range(n):
        if x[i] <= x_point <= x[i + 1]:
            dx = x_point - x[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return None


def evaluate_spline_full_range(x, a, b, c, d, num_points=100):
    """Вычисление сплайна на всем диапазоне для построения графика"""
    n = len(x) - 1
    x_full = []
    y_full = []

    for i in range(n):
        x_segment = np.linspace(x[i], x[i + 1], num_points)
        y_segment = []
        for x_point in x_segment:
            dx = x_point - x[i]
            # 3.11
            y_val = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
            y_segment.append(y_val)
        x_full.extend(x_segment)
        y_full.extend(y_segment)

    return np.array(x_full), np.array(y_full)

x_spline = np.array([0.0, 1.7, 3.4, 5.1, 6.8])
y_spline = np.array([0.0, 1.3038, 1.8439, 2.2583, 2.6077])
x_star_spline = 3.0

a, b_coef, c, d = cubic_spline_coefficients(x_spline, y_spline)
spline_value = evaluate_spline(x_spline, y_spline, x_star_spline, a, b_coef, c, d)

print(f"Узлы: x = {x_spline}")
print(f"Значения: y = {y_spline}")
print(f"X* = {x_star_spline}")
print(f"Значение сплайна в X*: {spline_value:.6f}")

print("\nКоэффициенты сплайна:")
for i in range(len(a)):
    print(f"Интервал [{x_spline[i]}, {x_spline[i + 1]}]:")
    print(f"  a_{i + 1} = {a[i]:.6f}, b_{i + 1} = {b_coef[i]:.6f}, c_{i + 1} = {c[i]:.6f}, d_{i + 1} = {d[i]:.6f}")

plt.figure(figsize=(9, 5))

plt.subplot(1, 1, 1)
colors = ['red', 'blue', 'green', 'orange']

for i in range(len(a)):
    x_segment = np.linspace(x_spline[i], x_spline[i + 1], 50)
    y_segment = []
    for x_point in x_segment:
        dx = x_point - x_spline[i]
        y_val = a[i] + b_coef[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
        y_segment.append(y_val)

    plt.plot(x_segment, y_segment, color=colors[i % len(colors)], linewidth=2,
             label=f'Интервал {i + 1}: [{x_spline[i]}, {x_spline[i + 1]}]')

plt.scatter(x_spline, y_spline, color='black', s=80, zorder=5, label='Узлы')
plt.scatter([x_star_spline], [spline_value], color='purple', s=100, marker='*',
            zorder=6, label=f'X* = {x_star_spline}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Отдельные сегменты кубического сплайна')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()