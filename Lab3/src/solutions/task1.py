import numpy as np
import math
import matplotlib.pyplot as plt

import Lab1.src.solutions.task3_SI as lq


# 3.5
def lagrange_interpolation(x, y, x_point):
    n = len(x)
    result = 0.0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_point - x[j]) / (x[i] - x[j])
        result += term

    return result


# 3.7
def divided_differences(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])

    return coef[0, :]


# 3.8
def newton_interpolation(x, y, x_point):
    n = len(x)
    coef = divided_differences(x, y)
    result = coef[0]

    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x_point - x[j])
        result += term

    return result

# Вариант a)
x_a = np.array([0, 1.7, 3.4, 5.1])
y_a = np.sqrt(x_a)
x_star = 3.0

lagrange_a = lagrange_interpolation(x_a, y_a, x_star)
newton_a = newton_interpolation(x_a, y_a, x_star)
exact_a = np.sqrt(x_star)

print(f"Вариант a):")
print(f"X_i = {x_a}")
print(f"Y_i = {y_a}")
print(f"X* = {x_star}")
print(f"Точное значение: {exact_a:.6f}")
print(f"Лагранж: {lagrange_a:.6f}, погрешность: {abs(lagrange_a - exact_a):.6f}")
print(f"Ньютон: {newton_a:.6f}, погрешность: {abs(newton_a - exact_a):.6f}")

# Вариант б)
x_b = np.array([0, 1.7, 4.0, 5.1])
y_b = np.sqrt(x_b)

lagrange_b = lagrange_interpolation(x_b, y_b, x_star)
newton_b = newton_interpolation(x_b, y_b, x_star)

print(f"\nВариант б):")
print(f"X_i = {x_b}")
print(f"Y_i = {y_b}")
print(f"X* = {x_star}")
print(f"Точное значение: {exact_a:.6f}")
print(f"Лагранж: {lagrange_b:.6f}, погрешность: {abs(lagrange_b - exact_a):.6f}")
print(f"Ньютон: {newton_b:.6f}, погрешность: {abs(newton_b - exact_a):.6f}")

plt.figure(figsize=(12, 5))

# Создаем гладкие кривые для отображения
x_smooth = np.linspace(0.1, 5.5, 500)
y_exact = np.sqrt(x_smooth)

y_lagrange_a = [lagrange_interpolation(x_a, y_a, xi) for xi in x_smooth]
y_newton_a = [newton_interpolation(x_a, y_a, xi) for xi in x_smooth]

y_lagrange_b = [lagrange_interpolation(x_b, y_b, xi) for xi in x_smooth]
y_newton_b = [newton_interpolation(x_b, y_b, xi) for xi in x_smooth]

# График 1: Вариант a)
plt.subplot(1, 2, 1)
plt.plot(x_smooth, y_exact, 'k-', linewidth=2, label='Точная функция: √x')
plt.plot(x_smooth, y_lagrange_a, 'b--', linewidth=1.5, label='Лагранж')
plt.plot(x_smooth, y_newton_a, 'r:', linewidth=1.5, label='Ньютон')
plt.scatter(x_a, y_a, color='red', s=80, zorder=5, label='Узлы интерполяции')
plt.scatter([x_star], [exact_a], color='green', s=100, marker='*', zorder=6, label=f'X* = {x_star}')
plt.axvline(x=x_star, color='green', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Вариант a) - Интерполяция √x')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 5.5)

# График 2: Вариант б)
plt.subplot(1, 2, 2)
plt.plot(x_smooth, y_exact, 'k-', linewidth=2, label='Точная функция: √x')
plt.plot(x_smooth, y_lagrange_b, 'b--', linewidth=1.5, label='Лагранж')
plt.plot(x_smooth, y_newton_b, 'r:', linewidth=1.5, label='Ньютон')
plt.scatter(x_b, y_b, color='red', s=80, zorder=5, label='Узлы интерполяции')
plt.scatter([x_star], [exact_a], color='green', s=100, marker='*', zorder=6, label=f'X* = {x_star}')
plt.axvline(x=x_star, color='green', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Вариант б) - Интерполяция √x')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 5.5)

plt.tight_layout()
plt.show()

# Вывод результатов в точке X*
print(f"\nСравнение в точке X* = {x_star}:")
print(f"Точное значение: {exact_a:.6f}")
print(f"Вариант a) - Лагранж: {lagrange_a:.6f}, погрешность: {abs(lagrange_a - exact_a):.6f}")
print(f"Вариант б) - Лагранж: {lagrange_b:.6f}, погрешность: {abs(lagrange_b - exact_a):.6f}")
print(f"Разница между вариантами: {abs(lagrange_a - lagrange_b):.6f}")
