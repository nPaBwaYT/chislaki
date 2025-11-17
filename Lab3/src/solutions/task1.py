import numpy as np


def lagrange_interpolation(x, y, x_point):
    """Интерполяционный многочлен Лагранжа"""
    n = len(x)
    result = 0.0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_point - x[j]) / (x[i] - x[j])
        result += term

    return result


def divided_differences(x, y):
    """Разделенные разности для многочлена Ньютона"""
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])

    return coef[0, :]


def newton_interpolation(x, y, x_point):
    """Интерполяционный многочлен Ньютона"""
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