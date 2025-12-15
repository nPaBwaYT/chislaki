import numpy as np

import Lab1.src.solutions.task3_SI as lq

# 3.23
def rectangle_method(f, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        result += f(x_mid)
    return result * h


# 3.25
def trapezoidal_method(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


# 3.28
def simpson_method(f, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return result * h / 3


# 3.30
def runge_romberg(F_h, F_kh, k, p):
    return F_h + (F_h - F_kh) / (k ** p - 1)


def f(x):
    return 1 / (3 * x ** 2 + 4 * x + 2)


# Функции для оценки производных
def f_second_derivative(x):
    """Вторая производная функции f(x) = 1/(3x² + 4x + 2)"""
    # Аналитически вычисленная вторая производная
    denom = (3*x**2 + 4*x + 2)
    return (2*(18*x**2 + 24*x + 8)) / (denom**3) - (6*(6*x + 4)**2) / (denom**4)


def f_fourth_derivative(x):
    """Четвертая производная функции f(x) = 1/(3x² + 4x + 2)"""
    # Упрощенная оценка четвертой производной
    # Для сложных функций можно использовать численное дифференцирование
    denom = (3*x**2 + 4*x + 2)
    return 24*(324*x**4 + 864*x**3 + 864*x**2 + 384*x + 64) / (denom**5)


def estimate_error_rectangle(f_second, a, b, h):
    """Оценка погрешности метода прямоугольников по формуле 3.24"""
    # 3.24 - R ≤ (1/24) * h² * M₂ * (b - a)
    x_samples = np.linspace(a, b, 1000)
    M_2 = max(abs(f_second(x)) for x in x_samples)
    error_bound = (1/24) * h**2 * M_2 * (b - a)
    return error_bound


def estimate_error_trapezoidal(f_second, a, b, h):
    """Оценка погрешности метода трапеций по формуле 3.26"""
    # 3.26 - R ≤ (b - a)/12 * h² * M₂
    x_samples = np.linspace(a, b, 1000)
    M_2 = max(abs(f_second(x)) for x in x_samples)
    error_bound = (b - a) / 12 * h**2 * M_2
    return error_bound


def estimate_error_simpson(f_fourth, a, b, h):
    """Оценка погрешности метода Симпсона по формуле 3.29"""
    # 3.29 - R ≤ (b - a)/180 * h⁴ * M₄
    x_samples = np.linspace(a, b, 1000)
    M_4 = max(abs(f_fourth(x)) for x in x_samples)
    error_bound = (b - a) / 180 * h**4 * M_4
    return error_bound


a, b = -2, 2
h1, h2 = 1.0, 0.5

n1 = int((b - a) / h1)
n2 = int((b - a) / h2)

# Вычисление интегралов с разными шагами
rect_h1 = rectangle_method(f, a, b, n1)
rect_h2 = rectangle_method(f, a, b, n2)

trap_h1 = trapezoidal_method(f, a, b, n1)
trap_h2 = trapezoidal_method(f, a, b, n2)

simp_h1 = simpson_method(f, a, b, n1)
simp_h2 = simpson_method(f, a, b, n2)

# Уточнение методом Рунге-Ромберга
rect_refined = runge_romberg(rect_h2, rect_h1, 2, 2)
trap_refined = runge_romberg(trap_h2, trap_h1, 2, 2)
simp_refined = runge_romberg(simp_h2, simp_h1, 2, 4)

print(f"Функция: y = 1/(3x² + 4x + 2)")
print(f"Интервал: [{a}, {b}]")
print(f"Шаг h1 = {h1}, шаг h2 = {h2}")

print(f"\nМетод прямоугольников:")
print(f"  h1: {rect_h1:.6f}")
print(f"  h2: {rect_h2:.6f}")
print(f"  Уточненное: {rect_refined:.6f}")

print(f"\nМетод трапеций:")
print(f"  h1: {trap_h1:.6f}")
print(f"  h2: {trap_h2:.6f}")
print(f"  Уточненное: {trap_refined:.6f}")

print(f"\nМетод Симпсона:")
print(f"  h1: {simp_h1:.6f}")
print(f"  h2: {simp_h2:.6f}")
print(f"  Уточненное: {simp_refined:.6f}")

# Оценка погрешности по разностям
print(f"\nПогрешность (по разности результатов):")
print(f"Прямоугольники: {abs(rect_refined - rect_h1):.6f}")
print(f"Трапеции: {abs(trap_refined - trap_h1):.6f}")
print(f"Симпсон: {abs(simp_refined - simp_h1):.6f}")

# Оценка погрешности по формулам из методички
print(f"\nОценка погрешностей:")
print("Прямоугольники:")
error_rect_h1 = estimate_error_rectangle(f_second_derivative, a, b, h1)
error_rect_h2 = estimate_error_rectangle(f_second_derivative, a, b, h2)
print(f"  h1: {error_rect_h1:.6f}")
print(f"  h2: {error_rect_h2:.6f}")

print("Трапеция:")
error_trap_h1 = estimate_error_trapezoidal(f_second_derivative, a, b, h1)
error_trap_h2 = estimate_error_trapezoidal(f_second_derivative, a, b, h2)
print(f"  h1: {error_trap_h1:.6f}")
print(f"  h2: {error_trap_h2:.6f}")

print("Симпсон:")
error_simp_h1 = estimate_error_simpson(f_fourth_derivative, a, b, h1/2)  # Для Симпсона h = (b-a)/(2n)
error_simp_h2 = estimate_error_simpson(f_fourth_derivative, a, b, h2/2)
print(f"  h1: {error_simp_h1:.6f}")
print(f"  h2: {error_simp_h2:.6f}")
