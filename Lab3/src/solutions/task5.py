def rectangle_method(f, a, b, n):
    """Метод прямоугольников"""
    h = (b - a) / n
    result = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        result += f(x_mid)
    return result * h


def trapezoidal_method(f, a, b, n):
    """Метод трапеций"""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def simpson_method(f, a, b, n):
    """Метод Симпсона"""
    if n % 2 != 0:
        n += 1  # Делаем n четным
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return result * h / 3


def runge_romberg(F_h, F_kh, k, p):
    """Метод Рунге-Ромберга для уточнения результата"""
    return F_h + (F_h - F_kh) / (k ** p - 1)


def f(x):
    return 1 / (3 * x ** 2 + 4 * x + 2)


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

# Оценка погрешности
print(f"\nРазности между шагами (оценка погрешности):")
print(f"Прямоугольники: {abs(rect_h2 - rect_h1):.6f}")
print(f"Трапеции: {abs(trap_h2 - trap_h1):.6f}")
print(f"Симпсон: {abs(simp_h2 - simp_h1):.6f}")