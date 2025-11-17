import numpy as np

def numerical_derivative(x, y, x_point, method='central'):
    """Численное дифференцирование"""
    n = len(x)

    # Находим индекс точки
    idx = np.where(np.abs(x - x_point) < 1e-10)[0][0]

    if method == 'forward' and idx < n - 1:
        # Правосторонняя производная
        return (y[idx + 1] - y[idx]) / (x[idx + 1] - x[idx])

    elif method == 'backward' and idx > 0:
        # Левосторонняя производная
        return (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])

    elif method == 'central' and idx > 0 and idx < n - 1:
        # Центральная разность (второй порядок)
        return (y[idx + 1] - y[idx - 1]) / (x[idx + 1] - x[idx - 1])

    else:
        return None


def numerical_second_derivative(x, y, x_point):
    """Численная вторая производная"""
    n = len(x)
    idx = np.where(np.abs(x - x_point) < 1e-10)[0][0]

    if idx > 0 and idx < n - 1:
        h1 = x[idx] - x[idx - 1]
        h2 = x[idx + 1] - x[idx]
        return ((y[idx + 1] - y[idx]) / h2 - (y[idx] - y[idx - 1]) / h1) / ((h1 + h2) / 2)

    return None


x_der = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
y_der = np.array([1.7722, 1.5708, 1.3694, 1.1593, 0.9273])
x_star_der = 0.2

# Первая производная
deriv_forward = numerical_derivative(x_der, y_der, x_star_der, 'forward')
deriv_backward = numerical_derivative(x_der, y_der, x_star_der, 'backward')
deriv_central = numerical_derivative(x_der, y_der, x_star_der, 'central')

# Вторая производная
second_deriv = numerical_second_derivative(x_der, y_der, x_star_der)

print(f"x = {x_der}")
print(f"y = {y_der}")
print(f"X* = {x_star_der}")

print(f"\nПервая производная:")
print(f"Левосторонняя: {deriv_backward:.6f}")
print(f"Правосторонняя: {deriv_forward:.6f}")
print(f"Центральная разность: {deriv_central:.6f}")

print(f"\nВторая производная: {second_deriv:.6f}")