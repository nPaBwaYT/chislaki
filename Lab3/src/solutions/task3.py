import numpy as np

def least_squares(x, y, degree):
    """Метод наименьших квадратов"""
    n = len(x)

    # Построение матрицы системы
    A = np.zeros((degree + 1, degree + 1))
    b = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x ** (i + j))
        b[i] = np.sum(y * x ** i)

    # Решение системы
    coefficients = np.linalg.solve(A, b)
    return coefficients


def polynomial_value(x, coeffs):
    """Вычисление значения многочлена"""
    result = 0
    for i, coef in enumerate(coeffs):
        result += coef * x ** i
    return result


x_ls = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y_ls = np.array([1.0, 1.0032, 1.0512, 1.2592, 1.8192, 3.0])

# Многочлен 1-й степени
coeffs_1 = least_squares(x_ls, y_ls, 1)
y_pred_1 = polynomial_value(x_ls, coeffs_1)
error_1 = np.sum((y_ls - y_pred_1) ** 2)

# Многочлен 2-й степени
coeffs_2 = least_squares(x_ls, y_ls, 2)
y_pred_2 = polynomial_value(x_ls, coeffs_2)
error_2 = np.sum((y_ls - y_pred_2) ** 2)

print("Исходные данные:")
print(f"x = {x_ls}")
print(f"y = {y_ls}")

print(f"\nМногочлен 1-й степени:")
print(f"F₁(x) = {coeffs_1[0]:.6f} + {coeffs_1[1]:.6f}*x")
print(f"Сумма квадратов ошибок: {error_1:.6f}")

print(f"\nМногочлен 2-й степени:")
print(f"F₂(x) = {coeffs_2[0]:.6f} + {coeffs_2[1]:.6f}*x + {coeffs_2[2]:.6f}*x^2")
print(f"Сумма квадратов ошибок: {error_2:.6f}")

# Вычисленные значения
print("\nВычисленные значения:")
print("x\t\tИсходное y\tF₁(x)\t\tF₂(x)")
for i in range(len(x_ls)):
    print(f"{x_ls[i]}\t\t{y_ls[i]}\t\t{y_pred_1[i]:.6f}\t{y_pred_2[i]:.6f}")