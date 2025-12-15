import numpy as np
import matplotlib.pyplot as plt

import Lab1.src.solutions.task3_SI as lq


# 3.17
def least_squares(x, y, degree):
    n = len(x)

    # Построение матрицы системы
    A = np.zeros((degree + 1, degree + 1))
    b = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x ** (i + j))
        b[i] = np.sum(y * x ** i)

    # Решение системы
    coefficients = lq.main(A, B)
    return coefficients


def polynomial_value(x, coeffs):
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

plt.figure(figsize=(10, 6))

plt.scatter(x_ls, y_ls, color='black', s=50, zorder=5, label='Исходные данные')

x_smooth = np.linspace(0, 1, 100)
y_smooth_1 = polynomial_value(x_smooth, coeffs_1)
y_smooth_2 = polynomial_value(x_smooth, coeffs_2)

plt.plot(x_smooth, y_smooth_1, 'b-', linewidth=2, label=f'F₁(x) (линейная) - ошибка: {error_1:.4f}')
plt.plot(x_smooth, y_smooth_2, 'r--', linewidth=2, label=f'F₂(x) (квадратичная) - ошибка: {error_2:.4f}')

plt.scatter(x_ls, y_pred_1, color='blue', s=30, alpha=0.7, marker='x')
plt.scatter(x_ls, y_pred_2, color='red', s=30, alpha=0.7, marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Метод наименьших квадратов\nАппроксимация многочленами 1-й и 2-й степени')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

textstr = '\n'.join((
    f'F₁(x) = {coeffs_1[0]:.4f} + {coeffs_1[1]:.4f}·x',
    f'F₂(x) = {coeffs_2[0]:.4f} + {coeffs_2[1]:.4f}·x + {coeffs_2[2]:.4f}·x²'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.show()

plt.figure(figsize=(10, 4))

errors_1 = y_ls - y_pred_1
errors_2 = y_ls - y_pred_2

plt.subplot(1, 2, 1)
plt.stem(x_ls, errors_1, basefmt=" ")
plt.title('Ошибки линейной аппроксимации')
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.stem(x_ls, errors_2, basefmt=" ")
plt.title('Ошибки квадратичной аппроксимации')
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
