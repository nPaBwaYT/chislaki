import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import Lab1.src.solutions.task3_SI as lq


def euler_method_system(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], h: float):
    t0, tf = t_span
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


def euler_cauchy_method(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], h: float):
    t0, tf = t_span
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        y_pred = y[i] + h * f(t[i], y[i])
        y[i + 1] = y[i] + h * 0.5 * (f(t[i], y[i]) + f(t[i] + h, y_pred))

    return t, y


def improved_euler_method(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], h: float):
    t0, tf = t_span
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        y_half = y[i] + 0.5 * h * f(t[i], y[i])
        t_half = t[i] + 0.5 * h
        y[i + 1] = y[i] + h * f(t_half, y_half)

    return t, y


def runge_kutta_4_system(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], h: float):
    t0, tf = t_span
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y


def adams_4_system(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], h: float):
    t0, tf = t_span
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(3):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    f_vals = np.zeros((n, len(y0)))
    for i in range(4):
        f_vals[i] = f(t[i], y[i])

    for i in range(3, n - 1):
        y[i + 1] = y[i] + h / 24 * (55 * f_vals[i] - 59 * f_vals[i - 1] +
                                    37 * f_vals[i - 2] - 9 * f_vals[i - 3])
        f_vals[i + 1] = f(t[i + 1], y[i + 1])

        for _ in range(2):
            y_corr = y[i] + h / 24 * (9 * f(t[i + 1], y[i + 1]) + 19 * f_vals[i] -
                                      5 * f_vals[i - 1] + f_vals[i - 2])
            y[i + 1] = y_corr
            f_vals[i + 1] = f(t[i + 1], y[i + 1])

    return t, y


def runge_romberg_error(y_h, y_2h, p=4):
    error = np.abs(y_h - y_2h) / (2 ** p - 1)
    return error


def cauchy_problem(t, y):
    y1, y2 = y
    return np.array([y2, 4 * t * y2 - (4 * t ** 2 - 2) * y1])


def exact_solution_cauchy(x):
    return (1 + x) * np.exp(x ** 2)


print("ЗАДАЧА КОШИ")

y0_cauchy = np.array([1.0, 1.0])
t_span = (0.0, 1.0)
h = 0.1

t_euler, y_euler = euler_method_system(cauchy_problem, y0_cauchy, t_span, h)
y_euler_solution = y_euler[:, 0]

t_euler_cauchy, y_euler_cauchy = euler_cauchy_method(cauchy_problem, y0_cauchy, t_span, h)
y_euler_cauchy_solution = y_euler_cauchy[:, 0]

t_improved, y_improved = improved_euler_method(cauchy_problem, y0_cauchy, t_span, h)
y_improved_solution = y_improved[:, 0]

t_rk4, y_rk4 = runge_kutta_4_system(cauchy_problem, y0_cauchy, t_span, h)
y_rk4_solution = y_rk4[:, 0]

t_adams, y_adams = adams_4_system(cauchy_problem, y0_cauchy, t_span, h)
y_adams_solution = y_adams[:, 0]

y_exact_cauchy = exact_solution_cauchy(t_rk4)

from scipy.interpolate import interp1d
t_2h, y_2h = runge_kutta_4_system(cauchy_problem, y0_cauchy, t_span, 2 * h)
y_2h_interp = interp1d(t_2h, y_2h[:, 0], kind='cubic')(t_rk4)
error_rr = runge_romberg_error(y_rk4_solution, y_2h_interp)

print("\nСравнение решений в конечной точке x=1.0:")
print(f"Точное решение: y(1.0) = {y_exact_cauchy[-1]:.10f}")
print(f"Метод Эйлера:         y(1.0) = {y_euler_solution[-1]:.10f}, погрешность = {abs(y_euler_solution[-1] - y_exact_cauchy[-1]):.2e}")
print(f"Метод Эйлера-Коши:    y(1.0) = {y_euler_cauchy_solution[-1]:.10f}, погрешность = {abs(y_euler_cauchy_solution[-1] - y_exact_cauchy[-1]):.2e}")
print(f"Улучшенный Эйлер:     y(1.0) = {y_improved_solution[-1]:.10f}, погрешность = {abs(y_improved_solution[-1] - y_exact_cauchy[-1]):.2e}")
print(f"Метод РК4:            y(1.0) = {y_rk4_solution[-1]:.10f}, погрешность = {abs(y_rk4_solution[-1] - y_exact_cauchy[-1]):.2e}")
print(f"Метод Адамса:         y(1.0) = {y_adams_solution[-1]:.10f}, погрешность = {abs(y_adams_solution[-1] - y_exact_cauchy[-1]):.2e}")

print(f"\nОценка погрешности методом Рунге-Ромберга для РК4: max = {np.max(error_rr):.2e}")

print("\nx\t\tЭйлер    \t\tЭйлер-Коши    \tУлучшенный  \tРК4        \t\tАдамс    \t\tТочное")
for i in range(len(t_rk4)):
    print(f"{t_rk4[i]:.1f}\t\t{y_euler_solution[i]:.6f}\t\t{y_euler_cauchy_solution[i]:.6f}\t\t"
          f"{y_improved_solution[i]:.6f}\t\t{y_rk4_solution[i]:.6f}\t\t{y_adams_solution[i]:.6f}\t\t{y_exact_cauchy[i]:.6f}")


def shooting_method(f: Callable, bc_left: Callable, bc_right: Callable,
                    t_span: Tuple[float, float], h: float,
                    eta_guess1: float, eta_guess2: float, tol: float = 1e-6, max_iter: int = 100):
    t0, tf = t_span

    def solve_cauchy(eta):
        y0 = bc_left(eta)
        t, y = runge_kutta_4_system(f, y0, t_span, h)
        return t, y, bc_right(y[-1])

    eta = [eta_guess1, eta_guess2]
    phi = []

    for i in range(2):
        t, y, phi_val = solve_cauchy(eta[i])
        phi.append(phi_val)

    for iter_count in range(max_iter):
        if abs(phi[-1]) < tol:
            break

        eta_new = eta[-1] - (eta[-1] - eta[-2]) / (phi[-1] - phi[-2]) * phi[-1]

        t, y, phi_new = solve_cauchy(eta_new)

        eta.append(eta_new)
        phi.append(phi_new)

        if len(eta) > 10:
            eta = eta[-10:]
            phi = phi[-10:]

    return t, y, eta[-1], phi[-1], iter_count


def finite_difference_method(p, q, f_func, bc_left, bc_right,
                             t_span: Tuple[float, float], h: float):
    a, b = t_span
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    for i in range(1, n):
        A[i, i - 1] = 1 - p(x[i]) * h / 2
        A[i, i] = -2 + h ** 2 * q(x[i])
        A[i, i + 1] = 1 + p(x[i]) * h / 2
        B[i] = h ** 2 * f_func(x[i])

    # Левое
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    B[0] = 2 * h * (-1)

    # Правое
    A[n, n - 2] = 1
    A[n, n - 1] = -4
    A[n, n] = 3 + 2 * 2 * h
    B[n] = 2 * h * 3

    y = lq.main(A, B)

    return x, y


def boundary_problem(t, y):
    y1, y2 = y
    if abs(2 * t + 1) < 1e-10:
        denominator = 1e-10
    else:
        denominator = 2 * t + 1
    return np.array([y2, (-4 * t * y2 + 4 * y1) / denominator])


def bc_left_boundary(eta):
    return np.array([eta, -1.0])


def bc_right_boundary(y_final):
    y1, y2 = y_final
    return y2 + 2 * y1 - 3


def exact_solution_boundary(x):
    return x + np.exp(-2 * x)


def p_func(x):
    return 4 * x / (2 * x + 1)


def q_func(x):
    return -4 / (2 * x + 1)


def f_func(x):
    return 0


print("\n" + "=" * 60)
print("КРАЕВАЯ ЗАДАЧА")

t_span_boundary = (0.0, 1.0)
h_boundary = 0.1

print("\nМетод стрельбы:")
t_shooting, y_shooting, eta_final, phi_final, iter_count = shooting_method(
    boundary_problem, bc_left_boundary, bc_right_boundary,
    t_span_boundary, h_boundary, eta_guess1=0.5, eta_guess2=1.0
)

y_shooting_solution = y_shooting[:, 0]
print(f"Найденный параметр eta = {eta_final:.6f}")
print(f"Невязка на правой границе = {phi_final:.2e}")
print(f"Количество итераций = {iter_count}")

print("\nКонечно-разностный метод:")
x_fd, y_fd = finite_difference_method(p_func, q_func, f_func,
                                      bc_left_boundary, bc_right_boundary,
                                      t_span_boundary, h_boundary)

y_exact_boundary = exact_solution_boundary(t_shooting)

print("\nСравнение решений в нескольких точках:")
print("x\t\tСтрельба\t\tКонечно-разн.\t\tТочное")
for i in range(0, len(t_shooting), 2):
    print(f"{t_shooting[i]:.1f}\t\t{y_shooting_solution[i]:.8f}\t\t{y_fd[i]:.8f}\t\t{y_exact_boundary[i]:.8f}")

print(f"\nПогрешности в конечной точке x=1.0:")
print(f"Метод стрельбы:    {abs(y_shooting_solution[-1] - y_exact_boundary[-1]):.2e}")
print(f"Конечно-разностный: {abs(y_fd[-1] - y_exact_boundary[-1]):.2e}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(t_rk4, y_exact_cauchy, 'k-', linewidth=2, label='Точное решение')
ax1.plot(t_euler, y_euler_solution, 'b--', marker='o', markersize=3, label='Эйлер')
ax1.plot(t_euler_cauchy, y_euler_cauchy_solution, 'c--', marker='s', markersize=3, label='Эйлер-Коши')
ax1.plot(t_improved, y_improved_solution, 'm--', marker='^', markersize=3, label='Улучшенный Эйлер')
ax1.plot(t_rk4, y_rk4_solution, 'r--', marker='d', markersize=3, label='РК4')
ax1.plot(t_adams, y_adams_solution, 'g--', marker='v', markersize=3, label='Адамс')
ax1.set_xlabel('x')
ax1.set_ylabel('y(x)')
ax1.set_title('Задача Коши: Сравнение методов')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

ax2 = axes[0, 1]
ax2.plot(t_euler, np.abs(y_euler_solution - exact_solution_cauchy(t_euler)),
         'b-', label='Эйлер')
ax2.plot(t_euler_cauchy, np.abs(y_euler_cauchy_solution - exact_solution_cauchy(t_euler_cauchy)),
         'c-', label='Эйлер-Коши')
ax2.plot(t_improved, np.abs(y_improved_solution - exact_solution_cauchy(t_improved)),
         'm-', label='Улучшенный Эйлер')
ax2.plot(t_rk4, np.abs(y_rk4_solution - y_exact_cauchy),
         'r-', label='РК4')
ax2.plot(t_adams, np.abs(y_adams_solution - y_exact_cauchy),
         'g-', label='Адамс')
ax2.set_xlabel('x')
ax2.set_ylabel('Абсолютная погрешность')
ax2.set_title('Задача Коши: Погрешности методов')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

ax3 = axes[1, 0]
ax3.plot(t_shooting, y_exact_boundary, 'k-', linewidth=2, label='Точное решение')
ax3.plot(t_shooting, y_shooting_solution, 'b--', marker='o', markersize=4, label='Метод стрельбы')
ax3.plot(x_fd, y_fd, 'r--', marker='s', markersize=4, label='Конечно-разностный')
ax3.set_xlabel('x')
ax3.set_ylabel('y(x)')
ax3.set_title('Краевая задача: Сравнение методов')
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4 = axes[1, 1]
ax4.plot(t_shooting, np.abs(y_shooting_solution - y_exact_boundary),
         'b-', label='Метод стрельбы')
ax4.plot(x_fd, np.abs(y_fd - exact_solution_boundary(x_fd)),
         'r-', label='Конечно-разностный')
ax4.set_xlabel('x')
ax4.set_ylabel('Абсолютная погрешность')
ax4.set_title('Краевая задача: Погрешности методов')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()