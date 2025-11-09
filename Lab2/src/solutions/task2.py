import numpy as np
import matplotlib.pyplot as plt


def simple_iteration_method(x0, epsilon=1e-8, max_iter=100):
    def phi(x):
        x1, x2 = x
        phi1 = np.sqrt(4 - x2 ** 2)
        phi2 = np.log(x1 + 2)
        return np.array([phi1, phi2])

    def phi_deriv(x):
        x1, x2 = x
        phi_deriv11 = 0
        phi_deriv12 = 1 / (2 * np.sqrt(4 - x2 ** 2)) * (-2) * x2
        phi_deriv21 = 1 / (x1 + 2)
        phi_deriv22 = 0
        return np.array([[phi_deriv11, phi_deriv12], [phi_deriv21, phi_deriv22]])

    x = x0.copy()
    iterations_data = {
        'iteration': [],
        'delta_x_norm': [],
        'f_norm': [],
        'x1_values': [],
        'x2_values': []
    }

    phi_der = phi_deriv(x)
    q = max(phi_der[0][0] + phi_der[0][1], phi_der[1][0] + phi_der[1][1])

    for k in range(max_iter):
        iterations_data['iteration'].append(k)
        iterations_data['x1_values'].append(x[0])
        iterations_data['x2_values'].append(x[1])

        f_val = f(x)
        iterations_data['f_norm'].append(np.linalg.norm(f_val))

        x_new = phi(x)

        delta_x = np.linalg.norm(x_new - x)

        iterations_data['delta_x_norm'].append(delta_x)

        if q / (1 - q) * delta_x < epsilon:
            break

        x = x_new
    else:
        print(f"Достигнуто максимальное количество итераций ({max_iter})")

    return x, iterations_data


def newton_method(x0, epsilon=1e-8, max_iter=50):
    def f(x):
        x1, x2 = x
        f1 = x1 ** 2 + x2 ** 2 - 4
        f2 = x1 - np.exp(x2) + 2
        return np.array([f1, f2])

    def jacobian(x):
        x1, x2 = x
        J11 = 2 * x1  # df1/dx1
        J12 = 2 * x2  # df1/dx2
        J21 = 1  # df2/dx1
        J22 = -np.exp(x2)  # df2/dx2
        return np.array([[J11, J12], [J21, J22]])

    x = x0.copy()
    iterations_data = {
        'iteration': [],
        'delta_x_norm': [],
        'f_norm': [],
        'x1_values': [],
        'x2_values': []
    }

    for k in range(max_iter):
        f_val = f(x)
        J = jacobian(x)

        iterations_data['iteration'].append(k)
        iterations_data['f_norm'].append(np.linalg.norm(f_val))
        iterations_data['x1_values'].append(x[0])
        iterations_data['x2_values'].append(x[1])

        if k > 0:
            delta_x = np.linalg.norm(x - prev_x)
            iterations_data['delta_x_norm'].append(delta_x)
            if delta_x < epsilon:
                break
        else:
            iterations_data['delta_x_norm'].append(np.nan)

        delta_x_vec = np.linalg.solve(J, -f_val)
        prev_x = x.copy()

        x = x + delta_x_vec

    else:
        print(f"Достигнуто максимальное количество итераций ({max_iter})")

    if len(iterations_data['delta_x_norm']) < len(iterations_data['iteration']):
        iterations_data['delta_x_norm'].append(np.linalg.norm(x - prev_x))

    return x, iterations_data


def f(x):
    x1, x2 = x
    f1 = x1 ** 2 + x2 ** 2 - 4
    f2 = x1 - np.exp(x2) + 2
    return np.array([f1, f2])


def plot_graphical_localization():
    print("Графическая локализация корней:")

    # Строим кривые
    theta = np.linspace(0, 2 * np.pi, 100)
    x1_circle = 2 * np.cos(theta)  # x1^2 + x2^2 = 4
    x2_circle = 2 * np.sin(theta)

    x2_exp = np.linspace(-2, 2, 100)
    x1_exp = np.exp(x2_exp) - 2  # x1 = exp(x2) - 2

    plt.figure(figsize=(10, 8))
    plt.plot(x1_circle, x2_circle, 'b-', label=r'$x_1^2 + x_2^2 = 4$', linewidth=2)
    plt.plot(x1_exp, x2_exp, 'r-', label=r'$x_1 = e^{x_2} - 2$', linewidth=2)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    x0 = np.array([1.5, 1.5])
    plt.plot(x0[0], x0[1], 'go', markersize=10,
             label='Начальное приближение\n')

    plt.grid(True, alpha=0.3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Графическая локализация корней\n')
    plt.legend()
    plt.axis('equal')
    plt.show()

    return x0


def plot_convergence_comparison(newton_data, iteration_data, epsilon):
    plt.figure(figsize=(12, 5))

    # График 1: Норма разности приближений
    plt.subplot(1, 2, 1)
    # Метод Ньютона
    iterations_newton = newton_data['iteration'][1:]
    delta_norms_newton = newton_data['delta_x_norm'][1:]
    plt.semilogy(iterations_newton, delta_norms_newton, 'b-o', linewidth=2, label='Метод Ньютона')

    # Метод простой итерации
    iterations_iter = iteration_data['iteration'][1:]
    delta_norms_iter = iteration_data['delta_x_norm'][1:]
    plt.semilogy(iterations_iter, delta_norms_iter, 'r-s', linewidth=2, label='Метод простой итерации')

    plt.xlabel('Номер итерации, k')
    plt.ylabel('||x^(k+1) - x^(k)||')
    plt.title('Сходимость по разности приближений')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # График 2: Норма невязки
    plt.subplot(1, 2, 2)
    plt.semilogy(newton_data['iteration'], newton_data['f_norm'], 'b-o', linewidth=2, label='Метод Ньютона')
    plt.semilogy(iteration_data['iteration'], iteration_data['f_norm'], 'r-s', linewidth=2,
                 label='Метод простой итерации')
    plt.xlabel('Номер итерации, k')
    plt.ylabel('||f(x^(k))||')
    plt.title('Сходимость по норме невязки')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    epsilon = 1e-9
    x0 = plot_graphical_localization()

    print(f"Начальное приближение: x1 = {x0[0]:.4f}, x2 = {x0[1]:.4f}")

    newton_solution, newton_data = newton_method(x0, epsilon=epsilon, max_iter=50)

    iteration_solution, iteration_data = simple_iteration_method(x0, epsilon=epsilon, max_iter=100)

    # Вывод результатов

    print(f"Метод Ньютона:")
    print(f"  x1 = {newton_solution[0]:.10f}")
    print(f"  x2 = {newton_solution[1]:.10f}")
    print(f"  Проверка: f1(x) = {f(newton_solution)[0]:.2e}, f2(x) = {f(newton_solution)[1]:.2e}")

    print(f"Метод простой итерации:")
    print(f"  x1 = {iteration_solution[0]:.10f}")
    print(f"  x2 = {iteration_solution[1]:.10f}")
    print(f"  Проверка: f1(x) = {f(iteration_solution)[0]:.2e}, f2(x) = {f(iteration_solution)[1]:.2e}")

    plot_convergence_comparison(newton_data, iteration_data, epsilon)
