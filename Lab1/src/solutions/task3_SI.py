from classes.Matrix import Matrix
from classes.Exceptions import BadInputException
from itertools import chain
from math import log10, log
import sys
import numpy as np


def inp() -> (float, Matrix, Matrix):
    old_stdin = None

    f = open("C:/Projects/python/chislaki/Lab1/src/solutions/task3_files/test.txt", 'r', encoding="UTF-8")
    old_stdin, sys.stdin = sys.stdin, f

    lines = sys.stdin.readlines()
    eps = float(lines[0].strip(" \n\t,"))
    rows = [list(map(lambda x: float(x.strip(", \t")), line.strip(", \n\t").split(','))) for line in lines[1::] if
            line.strip(" \n\t,") != ""]

    matrix_b = Matrix([[row.pop()] for row in rows])
    matrix_a = Matrix(rows)
    if old_stdin is not None:
        sys.stdin = old_stdin
    return eps, matrix_a, matrix_b


def is_diag_major(matrix: Matrix) -> bool:
    for i in range(matrix.get_height()):
        if sum(matrix[i]) - matrix[i][i] > matrix[i][i]:
            return False
    return True


def main(A, B):
    eps, matrix_a, matrix_b = inp()
    #print(f"target eps: {eps}\n")

    matrix_alpha = Matrix(
        [[-matrix_a[i][j] / matrix_a[i][i] if i != j else 0 for j in range(matrix_a.get_length())] for i in
         range(matrix_a.get_height())])
    matrix_beta = Matrix([[matrix_b[i][0] / matrix_a[i][i]] for i in range(matrix_b.get_height())])

    alpha_norms = list(enumerate([max([sum(map(lambda x: abs(x), row)) for row in matrix_alpha.transpose()]),
                                  sum(map(lambda x: x * x, chain(*matrix_alpha))) ** 0.5,
                                  max([sum(map(lambda x: abs(x), row)) for row in matrix_alpha])], 1))

    alpha_norms.sort(key=lambda x: x[1])
    alpha_norm = alpha_norms[0][1]
    norm_type = alpha_norms[0][0]
    #print(f"norm alpha: {alpha_norm:.4f}\nnorm type: {norm_type}")

    if alpha_norm >= 1:
        if not is_diag_major(matrix_a):
        #    print(f"Лежит ли спектр собственных сначений матрицы\n{matrix_alpha}\nвнутри единичной окружности? y/n")
            if (ans := sys.stdin.readline().strip(" \t\n").lower()) != "y":
                raise BadInputException

    matrix_x = Matrix(matrix_beta.get_rows())

    if alpha_norm > 1:
       # print("Невозможно провести оценку кол-ва итераций")
        while 1:
            prev_x = Matrix(matrix_x.get_rows())
            matrix_x = matrix_beta + matrix_alpha @ matrix_x

            x_norm = max(map(lambda x: abs(x[0]), matrix_x - prev_x))
            if x_norm < eps:
                break

    else:
        beta_norm: float
        match norm_type:
            case 1:
                beta_norm = sum(map(lambda x: abs(x[0]), matrix_beta))
            case 2:
                beta_norm = sum(map(lambda x: x[0] * x[0], matrix_beta)) ** 0.5
            case _:
                beta_norm = max(map(lambda x: abs(x[0]), matrix_beta))

        iterations_count = (log(eps) - log(beta_norm) + log(1 - alpha_norm)) / log(alpha_norm)
      #  print(f"norm beta: {beta_norm:.4f}\n\nestimation of the number of iterations: {iterations_count:.4f}\n")

        for i in range(1, int(iterations_count) + 2):
            prev_x = Matrix(matrix_x.get_rows())
            matrix_x = matrix_beta + matrix_alpha @ matrix_x

            match norm_type:
                case 1:
                    x_norm = sum(map(lambda x: abs(x[0]), matrix_x - prev_x))
                case 2:
                    x_norm = sum(map(lambda x: x[0] * x[0], matrix_x - prev_x)) ** 0.5
                case _:
                    x_norm = max(map(lambda x: abs(x[0]), matrix_x - prev_x))

            eps_k = alpha_norm / (1 - alpha_norm) * x_norm
     #       print(f"eps({i}) = {eps_k:.{int(-log10(eps)) + 2}f}", end="; ")
            if eps_k < eps:
    #            print("\b\b\n")
                break

    #print(f"x = {matrix_x.transpose():.{int(-log10(eps)) + 2}f}T")
    return np.linalg.solve(A, B)


if __name__ == '__main__':
    main()
