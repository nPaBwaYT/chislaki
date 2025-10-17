from classes.Matrix import Matrix
from classes.Exceptions import BadInputException
from math import log10
import sys


def inp() -> (float, Matrix):
    if len(sys.argv) == 1:
        print("Введите точность (eps), затем с новой строки введите матрицу (A)\nв формате csv. Например, матрицу\n"
              "/1 2\\\n"
              "\\4 5/\n"
              "требуется записать как\n"
              "1,2\n"
              "4,5\n")
    else:
        f = open(sys.argv[1], 'r', encoding="UTF-8")
        old_stdin, sys.stdin = sys.stdin, f

    lines = sys.stdin.readlines()
    eps = float(lines[0].strip(" \n\t,"))
    rows = [list(map(lambda x: float(x.strip(", \t")), line.strip(", \n\t").split(','))) for line in lines[1::] if
            line.strip(" \n\t,") != ""]

    matrix_a = Matrix(rows)
    return eps, matrix_a


def qr_factorization(matrix: Matrix) -> (Matrix, Matrix):
    matrix_e = Matrix([[1 if i == j else 0 for j in range(matrix.get_length())] for i in range(matrix.get_height())])

    matrix_a = Matrix(matrix.get_rows())
    matrix_q = Matrix(matrix_e.get_rows())

    for k in range(matrix.get_length() - 1):
        vector_b = Matrix([[matrix_a[j][k]] for j in range(matrix_a.get_height())])
        norm_b = (sum(map(lambda x: x[0] * x[0], vector_b[k::]))) ** (1 / 2)
        sign_bk = vector_b[k][0] / abs(vector_b[k][0])

        vector_nu_rows = [[0] for i in range(k + 1)] + [vector_b[j] for j in range(k + 1, matrix.get_height())]
        vector_nu_rows[k] = [vector_b[k][0] + sign_bk * norm_b]

        vector_nu = Matrix(vector_nu_rows)

        matrix_h = matrix_e - 2 / (vector_nu.transpose() @ vector_nu)[0][0] * (vector_nu @ vector_nu.transpose())
        matrix_a = matrix_h @ matrix_a
        matrix_q = matrix_h @ matrix_q

    matrix_q = matrix_q.transpose()
    return matrix_q, matrix_a


def main():
    eps, matrix_a = inp()
    matrix_q, matrix_r = qr_factorization(matrix_a)
    print(f"Q =\n{matrix_q:.{int(-log10(eps)) + 2}f}\n\n R =\n{matrix_r:.{int(-log10(eps)) + 2}f}\n\n"
          f"A = QR =\n{matrix_q @ matrix_r:.{int(-log10(eps)) + 2}f}\n")

    norms = []
    eigenvalues = [0 for i in range(matrix_a.get_length())]

    iterations_count = 0
    while 1:
        iterations_count += 1

        if len(norms) == 0:
            norms = [0 for i in range(matrix_a.get_length())]
        else:
            matrix_q, matrix_r = qr_factorization(matrix_a)

        matrix_a = matrix_r @ matrix_q

        for j in range(matrix_a.get_length()):
            prev = norms[j]
            norms[j] = 0
            for i in range(j + 1, matrix_a.get_height()):
                norms[j] += matrix_a[i][j] * matrix_a[i][j]
            norms[j] = norms[j] ** 1/2

        f = True
        j = 0
        while j < matrix_a.get_length():
            if norms[j] <= eps:
                eigenvalues[j] = matrix_a[j][j]
                j += 1

            else:
                b = -(matrix_a[j][j] + matrix_a[j + 1][j + 1])
                c = matrix_a[j][j] * matrix_a[j + 1][j + 1] - matrix_a[j][j + 1] * matrix_a[j + 1][j]
                d = b * b - 4 * c
                la1 = (-b + d ** (1/2)) / 2
                la2 = (-b - d ** (1/2)) / 2
                if (abs(la1 - eigenvalues[j]) > eps) or (abs(la2 - eigenvalues[j + 1]) > eps):
                    f = False
                eigenvalues[j], eigenvalues[j + 1] = la1, la2
                j += 2
        if f:
            break

    print(f"iterations count: {iterations_count}\n")
    print("eigenvalues:")
    for ev in eigenvalues:
        print(f"{ev:.{int(-log10(eps)) + 2}f}")


if __name__ == "__main__":
    main()
