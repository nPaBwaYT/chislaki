from Exceptions import BadInputException
from classes.Matrix import Matrix
import sys


def inp() -> (Matrix, Matrix):
    if len(sys.argv) == 1:
        print("Введите трёхдиагональную матрицу СЛАУ (A|b) (только ненулевые элементы)\nв формате csv. "
              "Например, матрицу\n"
              "/1 2 0 0 │ 5\\\n"
              "│2 3 4 0 │ 1│\n"
              "│0 4 5 1 │ 2│\n"
              "\\0 0 1 2 │ 3/\n"
              "требуется записать как\n"
              "1,2,5\n"
              "2,3,4,1\n"
              "4,5,1,2\n"
              "1,2,3")
    else:
        f = open(sys.argv[1], 'r', encoding="UTF-8")
        old_stdin, sys.stdin = sys.stdin, f

    lines = sys.stdin.readlines()
    height = len(lines)

    rows = [[0 for i in range(index - 1)] + list(map(lambda x: float(x.strip(", \t")), line.strip(", \n\t").split(',')))
            for (index, line) in enumerate(lines) if line.strip(" \n\t") != ""]
    matrix_b = Matrix([[row.pop()] for row in rows])

    for row_index in range(height):
        rows[row_index] += [0 for i in range(height - row_index - 2)]

    matrix_a = Matrix(rows)
    return matrix_a, matrix_b


def main():
    matrix_a, matrix_b = inp()
    det_a = 1

    coefficients_p, coefficients_q = [0], [0]
    for i in range(matrix_a.get_height()):  # индексация с 0
        coeff_c = matrix_a[i][i + 1] if i + 1 < matrix_a.get_height() else 0
        coeff_b = matrix_a[i][i]
        coeff_a = matrix_a[i][i - 1] if i > 0 else 0
        coeff_d = matrix_b[i][0]

        det_a *= coeff_b + coeff_a * coefficients_p[i]
        if det_a == 0:
            raise BadInputException

        coefficients_p.append(-coeff_c / (coeff_b + coeff_a * coefficients_p[i]))
        coefficients_q.append((coeff_d - coeff_a * coefficients_q[i]) / (coeff_b + coeff_a * coefficients_p[i]))

    rows_x = [[0] for i in range(matrix_a.get_height() + 1)]
    for j in range(matrix_a.get_height(), 0, -1):
        rows_x[j - 1][0] = coefficients_p[j] * rows_x[j][0] + coefficients_q[j]

    rows_x.pop()
    matrix_x = Matrix(rows_x)
    print(f"det(A) = {det_a:.4f}\n{matrix_x.transpose()}T", sep='\n')

    print(f"\nb = {matrix_b.transpose()}T\nAx = {(matrix_a @ matrix_x).transpose()}T")


if __name__ == "__main__":
    main()
