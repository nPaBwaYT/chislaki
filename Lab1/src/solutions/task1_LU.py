from classes.Matrix import Matrix
from classes.Exceptions import BadInputException
import sys


def inp() -> (Matrix, Matrix):
    if len(sys.argv) == 1:
        print("Введите матрицу СЛАУ (A|b) (матрица A должна быть невырожденной)\nв формате csv. Например, матрицу\n"
              "/1 2 │ 3\\\n"
              "\\4 5 │ 6/\n"
              "требуется записать как\n"
              "1,2,3\n"
              "4,5,6\n")
    else:
        f = open(sys.argv[1], 'r', encoding="UTF-8")
        old_stdin, sys.stdin = sys.stdin, f

    lines = sys.stdin.readlines()
    rows = [list(map(lambda x: float(x.strip(", \t")), line.strip(", \n\t").split(','))) for line in lines if
            line.strip(" \n\t") != ""]

    matrix_b = Matrix([[row.pop()] for row in rows])
    matrix_a = Matrix(rows)
    return matrix_a, matrix_b


def lu_factorization(matrix: Matrix, vector: Matrix) -> (Matrix, Matrix):
    matrix_u = Matrix(matrix.get_rows())
    matrix_l = Matrix([[1 if i == j else 0 for i in range(matrix.get_length())] for j in range(matrix.get_height())])
    matrix_y = Matrix(vector.get_rows())

    inverse_a_rows = [[1 if i == j else 0 for i in range(matrix.get_length())] for j in range(matrix.get_height())]
    inverse_matrix = Matrix(inverse_a_rows)

    for row_index in range(matrix.get_height()):
        for from_index in range(row_index + 1, matrix.get_length()):
            m = matrix_u[from_index][row_index] / matrix_u[row_index][row_index]
            matrix_u.subtract_rows(row_index, from_index, m)
            matrix_y.subtract_rows(row_index, from_index, m)
            inverse_matrix.subtract_rows(row_index, from_index, m)
            matrix_l[from_index][row_index] = -inverse_matrix[from_index][row_index]

    matrix_lu = Matrix([[matrix_l[i][j] if i > j else matrix_u[i][j] for j in range(matrix.get_length())] for i in
                        range(matrix.get_height())])

    return matrix_lu, matrix_y, inverse_matrix


def main():
    matrix_a, matrix_b = inp()
    a = 1

    for col_index in range(matrix_a.get_length()):
        for row_index in range(col_index, matrix_a.get_height()):
            if matrix_a[row_index][col_index] > matrix_a[col_index][col_index]:
                matrix_a.swap_rows(col_index, col_index)
                matrix_b.swap_rows(col_index, col_index)
                a *= -1

    matrix_lu, matrix_y, inverse_matrix = lu_factorization(matrix_a, matrix_b)
    det = 1
    for i in range(matrix_lu.get_height()):
        det *= matrix_lu[i][i]
        if det == 0:
            raise BadInputException

    print(f"det(A) = {det:.4f}\n\nLU-matrix:\n{matrix_lu}\n\ny = {matrix_y.transpose()}T\n")

    x_rows = [[0] for i in range(matrix_lu.get_height())]
    matrix_x = Matrix(matrix_y.get_rows())
    matrix_lu_buff = Matrix([[matrix_lu[i][j] if j >= i else 0 for j in range(matrix_lu.get_length())] for i in
                             range(matrix_lu.get_height())])

    for row_index in range(matrix_lu.get_height() - 1, -1, -1):
        for from_index in range(row_index - 1, -1, -1):
            m = matrix_lu_buff[from_index][row_index] / matrix_lu_buff[row_index][row_index]
            matrix_lu_buff.subtract_rows(row_index, from_index, m)
            inverse_matrix.subtract_rows(row_index, from_index, m)
            matrix_x.subtract_rows(row_index, from_index, m)

        inverse_matrix.multiply_row(row_index, 1/matrix_lu_buff[row_index][row_index])
        matrix_x.multiply_row(row_index, 1/matrix_lu_buff[row_index][row_index])

    print(f"Inverse matrix:\n{inverse_matrix}\n\nx = {matrix_x.transpose()}T")

    print(f"\nb = {matrix_b.transpose()}T\nAx = {(matrix_a @ matrix_x).transpose()}T\n")
    print(f"A * A^-1 =\n{matrix_a @ inverse_matrix}\n")


if __name__ == "__main__":
    main()
