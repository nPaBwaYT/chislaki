from classes.Matrix import Matrix
import sys


def inp() -> (Matrix, Matrix):
    if len(sys.argv) == 1:
        sys.stdout.write("Введите матрицу СЛАУ (A|b) (матрица A должна быть невырожденной)\nв формате csv. Например, матрицу\n"
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


def lu_factorization(matrix: Matrix) -> Matrix:
    lu_rows = []
    for row_index in range(matrix.get_height()):
        lu_rows.append([])
        for col_index in range(matrix.get_length()):
            if col_index >= row_index:
                u_element = matrix[row_index][col_index] - sum(
                    [lu_rows[row_index][index] * lu_rows[index][col_index] for index in range(row_index)])
                lu_rows[row_index].append(u_element)

            else:
                l_element = (matrix[row_index][col_index] - sum(
                    [lu_rows[row_index][index] * lu_rows[index][col_index] for index in range(col_index)])) / \
                            lu_rows[col_index][col_index]
                lu_rows[row_index].append(l_element)

    matrix_lu = Matrix(lu_rows)

    return matrix_lu


def main():
    matrix_a, matrix_b = inp()

    for col_index in range(matrix_a.get_length()):
        for row_index in range(col_index, matrix_a.get_height()):
            if matrix_a[row_index][col_index] > matrix_a[col_index][col_index]:
                matrix_a.swap_rows(col_index, col_index)

    matrix_lu = lu_factorization(matrix_a)
    print(f"LU-matrix:\n{matrix_lu}\n")

    y_rows = []
    for row_index in range(matrix_lu.get_height()):
        row = matrix_b[row_index][0] - sum(
            [matrix_lu[row_index][col_index] * y_rows[col_index][0] for col_index in range(row_index)])
        y_rows.append([row])

    matrix_y = Matrix(y_rows)
    print(f"y = {matrix_y.transpose()}T")

    x_rows = [[0] for i in range(matrix_lu.get_height())]
    for row_index in range(matrix_lu.get_height() - 1, -1, -1):
        row = ((matrix_y[row_index][0] - sum([matrix_lu[row_index][col_index] * x_rows[col_index][0] for col_index in
                                             range(row_index + 1, matrix_lu.get_length())])) /
               matrix_lu[row_index][row_index])
        x_rows[row_index] = [row]
    matrix_x = Matrix(x_rows)

    print(f"x = {matrix_x.transpose()}T")


if __name__ == "__main__":
    main()
