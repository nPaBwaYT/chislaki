from classes.Matrix import Matrix
from sys import stdin, stdout


def inp() -> (Matrix, Matrix):
    lines = stdin.readlines()
    rows = [list(map(lambda x: float(x.strip(" \t")), line.strip(" \n\t").split(','))) for line in lines]
    matrix_b = Matrix([[row.pop()] for row in rows])
    matrix_a = Matrix(rows)
    return matrix_a, matrix_b


def main():
    matrix_a, matrix_b = inp()
    matrix_l = Matrix([[0 for col_index in range(matrix_a.get_height())] for row_index in range(matrix_a.get_length())])
    for index in range(matrix_l.get_height()):
        matrix_l[index][index] = 1

    print(matrix_a, matrix_b, matrix_l, sep='\n')


if __name__ == "__main__":
    main()
