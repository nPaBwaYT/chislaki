from typing import List, Tuple
from classes.Exceptions import BadInputException
from math import log10


class Matrix:
    """Matrix class, nothing special"""

    def __init__(self, rows: List[List[int | float]] | Tuple[Tuple[int | float]]):
        """Initialisation"""
        first_row_length = len(rows[0])
        for row in rows[1::]:
            if len(row) != first_row_length:
                raise BadInputException

        self.rows = rows.copy()
        self.length = first_row_length
        self.height = len(rows)

    def __getitem__(self, item):
        return self.rows[item]

    def __setitem__(self, key, value):
        self.rows[key] = value

    def __add__(self, other):
        """Addition"""
        if not isinstance(other, Matrix):
            raise TypeError

        if not (self.length == other.length and self.height == other.height):
            raise TypeError

        result_matrix_rows = [
            [this_element + other_element for (this_element, other_element) in zip(this_row, other_row)] for
            (this_row, other_row) in zip(self.rows, other)]

        result_matrix = Matrix(result_matrix_rows)
        return result_matrix

    def __iadd__(self, other):
        """Increment addition"""
        if not isinstance(other, Matrix):
            raise TypeError

        if not (self.length == other.length and self.height == other.height):
            raise TypeError

        for (row_index, row) in enumerate(other, 0):
            for (col_index, element) in enumerate(row, 0):
                self[row_index][col_index] += element

        return self

    def __sub__(self, other):
        """Subtraction"""
        if not isinstance(other, Matrix):
            raise TypeError

        if not (self.length == other.length and self.height == other.height):
            raise TypeError

        result_matrix_rows = [
            [this_element - other_element for (this_element, other_element) in zip(this_row, other_row)] for
            (this_row, other_row) in zip(self.rows, other)]

        result_matrix = Matrix(result_matrix_rows)
        return result_matrix

    def __isub__(self, other):
        """Increment subtraction"""
        if not isinstance(other, Matrix):
            raise TypeError

        if not (self.length == other.length and self.height == other.height):
            raise TypeError

        for (row_index, row) in enumerate(other, 0):
            for (col_index, element) in enumerate(row, 0):
                self[row_index][col_index] -= element

        return self

    def __mul__(self, other: int | float):
        """Multiplication by number"""
        result_matrix = Matrix(self.rows)
        for row_index in range(self.height):
            for col_index in range(self.length):
                result_matrix.rows[row_index][col_index] *= other
        return result_matrix

    def __rmul__(self, other: int | float):
        """Multiplication by number"""
        result_matrix = Matrix(self.rows)
        for row_index in range(self.height):
            for col_index in range(self.length):
                result_matrix.rows[row_index][col_index] *= other
        return result_matrix

    def __imul__(self, other):
        """Increment multiplication by number"""
        for row_index in range(self.height):
            for col_index in range(self.length):
                self[row_index][col_index] *= other
        return self

    def __matmul__(self, other):
        """Matrix multiplication"""
        if not isinstance(other, Matrix):
            raise TypeError

        if self.length != other.height:
            raise TypeError

        result_matrix_rows = [
            [sum([self[row_index][index] * other[index][col_index] for index in range(self.length)]) for
             col_index in range(other.length)] for row_index in range(self.height)]

        result_matrix = Matrix(result_matrix_rows)
        return result_matrix

    def __imatmul__(self, other):
        """Increment matrix multiplication"""
        if not isinstance(other, Matrix):
            raise TypeError

        if self.length != other.height:
            raise TypeError

        result_matrix_rows = [
            [sum([self[row_index][index] * other[index][col_index] for index in range(self.length)]) for
             col_index in range(other.length)] for row_index in range(self.height)]

        self.rows = result_matrix_rows
        self.length = other.length
        return self

    def __neg__(self):
        """Unar minus"""

        result_matrix_rows = [[-element for element in row] for row in self.rows]

        result_matrix = Matrix(result_matrix_rows)
        return result_matrix

    def __str__(self):
        """Representation"""
        if self.height == 0:
            return "()"
        if self.height == 1:
            return f"({' '.join(list(map(lambda x: str(x.__round__(4)), self[0])))})"

        rows = ["/"] + ["│" for i in range(1, self.height - 1)] + ["\\"]
        for col_index in range(self.length):
            col_width = max([len(str(self[row_index][col_index].__round__(4))) for row_index in range(self.height)])

            for row_index in range(self.height):
                if col_index != 0:
                    rows[row_index] += " "

                rows[row_index] += f"{self[row_index][col_index].__round__(4): ^{col_width}}"

        rows[0] += "\\"
        rows[self.height - 1] += "/"
        for i in range(1, self.height - 1):
            rows[i] += "│"

        result = "\n".join(rows)
        return result

    def transpose(self):
        """Matrix transposition"""
        result_matrix_rows = [[self[row_index][col_index] for row_index in range(self.height)] for col_index in
                              range(self.length)]
        result_matrix = Matrix(result_matrix_rows)
        return result_matrix

    def determinant(self) -> int | float:
        """Determinant"""
        if self.length != self.height:
            raise TypeError

        if self.height == 1:
            return self[0][0]

        minor_matrices = [Matrix([row[0:col_index] + row[col_index + 1::] for row in self.rows[1::]]) for col_index in
                          range(self.length)]

        return sum(
            [(-1) ** index * self[0][index] * minor_matrices[index].determinant() for index in range(self.height)])

    def __invert__(self):
        """Inverse matrix"""
        det = self.determinant()

        if det == 0:
            raise ZeroDivisionError

        inverse_matrix = Matrix(
            [[self.alg_comp(row_index, col_index) / det for col_index in range(self.length)] for row_index in
             range(self.height)]).transpose()

        return inverse_matrix

    def alg_comp(self, row_index: int, col_index: int) -> int | float:
        """Finds algebraic completion for element at [row][col]"""
        return (-1) ** (row_index + col_index) * Matrix([row[0:col_index] + row[col_index + 1::] for row in (
                self.rows[0:row_index] + self.rows[row_index + 1::])]).determinant()

    def swap_rows(self, first_index: int, second_index: int):
        self[first_index], self[second_index] = self[second_index], self[first_index]
        return self

    def swap_columns(self, first_index: int, second_index: int):
        for row in self.rows:
            row[first_index], row[second_index] = row[second_index], row[first_index]
        return self

    def multiply_row(self, row_index: int, multiplier: int | float):
        self[row_index] = map(lambda x: x * multiplier, self[row_index])
        return self

    def subtract_rows(self, row_index: int, from_index: int, multiplier: int | float):
        self[from_index] = list(map(lambda x: x[0] - x[1] * multiplier, zip(self[from_index], self[row_index])))
        return self

    def get_height(self) -> int:
        return self.height

    def get_length(self) -> int:
        return self.length

    def get_rows(self) -> List[List[int | float]]:
        return self.rows


def main():
    """Tests"""
    a = Matrix([[1, 2],
                [-1.5, 1],
                [1, 1]])
    b = Matrix([[1, 2, 3],
                [4, 5, 6]])
    print(a @ b, a - b.transpose(), sep="\n")

    a = Matrix([[1, 2, 1],
                [-1, -1, 2],
                [2, 2, -2]])
    print(~a)
    a.swap_columns(1, 2)
    print(a, ~a, a.determinant(), sep='\n')

    """"""
    print("—————————————————————————————————")

    a = Matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    p = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    a.swap_rows(0, 1)
    p.swap_rows(0, 1)
    print(a, ~p @ a, sep='\n')

    """"""
    print("—————————————————————————————————")

    a = Matrix([[1, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 0, 1, 0],
                [2, 0, 0, 1]])
    b = Matrix([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 3, 1]])
    print(a @ b)


if __name__ == "__main__":
    main()
