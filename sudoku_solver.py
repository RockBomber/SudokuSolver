#!/usr/bin/env python3
from __future__ import annotations

import abc
import argparse
import csv
import logging
from math import sqrt
from pprint import pformat
from typing import ClassVar

logger = logging.getLogger(__name__)

SUDOKU_SIZE = 9
SQUARE_SIZE = int(sqrt(SUDOKU_SIZE))


class AbstractContainer(abc.ABC):
    """
    Abstraction for containers that contain cells of Sudoku.
    Such containers can be rows or columns or squares.
    """

    def __init__(self):
        self.cells: list[Cell | None] = [None] * SUDOKU_SIZE

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cells})"

    def __contains__(self, value: int) -> bool:
        """
        Returns True if the value already exists among other values.
        """
        for cell in self.cells:
            if isinstance(cell.value, int) and value == cell.value:
                return True
        return False

    def is_in_candidates(self, candidate: int) -> bool:
        """
        Returns True if the candidate already exists among other candidates.
        """
        for cell in self.cells:
            if isinstance(cell.value, list) and candidate in cell.value:
                return True
        return False


class AbstactLine(AbstractContainer):
    """
    Abstraction for lines such as rows or columns.
    """
    instances: ClassVar[list]

    def __init__(self, pos: int):
        super().__init__()
        self.pos = pos

    def add(self, cell: Cell, pos: int):
        """
        Adds a cell to the line at the specified position.
        """
        if cell not in self.cells:
            self.cells[pos] = cell
        else:
            exists_pos = self.cells.index(cell)
            raise ValueError(
                f"The number {cell.value} already exists in the line at "
                f"index {exists_pos}"
            )

    @classmethod
    def get(cls, pos: int):
        """
        Returns an instance of line at the specified position.
        """
        if not cls.instances:
            for inner_pos in range(SUDOKU_SIZE):
                cls.instances.append(cls(inner_pos))
        return cls.instances[pos]


class Row(AbstactLine):
    """
    Contains row of cells.
    """
    instances = list()


class Column(AbstactLine):
    """
    Contains column of cells.
    """
    instances = list()


class Square(AbstractContainer):
    """
    Contains a square of 3x3 cells.
    """
    instances: ClassVar[list] = list()

    def __init__(self, pos_row: int, pos_column: int):
        super().__init__()
        self.pos_row = pos_row
        self.pos_column = pos_column

    @staticmethod
    def cell_index(pos_row: int, pos_column: int) -> int:
        """
        Converts the cell position to an array index.
        """
        pos_row, pos_column = pos_row % SQUARE_SIZE, pos_column % SQUARE_SIZE
        index = SQUARE_SIZE * pos_row + pos_column
        return index

    @staticmethod
    def square_index(pos_row: int, pos_column: int) -> int:
        """
        Converts the square position to an array index.
        """
        pos_row, pos_column = pos_row // SQUARE_SIZE, pos_column // SQUARE_SIZE
        index = SQUARE_SIZE * pos_row + pos_column
        return index

    def add(self, cell: Cell, pos_row: int, pos_column: int):
        """
        Adds a cell to the square at the specified position.
        """
        index = self.cell_index(pos_row, pos_column)
        if cell not in self.cells:
            self.cells[index] = cell
        else:
            exists_pos = self.cells.index(cell)
            raise ValueError(
                f"The number {cell.value} already exists in the square at "
                f"index {exists_pos}"
            )

    @classmethod
    def get(cls, pos_row: int, pos_column: int):
        """
        Returns an instance of line at the specified position.
        """
        if not cls.instances:
            for inner_pos in range(SUDOKU_SIZE):
                cls.instances.append(cls(pos_row, pos_column))
        index = cls.square_index(pos_row, pos_column)
        return cls.instances[index]


class Cell:
    def __init__(
        self,
        value: int | list[int] | None,
        pos_row: int,
        pos_column: int,
    ):
        self.value = value
        self.pos_row = pos_row
        self.pos_column = pos_column
        # init row
        self.row = Row.get(pos_row)
        self.row.add(self, pos_column)
        # init column
        self.column = Column.get(pos_column)
        self.column.add(self, pos_row)
        # init square
        self.square = Square.get(pos_row, pos_column)
        self.square.add(self, pos_row, pos_column)

    def create_candidates(self):
        """
        Creates lists of candidates based on existing numbers in a line,
        column, and square.
        If the resulting list contains only one number, then the number will
        be assigned to the value.
        """
        if self.value is None:
            self.value = list()
            for candidate in range(1, SUDOKU_SIZE+1):
                if not (
                    candidate in self.row
                    or candidate in self.column
                    or candidate in self.square
                ):
                    self.value.append(candidate)
            if len(self.value) == 1:
                self.value = self.value[0]

    def check(self) -> bool:
        """
        Returns True if cell contains number.
        Otherwise, it checks the candidates to see if they can be reduced or
        become a value.
        """
        if isinstance(self.value, int):
            return True
        elif isinstance(self.value, list):
            for candidate in self.value:
                if (
                    candidate in self.row
                    or candidate in self.column
                    or candidate in self.square
                ):
                    self.value.remove(candidate)
                    if len(self.value) == 1:
                        self.value = self.value[0]
                        return True
                elif not (
                    self.row.is_in_candidates(candidate)
                    or self.column.is_in_candidates(candidate)
                    or self.square.is_in_candidates(candidate)
                ):
                    self.value = candidate
                    return True
            return False
        else:
            raise ValueError(f"Wrong value type: {type(self.value)}")

    def __eq__(self, other) -> bool:
        """
        Compares cells by value only.
        """
        if isinstance(other, Cell) and self.value is not None:
            return self.value == other.value
        else:
            return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.value}, {self.pos_row}, {self.pos_column})"
        )


class SudokuSolver:

    def __init__(self, sudoku: list[list[int | None]]):
        """
        Creates cells from a Sudoku grid.
        """
        self.cells = list()
        for pos_row, row in enumerate(sudoku):
            for pos_column, value in enumerate(row):
                self.cells.append(Cell(value, pos_row, pos_column))
        for cell in self.cells:
            cell.create_candidates()

    def get_grid(self) -> list[list]:
        """
        Returns a Sudoku grid from a list of cells.
        """
        grid = [[] for _ in range(SUDOKU_SIZE)]
        for idx, cell in enumerate(self.cells):
            grid[idx // SUDOKU_SIZE].append(cell.value)
        return grid

    def solve(self, max_iterates: int = 100) -> bool:
        """
        Solves a Sudoku.
        """
        iteration = 0
        while iteration < max_iterates:
            iteration += 1
            all_resolved = True
            for cell in self.cells:
                result = cell.check()
                if result is False:
                    all_resolved = False
            if all_resolved:
                logger.info(
                    f"Sudoku has been solved in {iteration} attempts."
                )
                return True
        else:
            logger.warning(
                f"Couldn't solve Sudoku after {iteration} attempts."
            )
            return False

    def __repr__(self):
        return f"{self.__class__.__name__}({pformat(self.cells)})"


def load_from_csv(fileobj) -> list[list[int | None]]:
    """
    :param fileobj: object of file of csv file.
     Csv file must contain 9 rows and 9 colums.
     Each cell must contain a number from 1 to 9 or be empty.
    :return: read grid of Sudoku.
    """
    sudoku = []
    reader = csv.reader(fileobj)
    for row in reader:
        line = []
        for value in row:
            if value and value.isdigit():
                value = int(value)
                if not (1 <= value <= SUDOKU_SIZE):
                    raise ValueError(
                        f"The number must be between 1 and {SUDOKU_SIZE}, "
                        f"not {value}"
                    )
                line.append(value)
            else:
                line.append(None)
        if len(line) != SUDOKU_SIZE:
            raise ValueError(
                f"Each Sudoku line must contain {SUDOKU_SIZE} number, "
                f"but given {line}"
            )
        sudoku.append(line)
    if length := len(sudoku) != SUDOKU_SIZE:
        raise ValueError(
            f"Sudoku must be {SUDOKU_SIZE} lines, but given {length}"
        )
    return sudoku


def solve(fileobj):
    sudoku = load_from_csv(fileobj)
    logger.info(f"Loaded Sudoku grid:\n{pformat(sudoku)}")
    s = SudokuSolver(sudoku)
    s.solve()
    logger.info(f"Solved Sudoku grid:\n{pformat(s.get_grid())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType("r"))
    args = parser.parse_args()
    solve(args.file)
