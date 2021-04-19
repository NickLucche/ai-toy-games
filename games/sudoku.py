from games import CSP
from typing import List

class Sudoku(CSP):
    _number_of_variables=81

    def __init__(self, var_names: List[str]=None) -> None:
        super().__init__(var_names=var_names)
        # this is pretty unusual, as we're gonna use the backtracking search
        # algorithm to effectively search for a solvable sudoku to generate the board
