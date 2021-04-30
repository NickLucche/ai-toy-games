from typing import List, Tuple, Callable
import numpy as np

class CSP:
    _number_of_variables: int  # to be defined by specific problem
    _var_names: List[str]
    # hold current (partial) assignment
    _state: np.ndarray

    def __init__(self, var_names: List[str] = None) -> None:
        if var_names:
            assert len(var_names) == self._number_of_variables
            self._var_names = var_names
        else:
            self._var_names = [
                f'X_{i}' for i in range(self._number_of_variables)
            ]

    @property
    def variable_names(self):
        return self._var_names

    @property
    def unassigned_vars(self):
        raise NotImplementedError()

    @property
    def assigned_vars(self):
        raise NotImplementedError()

    @property
    def assignment(self):
        return self._state.copy()

    def _check_consistency(self):
        raise NotImplementedError()

    @property
    def is_solution(self):
        raise NotImplementedError()

    @staticmethod
    def csp(n: int):
        # returns problem in CSP format (X, D, C), where X is a set of variables,
        # D is the set of variables' domains and C is a function testing constraints
        # satisfaction between two variables (given their respective domains).
        raise NotImplementedError()

from games.n_queens import NQueens
from games.snake import Snake
from games.sudoku import Sudoku
__all__ = ['CSP', 'NQueens', 'Snake', 'Sudoku']
