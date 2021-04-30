from typing import List
import numpy as np
import cv2
import sys, os
curdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curdir, os.pardir)))
from games import CSP


class NQueens(CSP):
    # solutions exists for all natural numbers n with the exception of n = 2 and n = 3.
    size: int
    _state: np.ndarray
    _var_names: List[str]

    def __init__(self,
                 n: int = 8,
                 init_state: np.ndarray = None,
                 var_names: List[str] = None,
                 render_width: int = 800,
                 render_height: int = 800) -> None:
        assert n > 3, "N Queens problem has no solution for board with size 2 or 3"
        self.size = n
        self._number_of_variables = n
        self._render_shape = (render_width, render_height)
        self._block_size = (render_width // n, render_height // n)
        offset = (render_width % self._block_size[0],
                  render_height % self._block_size[1])
        self._render_grid = (np.linspace(offset[0],
                                         render_width - offset[0],
                                         n + 1,
                                         dtype=np.int32),
                             np.linspace(offset[1],
                                         render_height - offset[1],
                                         n + 1,
                                         dtype=np.int32))
        super().__init__(var_names=var_names)
        # initialize state with empty assignment (all -1)
        if init_state is not None:
            self._state = np.empty_like(init_state)
            self._state[:] = init_state
        else:
            self._state = np.ones(n, dtype=np.int32) * (-1)

    @property
    def unassigned_vars(self):
        return np.where(self._state == -1)[0]

    @property
    def assigned_vars(self):
        return np.where(self._state > -1)[0]

    def sample_from_domain(self, n_values: int = 1):
        return np.random.choice(self.size, n_values)

    def _check_consistency(self):
        if len(self._state[self._state > -1]) <= 1:
            return True
        # check whether current (partial) assignment satisfies all constraints (queens do not attack each other)
        vals = np.sort(self._state[self._state > -1])
        # can't be on same row (all rows must be different, we sort first)
        for i in range(len(vals) - 1):
            if vals[i] == vals[i + 1]:
                return False
        # can't be on same diagonal (horizontal and vertical distances do differ)
        for i in range(self.size - 1):
            if self._state[i] == -1:
                continue
            for j in range(i + 1, self.size):
                if self._state[j] > -1 and (j - i) == np.abs(self._state[i] -
                                                             self._state[j]):
                    return False
        return True

    def step(self, action):
        # action defined as (column/queen, row in which we place it)
        # one can de-assign vars by assigning -1 to that
        assert action[0] >= 0 and action[0] < self.size
        assert action[1] >= -1 and action[1] < self.size
        self._state[action[0]] = action[1]

        return self.state, self._check_consistency()

    @property
    def state(self):
        return self._state.copy()

    def __repr__(self) -> str:
        return ', '.join([
            f'({k}:{v})' for k, v in zip(self._var_names, self._state)
            if v > -1
        ])

    @property
    def is_solution(self):
        return len(self.unassigned_vars) == 0 and self._check_consistency()

    @staticmethod
    def goal_test(state: np.ndarray):
        # state has to be a valid state
        return NQueens(n=len(state), init_state=state).is_solution

    @staticmethod
    def csp(n: int):
        # vars/X, domains/D, constraints/C expressed as a test function
        def constraint_check(x: int, dx: np.ndarray, y: int, dy: np.ndarray):
            """
            Given two variables and respective domains,
            checks whether binary constaints involving 
            those 2 variables hold for every value in dx.
            x, y are passed to retrieve specific constraints 
            for those two variables (in nqueens you only got
            global constraints tho). 
            """
            if len(dx) == 0 or len(dy) == 0:
                return False
            # row constraint
            unique_dy = np.unique(dy)
            unique_dx = np.unique(dx)
            if len(unique_dy) <= 1 and unique_dy[0] in unique_dx:
                return False
            # diagonal constraint
            row_dist = np.abs(x - y)
            for vx in unique_dx:
                violations = [
                    1 for vy in unique_dy if row_dist == np.abs(vx - vy)
                ]
                if len(violations) == len(unique_dy):
                    return False
            return True

        return np.arange(n), np.arange(n), constraint_check

    def render(self):
        canvas = np.ones(self._render_shape, dtype=np.uint8) * 255
        # draw grid
        for x, y in zip(*self._render_grid):
            cv2.line(canvas, (x, 0), (x, self._render_grid[1][-1]), (0, 0, 0),
                     thickness=3)
            cv2.line(canvas, (0, y), (self._render_grid[0][-1], y), (0, 0, 0),
                     thickness=3)

        for x, y in enumerate(self._state):
            if y > -1:
                canvas = cv2.circle(
                    canvas,
                    (x * self._block_size[0] + self._block_size[0] // 2,
                     y * self._block_size[1] + self._block_size[1] // 2),
                    self._block_size[0] // 4, (0, 0, 0),
                    thickness=-1)
        return canvas


if __name__ == '__main__':
    game = NQueens(n=16)
    for i in range(16):
        _, valid = game.step((i, i))
        print('Consistent step:', valid)
    cv2.imshow('N Queens', game.render())
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()