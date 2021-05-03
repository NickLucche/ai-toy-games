import os, sys

curdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curdir, os.pardir)))
from games import CSP
from typing import List
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from csp import *
import cv2


class Sudoku(CSP):
    # we've got 27 Aldiff constraints in Sudoku: one per row, one per column
    # and one per box (3x3 square)
    size = 81
    _number_of_variables = 81
    _state: np.ndarray
    _min_prefilled: int

    def __init__(
        self,
        number_of_prefilled: int,
        init_state: np.ndarray = None,
        var_names: List[str] = None,
        render_width: int = 900,
        render_height: int = 900,
    ) -> None:
        self._render_shape = (render_width, render_height)
        self._block_size = (render_width // 9, render_height // 9)
        offset = (
            render_width % self._block_size[0],
            render_height % self._block_size[1],
        )
        self._render_grid = (
            np.linspace(offset[0]//2, render_width - offset[0]//2, 9 + 1, dtype=np.int32),
            np.linspace(offset[1]//2, render_height - offset[1]//2, 9 + 1, dtype=np.int32),
        )
        super().__init__(var_names=var_names)
        self._prefilled = number_of_prefilled
        if init_state is not None:
            self._state = np.empty_like(init_state, dtype=np.int8)
            self._state[:] = init_state
        else:
            self._generate_init_state(unique_solution=False)
        # this is pretty unusual, as we're gonna use the backtracking search
        # algorithm to effectively search for a solvable sudoku to generate the board

        # strides that indicate the number of bytes to jump
        # in order to reach the next value in the dimension
        # check this to figure out numbers https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20#1f28
        s = self._state
        # this is a view so value will change as state is changed
        self.blocks = ast(
            s,
            shape=(3, 3, 3, 3),
            strides=(9 * 3 * s.itemsize, 3 * s.itemsize, 9 * s.itemsize, s.itemsize),
        )

    @property
    def unassigned_vars(self):
        return np.where(self._state.reshape(-1,) == -1)[0]

    @property
    def assigned_vars(self):
        return np.where(self._state.reshape(-1,) > -1)[0]

    def _check_consistency(self):
        # each row and col must contain different numbers
        state_transpose = self._state.T
        for i in range(self._state.shape[0]):
            # ignore empty entries
            row = self._state[i, :][self._state[i, :] > -1]
            col = state_transpose[i, :][state_transpose[i, :] > -1]
            if len(row) != len(np.unique(row)):
                return False
            if len(col) != len(np.unique(col)):
                return False

        # check box constraint
        for i in range(3):
            for j in range(3):
                box = self.blocks[i, j][self.blocks[i, j] > -1]
                if len(box) != len(np.unique(box)):
                    return False
        return True

    def _generate_init_state(self, unique_solution=False):
        # start from *SOLVABLE* fully filled grid
        graph = ConstraintGraph(self)
        # empty state run with
        self._state = np.ones(81, dtype=np.int8).reshape(9, 9) * (-1)
        s = self._state
        self.blocks = ast(
            s,
            shape=(3, 3, 3, 3),
            strides=(9 * 3 * s.itemsize, 3 * s.itemsize, 9 * s.itemsize, s.itemsize),
        )
        # TODO: some randomness to find different initial solution..?
        solution = backtrack_search(
            self, graph, mrv_heuristics, generate_domain_values, forward_checking
        )
        if solution is None:
            raise Exception("Can't find starting solution for Sudoku!")
        print("Initial solution found:", self._state)

        prefilled = 81
        while prefilled > self._prefilled:
            # remove one value from grid
            perm = np.random.permutation(self.assigned_vars).astype(np.int8)
            found = False
            for var in perm:
                value = self._state.reshape(-1)[var]
                print("variable", var)
                self.step((var, -1))
                # check if sudoku can be solved
                sol = None
                if unique_solution:
                    # count number of solutions and make sure it's a single one
                    pass
                else:
                    # o/w current state is substituted with a solution
                    ss = Sudoku(self._prefilled, init_state=self._state)
                    graph = ConstraintGraph(ss)
                    # any solution is fine
                    sol = backtrack_search(
                        ss,
                        graph,
                        mrv_heuristics,
                        generate_domain_values,
                        forward_checking,
                    )
                # if no solution exists remove assignment and test again
                if sol is None:
                    print(f"No solution found removing variable {var}")
                    self.step((var, value))
                else:
                    found = True
                    break
            if not found:
                raise Exception(
                    f"Cannot instatiate a valid Sudoku with {prefilled} pre-filled values"
                )
            # another value
            prefilled -= 1

    def step(self, action):
        # action couple composed of:
        #  - variable number if int otherwise (i, j) indexing if tuple
        #  - value to assign to variable
        # one can de-assign vars by assigning -1 to that
        # assert action[0] >= 0 and action[0] < self._number_of_variables
        assert action[1] >= -1 and action[1] < 9

        if type(action[0]) is tuple:
            self._state[action[0]] = action[1]
        else:
            self._state.reshape(-1,)[action[0]] = action[1]
        # else:
        # raise Exception("Invalid variable type")

        return self.state, self._check_consistency()

    @property
    def state(self):
        return self._state.copy()

    @property
    def is_solution(self):
        return len(self.unassigned_vars) == 0 and self._check_consistency()

    @staticmethod
    def csp(n: int):
        # vars/X, domains/D, constraints/C expressed as a test function
        def constraint_check(x: int, dx: np.ndarray, y: int, dy: np.ndarray):
            """
            Given two variables and respective domains,
            checks whether binary constaints involving 
            those 2 variables hold for every value in dx.
            x, y are passed to retrieve specific constraints 
            for those two variables. 
            """
            if len(dx) == 0 or len(dy) == 0:
                return False
            # check if on same row or col constraint
            unique_dy = np.unique(dy)
            if len(unique_dy) > 1:
                return True

            unique_dx = np.unique(dx)
            if x // 9 == y // 9 or x % 9 == y % 9:
                if unique_dy[0] in unique_dx:
                    return False
            # check if on same box
            sud = np.arange(81).reshape(9, 9).astype(np.int8)
            blocks = ast(
                sud,
                shape=(3, 3, 3, 3),
                strides=(
                    9 * 3 * sud.itemsize,
                    3 * sud.itemsize,
                    9 * sud.itemsize,
                    sud.itemsize,
                ),
            )
            # if (x//9 == y//9 and np.abs(x-y)<3) or (x%9 == y%9 and np.abs(x-y)<=9*2):
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    if x in blocks and y in blocks:
                        # check block constraint
                        if unique_dy[0] in unique_dx:
                            return False
                        else:
                            return True
            return True

        return np.arange(n), np.arange(9), constraint_check

    def render(self):
        canvas = np.ones(self._render_shape, dtype=np.uint8) * 255
        # draw grid
        for i, (x, y) in enumerate(zip(*self._render_grid)):
            thick = 3 if i % 3 == 0 else 1
            cv2.line(
                canvas,
                (x, 0),
                (x, self._render_grid[1][-1]),
                (0, 0, 0),
                thickness=thick,
            )
            cv2.line(
                canvas,
                (0, y),
                (self._render_grid[0][-1], y),
                (0, 0, 0),
                thickness=thick,
            )

        for x in range(self._state.shape[1]):
            for y in range(self._state.shape[0]):
                val = self._state[y, x]
                if val > -1:
                    val += 1
                    pos = (
                        x * self._block_size[0] + self._block_size[0] // 4,
                        int(y * self._block_size[1] + self._block_size[1] * 0.7),
                    )
                    canvas = cv2.putText(
                        canvas,
                        f"{val}",
                        pos,
                        cv2.QT_FONT_BLACK,
                        2,
                        (0, 0, 0),
                        6,
                        cv2.LINE_AA,
                    )

        return canvas


if __name__ == "__main__":
    wname = "Sudoku"
    print("Generating Sudoku...")
    s = Sudoku(number_of_prefilled=30)
    cv2.namedWindow(wname)
    nums = [ord(str(i)) for i in range(1, 10)]
    number = -1
    def onMouse(event, x, y, flags, param):
        global number
        if event == cv2.EVENT_LBUTTONDOWN:
            # get cell at x, y
            row, col = int(y / s._block_size[1]), int(x / s._block_size[0])
            _, done = s.step(((row, col), number))
            number = -1
            if s.is_solution:
                print("Congratulations you solved it!")
            elif len(s.unassigned_vars) == 0 and not done:
                print("That doesn't look right, please re-try!")

    cv2.setMouseCallback(wname, onMouse)

    while True:
        cv2.imshow(wname, s.render())
        k = cv2.waitKey(100) & 0xFF
        # q or esc to quit
        if k == 27 or k == ord("q"):
            break
        # no key
        elif k==255:
            continue
        try:
            # press a number before clicking to insert it
            number = nums.index(k)
            print("number", number)
        except:
            # press any other button but numbers 1 to 9 to remove inserted numbers
            number = -1

