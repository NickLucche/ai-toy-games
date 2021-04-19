import numpy as np
import enum

# i, j indexing
dir_offset = [
    np.asarray([i, j], dtype=np.int32)
    for i, j in [(-1, 0), (0, 1), (1, 0), (0, -1)]
]


class Direction(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SBPuzzle:
    def __init__(self, n: int = 3, init_state: np.ndarray = None) -> None:
        self.size = n
        # initialize state
        if init_state is not None:
            # assume init state is solvable
            assert init_state.shape[0] == n == init_state.shape[1]
            # copy state by value o/w reference will be kept and search is messed up
            self._state = np.empty_like(init_state)
            self._state[:] = init_state
        else:
            self._state = np.random.choice(n * n,
                                           size=n * n, replace=False).reshape(
                                               n, n).astype(np.uint8)
            # initialize with solvable state
            while not self.is_solvable:
                print(
                    f"Initial state \n{self._state}\nis not solvable, re-trying.."
                )
                self._state = np.random.choice(n * n,
                                               size=n * n,
                                               replace=False).reshape(
                                                   n, n).astype(np.uint8)

        self.pos_pointer = np.array(np.where(self._state == 0),
                                    dtype=np.int32).reshape(2, )

    def _slide(self, index_offset: list):
        next_pos = self.pos_pointer + index_offset
        self._state[self.pos_pointer[0],
                    self.pos_pointer[1]] = self._state[next_pos[0],
                                                       next_pos[1]]
        self._state[next_pos[0], next_pos[1]] = 0
        self.pos_pointer = next_pos
        return self._state.copy()

    def step(self, direction: Direction):
        # flat vector indexing
        position = self.pos_pointer[0] * self.size + self.pos_pointer[1]
        # print(self.pos_pointer, position)
        if direction == Direction.UP:
            # check if you can't go up
            if position < self.size:
                return None
        elif direction == Direction.DOWN:
            if position > (self.size * self.size - self.size - 1):
                return None
        elif direction == Direction.LEFT:
            if position % self.size == 0:
                return None
        elif direction == Direction.RIGHT:
            if position % self.size == (self.size - 1):
                return None
        else:
            raise Exception("Unknown Direction")

        return self._slide(dir_offset[direction.value])

    @property
    def state(self):
        # very important to return copy of state o/w you'll change parent state as you search
        return self._state.copy()

    @property
    def is_solution(self):
        return self.goal_test(self._state)

    def _count_inversions(self, arr):
        # arr here is flattened
        inv_count = 0
        for i in range(self.size * self.size - 1):
            if arr[i] == 0:
                continue
            for j in range(i + 1, self.size * self.size):
                # count pairs(i, j) such that i appears
                # before j, but i > j.
                if (arr[j] > 0 and arr[i] > arr[j]):
                    inv_count += 1
        return inv_count

    @property
    def is_solvable(self):
        # This function returns true if given puzzle is solvable. Adapted from https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/

        # Count inversions in given 8 puzzle
        inv_count = self._count_inversions(self.state.reshape(-1, ))
        # If grid is odd, return true if inversion count is even
        if self.size % 2 == 1:
            return inv_count % 2 == 0
        else:  # even grid
            # get row of blank tile from bottom
            row, _ = np.where(self._state == 0)
            # check whether it's an odd or even position from bottom
            is_even = (self.size - row) % 2 == 0
            # if even and inv count is odd return True
            if is_even:
                return inv_count % 2 == 1
            else:
                return inv_count % 2 == 0

    @staticmethod
    def goal_test(state: np.ndarray):
        # state has to be a valid state
        goal = np.arange(np.multiply(*state.shape)).astype(np.uint8).reshape(
            state.shape)
        return np.sum(goal - state) == 0

    @staticmethod
    def goal_state(N: int):
        # returns goal state for the sliding block puzzle game of size NxN
        return np.arange(N * N).astype(np.uint8).reshape(N, N)
