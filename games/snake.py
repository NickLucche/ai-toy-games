from typing import List, Tuple, Union
import numpy as np
import enum
import cv2

# search here is done by finding path from snake to food, each time new food is added.
# Problem is that shortest path is not optimal (you end up hitting yourself) so you can use longest path finding + hamiltonian cycles
dir_offset = [
    np.asarray([x, y], dtype=np.int8)
    for x, y in [(0, -1), (1, 0), (0, 1), (-1, 0)]
]


class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Snake:
    size: int
    # body implemented as a list rather than a linked list
    # assuming x, y format with origin in top-left corner
    body: List[np.ndarray] = []
    body_lenght: int
    food: np.ndarray = None
    current_dir: np.ndarray
    done = False
    score = 0

    def __init__(self,
                 n: int,
                 body_len: int = 4,
                 walls=True,
                 render_width=800,
                 render_height=800) -> None:
        self.size = n
        self._initialize_body(body_len)
        self.body_lenght = body_len
        self.walls = walls
        self._render_shape = (render_width, render_height)
        self._block_size = (render_width // n, render_height // n)

    def _initialize_body(self, body_len):
        assert self.size // 2 > body_len, 'Body too long to start from center!'
        mid = self.size // 2
        for i in range(body_len):
            self.body.append(np.array([mid - i, mid], dtype=np.int32))
        # going E/Right
        self.current_dir = np.array([1, 0], dtype=np.int8)

    def step(self, action: Union[Action, int]):
        if self.done:
            return None, self.done
        if type(action) is int:
            action = Action(action)
        direction = self.current_dir if action is None else dir_offset[
            action.value]
        # can't go opposite direction
        if self.current_dir @ direction == -1:
            direction = self.current_dir
        else:
            self.current_dir = direction
        # check wall
        newhead = self.body[0] + direction
        if self.walls and ((newhead >= self.size).any() or
                           (newhead <= -1).any()):
            # you're dead
            self.done = True
            return None, self.done
        # move snake
        if not self._move_body(direction):
            # you're dead
            self.done = True
            return None, self.done
        # spawn food (simple logic)
        if self.food is None:
            self._spawn_food()

        return [block.copy() for block in self.body], self.done

    def _move_body(self, offset: Tuple[int, int]):
        newhead = np.remainder(self.body[0] + offset, self.size)
        # check food
        if self.food is not None and (newhead == self.food).all():
            self.food = None
            self.body.append(self.body[-1].copy())
            self.body_lenght += 1
            self.score += 1
            print("Score:", self.score)

        for i in range(self.body_lenght - 1, 0, -1):
            # check collision as you move body
            if (newhead == self.body[i]).all():
                return False
            self.body[i][:] = self.body[i - 1]
        # update head too
        self.body[0] = newhead
        return True

    def _spawn_food(self):
        # for now spawn food anywhere even on snake body
        self.food = np.random.randint(0, self.size - 1, 2)

    def render(self, normalize=False):
        canvas = np.ones((*self._render_shape, 3), dtype=np.uint8) * 255

        # draw snake
        for block in self.body:
            top_leftc = (int(block[0] * self._block_size[0]),
                         int(block[1] * self._block_size[1]))
            btm_rightc = (top_leftc[0] + self._block_size[0],
                          top_leftc[1] + self._block_size[1])
            canvas = cv2.rectangle(canvas, top_leftc, btm_rightc, (50, 150, 0),
                                   -1)

        # draw food
        if self.food is not None:
            canvas = cv2.circle(
                canvas,
                tuple([
                    int(self.food[i] * self._block_size[i] +
                        self._block_size[i] / 2) for i in range(2)
                ]), self._block_size[0] // 2, (0, 50, 150), -1)

        if normalize:
            canvas = canvas.astype(np.float32) / 255.
        return canvas

    @property
    def won(self):
        # TODO: win condition?
        return False


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
DOWN_KEY = "s"
QUIT = "q"
if __name__ == '__main__':
    continuous_movement = True
    speed = 15
    snake = Snake(30, walls=False, render_width=900, render_height=900)
    cv2.imshow('Snake', snake.render())

    while not snake.done:
        keystroke = cv2.waitKey(
            int(1 / speed * 1e3) if continuous_movement else 0)

        # no key pressed
        if keystroke == -1:
            action = None
        # ord gets unicode from one-char string
        elif keystroke == ord(FORWARD_KEY):
            action = Action.UP
        elif keystroke == ord(LEFT_KEY):
            action = Action.LEFT
        elif keystroke == ord(RIGHT_KEY):
            action = Action.RIGHT
        elif keystroke == ord(DOWN_KEY):
            action = Action.DOWN
        elif keystroke == ord(QUIT) or keystroke == 27:
            break
        else:
            print("INVALID KEY")
            continue
        snake.step(action)
        # snake.step(list(Action)[np.random.randint(4)])
        cv2.imshow('Snake', snake.render())

    cv2.destroyAllWindows()