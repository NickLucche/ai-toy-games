from typing import List, Tuple, Union
import numpy as np
import enum
import cv2

# search here is done by finding path from snake to food, each time new food is added.
# Problem is that shortest path is not optimal (you end up hitting yourself) so you can use longest path finding + hamiltonian cycles
dir_offset = [
    np.asarray([x, y], dtype=np.int8) for x, y in [(0, -1), (1, 0), (0, 1), (-1, 0)]
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
    body: List[np.ndarray]
    body_lenght: int
    food: np.ndarray
    current_dir: np.ndarray

    def __init__(
        self, n: int, body_len: int = 4, walls=True, render_width=800, render_height=800
    ) -> None:
        self.size = n
        self.init_body_lenght = body_len
        self._initialize()
        self.walls = walls
        self._render_shape = (render_width, render_height)
        self._block_size = (render_width // n, render_height // n)

    def _initialize(self):
        self.body_lenght = self.init_body_lenght
        assert self.size // 2 > self.body_lenght, "Body too long to start from center!"
        self.food = None
        self.done = False
        self.score = 0

        mid = self.size // 2
        self.body = []
        for i in range(self.body_lenght):
            self.body.append(np.array([mid - i, mid], dtype=np.int32))
        # going E/Right
        self.current_dir = np.array([1, 0], dtype=np.int8)

    def step(self, action: Union[Action, int]):
        if self.done:
            return None, self.done
        if not isinstance(action, Action):
            action = Action(action)
        direction = self.current_dir if action is None else dir_offset[action.value]
        # can't go opposite direction
        if self.current_dir @ direction == -1:
            direction = self.current_dir
        else:
            self.current_dir = direction
        # check wall
        newhead = self.body[0] + direction
        if self.walls and ((newhead >= self.size).any() or (newhead <= -1).any()):
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
            # print("Score:", self.score)

        for i in range(self.body_lenght - 1, 0, -1):
            # check collision as you move body
            if (newhead == self.body[i]).all():
                return False
            self.body[i][:] = self.body[i - 1]
        # update head too
        self.body[0] = newhead
        return True

    def _spawn_food(self):
        # TODO: for now spawn food anywhere even on snake body
        self.food = np.random.randint(0, self.size - 1, 2)

    def render(self, normalize=False, render_shape: Tuple[int, int] = None):
        render_shape = self._render_shape if render_shape is None else render_shape
        block_size = (
            self._block_size
            if render_shape is None
            else (render_shape[0] // self.size, render_shape[1] // self.size)
        )

        canvas = np.ones((*render_shape, 3), dtype=np.uint8) * 255

        # draw snake
        for block in self.body:
            top_leftc = (int(block[0] * block_size[0]), int(block[1] * block_size[1]))
            btm_rightc = (top_leftc[0] + block_size[0], top_leftc[1] + block_size[1])
            canvas = cv2.rectangle(canvas, top_leftc, btm_rightc, (50, 150, 0), -1)

        # draw food
        if self.food is not None:
            canvas = cv2.circle(
                canvas,
                tuple(
                    [
                        int(self.food[i] * block_size[i] + block_size[i] / 2)
                        for i in range(2)
                    ]
                ),
                block_size[0] // 2,
                (0, 50, 150),
                -1,
            )

        if normalize:
            canvas = canvas.astype(np.float32) / 255.0
        return canvas

    @property
    def won(self):
        return self.body_lenght >= (self.size * self.size)


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
DOWN_KEY = "s"
QUIT = "q"
if __name__ == "__main__":
    continuous_movement = True
    speed = 15
    make_gif = False
    if make_gif:
        from PIL import Image

        frames = []

    snake = Snake(30, 4, walls=False, render_width=900, render_height=900)
    cv2.imshow("Snake", snake.render())

    while not snake.done:
        keystroke = cv2.waitKey(int(1 / speed * 1e3) if continuous_movement else 0)

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
        frame = snake.render()
        cv2.imshow("Snake", frame)
        if make_gif:
            frames.append(frame)
    if snake.won:
        print("Congratulations you made it!")
    else:
        print("Darn, gotta try harder, better luck next time!")
    cv2.destroyAllWindows()

    if make_gif:
        frames = list(map(lambda img: Image.fromarray(img[..., ::-1]), frames))
        print("Writing GIF..")
        img, *imgs = frames
        img.save(
            fp="assets/snake.gif",
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )

