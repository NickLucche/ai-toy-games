from utils import manhattan_distance
from sliding_block_puzzle import Direction, SBPuzzle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from search_algorithms import *

NUM_COLOR = (0, 0, 0)

WIDTH = 900
N = 3
BLOCK_SIZE = WIDTH // N

xx, yy = np.meshgrid(np.linspace(BLOCK_SIZE / 2, WIDTH - BLOCK_SIZE / 2, N),
                     np.linspace(BLOCK_SIZE / 2, WIDTH - BLOCK_SIZE / 2, N))


def visualize_matrix(matrix: np.ndarray):
    # matrix[matrix > 0] = 255
    canvas = np.ones((WIDTH, WIDTH), dtype=np.uint8) * 255
    for x in range(0, WIDTH, BLOCK_SIZE):
        canvas = cv2.line(canvas, (x + 2, 0), (x + 2, WIDTH - 2), (0, 0, 0), 3)
        canvas = cv2.line(canvas, (0, x + 2), (WIDTH - 2, x + 2), (0, 0, 0), 3)
    # canvas = cv2.line(canvas, (WIDTH-2, 0), (WIDTH-2, WIDTH-2), (0,0,0), 3)
    # simple way t o scale matrix for viz, each pixel will be scaled in size
    # img = cv2.resize(matrix, (WIDTH, WIDTH), interpolation=cv2.INTER_NEAREST)
    for i in range(N):
        for j in range(N):
            number = matrix[i, j]
            pos = (int(xx[i, j]), int(yy[i, j]))
            if number == 0:
                top_leftc = (int(pos[0] - BLOCK_SIZE / 2), int(pos[1] - BLOCK_SIZE / 2))
                btm_rightc = (int(pos[0] + BLOCK_SIZE / 2), int(pos[1] + BLOCK_SIZE / 2))
                canvas = cv2.rectangle(canvas, top_leftc, btm_rightc, (0, 0, 0), -1)
            else:
                # color in bgr
                canvas = cv2.putText(canvas, str(number), pos,
                                     cv2.QT_FONT_BLACK, 2, NUM_COLOR, 6,
                                     cv2.LINE_AA)
    cv2.imshow('Sliding Block Puzzle', canvas)
    # plt.imshow(resized)
    # plt.show()


if __name__ == '__main__':
    visualize = True
    puzzle = SBPuzzle(N)
    # dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    dirs = list(Direction)
    print(puzzle.state)
    # search for solution
    # res, elapsed = bread_first_search(Node(puzzle.state, None, None))
    def heuristic(n: Node):
        return manhattan_distance(n.state, SBPuzzle.goal_state(n.state.shape[0]))
    res, elapsed = a_star_search(Node(puzzle.state, None, None), heuristic)
    if res is None:
        print(f"Search failed in {elapsed} seconds")
        exit(-1)
    actions, cost = res
    print(f"Search took {elapsed} seconds! Found path cost is {cost}")
    if visualize:
        for action in actions:
            # action = dirs[np.random.choice(4)]
            print(action)
            obs = puzzle.step(action)
            print(obs)
            visualize_matrix(obs)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
