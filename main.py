from utils import manhattan_distance
from games.sliding_block_puzzle import Direction, SBPuzzle
from games import *
from csp import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from search_algorithms import *
from argparse import ArgumentParser

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
                top_leftc = (int(pos[0] - BLOCK_SIZE / 2),
                             int(pos[1] - BLOCK_SIZE / 2))
                btm_rightc = (int(pos[0] + BLOCK_SIZE / 2),
                              int(pos[1] + BLOCK_SIZE / 2))
                canvas = cv2.rectangle(canvas, top_leftc, btm_rightc,
                                       (0, 0, 0), -1)
            else:
                # color in bgr
                canvas = cv2.putText(canvas, str(number), pos,
                                     cv2.QT_FONT_BLACK, 2, NUM_COLOR, 6,
                                     cv2.LINE_AA)
    cv2.imshow('Sliding Block Puzzle', canvas)
    # plt.imshow(resized)
    # plt.show()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-g',
                      '--game',
                      type=str,
                      help='game to solve',
                      choices=['snake', 'sbpuzzle', 'nqueens', 'sudoku'])
    args.add_argument('-s',
                      '--search-algorithm',
                      type=str,
                      help='search algorithm to use')
    args.add_argument('-n',
                      '--size',
                      type=int,
                      default=8,
                      help='size of the grid')
    args.add_argument('--no-visualize', help='do not visualize results', action='store_false', default=False)
    args = args.parse_args()
    visualize = not args.no_visualize
    
    if args.game == 'sbpuzzle':
        game = SBPuzzle(args.size)
    elif args.game == 'snake':
        game = Snake(args.size)
    elif args.game == 'nqueens':
        game = NQueens(args.size)
    elif args.game == 'sudoku':
        game = NQueens(args.size)
    else:
        game = None
    # dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    print(game.state)

    # search for solution
    if isinstance(game, SBPuzzle):
        dirs = list(Direction)
        # res, elapsed = bread_first_search(Node(puzzle.state, None, None))
        def heuristic(n: Node):
            return manhattan_distance(n.state,
                                    SBPuzzle.goal_state(n.state.shape[0]))

        res, elapsed = a_star_search(Node(game.state, None, None),
                                    heuristic)  #, path_cost=False)
        if res is None:
            print(f"Search failed in {elapsed} seconds")
            exit(-1)
        actions, cost = res
        print(f"Search took {elapsed} seconds! Found path cost is {cost}")
        if visualize:
            for action in actions:
                # action = dirs[np.random.choice(4)]
                print(action)
                obs = game.step(action)
                print(obs)
                visualize_matrix(obs)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
    elif isinstance(game, NQueens):
        const_graph = ConstraintGraph(game)
        def no_inferences(*args):
            return [], [] 
        start = time.time()
        solution = backtrack_search(game, const_graph, mrv_heuristics, generate_domain_values, forward_checking)
        elapsed = time.time()-start
        if solution is None:
            print(f"Unable to find solution! Elapsed Run-time: {elapsed}s")
        else:
            print(f"Solution found in {elapsed}s! ", solution.state)
            if visualize:
                cv2.imshow('N Queens', solution.render())
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                
