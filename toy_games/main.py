from toy_games.utils import manhattan_distance
from toy_games.games.sliding_block_puzzle import Direction, SBPuzzle
from toy_games.games import *
from toy_games.csp import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from toy_games.search_algorithms import *
from argparse import ArgumentParser

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-g',
                      '--game',
                      type=str,
                      help='game to solve',
                      required=True,
                      choices=['snake', 'sbpuzzle', 'nqueens', 'sudoku'])
    args.add_argument('-s',
                      '--search-algorithm',
                      type=str,
                      help='search algorithm to use',
                      choices=['bfs', 'a*', 'backtrack', 'local_search'])
    args.add_argument('-n',
                      '--size',
                      type=int,
                      default=8,
                      help='size of the grid')
    args.add_argument('--make-gif', type=str, required=False)
    args.add_argument('--no-visualize',
                      help='do not visualize results',
                      action='store_false',
                      default=False)
    args = args.parse_args()
    visualize = not args.no_visualize

    if args.game == 'sbpuzzle':
        game = SBPuzzle(args.size)
    elif args.game == 'snake':
        game = Snake(args.size)
    elif args.game == 'nqueens':
        game = NQueens(args.size)
    # elif args.game == 'sudoku':
    # game = Sudoku(args.size)
    else:
        raise Exception("Unknown game")
    # dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    if args.make_gif:
        from PIL import Image
        output_path = args.make_gif
    print("Searching for solution..")
    print(game.state)
    images = []

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
                img = game.render()
                images.append(Image.fromarray(img))
                if visualize:
                    cv2.imshow(args.game, img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        visualize = False
    elif isinstance(game, NQueens):
        const_graph = ConstraintGraph(game)

        def no_inferences(*args):
            return [], []

        start = time.time()
        solution = backtrack_search(game, const_graph, mrv_heuristics,
                                    generate_domain_values, forward_checking)
        elapsed = time.time() - start
        if solution is None:
            print(f"Unable to find solution! Elapsed Run-time: {elapsed}s")
        else:
            print(f"Solution found in {elapsed}s! ", solution.state)
            if visualize:
                cv2.imshow('N Queens', solution.render())
                cv2.waitKey(0)
    if visualize:
        cv2.destroyAllWindows()

    if args.make_gif:
        print("Writing GIF..")
        # head-rest unpacking
        img, *imgs = images
        # from https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=args.make_gif,
                 format='GIF',
                 append_images=imgs,
                 save_all=True,
                 duration=200,
                 loop=0)
