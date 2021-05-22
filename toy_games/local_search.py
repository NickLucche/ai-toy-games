from typing import Any, Callable, Tuple
from toy_games.csp import CSPNode, ConstraintGraph, CSP
import numpy as np
import random


# start from a complete (rnd) assignment, optimize to get solution, keep only current state in memory
def local_search(game: CSP,
                 csp: ConstraintGraph,
                 select_var_and_val: Callable[[CSP, ConstraintGraph],
                                              Tuple[int, Any]],
                 stop_walk: Callable[[int, CSP], bool] = None,
                 max_steps: int = 10000):
    if stop_walk is None:
        stop_walk = lambda i, _: i >= max_steps
    # sample random assignment
    for var in csp.nodes:
        game.step((var.var_id, var.sample_from_domain()))
    # print("random init assignment", game.assignment)
    i = 0
    while not stop_walk(i, game):
        if game.is_solution:
            print(f"Local search executed {i} steps")
            return True, game
        # select next variable to assign and a value from its domain
        # this is where all the local search logic goes
        var, value = select_var_and_val(game, csp)
        # print(f"Selected var-value pair {var}, {value}")
        game.step((var, value))
        i += 1
    print(f"Local search executed {i} steps")
    # in case it fails return `Fail`+current best "guess"
    return False, game


def min_conflicts(game: CSP, graph: ConstraintGraph):
    """
    Most popular local search heuristics for selecting the next 
    variable-value pair to assing during a local search.
    It does so by selecting a random conflicting variable
    and the value which minimizes the number of conflicts
    among the variable's domain.  
    """
    # print("Current assignment", game.assignment)

    def count_conflicts(x: CSPNode, x_val=None):
        x_val = game.assignment[x.var_id] if x_val is None else x_val
        conflicting_ys = []
        for y in graph.generate_neighbors(x):
            # assignment is always complete during local search
            y_val = game.assignment[y.var_id]
            if not graph.constraint_check(x.var_id, [x_val], y.var_id,
                                          [y_val]):
                conflicting_ys.append(y)
        return conflicting_ys

    # get conflicting variables
    conflicting_vars = []
    for x in graph.nodes:
        if x.n_conflicts < 0:
            # init n_conflicts
            x.n_conflicts = len(count_conflicts(x))
        if x.n_conflicts > 0:
            conflicting_vars.append(x)

    if not len(conflicting_vars):
        # you should be done already
        return None, None

    # choose rnd conflicting var
    x = random.choice(conflicting_vars)
    min_conflicts, value, conflicting_vars = np.inf, None, []
    for xval in x.domain:
        # allow current value to be re-chosen if it's min
        # if xval == current_xval:
            # continue
        conflicting_ys = count_conflicts(x, xval)
        if len(conflicting_ys) < min_conflicts:
            min_conflicts = len(conflicting_ys)
            value = [xval]
            conflicting_vars = [conflicting_ys]
        elif len(conflicting_ys) == min_conflicts:
            # for random tie break
            value.append(xval)
            conflicting_vars.append(conflicting_ys)

    # random tie break
    idx = random.choice(range(len(value)))

    # update conflict counters of conflicting variables too
    x.n_conflicts = min_conflicts
    for var in conflicting_vars[idx]:
        var.n_conflicts = len(count_conflicts(var))
    return x.var_id, value[idx]


if __name__ == '__main__':
    from toy_games.games.n_queens import NQueens
    import cv2
    queens = NQueens(n=32)
    csp = ConstraintGraph(queens)
    done, game = local_search(queens, csp, min_conflicts, max_steps=int(1e6))
    if done:
        print("Local Search solved the problem!")
    else:
        print("Local Search did not solve the problem :(")

    cv2.imshow('N Queens', game.render())
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()