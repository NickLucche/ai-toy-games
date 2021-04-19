from typing import Callable
import numpy as np
from games.sliding_block_puzzle import Direction, SBPuzzle
from collections import deque
import time
from heapq import heapify, heappop, heappush


class Node:
    state: np.ndarray
    action: Direction
    parent: 'Node'
    path_cost: float  # g in literature
    f: float  # f(n) = g(n) + h(n)
    _hash: str

    def __init__(self,
                 state: np.ndarray,
                 action: Direction,
                 parent: 'Node',
                 step_cost: float = 1) -> None:
        self.state = state
        self.action = action
        self.parent = parent
        self.path_cost = parent.path_cost + step_cost if parent else 0.
        # heuristic estimate to be added after init
        self.f = self.path_cost
        self._hash = hash(str(self.state))

    def generate_children(self, test_goal_on_gen=False):
        for action in Direction:
            sbp = SBPuzzle(self.state.shape[0], init_state=self.state)
            child = Node(sbp.step(action), action, self)  # TODO: step cost
            # filter out 'invalid' actions, really important!!
            if child.state is None:
                continue
            else:
                yield (child, SBPuzzle.goal_test(
                    child.state)) if test_goal_on_gen else child

    @property
    def hash(self):
        return self._hash

    # these methods are called to decide node ordering in priority queue
    def __lt__(self, other: 'Node'):
        return self.f < other.f

    def __lte__(self, other: 'Node'):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f


def solution(node: Node):
    # builds sequence of actions which led to goal state
    actions = []
    cost = node.path_cost
    print("Solution", node.state)
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent

    return reversed(actions), cost


# sbpuzzle can easily loop if you can select actions like right-left-right-left..indefinitely
# need an "explored set"!
def bread_first_search(init_node: Node):
    if SBPuzzle.goal_test(init_node.state):
        return solution(init_node), 0.
    frontier = deque()
    explored_set = set()
    frontier.append(init_node)
    start = time.time()
    while len(frontier):
        node: Node = frontier.popleft()
        # add node we're going to expand to explored_set
        explored_set.add(node.hash)
        for child, goal in node.generate_children(test_goal_on_gen=True):
            if not child.hash in explored_set:
                if goal:
                    return solution(child), time.time() - start
                else:
                    frontier.append(child)

    return None, time.time() - start


# tree-search
def depth_limited_search(init_node: Node, depth: int = 10):
    if SBPuzzle.goal_test(init_node.state):
        return solution(init_node), 0.
    elif depth == 0:
        return None, 0.

    start = time.time()
    for child in init_node.generate_children():
        res = depth_limited_search(child, depth - 1)
        if res is not None:
            return res, time.time() - start

    return None, time.time() - start


def a_star_search(init_node: Node,
                  h: Callable[[Node], float],
                  path_cost: bool = True,
                  heuristic: bool = True):
    """ A* search algorithm; the implementation is the same as the Uniform-Cost-Search algorithm,
        but here we're guided by a heuristic function h, used to compute `f(n)=g(n)+h(n)`,
        which expresses priority of Node n in a priority queue from which we select
        nodes to expand.
    Args:
        init_node (Node): Initial node.
        h (Callable[[Node], float]): Heuristic function to use during search.
    Returns:
        A solution to the search (None in case of failure) as in a sequence of actions needed
        to reach goal state along with cost and elapsed time of search.
        (actions, cost), time.
    """

    # this can be generalized to find path between any starting state s1 and
    # any destination/goal state s2 like here https://www.geeksforgeeks.org/a-search-algorithm/

    def f(n: Node):
        #  options to build greedy best-first and lowest cost first
        return (n.path_cost if path_cost else 0.) + (h(n) if heuristic else 0.)

    # by default, lowest valued entries are retrieved first
    frontier = []
    # for efficient membership testing (check if node is in frontier)
    frontier_set = {}

    def add_to_frontier(_node: Node):
        # NOTE: elements in heapq should be comparable or first one unique (refer to: https://stackoverflow.com/a/53554555/4991653)
        # add it to the nodes to be explored with priority given by f
        _node.f = f(_node)
        heappush(frontier, _node)
        frontier_set[_node.hash] = _node

    # note that we don't need the node to store f, since f is only needed to express node priority.
    # We'll return actual cost anyway once we compute solution.
    add_to_frontier(init_node)

    explored_set = set()
    start = time.time()
    while len(frontier):
        node = heappop(frontier)  
        frontier_set.pop(node.hash) # pop from set too?
        # consider goal state as you pop from queue rather than when generating successors, as there's no
        # guarantee successors will have (in general) a good f; here we're sure node is the best guess atm
        if SBPuzzle.goal_test(node.state):
            # return path along with actual cost
            return solution(node), time.time() - start

        explored_set.add(node.hash)
        # generate node's successors
        for child in node.generate_children(test_goal_on_gen=False):
            if not child.hash in explored_set and not child.hash in frontier_set:
                add_to_frontier(child)
            # check whether node is already in frontier BUT with higher cost, thus this is a better path
            elif child.hash in frontier_set and frontier_set[
                    child.hash].path_cost > child.path_cost:
                # replace frontier node
                frontier.remove(frontier_set[child.hash])
                heapify(frontier)  # this is linear time
                add_to_frontier(child)

    return None, time.time() - start


# iterativa deepening a*
def ida_search(init_node: Node,
                  h: Callable[[Node], float],
                  path_cost: bool = True,
                  heuristic: bool = True):
    pass