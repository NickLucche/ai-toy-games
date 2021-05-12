from toy_games.games import CSP
from toy_games.games.n_queens import NQueens
from typing import Callable, Iterable, List, Tuple, Union, Any
import numpy as np
from collections import deque


class CSPNode:
    # variable
    var_id: int
    domain: np.ndarray
    var_name: str

    def __init__(self, var: int, domain, var_name: str = None) -> None:
        self.var_id = var
        self.var_name = str(var) if var_name is None else var_name
        # copy domain
        if isinstance(domain, np.ndarray):
            self.domain = np.empty_like(domain)
            self.domain[:] = domain
        elif type(domain) is list:
            self.domain = np.asarray(domain, dtype=np.int32)
        else:
            raise Exception("Unrecognized type for domain", domain)

    def copy(self):
        return CSPNode(self.var_id, self.domain.copy())

    def remove_value_from_domain(self, value: int):
        self.domain = np.asarray([vx for vx in self.domain if vx != value],
                                 dtype=np.int32)

    def add_value_to_domain(self, value: int):
        self.domain = np.append(self.domain, value)

    def __eq__(self, o: object) -> bool:
        return self.var_id == o.var_id


class ConstraintGraph:
    nodes: List[CSPNode] = []
    arcs: List[Tuple[CSPNode, CSPNode]] = []
    constraint_check: Callable[[int, np.ndarray, int, np.ndarray], bool]

    def __init__(self,
                 game: CSP,
                 generate_arcs=True,
                 from_string: str = None) -> None:
        x, d, c = game.csp(game.size)
        self.constraint_check = c
        self._arcs_flag = generate_arcs
        # they all have same domain
        if d.ndim == 1:
            self.nodes = [CSPNode(var, d) for var in x]
        else:
            self.nodes = [CSPNode(var, d[i]) for i, var in enumerate(x)]
        # index var_id: position
        self._var_index = {
            str(var.var_id): i
            for i, var in enumerate(self.nodes)
        }
        self._var_names = {
            str(var.var_name): i
            for i, var in enumerate(self.nodes)
        }

        if generate_arcs:
            if isinstance(game, NQueens):
                self.arcs = [(var, othervar) for var in self.nodes
                             for othervar in self.nodes if var != othervar]

    def generate_neighbors(self,
                           node: CSPNode,
                           exclude_neighbors: List[CSPNode] = None) -> CSPNode:
        # returns neighbors of a node assuming arcs is 'ordered' wrt first arc
        # (e.g [(x1, _)..(x2,)..(xn, _)])
        if not self._arcs_flag:
            raise Exception(
                "Can't generate neighbors of node if arcs haven't been generated!"
            )
        aiter = iter(self.arcs)
        for x, y in aiter:
            if x == node:
                curr = x
                while curr == x:
                    if exclude_neighbors is None or y not in exclude_neighbors:
                        # yield curr, y
                        yield y
                    try:
                        curr, y = next(aiter)
                    except StopIteration as e:
                        return
                return

    def update_node(self, node: CSPNode):
        """ Substitues node with variable id equal to that of `node` with `node`.
            Returns True if node was swapped, False otherwise.
        """
        n = len(self.arcs)
        if not str(node.var_id) in self._var_index:
            return False
        if not self._arcs_flag:
            return True
        self.nodes[self._var_index[str(node.var_id)]] = node
        # assume arcs is ordered wrt first node
        for i in range(n):
            x, y = self.arcs[i]
            if x == node:
                for j in range(i, n):
                    if self.arcs[j][0] != x:
                        return True
                    self.arcs[j] = (node, y)
        return False

    def get_node(self, node_id: str):
        if not str(node_id) in self._var_index:
            return None
        return self.nodes[self._var_index[str(node_id)]]

    def _from_str(self, s: str):
        # TODO:
        x, d, c = s.replace(' ', '').split('|')

        cons_list = []
        for constr in c.split(','):
            if constr.lower() == 'aldiff':
                self.arcs.extend([(var, othervar) for var in self.nodes
                                  for othervar in self.nodes
                                  if var != othervar])
                cons_list.extend([((self.nodes[i], self.nodes[j]),
                                   f'self.nodes[{i}]!=self.nodes[{j}]')
                                  for i in range(len(self.nodes))
                                  for j in range(len(self.nodes))
                                  if self.nodes[i] != self.nodes[j]])
                break
            for op in ['>=', '<=', '>', '<', '==', '!=']:
                if op in constr:
                    a, b = constr.split(op)
                    cons_list.append(((
                        (self.nodes[self._var_names[a]],
                         self.nodes[self._var_names[b]]),
                    ), f'self.nodes[{self._var_names[a]}]{op}self.nodes[{self._var_names[b]}]'
                                      ))
                break
        # def _ccheck(x, dx, y, dy):
        # for vars, con in cons_list:
        # if vars[0] == self.get_node(x) and vars[1] == self.get_node(y):
        # eval usage    
        # self.constraint_check


# arc consistency 3
def ac3(csp_graph: ConstraintGraph):
    """
    Returns an arc-consistent version of a CSP problem or None in case the 
    problem is unsolvable (some variable domain is empty).
    """
    # queue of arcs, initially all arcs
    arcsq = deque(len(csp_graph.arcs))
    arcsq.extend(csp_graph.arcs)
    while len(arcsq):
        (x, y) = arcsq.pop()
        # revise binary constraints x-y
        new_domain = []
        for vx in x.domain:
            if csp_graph.constraint_check(x.var_id, [vx], y.var_id, y.domain):
                new_domain.append(vx)
        if len(new_domain) == 0:
            # unsolvable csp
            return None
        # if revised
        if len(new_domain) != len(x.domain):
            newx = CSPNode(x.var_id, new_domain)
            csp_graph.update_node(newx)
            for xk in csp_graph.generate_neighbors(x, exclude_neighbors=[y]):
                arcsq.append((xk, x))
    return csp_graph


def backtrack_search(
        game: CSP, csp: ConstraintGraph,
        select_unassigned_variable: Callable[[CSP], CSPNode],
        order_domain_values: Callable[[CSP, ConstraintGraph, CSPNode],
                                      Iterable], inference: Callable):
    """
    Classical incremental formulation of CSP as search problem:
        - States are partial assignments.
        - Initial state is the empty assignment.
        - Goal state is a complete and consistent (satisfies all constraints) assignment.
        - Actions consist in assigning to a specific unassigned variable $X_i$ a value $\in D_i$
    Backtrack is a depth-first search algorithm which assigns a value to a variable at each step,
    and backtracks when a variable has no legal values to assign.
    We assume `game` to hold the assignment as its state, and we assume the (initial) empty
    assignment to be the state with all -1.
    """
    # goal-test
    if game.is_solution:
        return game
    # 1. Select (unassigned) variable to assign
    var = select_unassigned_variable(game, csp)
    # 2. Select value from respective domain wrt some ordering
    for value in order_domain_values(game, csp, var):
        # assign variable
        print("Assignment tested:", var.var_id, value)
        curr_ass, consistent = game.step((var.var_id, value))
        inferences = None
        if consistent:
            # var.remove_value_from_domain(value) #this breaks everything as it loops algorithm
            # 3. Run inferences, here we expect `inference` to return a list of modified
            # variable nodes (with updated domains), along with previous nodes in order to undo
            # inferences
            inferences = inference(game, csp, (var, value))
            # None inferences stands for a failure
            if inferences is not None:
                for new_node in inferences[0]:
                    # print("Updating node", new_node.var_id, new_node.domain)
                    csp.update_node(new_node)
                # print('current state', game.state)
                res = backtrack_search(game, csp, select_unassigned_variable,
                                       order_domain_values, inference)
                if res is not None:
                    return res
        # remove assignment and inferences
        # var.add_value_to_domain(value)
        game.step((var.var_id, -1))
        # we expect inferences to be able to undo previous inferences
        if inferences is not None:
            for old_node in inferences[1]:
                csp.update_node(old_node)

    return None


# minimum remaining values
def mrv_heuristics(game: CSP, csp: ConstraintGraph) -> CSPNode:
    """ 
    Choose the *unassigned* variable with the fewest “legal” remaining values in its domain,
    aka 'fail-first' heuristics.
    """
    dom_lengths = np.asarray([len(node.domain) for node in csp.nodes],
                             dtype=np.int32)
    # mask assigned variables so that they're not chosen
    assert len(dom_lengths) > len(game.assigned_vars)
    dom_lengths[game.assigned_vars] = np.iinfo(np.int32).max
    _min = np.argmin(dom_lengths)
    return csp.nodes[_min]


def least_constraining_values(game: CSP, csp: ConstraintGraph, var: CSPNode):
    """
    Least constraining values heuristics, prefers the value that rules out the fewest choices
    for the neighboring variables in the constraint graph, 'fail-last' approach.
    Generates domain values for variable `var` following order described above.
    """
    # for value in var.domain:
    # for neigh in csp.generate_neighbors(var):
    # csp.constraint_check()
    pass


def generate_domain_values(game: CSP, csp: ConstraintGraph, var: CSPNode):
    # get all values in domain at current timestep so we avoid incurring
    # in modifications as we iterate over it
    return list(var.domain)


def forward_checking(game: CSP, csp: ConstraintGraph, ass: Tuple[CSPNode,
                                                                 int]):
    """ 
    Estabilish arc-consistency for the variable we just assigned: for each *unassigned variable*
    Y that is connected to X by a constraint, delete from Y’s domain any value that is inconsistent
    with the value chosen for X.
    
    Note: you shouldn't run FC if you already made the graph arc-consistent with
    ac3 beforehand, this won't do anything.
    """
    x, value = ass
    # old nodes used for un-doing inferences
    old_nodes = []
    new_nodes = []
    # exclude already assigned variables while generating neighbors
    assigned_variables = [csp.get_node(var) for var in game.assigned_vars]
    for y in csp.generate_neighbors(x, exclude_neighbors=assigned_variables):
        new_y_domain = [
            vy for vy in y.domain
            if csp.constraint_check(x.var_id, [value], y.var_id, [vy])
        ]
        # failure detection, there's some variable which has no legal values
        if len(new_y_domain) == 0:
            print(
                f"Forward checking detected an arc inconsistency between node {x.var_id} and {y.var_id}"
            )
            return None
        # avoid updating variable if there's no change
        if len(new_y_domain) != len(y.domain):
            new_nodes.append(CSPNode(y.var_id, new_y_domain))
        old_nodes.append(y.copy())

    return new_nodes, old_nodes


def maintaining_arc_consistency(game: CSP, csp: ConstraintGraph,
                                ass: Tuple[CSPNode, int]):
    """
    Yet another inference heuristics which runs ac-3 on the 
    *unassigned* neighbors of X_i and keeps tracks of changed
    nodes so that we can undo inferences.
    """
    x, value = ass
    old_nodes = []
    new_nodes = []
    # exclude already assigned variables while generating neighbors
    assigned_variables = [csp.get_node(var) for var in game.assigned_vars]
    unassign_neigh = csp.generate_neighbors(
        x, exclude_neighbors=assigned_variables)
    arcsq = deque(len(csp.arcs))
    # initialize queue with (y, x) with y unassigned neighbor of x
    arcsq.extend([(y, x) for y in unassign_neigh])
    while len(arcsq):
        (y, xk) = arcsq.pop()
        # check if we popped the variable we just assigned
        xk_domain = value if xk == x else xk.domain
        # revise binary constraints x-y
        new_domain = [
            vy for vy in y.domain
            if csp.constraint_check(y.var_id, [vy], xk.var_id, xk_domain)
        ]
        if len(new_domain) == 0:
            # unsolvable csp
            print(
                f"MAC detected an arc inconsistency between node {y.var_id} and {xk.var_id}"
            )
            return None
        # if revised
        if len(new_domain) != len(y.domain):
            new_nodes.append(CSPNode(y.var_id, new_domain))
            for yk in csp.generate_neighbors(y, exclude_neighbors=[xk]):
                arcsq.append((yk, y))

        old_nodes.append(y.copy())
    return new_nodes, old_nodes
