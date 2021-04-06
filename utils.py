import numpy as np

def manhattan_distance(state: np.ndarray, goal: np.ndarray):
    """ Returns Manhattan/City Block/Snake Distance (L1 norm) 
    between `state` and `goal`.
    """
    return np.sum(np.abs(state-goal))