import unittest
from games.n_queens import NQueens
import numpy as np

class NQueensTest(unittest.TestCase):

    def test_empty_repr(self):
        game = NQueens()
        self.assertEqual(game.__repr__(), '')

    def test_2_assignments(self):
        game = NQueens()
        state, consistency = game.step((0, 0))
        self.assertTrue(consistency)
        self.assertTrue((state == np.array([0]+[-1]*7)).all())
        self.assertTrue(0 not in game.unassigned_vars)
        state, consistency = game.step((1, 2))
        self.assertTrue(consistency)
        self.assertTrue((state == np.array([0, 2]+[-1]*6)).all())

    def test_row_inc(self):
        game = NQueens()
        state, consistency = game.step((0, 0))
        state, consistency = game.step((0, 0))
        self.assertTrue(consistency)
        self.assertIn(0, game.assigned_vars)
        self.assertNotIn(0, game.unassigned_vars)

        state, consistency = game.step((1, 0))
        self.assertFalse(consistency)

    def test_diag_inc(self):
        game = NQueens()
        state, consistency = game.step((0, 0))
        state, consistency = game.step((7, 7))
        self.assertFalse(consistency)





