import unittest
import numpy as np
from pylinkage.optimization.collections import Agent, MutableAgent


class TestAgent(unittest.TestCase):
    """Test an Agent object."""
    def test_definition(self):
        """Test if we can faithfully define an agent object."""
        score = (np.random.rand() - 0.5) * 100
        dimensions = np.random.rand(np.random.randint(10))
        init_positions = np.random.rand(np.random.randint(10))
        agent = Agent(score, dimensions, init_positions)
        self.assertEqual(score, agent.score)
        self.assertTupleEqual(tuple(dimensions), tuple(agent.dimensions))
        self.assertTupleEqual(tuple(init_positions), tuple(agent.init_positions))
        for i, val in enumerate([dimensions, init_positions]):
            self.assertTupleEqual(tuple(val), tuple(agent[i + 1]))


class TestMutableAgent(unittest.TestCase):
    """Test a Mutable agent object."""
    def test_definition(self):
        """Test if we can faithfully define an agent object."""
        score = (np.random.rand() - 0.5) * 100
        dimensions = np.random.rand(np.random.randint(10))
        init_positions = np.random.rand(np.random.randint(10))
        agent = MutableAgent(score, dimensions, init_positions)
        self.assertEqual(agent.score, score)
        self.assertTupleEqual(tuple(dimensions), tuple(agent.dimensions))
        self.assertTupleEqual(tuple(init_positions), tuple(agent.init_positions))
        for i, val in enumerate([dimensions, init_positions]):
            self.assertTupleEqual(tuple(val), tuple(agent[i + 1]))

    def test_assignment(self):
        """Assignment test of a MutableAgent object."""
        agent = MutableAgent()
        score = (np.random.rand() - 0.5) * 100
        dimensions = np.random.rand(np.random.randint(10))
        init_positions = np.random.rand(np.random.randint(10))
        for i, val in enumerate([score, dimensions, init_positions]):
            agent[i] = val
        self.assertEqual(agent[0], score)
        for i, val in enumerate([dimensions, init_positions]):
            self.assertTupleEqual(tuple(val), tuple(agent[i + 1]))


if __name__ == '__main__':
    unittest.main()
