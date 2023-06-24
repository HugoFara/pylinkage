class MutableAgent(list):
    """
    A custom class that is mutable, subscriptable, and supports index assignment.

    You should only use it as a dictionary of 3 elements. No backward compatibility guaranty on other use cases.
    """
    score: float
    dimensions: tuple
    init_positions: tuple

    def __init__(self, score=None, dimensions=None, init_position=None):
        super().__init__((score, dimensions, init_position))
        self.score = score
        self.dimensions = dimensions
        self.init_positions = init_position

    def __setitem__(self, key, value):
        """
        Allow index assignment.
        """
        # If the key is an integer, treat it as an index.
        if key == 0:
            self.score = value
        elif key == 1:
            self.dimensions = value
        elif key == 2:
            self.init_positions = value
        else:
            raise IndexError()

    def __getitem__(self, key):
        """
        Allow subscripting.
        """
        if isinstance(key, int):
            # If the key is an integer, treat it as an index.
            if key == 0:
                return self.score
            if key == 1:
                return self.dimensions
            if key == 2:
                return self.init_positions
            raise IndexError()

    def __repr__(self):
        return f"Agent(score={self.score}, dimensions={self.dimensions}, init_positions={self.init_positions})"
