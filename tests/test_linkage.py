"""
Test cases for a Linkage.
"""
import unittest
import pylinkage as pl


class TestLinkage(unittest.TestCase):
    """Test cases for linkages."""
    def __init__(self, method_name):
        super().__init__(method_name)
        # Main motor
        self.crank = pl.Crank(
            0, 1,
            joint0=(0, 0),  # Fixed to a single point in space
            angle=0.31, distance=1,
            name="B"
        )
        # Close the loop
        self.pin = pl.Pivot(
            3, 2,
            joint0=self.crank, joint1=(3, 0),
            distance0=3, distance1=1, name="C"
        )

    def test_definition(self):
        """Test if a linkage can be defined."""
        # Linkage definition
        my_linkage = pl.Linkage(
            joints=[self.crank, self.pin],
            order=[self.crank, self.pin],
            name="My four-bar linkage"
        )
        self.assertTupleEqual((self.crank, self.pin), my_linkage.joints)


if __name__ == '__main__':
    unittest.main()
