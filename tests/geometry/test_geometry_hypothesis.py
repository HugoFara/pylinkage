"""Property-based tests for geometry module using hypothesis."""

import math
import unittest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pylinkage.geometry import (
    circle_intersect,
    circle_line_from_points_intersection,
    circle_line_intersection,
    cyl_to_cart,
    get_nearest_point,
    intersection,
    norm,
    sqr_dist,
)
from pylinkage.geometry.core import dist, line_from_points

# Strategy for valid coordinates (avoiding extreme values)
coord_st = st.tuples(
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
)

# Strategy for valid radius
radius_st = st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False)

# Strategy for circles
circle_st = st.tuples(
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    radius_st,
)


class TestSquaredDistance(unittest.TestCase):
    """Property tests for sqr_dist function."""

    @given(coord_st, coord_st)
    def test_sqr_dist_non_negative(self, p1, p2):
        """Squared distance is always non-negative."""
        result = sqr_dist(p1, p2)
        self.assertGreaterEqual(result, 0)

    @given(coord_st)
    def test_sqr_dist_same_point_is_zero(self, p):
        """Distance from a point to itself is zero."""
        self.assertEqual(sqr_dist(p, p), 0)

    @given(coord_st, coord_st)
    def test_sqr_dist_symmetric(self, p1, p2):
        """Squared distance is symmetric."""
        self.assertAlmostEqual(sqr_dist(p1, p2), sqr_dist(p2, p1))

    @given(coord_st, coord_st)
    def test_sqr_dist_matches_dist_squared(self, p1, p2):
        """Squared distance matches math.dist squared."""
        self.assertAlmostEqual(sqr_dist(p1, p2), dist(p1, p2) ** 2, places=8)


class TestGetNearestPoint(unittest.TestCase):
    """Property tests for get_nearest_point function."""

    @given(coord_st, coord_st, coord_st)
    def test_nearest_point_is_one_of_inputs(self, ref, p1, p2):
        """Result is always one of the two candidate points."""
        result = get_nearest_point(ref, p1, p2)
        self.assertIn(result, [ref, p1, p2])

    @given(coord_st, coord_st)
    def test_nearest_point_ref_equals_first(self, ref, p2):
        """If reference equals first point, return reference."""
        result = get_nearest_point(ref, ref, p2)
        self.assertEqual(result, ref)

    @given(coord_st, coord_st)
    def test_nearest_point_ref_equals_second(self, ref, p1):
        """If reference equals second point, return reference."""
        result = get_nearest_point(ref, p1, ref)
        self.assertEqual(result, ref)


class TestNorm(unittest.TestCase):
    """Property tests for norm function."""

    @given(coord_st)
    def test_norm_non_negative(self, vec):
        """Norm is always non-negative."""
        self.assertGreaterEqual(norm(vec), 0)

    def test_norm_zero_vector(self):
        """Norm of zero vector is zero."""
        self.assertEqual(norm((0, 0)), 0)

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    def test_norm_matches_math(self, x, y):
        """Norm matches expected formula."""
        expected = math.sqrt(x ** 2 + y ** 2)
        self.assertAlmostEqual(norm((x, y)), expected)


class TestCylToCart(unittest.TestCase):
    """Property tests for cyl_to_cart function."""

    @given(
        st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
    )
    def test_cyl_to_cart_distance_preserved(self, radius, theta):
        """Distance from origin equals radius."""
        x, y = cyl_to_cart(radius, theta)
        computed_dist = math.sqrt(x ** 2 + y ** 2)
        self.assertAlmostEqual(computed_dist, radius, places=10)

    @given(
        st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
        coord_st,
    )
    def test_cyl_to_cart_with_origin(self, radius, theta, origin):
        """Distance from origin equals radius when using custom origin."""
        x, y = cyl_to_cart(radius, theta, origin)
        computed_dist = math.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
        self.assertAlmostEqual(computed_dist, radius, places=10)

    def test_cyl_to_cart_zero_angle(self):
        """Zero angle should give positive x-axis direction."""
        x, y = cyl_to_cart(1, 0)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)

    def test_cyl_to_cart_90_degree(self):
        """90 degree angle should give positive y-axis direction."""
        x, y = cyl_to_cart(1, math.pi / 2)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 1, places=10)


class TestLineFromPoints(unittest.TestCase):
    """Property tests for line_from_points function."""

    @given(coord_st, coord_st)
    def test_line_contains_both_points(self, p1, p2):
        """Both input points should satisfy the line equation."""
        assume(p1 != p2)  # Avoid degenerate case
        a, b, c = line_from_points(p1, p2)
        # ax + by + c = 0 for both points
        val1 = a * p1[0] + b * p1[1] + c
        val2 = a * p2[0] + b * p2[1] + c
        self.assertAlmostEqual(val1, 0, places=8)
        self.assertAlmostEqual(val2, 0, places=8)

    def test_line_from_same_point_gives_warning(self):
        """Same point input should give warning."""
        import warnings
        p = (1.0, 2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = line_from_points(p, p)
            self.assertEqual(len(w), 1)
            self.assertEqual(result, (0, 0, 0))


class TestCircleIntersect(unittest.TestCase):
    """Property tests for circle_intersect function."""

    @given(circle_st)
    def test_same_circle_returns_type_3(self, circle):
        """Same circle should return type 3 (coincident)."""
        result = circle_intersect(circle, circle)
        self.assertEqual(result[0], 3)

    @given(circle_st, circle_st)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_intersection_count_valid(self, c1, c2):
        """Intersection count should be 0, 1, 2, or 3."""
        result = circle_intersect(c1, c2)
        self.assertIn(result[0], [0, 1, 2, 3])

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_far_apart_circles_no_intersection(self, x1, y1, r1, r2):
        """Circles far apart should have no intersection."""
        # Place second circle far enough away
        dist_between = r1 + r2 + 10
        c1 = (x1, y1, r1)
        c2 = (x1 + dist_between, y1, r2)
        result = circle_intersect(c1, c2)
        self.assertEqual(result[0], 0)

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=5, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=3, allow_nan=False, allow_infinity=False),
    )
    def test_inner_circle_no_intersection(self, x, y, r_outer, r_inner):
        """Circle inside another should have no intersection."""
        c_outer = (x, y, r_outer)
        c_inner = (x, y, r_inner)
        result = circle_intersect(c_outer, c_inner)
        self.assertEqual(result[0], 0)


class TestCircleLineIntersection(unittest.TestCase):
    """Property tests for circle_line_intersection function."""

    def test_line_through_center_has_two_intersections(self):
        """A line through circle center has two intersections."""
        circle = (0, 0, 5)
        line = (1, 0, 0)  # x = 0, vertical line through center
        result = circle_line_intersection(circle, line)
        self.assertEqual(len(result), 2)

    def test_tangent_line_has_one_intersection(self):
        """A tangent line has one intersection."""
        circle = (0, 0, 1)
        line = (1, 0, -1)  # x = 1, tangent at (1, 0)
        result = circle_line_intersection(circle, line)
        # Should be 1 or 2 very close points
        self.assertIn(len(result), [1, 2])

    @given(circle_st)
    def test_far_line_no_intersection(self, circle):
        """A line far from circle has no intersection."""
        # Line far above the circle
        far_offset = circle[2] + 1000
        line = (0, 1, -(circle[1] + far_offset))  # y = circle[1] + far_offset
        result = circle_line_intersection(circle, line)
        self.assertEqual(len(result), 0)


class TestCircleLineFromPointsIntersection(unittest.TestCase):
    """Property tests for circle_line_from_points_intersection."""

    def test_line_through_center_two_points(self):
        """Line through circle center gives two intersections."""
        circle = (0, 0, 2)
        p1 = (-5, 0)
        p2 = (5, 0)
        result = circle_line_from_points_intersection(circle, p1, p2)
        self.assertEqual(len(result), 2)
        # Both points should be at distance radius from center
        for point in result:
            d = math.sqrt(point[0] ** 2 + point[1] ** 2)
            self.assertAlmostEqual(d, 2, places=10)


class TestIntersection(unittest.TestCase):
    """Property tests for the intersection function."""

    @given(coord_st)
    def test_same_point_intersection(self, p):
        """Same point intersects with itself."""
        result = intersection(p, p)
        self.assertEqual(result, p)

    @given(coord_st, coord_st)
    def test_different_points_no_intersection(self, p1, p2):
        """Different points don't intersect (without tolerance)."""
        assume(p1 != p2)
        result = intersection(p1, p2, tol=0)
        self.assertIsNone(result)

    @given(circle_st)
    def test_same_circle_intersection(self, circle):
        """Same circle intersects with itself."""
        result = intersection(circle, circle)
        # Should return something (could be empty tuple or circle)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
