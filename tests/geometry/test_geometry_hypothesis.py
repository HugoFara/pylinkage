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
from pylinkage.geometry.secants import (
    INTERSECTION_NONE,
    INTERSECTION_ONE,
    INTERSECTION_SAME,
    INTERSECTION_TWO,
)

# Strategy for valid coordinates
float_st = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)

# Strategy for valid radius
radius_st = st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False)


class TestSquaredDistance(unittest.TestCase):
    """Property tests for sqr_dist function."""

    @given(float_st, float_st, float_st, float_st)
    def test_sqr_dist_non_negative(self, x1, y1, x2, y2):
        """Squared distance is always non-negative."""
        result = sqr_dist(x1, y1, x2, y2)
        self.assertGreaterEqual(result, 0)

    @given(float_st, float_st)
    def test_sqr_dist_same_point_is_zero(self, x, y):
        """Distance from a point to itself is zero."""
        self.assertEqual(sqr_dist(x, y, x, y), 0)

    @given(float_st, float_st, float_st, float_st)
    def test_sqr_dist_symmetric(self, x1, y1, x2, y2):
        """Squared distance is symmetric."""
        self.assertAlmostEqual(sqr_dist(x1, y1, x2, y2), sqr_dist(x2, y2, x1, y1))

    @given(float_st, float_st, float_st, float_st)
    def test_sqr_dist_matches_dist_squared(self, x1, y1, x2, y2):
        """Squared distance matches dist squared."""
        self.assertAlmostEqual(sqr_dist(x1, y1, x2, y2), dist(x1, y1, x2, y2) ** 2, places=8)


class TestGetNearestPoint(unittest.TestCase):
    """Property tests for get_nearest_point function."""

    @given(float_st, float_st, float_st, float_st, float_st, float_st)
    def test_nearest_point_is_one_of_inputs(self, ref_x, ref_y, p1_x, p1_y, p2_x, p2_y):
        """Result is always one of the two candidate points."""
        result = get_nearest_point(ref_x, ref_y, p1_x, p1_y, p2_x, p2_y)
        self.assertIn(result, [(p1_x, p1_y), (p2_x, p2_y)])

    @given(float_st, float_st, float_st, float_st)
    def test_nearest_point_ref_equals_first(self, ref_x, ref_y, p2_x, p2_y):
        """If reference equals first point, return first point (or very close)."""
        result = get_nearest_point(ref_x, ref_y, ref_x, ref_y, p2_x, p2_y)
        # With floating point, either p1 or a very close p2 might be returned
        # when both are essentially at the same distance
        d_to_ref = sqr_dist(result[0], result[1], ref_x, ref_y)
        d_to_p2 = sqr_dist(result[0], result[1], p2_x, p2_y)
        self.assertTrue(d_to_ref <= d_to_p2 or d_to_ref < 1e-20)

    @given(float_st, float_st, float_st, float_st)
    def test_nearest_point_ref_equals_second(self, ref_x, ref_y, p1_x, p1_y):
        """If reference equals second point, return second point (or very close)."""
        result = get_nearest_point(ref_x, ref_y, p1_x, p1_y, ref_x, ref_y)
        # With floating point, either p2 or a very close p1 might be returned
        d_to_ref = sqr_dist(result[0], result[1], ref_x, ref_y)
        d_to_p1 = sqr_dist(result[0], result[1], p1_x, p1_y)
        self.assertTrue(d_to_ref <= d_to_p1 or d_to_ref < 1e-20)


class TestNorm(unittest.TestCase):
    """Property tests for norm function."""

    @given(float_st, float_st)
    def test_norm_non_negative(self, x, y):
        """Norm is always non-negative."""
        self.assertGreaterEqual(norm(x, y), 0)

    def test_norm_zero_vector(self):
        """Norm of zero vector is zero."""
        self.assertEqual(norm(0.0, 0.0), 0)

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    def test_norm_matches_math(self, x, y):
        """Norm matches expected formula."""
        expected = math.sqrt(x ** 2 + y ** 2)
        self.assertAlmostEqual(norm(x, y), expected)


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
        float_st,
        float_st,
    )
    def test_cyl_to_cart_with_origin(self, radius, theta, ori_x, ori_y):
        """Distance from origin equals radius when using custom origin."""
        x, y = cyl_to_cart(radius, theta, ori_x, ori_y)
        computed_dist = math.sqrt((x - ori_x) ** 2 + (y - ori_y) ** 2)
        self.assertAlmostEqual(computed_dist, radius, places=10)

    def test_cyl_to_cart_zero_angle(self):
        """Zero angle should give positive x-axis direction."""
        x, y = cyl_to_cart(1.0, 0.0)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)

    def test_cyl_to_cart_90_degree(self):
        """90 degree angle should give positive y-axis direction."""
        x, y = cyl_to_cart(1.0, math.pi / 2)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 1, places=10)


class TestLineFromPoints(unittest.TestCase):
    """Property tests for line_from_points function."""

    @given(float_st, float_st, float_st, float_st)
    def test_line_contains_both_points(self, x1, y1, x2, y2):
        """Both input points should satisfy the line equation."""
        assume(x1 != x2 or y1 != y2)  # Avoid degenerate case
        a, b, c = line_from_points(x1, y1, x2, y2)
        # ax + by + c = 0 for both points
        val1 = a * x1 + b * y1 + c
        val2 = a * x2 + b * y2 + c
        self.assertAlmostEqual(val1, 0, places=8)
        self.assertAlmostEqual(val2, 0, places=8)

    def test_line_from_same_point_gives_zeros(self):
        """Same point input should give zeros."""
        result = line_from_points(1.0, 2.0, 1.0, 2.0)
        self.assertEqual(result, (0.0, 0.0, 0.0))


class TestCircleIntersect(unittest.TestCase):
    """Property tests for circle_intersect function."""

    @given(float_st, float_st, radius_st)
    def test_same_circle_returns_type_3(self, x, y, r):
        """Same circle should return type 3 (coincident)."""
        result = circle_intersect(x, y, r, x, y, r, tol=0.01)
        self.assertEqual(result[0], INTERSECTION_SAME)

    @given(float_st, float_st, radius_st, float_st, float_st, radius_st)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_intersection_count_valid(self, x1, y1, r1, x2, y2, r2):
        """Intersection count should be 0, 1, 2, or 3."""
        result = circle_intersect(x1, y1, r1, x2, y2, r2)
        self.assertIn(result[0], [INTERSECTION_NONE, INTERSECTION_ONE, INTERSECTION_TWO, INTERSECTION_SAME])

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_far_apart_circles_no_intersection(self, x1, y1, r1, r2):
        """Circles far apart should have no intersection."""
        dist_between = r1 + r2 + 10
        result = circle_intersect(x1, y1, r1, x1 + dist_between, y1, r2)
        self.assertEqual(result[0], INTERSECTION_NONE)

    @given(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=5, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=3, allow_nan=False, allow_infinity=False),
    )
    def test_inner_circle_no_intersection(self, x, y, r_outer, r_inner):
        """Circle inside another should have no intersection."""
        result = circle_intersect(x, y, r_outer, x, y, r_inner)
        self.assertEqual(result[0], INTERSECTION_NONE)


class TestCircleLineIntersection(unittest.TestCase):
    """Property tests for circle_line_intersection function."""

    def test_line_through_center_has_two_intersections(self):
        """A line through circle center has two intersections."""
        # x = 0, vertical line through center
        result = circle_line_intersection(0.0, 0.0, 5.0, 1.0, 0.0, 0.0)
        self.assertEqual(result[0], INTERSECTION_TWO)

    def test_tangent_line_has_one_intersection(self):
        """A tangent line has one intersection."""
        # x = 1, tangent at (1, 0)
        result = circle_line_intersection(0.0, 0.0, 1.0, 1.0, 0.0, -1.0)
        # Should be 1 or 2 very close points
        self.assertIn(result[0], [INTERSECTION_ONE, INTERSECTION_TWO])

    @given(float_st, float_st, radius_st)
    def test_far_line_no_intersection(self, cx, cy, r):
        """A line far from circle has no intersection."""
        # Line far above the circle: y = cy + r + 1000
        far_offset = r + 1000
        result = circle_line_intersection(cx, cy, r, 0.0, 1.0, -(cy + far_offset))
        self.assertEqual(result[0], INTERSECTION_NONE)


class TestCircleLineFromPointsIntersection(unittest.TestCase):
    """Property tests for circle_line_from_points_intersection."""

    def test_line_through_center_two_points(self):
        """Line through circle center gives two intersections."""
        result = circle_line_from_points_intersection(0.0, 0.0, 2.0, -5.0, 0.0, 5.0, 0.0)
        self.assertEqual(result[0], INTERSECTION_TWO)
        # Both points should be at distance radius from center
        d1 = math.sqrt(result[1] ** 2 + result[2] ** 2)
        d2 = math.sqrt(result[3] ** 2 + result[4] ** 2)
        self.assertAlmostEqual(d1, 2, places=10)
        self.assertAlmostEqual(d2, 2, places=10)


class TestIntersection(unittest.TestCase):
    """Property tests for the intersection function."""

    @given(float_st, float_st)
    def test_same_point_intersection(self, x, y):
        """Same point intersects with itself."""
        p = (x, y)
        result = intersection(p, p)
        self.assertEqual(result, p)

    @given(float_st, float_st, float_st, float_st)
    def test_different_points_no_intersection(self, x1, y1, x2, y2):
        """Different points don't intersect (without tolerance)."""
        assume(x1 != x2 or y1 != y2)
        p1 = (x1, y1)
        p2 = (x2, y2)
        result = intersection(p1, p2, tol=0)
        self.assertIsNone(result)

    @given(float_st, float_st, radius_st)
    def test_same_circle_intersection(self, x, y, r):
        """Same circle intersects with itself."""
        circle = (x, y, r)
        result = intersection(circle, circle, tol=0.01)
        # Should return something (the circle itself)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
