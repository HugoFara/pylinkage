"""Tests for Mechanism velocity/acceleration kinematics.

Covers ``set_input_velocity`` / ``get_velocities`` / ``get_accelerations`` /
``step_with_derivatives`` on the ``pylinkage.mechanism.Mechanism`` class.
"""

import math

import pytest

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    Mechanism,
    fourbar,
)


def _crank_and_mechanism() -> tuple[DriverLink, Mechanism]:
    """Build a Grashof four-bar and return its driver link plus mechanism."""
    m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
    driver = next(link for link in m.links if isinstance(link, DriverLink))
    return driver, m


# ---------------------------------------------------------------------------
# set_input_velocity
# ---------------------------------------------------------------------------


class TestSetInputVelocity:
    def test_stores_omega_on_driver(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        assert driver._omega == 10.0
        assert driver._alpha == 0.0

    def test_stores_alpha(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0, alpha=2.5)
        assert driver._alpha == 2.5

    def test_unknown_driver_raises(self) -> None:
        _, m = _crank_and_mechanism()
        # A driver link not registered with this mechanism.
        ground = GroundJoint(id="X", position=(0.0, 0.0))
        tip = GroundJoint(id="Y", position=(1.0, 0.0))
        rogue = DriverLink(id="rogue", joints=[ground, tip], motor_joint=ground)
        with pytest.raises(ValueError):
            m.set_input_velocity(rogue, omega=1.0)


# ---------------------------------------------------------------------------
# get_velocities / get_accelerations defaults
# ---------------------------------------------------------------------------


class TestKinematicAccessorDefaults:
    def test_velocities_none_before_step(self) -> None:
        _, m = _crank_and_mechanism()
        assert all(v is None for v in m.get_velocities())

    def test_accelerations_none_before_step(self) -> None:
        _, m = _crank_and_mechanism()
        assert all(a is None for a in m.get_accelerations())


# ---------------------------------------------------------------------------
# step_with_derivatives
# ---------------------------------------------------------------------------


class TestStepWithDerivatives:
    def test_yields_three_tuples_of_correct_length(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        n = len(m.joints)
        for pos, vel, acc in m.step_with_derivatives(iterations=3):
            assert len(pos) == n
            assert len(vel) == n
            assert len(acc) == n

    def test_grounds_have_zero_kinematics(self) -> None:
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)
        ground_indices = [i for i, j in enumerate(m.joints) if isinstance(j, GroundJoint)]
        assert ground_indices, "Expected at least one ground joint"
        for _pos, vel, acc in m.step_with_derivatives(iterations=1):
            for i in ground_indices:
                assert vel[i] == (0.0, 0.0)
                assert acc[i] == (0.0, 0.0)

    def test_crank_tip_speed_matches_omega_radius(self) -> None:
        driver, m = _crank_and_mechanism()
        omega = 10.0
        m.set_input_velocity(driver, omega=omega)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            assert vel[tip_idx] is not None
            speed = math.hypot(*vel[tip_idx])
            assert speed == pytest.approx(omega * driver.radius, rel=1e-9)

    def test_dependent_joint_velocity_computed(self) -> None:
        """The coupler/rocker connection should pick up a finite velocity."""
        driver, m = _crank_and_mechanism()
        m.set_input_velocity(driver, omega=10.0)

        non_ground_non_tip = [
            i
            for i, j in enumerate(m.joints)
            if not isinstance(j, GroundJoint) and j is not driver.output_joint
        ]
        assert non_ground_non_tip, "Need at least one driven joint"

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            for i in non_ground_non_tip:
                assert vel[i] is not None

    def test_acceleration_matches_centripetal_when_alpha_zero(self) -> None:
        driver, m = _crank_and_mechanism()
        omega = 5.0
        m.set_input_velocity(driver, omega=omega, alpha=0.0)

        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)
        ground = driver.motor_joint
        assert ground is not None

        for _pos, _vel, acc in m.step_with_derivatives(iterations=1):
            assert acc[tip_idx] is not None
            ax, ay = acc[tip_idx]
            mag = math.hypot(ax, ay)
            # |a| = omega² * r when alpha = 0 (pure centripetal)
            assert mag == pytest.approx(omega * omega * driver.radius, rel=1e-6)

    def test_default_omega_zero_yields_zero_tip_velocity(self) -> None:
        """No set_input_velocity → tip velocity should still resolve to zero."""
        driver, m = _crank_and_mechanism()
        tip = driver.output_joint
        assert tip is not None
        tip_idx = m.joints.index(tip)

        for _pos, vel, _acc in m.step_with_derivatives(iterations=1):
            assert vel[tip_idx] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Joint runtime fields are independent of equality
# ---------------------------------------------------------------------------


class TestJointKinematicFieldsDoNotBreakEquality:
    def test_equality_ignores_velocity(self) -> None:
        j1 = GroundJoint(id="A", position=(0.0, 0.0))
        j2 = GroundJoint(id="A", position=(1.0, 1.0))
        j1.velocity = (3.0, 4.0)
        assert j1 == j2  # equality is by id only

    def test_kinematics_default_to_none(self) -> None:
        j = GroundJoint(id="A", position=(0.0, 0.0))
        assert j.velocity is None
        assert j.acceleration is None


# ---------------------------------------------------------------------------
# Collinear anchors and unresolved anchors
# ---------------------------------------------------------------------------


def _collinear_ternary_mechanism() -> Mechanism:
    """A four-bar whose coupler point lies on the line through its anchors.

    ``P`` is a port of the rigid triangle ``(A, B, P)`` placed on the A-B
    line — the standard Chebyshev/Hoeken arrangement. Its two distance
    constraints are therefore parallel, so the constraint-intersection
    solver is singular even though the motion is fully determined.
    """
    from pylinkage.dimensions import Dimensions, DriverAngle
    from pylinkage.hypergraph import (
        Edge,
        Hyperedge,
        HypergraphLinkage,
        Node,
        NodeRole,
        to_mechanism,
    )

    hg = HypergraphLinkage(name="collinear_ternary")
    hg.add_node(Node("O1", role=NodeRole.GROUND))
    hg.add_node(Node("O2", role=NodeRole.GROUND))
    hg.add_node(Node("A", role=NodeRole.DRIVER))
    hg.add_node(Node("B", role=NodeRole.DRIVEN))
    hg.add_node(Node("P", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("O1_A", "O1", "A"))
    hg.add_edge(Edge("A_B", "A", "B"))
    hg.add_edge(Edge("O2_B", "O2", "B"))
    hg.add_edge(Edge("A_P", "A", "P"))
    hg.add_hyperedge(Hyperedge("triangle_P", ("A", "B", "P")))
    dims = Dimensions(
        node_positions={
            "O1": (0.0, 0.0), "O2": (3.0, 0.0), "A": (0.75, 0.0),
            "B": (2.0, -3.6), "P": (1.375, -1.8),
        },
        driver_angles={"A": DriverAngle(angular_velocity=-math.tau / 48)},
        edge_distances={"O1_A": 0.75, "A_B": 3.75, "O2_B": 3.75, "A_P": 1.875},
    )
    return to_mechanism(hg, dims)


def _dead_centre_mechanism() -> Mechanism:
    """An RRR dyad at full extension, with a further joint hanging off it.

    ``P`` is collinear with anchors ``A`` and ``O2``, which are *not*
    rigidly tied to each other — a genuine toggle, where the velocity is
    truly indeterminate. ``Q`` depends on ``P`` but is not itself
    collinear with its own anchors.
    """
    from pylinkage.dimensions import Dimensions, DriverAngle
    from pylinkage.hypergraph import (
        Edge,
        HypergraphLinkage,
        Node,
        NodeRole,
        to_mechanism,
    )

    hg = HypergraphLinkage(name="dead_centre")
    for node_id in ("O1", "O2", "O3"):
        hg.add_node(Node(node_id, role=NodeRole.GROUND))
    hg.add_node(Node("A", role=NodeRole.DRIVER))
    hg.add_node(Node("P", role=NodeRole.DRIVEN))
    hg.add_node(Node("Q", role=NodeRole.DRIVEN))
    for edge_id, source, target in (
        ("O1_A", "O1", "A"), ("A_P", "A", "P"), ("O2_P", "O2", "P"),
        ("P_Q", "P", "Q"), ("O3_Q", "O3", "Q"),
    ):
        hg.add_edge(Edge(edge_id, source, target))
    dims = Dimensions(
        node_positions={
            "O1": (0.0, 0.0), "O2": (3.0, 0.0), "O3": (3.2, 0.2),
            "A": (1.0, 0.0), "P": (1.8, 0.0), "Q": (2.5, 0.9),
        },
        driver_angles={"A": DriverAngle(angular_velocity=-math.tau / 48)},
        # |A - O2| == 2.0 == 0.8 + 1.2, so the dyad starts fully extended.
        edge_distances={
            "O1_A": 1.0, "A_P": 0.8, "O2_P": 1.2,
            "P_Q": math.hypot(0.7, 0.9), "O3_Q": math.hypot(0.7, 0.7),
        },
    )
    return to_mechanism(hg, dims)


class TestCollinearAnchors:
    """A joint sharing a rigid body with both anchors is not a dead centre."""

    def test_constraint_intersection_really_is_singular_here(self) -> None:
        """Guard the premise: the fallback is what resolves this joint."""
        from pylinkage.solver.velocity import solve_revolute_velocity

        m = _collinear_ternary_mechanism()
        for driver in m._driver_links:
            m.set_input_velocity(driver, 2.0)
        next(m.step_with_derivatives(iterations=1))

        joints = {j.id: j for j in m.joints}
        point, first, second = joints["P"], joints["A"], joints["B"]
        cross = (second.x - first.x) * (point.y - first.y) - (
            second.y - first.y
        ) * (point.x - first.x)
        assert cross == pytest.approx(0.0, abs=1e-12)

        assert first.velocity is not None and second.velocity is not None
        vx, vy = solve_revolute_velocity(
            point.x, point.y,
            first.x, first.y, first.velocity[0], first.velocity[1],
            second.x, second.y, second.velocity[0], second.velocity[1],
        )
        assert math.isnan(vx) and math.isnan(vy)

    def test_collinear_ternary_point_resolves(self) -> None:
        m = _collinear_ternary_mechanism()
        for driver in m._driver_links:
            m.set_input_velocity(driver, 2.0)
        index = [j.id for j in m.joints].index("P")

        for _pos, vel, acc in m.step_with_derivatives(iterations=12):
            assert vel[index] is not None
            assert acc[index] is not None

    def test_matches_finite_differences(self) -> None:
        """The recovered velocity is the true one, not merely non-None."""
        h = 1e-4
        m = _collinear_ternary_mechanism()
        for driver in m._driver_links:
            # omega must match the rate the driver actually steps at, or
            # the derivatives describe a different motion than the
            # positions they are compared against.
            m.set_input_velocity(driver, driver.angular_velocity)
        index = [j.id for j in m.joints].index("P")

        frames = list(m.step_with_derivatives(iterations=24, dt=h))
        for i in range(1, len(frames) - 1):
            before = frames[i - 1][0][index]
            after = frames[i + 1][0][index]
            vx, vy = frames[i][1][index]
            assert vx == pytest.approx((after[0] - before[0]) / (2 * h), rel=1e-4)
            assert vy == pytest.approx((after[1] - before[1]) / (2 * h), rel=1e-4)

    def test_genuine_dead_centre_stays_undefined(self) -> None:
        """Anchors that can move relative to each other still yield None —
        the fallback must not paper over a real singularity."""
        m = _dead_centre_mechanism()
        for driver in m._driver_links:
            m.set_input_velocity(driver, 2.0)
        index = [j.id for j in m.joints].index("P")

        # dt=0 holds the mechanism at its fully extended starting pose.
        _pos, vel, acc = next(m.step_with_derivatives(iterations=1, dt=0.0))
        assert vel[index] is None
        assert acc[index] is None


class TestUnresolvedAnchorsPropagate:
    """An unknown anchor velocity is not a zero one."""

    def test_dependent_of_undefined_joint_is_undefined(self) -> None:
        """Q hangs off the dead-centre joint P. Reading P's unknown
        velocity as zero would report Q as stationary, which is a
        plausible-looking answer rather than an honest one."""
        m = _dead_centre_mechanism()
        for driver in m._driver_links:
            m.set_input_velocity(driver, 2.0)
        ids = [j.id for j in m.joints]

        _pos, vel, acc = next(m.step_with_derivatives(iterations=1, dt=0.0))
        assert vel[ids.index("P")] is None
        assert vel[ids.index("Q")] is None
        assert acc[ids.index("Q")] is None

    def test_no_joint_outruns_its_anchors(self) -> None:
        """General invariant: a dependent joint may only report a velocity
        when the anchors it was solved from report one too.

        Ground joints and driver outputs are excluded — they are solved
        from the frame and the crank formula, not from anchor pairs.
        """
        from pylinkage.mechanism.joint import GroundJoint

        m = _dead_centre_mechanism()
        for driver in m._driver_links:
            m.set_input_velocity(driver, 2.0)
        driven_by_crank = {
            d.output_joint.id for d in m._driver_links if d.output_joint is not None
        }

        # dt=0 holds the toggle, so P and Q stay undefined and the
        # invariant has something to bite on.
        checked = 0
        for _pos, _vel, _acc in m.step_with_derivatives(iterations=4, dt=0.0):
            for joint in m.joints:
                if isinstance(joint, GroundJoint) or joint.id in driven_by_crank:
                    continue
                if joint.velocity is None:
                    continue
                anchors = []
                seen = set()
                for link in joint._links:
                    for other in link.joints:
                        if other is joint or other.id in seen:
                            continue
                        if not other.is_defined():
                            continue
                        if link.get_distance(joint, other) is None:
                            continue
                        anchors.append(other)
                        seen.add(other.id)
                for anchor in anchors[:2]:
                    checked += 1
                    assert anchor.velocity is not None, (
                        f"{joint.id} reported a velocity while anchor "
                        f"{anchor.id} did not"
                    )
        # The mechanism must actually contain undefined joints, or this
        # invariant would hold vacuously.
        assert any(j.velocity is None for j in m.joints)
