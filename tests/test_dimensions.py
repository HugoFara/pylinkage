"""Tests for dimensions.py to increase coverage.

Covers: DriverAngle, Dimensions construction, copy, validate_against,
get_node_position, get_edge_distance, get_driver_angle, get_hyperedge_distance.
"""


from pylinkage.dimensions import Dimensions, DriverAngle

# ---------------------------------------------------------------------------
# DriverAngle
# ---------------------------------------------------------------------------

class TestDriverAngle:
    def test_default_initial_angle(self):
        da = DriverAngle(angular_velocity=0.5)
        assert da.angular_velocity == 0.5
        assert da.initial_angle == 0.0

    def test_custom_initial_angle(self):
        da = DriverAngle(angular_velocity=0.1, initial_angle=1.57)
        assert da.angular_velocity == 0.1
        assert da.initial_angle == 1.57


# ---------------------------------------------------------------------------
# Dimensions construction
# ---------------------------------------------------------------------------

class TestDimensionsConstruction:
    def test_empty(self):
        d = Dimensions()
        assert d.node_positions == {}
        assert d.driver_angles == {}
        assert d.edge_distances == {}
        assert d.hyperedge_constraints == {}
        assert d.name == ""

    def test_with_data(self):
        d = Dimensions(
            node_positions={"A": (1.0, 2.0), "B": (3.0, 4.0)},
            driver_angles={"A": DriverAngle(0.1)},
            edge_distances={"AB": 2.83},
            hyperedge_constraints={"he1": {("A", "B"): 2.83}},
            name="test-dims",
        )
        assert d.name == "test-dims"
        assert len(d.node_positions) == 2
        assert len(d.driver_angles) == 1
        assert len(d.edge_distances) == 1
        assert len(d.hyperedge_constraints) == 1


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------

class TestDimensionsCopy:
    def test_copy_returns_new_object(self):
        d = Dimensions(
            node_positions={"A": (0.0, 0.0)},
            driver_angles={"A": DriverAngle(0.1, 0.5)},
            edge_distances={"AB": 1.0},
            hyperedge_constraints={"he1": {("A", "B"): 1.0}},
            name="original",
        )
        c = d.copy()
        assert c is not d
        assert c.name == "original"

    def test_copy_independent(self):
        """Modifying the copy does not affect the original."""
        d = Dimensions(
            node_positions={"A": (0.0, 0.0)},
            driver_angles={"A": DriverAngle(0.1, 0.5)},
            edge_distances={"AB": 1.0},
            hyperedge_constraints={"he1": {("A", "B"): 1.0}},
        )
        c = d.copy()
        c.node_positions["B"] = (1.0, 1.0)
        c.edge_distances["BC"] = 2.0
        c.driver_angles["B"] = DriverAngle(0.2)
        c.hyperedge_constraints["he2"] = {("B", "C"): 2.0}

        assert "B" not in d.node_positions
        assert "BC" not in d.edge_distances
        assert "B" not in d.driver_angles
        assert "he2" not in d.hyperedge_constraints

    def test_copy_driver_angle_independent(self):
        """DriverAngle objects are properly copied."""
        d = Dimensions(
            driver_angles={"A": DriverAngle(0.1, 0.5)},
        )
        c = d.copy()
        c.driver_angles["A"] = DriverAngle(0.9, 1.0)
        assert d.driver_angles["A"].angular_velocity == 0.1
        assert d.driver_angles["A"].initial_angle == 0.5


# ---------------------------------------------------------------------------
# validate_against
# ---------------------------------------------------------------------------

class TestValidateAgainst:
    def test_valid_dimensions(self):
        d = Dimensions(
            node_positions={"A": (0.0, 0.0), "B": (1.0, 0.0)},
            driver_angles={"B": DriverAngle(0.1)},
            edge_distances={"AB": 1.0},
        )
        errors = d.validate_against(["A", "B"], ["AB"])
        assert errors == []

    def test_unknown_node_in_positions(self):
        d = Dimensions(node_positions={"A": (0.0, 0.0), "X": (1.0, 1.0)})
        errors = d.validate_against(["A", "B"], [])
        assert any("X" in e and "node_positions" in e for e in errors)

    def test_unknown_node_in_driver_angles(self):
        d = Dimensions(driver_angles={"MISSING": DriverAngle(0.1)})
        errors = d.validate_against(["A"], [])
        assert any("MISSING" in e and "driver_angles" in e for e in errors)

    def test_unknown_edge_in_distances(self):
        d = Dimensions(edge_distances={"XY": 1.0})
        errors = d.validate_against([], ["AB"])
        assert any("XY" in e and "edge_distances" in e for e in errors)

    def test_unknown_hyperedge(self):
        d = Dimensions(hyperedge_constraints={"bad_he": {("A", "B"): 1.0}})
        errors = d.validate_against([], [], ["he1"])
        assert any("bad_he" in e and "hyperedge_constraints" in e for e in errors)

    def test_hyperedge_not_checked_when_none(self):
        """When hyperedge_ids is None, hyperedge_constraints are not validated."""
        d = Dimensions(hyperedge_constraints={"he1": {("A", "B"): 1.0}})
        errors = d.validate_against([], [])
        # No errors about hyperedges since hyperedge_ids was not provided
        assert not any("hyperedge" in e.lower() for e in errors)

    def test_multiple_errors(self):
        d = Dimensions(
            node_positions={"X": (0.0, 0.0)},
            driver_angles={"Y": DriverAngle(0.1)},
            edge_distances={"ZZ": 1.0},
        )
        errors = d.validate_against(["A"], ["AB"])
        assert len(errors) == 3


# ---------------------------------------------------------------------------
# get_node_position
# ---------------------------------------------------------------------------

class TestGetNodePosition:
    def test_existing_node(self):
        d = Dimensions(node_positions={"A": (1.0, 2.0)})
        assert d.get_node_position("A") == (1.0, 2.0)

    def test_missing_node(self):
        d = Dimensions()
        assert d.get_node_position("A") is None


# ---------------------------------------------------------------------------
# get_edge_distance
# ---------------------------------------------------------------------------

class TestGetEdgeDistance:
    def test_existing_edge(self):
        d = Dimensions(edge_distances={"AB": 5.0})
        assert d.get_edge_distance("AB") == 5.0

    def test_missing_edge(self):
        d = Dimensions()
        assert d.get_edge_distance("AB") is None


# ---------------------------------------------------------------------------
# get_driver_angle
# ---------------------------------------------------------------------------

class TestGetDriverAngle:
    def test_existing_driver(self):
        da = DriverAngle(0.1, 0.5)
        d = Dimensions(driver_angles={"A": da})
        result = d.get_driver_angle("A")
        assert result is da

    def test_missing_driver(self):
        d = Dimensions()
        assert d.get_driver_angle("A") is None


# ---------------------------------------------------------------------------
# get_hyperedge_distance
# ---------------------------------------------------------------------------

class TestGetHyperedgeDistance:
    def test_existing_constraint(self):
        d = Dimensions(
            hyperedge_constraints={"he1": {("A", "B"): 3.0, ("B", "C"): 4.0}}
        )
        # Should try both orderings -- canonical is (min, max)
        assert d.get_hyperedge_distance("he1", "A", "B") == 3.0
        assert d.get_hyperedge_distance("he1", "B", "A") == 3.0

    def test_missing_hyperedge(self):
        d = Dimensions()
        assert d.get_hyperedge_distance("he1", "A", "B") is None

    def test_missing_pair_in_existing_hyperedge(self):
        d = Dimensions(
            hyperedge_constraints={"he1": {("A", "B"): 3.0}}
        )
        assert d.get_hyperedge_distance("he1", "X", "Y") is None

    def test_ordering_normalization(self):
        """Key is (min, max), so (B, A) should also work if stored as (A, B)."""
        d = Dimensions(
            hyperedge_constraints={"he1": {("A", "B"): 7.0}}
        )
        assert d.get_hyperedge_distance("he1", "B", "A") == 7.0


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------


class TestDriverAngleSerialization:
    def test_round_trip(self):
        da = DriverAngle(angular_velocity=0.2, initial_angle=1.0)
        restored = DriverAngle.from_dict(da.to_dict())
        assert restored == da

    def test_from_dict_missing_initial_angle_defaults_to_zero(self):
        restored = DriverAngle.from_dict({"angular_velocity": 0.3})
        assert restored == DriverAngle(angular_velocity=0.3, initial_angle=0.0)


class TestDimensionsSerialization:
    def _sample(self) -> Dimensions:
        return Dimensions(
            node_positions={"A": (0.0, 0.0), "B": (1.0, 0.0)},
            driver_angles={"B": DriverAngle(0.1, 0.5)},
            edge_distances={"AB": 1.0},
            hyperedge_constraints={"h1": {("A", "B"): 3.0, ("A", "C"): 4.0}},
            name="demo",
        )

    def test_round_trip(self):
        original = self._sample()
        restored = Dimensions.from_dict(original.to_dict())
        assert restored == original

    def test_to_dict_is_json_safe(self):
        import json

        payload = self._sample().to_dict()
        # Must be round-trippable through json without custom encoders.
        reloaded = json.loads(json.dumps(payload))
        restored = Dimensions.from_dict(reloaded)
        assert restored == self._sample()

    def test_from_dict_empty(self):
        restored = Dimensions.from_dict({})
        assert restored == Dimensions()

    def test_from_dict_accepts_legacy_stringified_hyperedge_keys(self):
        """Legacy format stored hyperedge keys as ``"('A', 'B')"`` strings."""
        legacy = {
            "node_positions": {"A": [0.0, 0.0]},
            "hyperedge_constraints": {"h1": {"('A', 'B')": 3.0}},
        }
        restored = Dimensions.from_dict(legacy)
        assert restored.hyperedge_constraints == {"h1": {("A", "B"): 3.0}}
