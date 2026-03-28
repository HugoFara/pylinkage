"""Tests for linkage serialization module."""

import warnings

import pytest

import pylinkage as pl
from pylinkage.joints.joint import _StaticBase
from pylinkage.linkage.serialization import (
    _is_dependency_satisfied,
    _resolve_joint_ref,
    _serialize_joint_ref,
    joint_from_dict,
    joint_to_dict,
    linkage_from_dict,
    linkage_to_dict,
    load_from_json,
    save_to_json,
)

# ---------------------------------------------------------------------------
# _serialize_joint_ref
# ---------------------------------------------------------------------------


class TestSerializeJointRef:
    """Tests for _serialize_joint_ref."""

    def test_none_returns_none(self):
        """None joint returns None."""
        assert _serialize_joint_ref(None, ()) is None

    def test_joint_in_linkage_returns_ref(self):
        """Joint in the linkage list returns a ref dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), distance=1, angle=0.1, name="B")
        result = _serialize_joint_ref(crank, (crank,))
        assert result == {"ref": "B"}

    def test_implicit_static_returns_inline(self):
        """Implicit Static not in linkage returns inline dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            static = pl.Static(5, 3, name="S")
        result = _serialize_joint_ref(static, ())
        assert result is not None
        assert result["inline"] is True
        assert result["type"] == "Static"
        assert result["x"] == 5
        assert result["y"] == 3

    def test_non_static_not_in_linkage_returns_ref(self):
        """Non-static joint not in linkage returns ref by name."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), distance=1, angle=0.1, name="B")
        result = _serialize_joint_ref(crank, ())
        assert result == {"ref": "B"}


# ---------------------------------------------------------------------------
# joint_to_dict
# ---------------------------------------------------------------------------


class TestJointToDict:
    """Tests for joint_to_dict."""

    def test_static_joint(self):
        """Serialize a static joint."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            static = pl.Static(1, 2, name="S")
        d = joint_to_dict(static)
        assert d["name"] == "S"
        assert d["x"] == 1
        assert d["y"] == 2

    def test_crank_joint(self):
        """Serialize a crank joint with distance and angle."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), distance=1.5, angle=0.3, name="B")
        d = joint_to_dict(crank)
        assert d["distance"] == 1.5
        assert d["angle"] == 0.3

    def test_linkage_joints_none_defaults_to_empty(self):
        """When linkage_joints is None, it defaults to empty tuple."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            static = pl.Static(0, 0, name="S")
        d = joint_to_dict(static, linkage_joints=None)
        assert d["name"] == "S"

    def test_revolute_joint(self):
        """Serialize a revolute joint."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), distance=1, angle=0.1, name="B")
            rev = pl.Revolute(
                2, 1, joint0=crank, joint1=(3, 0),
                distance0=2.0, distance1=1.5, name="C",
            )
        d = joint_to_dict(rev, linkage_joints=(crank, rev))
        assert d["distance0"] == 2.0
        assert d["distance1"] == 1.5

    def test_prismatic_joint(self):
        """Serialize a prismatic joint."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), distance=1, angle=0.1, name="B")
            prismatic = pl.Prismatic(
                3, 0,
                joint0=crank,
                joint1=(0, 0),
                joint2=(10, 0),
                revolute_radius=3.0,
                name="slider",
            )
        d = joint_to_dict(prismatic, linkage_joints=(crank, prismatic))
        assert d["revolute_radius"] == 3.0
        assert "joint2" in d


# ---------------------------------------------------------------------------
# joint_from_dict
# ---------------------------------------------------------------------------


class TestJointFromDict:
    """Tests for joint_from_dict."""

    def test_unknown_type_raises(self):
        """Unknown joint type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown joint type"):
            joint_from_dict({"type": "Unknown", "name": "X", "x": 0, "y": 0}, {})

    def test_static_from_dict(self):
        """Reconstruct a static joint."""
        j = joint_from_dict({"type": "Static", "name": "S", "x": 1, "y": 2}, {})
        assert isinstance(j, _StaticBase)
        assert j.x == 1
        assert j.y == 2

    def test_crank_from_dict(self):
        """Reconstruct a crank joint."""
        j = joint_from_dict(
            {"type": "Crank", "name": "B", "x": 1, "y": 0, "distance": 1.5, "angle": 0.3},
            {},
        )
        assert isinstance(j, pl.Crank)
        assert j.r == 1.5

    def test_revolute_from_dict(self):
        """Reconstruct a revolute joint."""
        j = joint_from_dict(
            {
                "type": "Revolute",
                "name": "C",
                "x": 2,
                "y": 1,
                "distance0": 2.0,
                "distance1": 1.5,
            },
            {},
        )
        assert isinstance(j, pl.Revolute)
        assert j.r0 == 2.0
        assert j.r1 == 1.5

    def test_prismatic_from_dict(self):
        """Reconstruct a prismatic joint."""
        j = joint_from_dict(
            {
                "type": "Prismatic",
                "name": "S",
                "x": 3,
                "y": 0,
                "revolute_radius": 2.5,
            },
            {},
        )
        assert isinstance(j, pl.Prismatic)


# ---------------------------------------------------------------------------
# _resolve_joint_ref
# ---------------------------------------------------------------------------


class TestResolveJointRef:
    """Tests for _resolve_joint_ref."""

    def test_none_returns_none(self):
        assert _resolve_joint_ref(None, {}) is None

    def test_inline_creates_static(self):
        """Inline reference creates a new Static joint."""
        result = _resolve_joint_ref(
            {"inline": True, "type": "Static", "x": 5, "y": 3, "name": "S"},
            {},
        )
        assert result is not None
        assert result.x == 5
        assert result.y == 3

    def test_ref_resolves_from_dict(self):
        """Named ref resolves from joints_by_name dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s = pl.Static(0, 0, name="S")
        result = _resolve_joint_ref({"ref": "S"}, {"S": s})
        assert result is s

    def test_ref_missing_returns_none(self):
        """Named ref not in dict returns None."""
        result = _resolve_joint_ref({"ref": "missing"}, {})
        assert result is None

    def test_no_ref_no_inline_returns_none(self):
        """Dict with neither ref nor inline returns None."""
        result = _resolve_joint_ref({}, {})
        assert result is None


# ---------------------------------------------------------------------------
# _is_dependency_satisfied
# ---------------------------------------------------------------------------


class TestIsDependencySatisfied:
    """Tests for _is_dependency_satisfied."""

    def test_none_is_satisfied(self):
        assert _is_dependency_satisfied(None, {}) is True

    def test_inline_is_satisfied(self):
        assert _is_dependency_satisfied({"inline": True}, {}) is True

    def test_ref_present_is_satisfied(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s = pl.Static(0, 0, name="S")
        assert _is_dependency_satisfied({"ref": "S"}, {"S": s}) is True

    def test_ref_missing_not_satisfied(self):
        assert _is_dependency_satisfied({"ref": "missing"}, {}) is False


# ---------------------------------------------------------------------------
# Linkage round-trip serialization
# ---------------------------------------------------------------------------


class TestLinkageRoundTrip:
    """Test full linkage serialization/deserialization."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            rev = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0, name="C"
            )
            linkage = pl.Linkage(
                joints=[crank, rev],
                order=[crank, rev],
                name="TestFourBar",
            )
        return linkage

    def test_to_dict_structure(self, fourbar_linkage):
        """to_dict returns expected structure."""
        d = linkage_to_dict(fourbar_linkage)
        assert d["name"] == "TestFourBar"
        assert len(d["joints"]) == 2
        assert d["solve_order"] is not None

    def test_round_trip(self, fourbar_linkage):
        """Serialize and deserialize produces equivalent linkage."""
        d = linkage_to_dict(fourbar_linkage)
        restored = linkage_from_dict(d)
        assert restored.name == fourbar_linkage.name
        assert len(restored.joints) == len(fourbar_linkage.joints)
        for j_orig, j_new in zip(
            fourbar_linkage.joints, restored.joints, strict=False
        ):
            assert j_orig.name == j_new.name

    def test_json_file_round_trip(self, fourbar_linkage, tmp_path):
        """Save and load from JSON file."""
        path = tmp_path / "test_linkage.json"
        save_to_json(fourbar_linkage, path)
        loaded = load_from_json(path)
        assert loaded.name == fourbar_linkage.name
        assert len(loaded.joints) == len(fourbar_linkage.joints)

    def test_linkage_to_json_method(self, fourbar_linkage, tmp_path):
        """Test convenience methods on Linkage class."""
        path = tmp_path / "test_linkage.json"
        fourbar_linkage.to_json(str(path))
        loaded = pl.Linkage.from_json(str(path))
        assert loaded.name == fourbar_linkage.name

    def test_linkage_to_dict_method(self, fourbar_linkage):
        """Test to_dict/from_dict convenience methods."""
        d = fourbar_linkage.to_dict()
        restored = pl.Linkage.from_dict(d)
        assert restored.name == fourbar_linkage.name


class TestLinkageFromDictEdgeCases:
    """Edge cases for linkage_from_dict."""

    def test_unresolvable_deps_raises(self):
        """Circular dependencies raise ValueError."""
        data = {
            "name": "Bad",
            "joints": [
                {
                    "type": "Crank",
                    "name": "A",
                    "x": 0,
                    "y": 0,
                    "distance": 1,
                    "angle": 0.1,
                    "joint0": {"ref": "B"},
                },
                {
                    "type": "Crank",
                    "name": "B",
                    "x": 0,
                    "y": 0,
                    "distance": 1,
                    "angle": 0.1,
                    "joint0": {"ref": "A"},
                },
            ],
        }
        with pytest.raises(ValueError, match="Could not resolve"):
            linkage_from_dict(data)
