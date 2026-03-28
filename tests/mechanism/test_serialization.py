"""Tests for mechanism/serialization.py — dict/JSON serialization paths."""

import json

from pylinkage.mechanism import (
    DriverLink,
    GroundJoint,
    GroundLink,
    Link,
    Mechanism,
    PrismaticJoint,
    RevoluteJoint,
    is_legacy_format,
    joint_from_dict,
    joint_to_dict,
    link_from_dict,
    link_to_dict,
    mechanism_from_dict,
    mechanism_from_json,
    mechanism_to_dict,
    mechanism_to_json,
)
from pylinkage.mechanism.joint import TrackerJoint
from pylinkage.mechanism.link import ArcDriverLink

# ---------------------------------------------------------------------------
# joint_to_dict / joint_from_dict
# ---------------------------------------------------------------------------


class TestJointToDict:
    """Tests for joint_to_dict()."""

    def test_ground_joint(self):
        j = GroundJoint("O1", position=(0.0, 0.0))
        d = joint_to_dict(j)
        assert d["id"] == "O1"
        assert d["type"] == "ground"
        assert d["position"] == [0.0, 0.0]
        # name == id, so "name" key should be absent
        assert "name" not in d

    def test_revolute_joint(self):
        j = RevoluteJoint("A", position=(1.5, 2.5), name="Joint A")
        d = joint_to_dict(j)
        assert d["type"] == "revolute"
        assert d["name"] == "Joint A"

    def test_prismatic_joint(self):
        j = PrismaticJoint(
            "S",
            position=(1.0, 0.0),
            axis=(0.0, 1.0),
            line_point=(0.5, 0.5),
            slide_distance=3.0,
        )
        d = joint_to_dict(j)
        assert d["type"] == "prismatic"
        assert d["axis"] == [0.0, 1.0]
        assert d["line_point"] == [0.5, 0.5]
        assert d["slide_distance"] == 3.0

    def test_tracker_joint(self):
        j = TrackerJoint(
            "T",
            position=(0.0, 0.0),
            ref_joint1_id="A",
            ref_joint2_id="B",
            distance=1.5,
            angle=0.3,
        )
        d = joint_to_dict(j)
        assert d["type"] == "tracker"
        assert d["ref_joint1_id"] == "A"
        assert d["ref_joint2_id"] == "B"
        assert d["distance"] == 1.5
        assert d["angle"] == 0.3


class TestJointFromDict:
    """Tests for joint_from_dict()."""

    def test_ground_joint(self):
        d = {"id": "O1", "type": "ground", "position": [0.0, 0.0]}
        j = joint_from_dict(d)
        assert isinstance(j, GroundJoint)
        assert j.position == (0.0, 0.0)

    def test_revolute_joint(self):
        d = {"id": "A", "type": "revolute", "position": [1.0, 2.0], "name": "Joint A"}
        j = joint_from_dict(d)
        assert isinstance(j, RevoluteJoint)
        assert j.name == "Joint A"

    def test_prismatic_joint(self):
        d = {
            "id": "S",
            "type": "prismatic",
            "position": [1.0, 0.0],
            "axis": [0.0, 1.0],
            "line_point": [0.5, 0.5],
            "slide_distance": 3.0,
        }
        j = joint_from_dict(d)
        assert isinstance(j, PrismaticJoint)
        assert j.axis == (0.0, 1.0)
        assert j.line_point == (0.5, 0.5)
        assert j.slide_distance == 3.0

    def test_tracker_joint(self):
        d = {
            "id": "T",
            "type": "tracker",
            "position": [0.0, 0.0],
            "ref_joint1_id": "A",
            "ref_joint2_id": "B",
            "distance": 1.5,
            "angle": 0.3,
        }
        j = joint_from_dict(d)
        assert isinstance(j, TrackerJoint)
        assert j.ref_joint1_id == "A"
        assert j.distance == 1.5

    def test_default_type_is_revolute(self):
        d = {"id": "X", "position": [0.0, 0.0]}
        j = joint_from_dict(d)
        assert isinstance(j, RevoluteJoint)

    def test_roundtrip_prismatic(self):
        original = PrismaticJoint(
            "S", position=(1.0, 2.0), axis=(0.6, 0.8), line_point=(1.0, 1.0), slide_distance=5.0
        )
        d = joint_to_dict(original)
        restored = joint_from_dict(d)
        assert isinstance(restored, PrismaticJoint)
        assert restored.axis == (0.6, 0.8)

    def test_roundtrip_tracker(self):
        original = TrackerJoint(
            "T", position=(0.0, 0.0), ref_joint1_id="A", ref_joint2_id="B", distance=2.0, angle=1.0
        )
        d = joint_to_dict(original)
        restored = joint_from_dict(d)
        assert isinstance(restored, TrackerJoint)
        assert restored.ref_joint2_id == "B"
        assert restored.angle == 1.0


# ---------------------------------------------------------------------------
# link_to_dict / link_from_dict
# ---------------------------------------------------------------------------


class TestLinkToDict:
    """Tests for link_to_dict()."""

    def test_regular_link(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2])
        d = link_to_dict(link)
        assert d["type"] == "link"
        assert d["joints"] == ["A", "B"]
        assert "name" not in d  # name == id

    def test_ground_link(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        link = GroundLink("ground", joints=[j])
        d = link_to_dict(link)
        assert d["type"] == "ground"

    def test_driver_link(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        driver = DriverLink(
            "crank",
            joints=[motor, output],
            motor_joint=motor,
            angular_velocity=0.2,
            initial_angle=0.5,
        )
        d = link_to_dict(driver)
        assert d["type"] == "driver"
        assert d["angular_velocity"] == 0.2
        assert d["initial_angle"] == 0.5
        assert d["motor_joint"] == "O"

    def test_arc_driver_link(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        arc = ArcDriverLink(
            "arc",
            joints=[motor, output],
            motor_joint=motor,
            angular_velocity=0.1,
            arc_start=0.5,
            arc_end=2.5,
            initial_angle=1.0,
        )
        d = link_to_dict(arc)
        assert d["type"] == "arc_driver"
        assert d["arc_start"] == 0.5
        assert d["arc_end"] == 2.5
        assert d["motor_joint"] == "O"

    def test_link_with_custom_name(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[j1, j2], name="Coupler")
        d = link_to_dict(link)
        assert d["name"] == "Coupler"


class TestLinkFromDict:
    """Tests for link_from_dict()."""

    def test_regular_link(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        j2 = RevoluteJoint("B", position=(1.0, 0.0))
        joint_map = {"A": j1, "B": j2}
        d = {"id": "AB", "type": "link", "joints": ["A", "B"]}
        link = link_from_dict(d, joint_map)
        assert isinstance(link, Link)
        assert len(link.joints) == 2

    def test_ground_link(self):
        j = GroundJoint("O", position=(0.0, 0.0))
        joint_map = {"O": j}
        d = {"id": "ground", "type": "ground", "joints": ["O"]}
        link = link_from_dict(d, joint_map)
        assert isinstance(link, GroundLink)

    def test_driver_link(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        joint_map = {"O": motor, "A": output}
        d = {
            "id": "crank",
            "type": "driver",
            "joints": ["O", "A"],
            "motor_joint": "O",
            "angular_velocity": 0.2,
            "initial_angle": 0.5,
        }
        link = link_from_dict(d, joint_map)
        assert isinstance(link, DriverLink)
        assert link.angular_velocity == 0.2
        assert link.motor_joint == motor

    def test_arc_driver_link(self):
        motor = GroundJoint("O", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        joint_map = {"O": motor, "A": output}
        d = {
            "id": "arc",
            "type": "arc_driver",
            "joints": ["O", "A"],
            "motor_joint": "O",
            "angular_velocity": 0.1,
            "arc_start": 0.5,
            "arc_end": 2.5,
            "initial_angle": 1.0,
        }
        link = link_from_dict(d, joint_map)
        assert isinstance(link, ArcDriverLink)
        assert link.arc_start == 0.5
        assert link.arc_end == 2.5

    def test_missing_joint_ids_skipped(self):
        j1 = RevoluteJoint("A", position=(0.0, 0.0))
        joint_map = {"A": j1}
        d = {"id": "AB", "type": "link", "joints": ["A", "MISSING"]}
        link = link_from_dict(d, joint_map)
        assert len(link.joints) == 1  # MISSING is skipped

    def test_driver_motor_must_be_ground(self):
        """If motor_joint points to a non-ground joint, motor_joint stays None."""
        rev = RevoluteJoint("R", position=(0.0, 0.0))
        output = RevoluteJoint("A", position=(1.0, 0.0))
        joint_map = {"R": rev, "A": output}
        d = {
            "id": "crank",
            "type": "driver",
            "joints": ["R", "A"],
            "motor_joint": "R",
            "angular_velocity": 0.1,
        }
        link = link_from_dict(d, joint_map)
        assert isinstance(link, DriverLink)
        assert link.motor_joint is None


# ---------------------------------------------------------------------------
# mechanism_to_dict / mechanism_from_dict
# ---------------------------------------------------------------------------


class TestMechanismDictRoundtrip:
    """Tests for mechanism_to_dict and mechanism_from_dict."""

    def _build_four_bar(self):
        O1 = GroundJoint("O1", position=(0.0, 0.0))
        O2 = GroundJoint("O2", position=(2.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        B = RevoluteJoint("B", position=(2.5, 1.0))

        ground = GroundLink("ground", joints=[O1, O2])
        crank = DriverLink("crank", joints=[O1, A], motor_joint=O1, angular_velocity=0.1)
        coupler = Link("coupler", joints=[A, B])
        rocker = Link("rocker", joints=[O2, B])

        return Mechanism(
            name="FourBar",
            joints=[O1, O2, A, B],
            links=[ground, crank, coupler, rocker],
            ground=ground,
        )

    def test_roundtrip_four_bar(self):
        original = self._build_four_bar()
        data = mechanism_to_dict(original)
        restored = mechanism_from_dict(data)

        assert restored.name == "FourBar"
        assert len(restored.joints) == 4
        assert len(restored.links) == 4
        assert restored.ground is not None
        assert restored.ground.id == "ground"

    def test_dict_contains_ground_key(self):
        mech = self._build_four_bar()
        data = mechanism_to_dict(mech)
        assert data["ground"] == "ground"

    def test_dict_joint_types(self):
        mech = self._build_four_bar()
        data = mechanism_to_dict(mech)
        types = {j["id"]: j["type"] for j in data["joints"]}
        assert types["O1"] == "ground"
        assert types["O2"] == "ground"
        assert types["A"] == "revolute"
        assert types["B"] == "revolute"

    def test_dict_link_types(self):
        mech = self._build_four_bar()
        data = mechanism_to_dict(mech)
        types = {lnk["id"]: lnk["type"] for lnk in data["links"]}
        assert types["ground"] == "ground"
        assert types["crank"] == "driver"
        assert types["coupler"] == "link"

    def test_roundtrip_with_prismatic(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        S = PrismaticJoint("S", position=(1.0, 0.0), axis=(1.0, 0.0), line_point=(0.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        link = Link("OS", joints=[origin, S])

        mech = Mechanism(name="Slider", joints=[origin, S], links=[ground, link])
        data = mechanism_to_dict(mech)
        restored = mechanism_from_dict(data)

        pris = [j for j in restored.joints if isinstance(j, PrismaticJoint)]
        assert len(pris) == 1
        assert pris[0].axis == (1.0, 0.0)

    def test_roundtrip_with_arc_driver(self):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        arc = ArcDriverLink(
            "arc",
            joints=[origin, A],
            motor_joint=origin,
            angular_velocity=0.1,
            arc_start=0.5,
            arc_end=2.5,
            initial_angle=1.0,
        )

        mech = Mechanism(name="Arc", joints=[origin, A], links=[ground, arc])
        data = mechanism_to_dict(mech)
        restored = mechanism_from_dict(data)

        arc_links = [lnk for lnk in restored.links if isinstance(lnk, ArcDriverLink)]
        assert len(arc_links) == 1
        assert arc_links[0].arc_start == 0.5
        assert arc_links[0].arc_end == 2.5

    def test_mechanism_without_ground(self):
        A = RevoluteJoint("A", position=(0.0, 0.0))
        B = RevoluteJoint("B", position=(1.0, 0.0))
        link = Link("AB", joints=[A, B])

        mech = Mechanism(name="NoGround", joints=[A, B], links=[link])
        data = mechanism_to_dict(mech)
        assert "ground" not in data  # No ground link

        restored = mechanism_from_dict(data)
        assert restored.ground is None


# ---------------------------------------------------------------------------
# mechanism_to_json / mechanism_from_json
# ---------------------------------------------------------------------------


class TestJsonFileIO:
    """Tests for mechanism_to_json and mechanism_from_json."""

    def test_save_and_load_json(self, tmp_path):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        crank = DriverLink("crank", joints=[origin, A], motor_joint=origin, angular_velocity=0.1)
        mech = Mechanism(name="JsonTest", joints=[origin, A], links=[ground, crank], ground=ground)

        path = tmp_path / "mech.json"
        mechanism_to_json(mech, path)

        # File should exist and be valid JSON
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "JsonTest"

        # Load it back
        restored = mechanism_from_json(path)
        assert restored.name == "JsonTest"
        assert len(restored.joints) == 2

    def test_json_roundtrip_preserves_driver(self, tmp_path):
        origin = GroundJoint("O", position=(0.0, 0.0))
        A = RevoluteJoint("A", position=(1.0, 0.0))
        ground = GroundLink("ground", joints=[origin])
        crank = DriverLink("crank", joints=[origin, A], motor_joint=origin, angular_velocity=0.3)
        mech = Mechanism(name="D", joints=[origin, A], links=[ground, crank])

        path = tmp_path / "driver.json"
        mechanism_to_json(mech, path)
        restored = mechanism_from_json(path)

        drivers = [lnk for lnk in restored.links if isinstance(lnk, DriverLink)]
        assert len(drivers) == 1
        assert drivers[0].angular_velocity == 0.3


# ---------------------------------------------------------------------------
# is_legacy_format
# ---------------------------------------------------------------------------


class TestIsLegacyFormat:
    """Tests for is_legacy_format()."""

    def test_legacy_format_detected(self):
        data = {
            "joints": [
                {"type": "Static", "x": 0.0, "y": 0.0},
                {"type": "Crank", "x": 1.0, "y": 0.0},
            ]
        }
        assert is_legacy_format(data) is True

    def test_new_format_not_detected(self):
        data = {
            "joints": [
                {"id": "O", "type": "ground", "position": [0.0, 0.0]},
                {"id": "A", "type": "revolute", "position": [1.0, 0.0]},
            ]
        }
        assert is_legacy_format(data) is False

    def test_empty_joints(self):
        assert is_legacy_format({"joints": []}) is False

    def test_no_joints_key(self):
        assert is_legacy_format({}) is False
