from __future__ import annotations

from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
rosbag2_py = pytest.importorskip("rosbag2_py")

import time

import yaml
from arena_humansim.core.recorder import BagRecorder, default_record_dir
from arena_humansim.utils.bag_io import extract_agent_states
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentGestures as AgentGesturesMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import Gesture as GestureMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from rclpy.node import Node
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosgraph_msgs.msg import Clock

from tests.unit._bags import Agent, frame_msg, meta_msg

_LATCHED = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


@pytest.fixture
def node(rclpy_context):
    if rclpy_context is None:
        pytest.skip("rclpy unavailable")
    n = Node("recorder_unit_test")
    try:
        yield n
    finally:
        n.destroy_node()


def test_default_record_dir_under_cwd() -> None:
    assert default_record_dir().parent == Path.cwd() / "recordings"


def test_recorder_writes_messages(node: Node, tmp_path: Path) -> None:
    rec = BagRecorder(node, tmp_path / "run")

    pub_agents = node.create_publisher(AgentFrameMsg, "/agent_states", 10)
    pub_meta = node.create_publisher(AgentMetaMsg, "/agent_meta", _LATCHED)
    pub_gestures = node.create_publisher(AgentGesturesMsg, "/agent_gestures", _LATCHED)
    pub_geom = node.create_publisher(WorldGeometryMsg, "/world_geometry", _LATCHED)
    pub_clock = node.create_publisher(Clock, "/clock", 10)

    for _ in range(3):
        pub_agents.publish(AgentFrameMsg())
        pub_meta.publish(AgentMetaMsg())
        pub_gestures.publish(AgentGesturesMsg())
        pub_geom.publish(WorldGeometryMsg())
        pub_clock.publish(Clock())
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.05)

    rec.close()

    bag_dir = tmp_path / "run" / "bag"
    assert bag_dir.exists()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics_seen: set[str] = set()
    while reader.has_next():
        topic, _, _ = reader.read_next()
        topics_seen.add(topic)

    assert "/agent_states" in topics_seen
    assert "/agent_meta" in topics_seen
    assert "/agent_gestures" in topics_seen
    assert "/world_geometry" in topics_seen
    assert "/clock" in topics_seen
    assert types["/agent_states"] == "arena_humansim_msgs/msg/AgentFrame"
    assert types["/agent_meta"] == "arena_humansim_msgs/msg/AgentMeta"
    assert types["/agent_gestures"] == "arena_humansim_msgs/msg/AgentGestures"


def test_recorder_close_idempotent(node: Node, tmp_path: Path) -> None:
    rec = BagRecorder(node, tmp_path / "run")
    rec.close()
    rec.close()


def _spin_until(node: Node, done, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def test_recorder_keeps_meta_latched_before_it_started(node: Node, tmp_path: Path) -> None:
    agents = [Agent(3, 1.0, 2.0, vx=0.5, policy="orca", name="bob", handedness="l"), Agent(-1, 0.0, 0.0, policy="", kind=1)]
    policies = ["sfm", "orca"]
    pub_meta = node.create_publisher(AgentMetaMsg, "/agent_meta", _LATCHED)
    pub_gestures = node.create_publisher(AgentGesturesMsg, "/agent_gestures", _LATCHED)
    pub_agents = node.create_publisher(AgentFrameMsg, "/agent_states", 10)
    gestures = AgentGesturesMsg()
    gestures.agent_id = [3]
    gestures.gestures = [GestureMsg(slot="head", clip="nod")]
    pub_meta.publish(meta_msg(1_000_000_000, agents, policies))
    pub_gestures.publish(gestures)

    rec = BagRecorder(node, tmp_path / "run")
    _spin_until(node, lambda: rec._policies is not None)
    for i in range(3):
        pub_agents.publish(frame_msg(2_000_000_000 + i * 100_000_000, agents, policies))
    _spin_until(node, lambda: rec._first_policies is not None)
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    rec.close()

    manifest = yaml.safe_load((tmp_path / "run" / "recording.yaml").read_text())
    assert manifest["first_message_policies"] == {3: "orca", -1: ""}
    df = extract_agent_states(tmp_path / "run" / "bag")
    assert len(df) >= 2
    assert df["planner"].tolist() == ["orca", ""] * (len(df) // 2)
    assert df["x"].tolist()[:2] == [1.0, 0.0]

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(tmp_path / "run" / "bag"), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    counts: dict[str, int] = {}
    while reader.has_next():
        topic, _, _ = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
    assert counts["/agent_meta"] == 1
    assert counts["/agent_gestures"] == 1


def test_record_seed_keeps_the_latest_and_every_seed_in_order(node: Node, tmp_path: Path) -> None:
    rec = BagRecorder(node, tmp_path / "run", provenance={"seed": 3})
    rec.record_seed(11)
    rec.record_seed(12)
    rec.close()

    manifest = yaml.safe_load((tmp_path / "run" / "recording.yaml").read_text())
    assert manifest["seed"] == 12
    assert manifest["seeds"] == [3, 11, 12]
