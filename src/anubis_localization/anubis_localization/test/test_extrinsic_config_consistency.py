"""Static contract checks for the LiDAR-to-base calibration chain.

anubis_localization consumes ``init_R``/``init_T``, anubis_mapping's
keyframe descriptor consumes ``keyframe.lidar_to_base_T``. Keeping these
two copies numerically identical prevents a map/TF frame split that is
otherwise difficult to detect until a robot replay.

[PATCH -- architecture change from darmawan_ws] The original version of
this test also checked a THIRD copy: a static_transform_publisher node
in localization.launch.py that re-published this same transform
manually. That node has been removed entirely -- anubis_description now
publishes base_link -> livox_frame from the URDF/xacro (the actual
single source of truth for this robot's real mounting position), so
there is no launch-file copy left to duplicate or drift. This test now
checks the two config.yaml copies against THAT xacro value instead,
which is a strictly stronger guarantee than the old three-way check: it
verifies the configs match the robot's actual physical description, not
just each other.
"""

import math
import re
from pathlib import Path

import pytest


# test/ -> anubis_localization/ -> src/ -> workspace root
ROOT = Path(__file__).resolve().parents[3]
LOCALIZATION_CONFIG = ROOT / "src" / "anubis_localization" / "anubis_localization" / "config" / "config.yaml"
MAPPING_CONFIG = ROOT / "src" / "anubis_mapping" / "config" / "config.yaml"
SENSORS_XACRO = ROOT / "src" / "anubis_description" / "urdf" / "sensors.xacro"


def _numbers_after_key(text: str, key: str, count: int):
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*\[(.*?)\]", text, re.DOTALL)
    assert match, f"missing vector {key}"
    values = [float(token) for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    assert len(values) == count, f"{key}: expected {count} values, got {len(values)}"
    return values


def _livox_joint_xyz_from_xacro(text: str):
    # Matches <joint name="livox_joint" ...> ... <origin xyz="x y z" .../>
    # within the livox_mid360 macro -- deliberately scoped to that one
    # joint block so a future second <origin> elsewhere in the file
    # (e.g. on the camera joint) cannot be matched by mistake.
    joint_match = re.search(
        r'<joint\s+name="livox_joint"[^>]*>.*?<origin\s+xyz="([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"',
        text,
        re.DOTALL,
    )
    assert joint_match, "could not find livox_joint's <origin xyz=...> in sensors.xacro"
    return [float(joint_match.group(i)) for i in (1, 2, 3)]


def _assert_close(first, second, tolerance=2e-6):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left == pytest.approx(right, abs=tolerance)


def test_all_runtime_copies_share_one_rigid_lidar_to_base_transform():
    localization = LOCALIZATION_CONFIG.read_text(encoding="utf-8")
    mapping = MAPPING_CONFIG.read_text(encoding="utf-8")
    sensors_xacro = SENSORS_XACRO.read_text(encoding="utf-8")

    init_r = _numbers_after_key(localization, "init_R", 9)
    init_t = _numbers_after_key(localization, "init_T", 16)
    keyframe_t = _numbers_after_key(mapping, "lidar_to_base_T", 16)
    _assert_close(init_t, keyframe_t, tolerance=2e-8)
    _assert_close(init_r, [init_t[row * 4 + col] for row in range(3) for col in range(3)], tolerance=2e-8)

    rotation = [init_t[row * 4 : row * 4 + 3] for row in range(3)]
    for row in rotation:
        assert all(math.isfinite(value) for value in row)
    gram = [[sum(rotation[k][row] * rotation[k][col] for k in range(3)) for col in range(3)] for row in range(3)]
    for row in range(3):
        for col in range(3):
            assert gram[row][col] == pytest.approx(1.0 if row == col else 0.0, abs=2e-6)
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    assert determinant == pytest.approx(1.0, abs=2e-6)
    assert init_t[12:] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=2e-8)

    # Both config copies must also match the robot's actual physical
    # description (anubis_description/urdf/sensors.xacro), not just each
    # other -- see this file's module docstring for why this replaced
    # the old static_transform_publisher check.
    xacro_xyz = _livox_joint_xyz_from_xacro(sensors_xacro)
    _assert_close(xacro_xyz, init_t[3::4][:3], tolerance=2e-6)
