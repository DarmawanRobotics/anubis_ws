"""Pure math helpers for anubis_control.

Deliberately free of ROS and SDK imports so these can be unit-tested
without a running ROS graph or connected hardware -- see test/test_math_utils.py.
"""

import math
from typing import Optional, Tuple


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert ZYX roll/pitch/yaw (radians) to a normalized ROS quaternion (x, y, z, w)."""
    values = (float(roll), float(pitch), float(yaw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("RPY values must be finite")

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("RPY conversion produced a degenerate quaternion")
    return qx / norm, qy / norm, qz / norm, qw / norm


def world_velocity_to_body(vx_world: float, vy_world: float, yaw: float) -> Tuple[float, float]:
    """Rotate a planar world-frame velocity into the body (base_link) frame."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cy * vx_world + sy * vy_world,
        -sy * vx_world + cy * vy_world,
    )


def wrapped_yaw_rate(previous_yaw: float, current_yaw: float, dt: float) -> float:
    """Estimate yaw rate from two yaw samples, handling the -pi/pi wrap boundary."""
    if not all(math.isfinite(v) for v in (previous_yaw, current_yaw, dt)) or dt <= 0.0:
        return 0.0
    delta = math.atan2(
        math.sin(current_yaw - previous_yaw),
        math.cos(current_yaw - previous_yaw),
    )
    return delta / dt


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


def apply_deadzone(value: float, deadzone: float) -> float:
    """Snap small values to zero."""
    return 0.0 if abs(value) < deadzone else value


def localization_allows_motion(require_valid: bool, localization_valid: Optional[bool]) -> bool:
    """Fail closed: motion is blocked until an explicit `True` is received.

    A `None` (never received a status yet) or `False` both block motion
    when `require_valid` is True. Only an explicit `True` allows it.
    """
    return (not require_valid) or (localization_valid is True)


def normalize_battery_percentage(raw_value) -> Optional[float]:
    """Convert the SDK's 0-100 battery reading to BatteryState's 0.0-1.0 range.

    Returns None for anything out of range or non-numeric, so callers can
    skip publishing a bogus reading instead of clamping it silently.
    """
    try:
        percentage = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(percentage) or not (0.0 <= percentage <= 100.0):
        return None
    return percentage / 100.0


def format_vector(value: Optional[Tuple[float, ...]], precision: int = 4) -> str:
    """Format an optional vector for log messages; 'unavailable' if None."""
    if value is None:
        return "unavailable"
    return "[" + ",".join(f"{v:.{precision}f}" for v in value) + "]"
