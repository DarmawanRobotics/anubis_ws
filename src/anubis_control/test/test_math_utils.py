"""Unit tests for anubis_control.math_utils.

Pure-function tests -- no ROS graph, no hardware, no mocking needed.
Run with: colcon test --packages-select anubis_control
Or directly: python3 -m pytest test/test_math_utils.py -v
"""

import math

import pytest

from anubis_control.math_utils import (
    apply_deadzone,
    clamp,
    format_vector,
    localization_allows_motion,
    normalize_battery_percentage,
    rpy_to_quaternion,
    world_velocity_to_body,
    wrapped_yaw_rate,
)


class TestRpyToQuaternion:
    def test_identity(self):
        qx, qy, qz, qw = rpy_to_quaternion(0.0, 0.0, 0.0)
        assert qx == pytest.approx(0.0, abs=1e-9)
        assert qy == pytest.approx(0.0, abs=1e-9)
        assert qz == pytest.approx(0.0, abs=1e-9)
        assert qw == pytest.approx(1.0, abs=1e-9)

    def test_normalized(self):
        q = rpy_to_quaternion(0.3, -0.7, 1.9)
        norm = math.sqrt(sum(c * c for c in q))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_yaw_90deg_rotates_x_to_y(self):
        # A pure +90deg yaw quaternion, applied to the +X axis, should
        # land on +Y. This is a physical sanity check, not just "some
        # numbers came out".
        qx, qy, qz, qw = rpy_to_quaternion(0.0, 0.0, math.pi / 2)
        # Rotate vector (1,0,0) by the quaternion.
        vx, vy, vz = 1.0, 0.0, 0.0
        # v' = q * v * q^-1, computed directly for a unit quaternion.
        ux, uy, uz = qx, qy, qz
        s = qw
        dot_uv = ux * vx + uy * vy + uz * vz
        cross_x = uy * vz - uz * vy
        cross_y = uz * vx - ux * vz
        cross_z = ux * vy - uy * vx
        rx = 2 * dot_uv * ux + (s * s - (ux * ux + uy * uy + uz * uz)) * vx + 2 * s * cross_x
        ry = 2 * dot_uv * uy + (s * s - (ux * ux + uy * uy + uz * uz)) * vy + 2 * s * cross_y
        assert rx == pytest.approx(0.0, abs=1e-9)
        assert ry == pytest.approx(1.0, abs=1e-9)

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            rpy_to_quaternion(float("nan"), 0.0, 0.0)
        with pytest.raises(ValueError):
            rpy_to_quaternion(0.0, float("inf"), 0.0)


class TestWorldVelocityToBody:
    def test_zero_yaw_is_identity(self):
        vx, vy = world_velocity_to_body(1.0, 2.0, 0.0)
        assert vx == pytest.approx(1.0)
        assert vy == pytest.approx(2.0)

    def test_90deg_yaw(self):
        # Facing +90deg (yaw=pi/2): world +X should appear as body -Y.
        vx, vy = world_velocity_to_body(1.0, 0.0, math.pi / 2)
        assert vx == pytest.approx(0.0, abs=1e-9)
        assert vy == pytest.approx(-1.0, abs=1e-9)


class TestWrappedYawRate:
    def test_simple_rate(self):
        rate = wrapped_yaw_rate(0.0, 1.0, 1.0)
        assert rate == pytest.approx(1.0)

    def test_wraps_across_pi_boundary(self):
        # From +3.0 rad to -3.0 rad should be a SMALL rotation the short
        # way around (through pi), not a large jump straight through zero.
        rate = wrapped_yaw_rate(3.0, -3.0, 1.0)
        assert abs(rate) < 1.0  # short way around, not ~6 rad/s

    def test_zero_dt_returns_zero(self):
        assert wrapped_yaw_rate(0.0, 1.0, 0.0) == 0.0

    def test_non_finite_returns_zero(self):
        assert wrapped_yaw_rate(float("nan"), 1.0, 1.0) == 0.0


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, -1.0, 1.0) == 0.5

    def test_clamps_high(self):
        assert clamp(5.0, -1.0, 1.0) == 1.0

    def test_clamps_low(self):
        assert clamp(-5.0, -1.0, 1.0) == -1.0


class TestApplyDeadzone:
    def test_below_deadzone_snaps_to_zero(self):
        assert apply_deadzone(0.01, 0.05) == 0.0

    def test_above_deadzone_passes_through(self):
        assert apply_deadzone(0.2, 0.05) == 0.2

    def test_negative_below_deadzone(self):
        assert apply_deadzone(-0.01, 0.05) == 0.0


class TestLocalizationAllowsMotion:
    def test_not_required_always_allows(self):
        assert localization_allows_motion(False, None) is True
        assert localization_allows_motion(False, False) is True

    def test_required_and_none_blocks(self):
        # Never received a status yet -- must fail closed, not open.
        assert localization_allows_motion(True, None) is False

    def test_required_and_false_blocks(self):
        assert localization_allows_motion(True, False) is False

    def test_required_and_true_allows(self):
        assert localization_allows_motion(True, True) is True


class TestNormalizeBatteryPercentage:
    def test_valid_value(self):
        assert normalize_battery_percentage(75) == pytest.approx(0.75)

    def test_zero(self):
        assert normalize_battery_percentage(0) == pytest.approx(0.0)

    def test_hundred(self):
        assert normalize_battery_percentage(100) == pytest.approx(1.0)

    def test_out_of_range_rejected(self):
        assert normalize_battery_percentage(150) is None
        assert normalize_battery_percentage(-5) is None

    def test_non_numeric_rejected(self):
        assert normalize_battery_percentage("not a number") is None
        assert normalize_battery_percentage(None) is None


class TestFormatVector:
    def test_none_is_unavailable(self):
        assert format_vector(None) == "unavailable"

    def test_formats_values(self):
        result = format_vector((1.0, 2.5, -3.25), precision=2)
        assert result == "[1.00,2.50,-3.25]"
