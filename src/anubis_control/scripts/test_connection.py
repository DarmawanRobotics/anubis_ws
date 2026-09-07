#!/usr/bin/env python3
"""Interactive SDK connectivity test for the ZSL-1W robot -- standalone,
no ROS graph required. Run this before bringing up control_node to
confirm the SDK itself can talk to the robot.

Usage:
    python3 test_connection.py --local-ip 192.168.234.234
    python3 test_connection.py --local-ip 192.168.234.234 --sdk-model zsl-1

Test sequence:
    1. Load the SDK module
    2. initRobot() + wait for connection (handshake is async, retries)
    3. Read state (battery, position, orientation, control mode)
    4. Stand up
    5. Slow move forward, then stop (asks for confirmation first)
    6. Lie down
"""

import argparse
import sys
import time

from anubis_control.sdk_loader import SdkLoadError, load_sdk

DOG_STATE_NAMES = {0: "Damping", 1: "Standing", 3: "Moving"}


def wait_for_connection(dog, timeout_s: float = 5.0, poll_interval_s: float = 0.5) -> bool:
    """Same async-handshake race condition as control_node._wait_for_connection."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if dog.checkConnect():
                return True
        except Exception:
            pass
        time.sleep(poll_interval_s)
    return False


def run(local_ip: str, local_port: int, dog_ip: str, sdk_model: str) -> bool:
    print("=" * 60)
    print(" ZSL-1W SDK connectivity test")
    print("=" * 60)

    try:
        handle = load_sdk(model=sdk_model)
    except SdkLoadError as exc:
        print(f"[FAIL] {exc}")
        return False
    print(f"[INFO] SDK loaded: model={handle.model} arch={handle.arch}")
    print(f"[INFO] lib_path={handle.lib_path}")
    print(f"[INFO] Python: {sys.version}")

    dog = handle.module.HighLevel()

    print("\n[1/6] initRobot()")
    print(f"      local_ip={local_ip} local_port={local_port} dog_ip={dog_ip}")
    dog.initRobot(local_ip, local_port, dog_ip)
    print("      [PASS] initRobot() returned")

    print("\n[2/6] Connection check (retrying up to 5s -- handshake is async)")
    if wait_for_connection(dog):
        print("      [PASS] Connected")
    else:
        print("      [FAIL] Could not connect. Check:")
        print("         1. Is the robot powered on?")
        print(f"        2. Is the network reachable? ping {dog_ip}")
        print("         3. Does the robot's /opt/export/config/sdk_config.yaml")
        print(f"           target_ip match this machine'`s IP ({local_ip})?")
        return False

    print("\n[3/6] Read state")
    try:
        battery = dog.getBatteryPower()
        pos = dog.getPosition()
        rpy = dog.getRPY()
        mode = dog.getCurrentCtrlmode()
        print(f"      Battery: {battery}%")
        if battery < 20:
            print("      [WARN] Battery below 20%, consider charging first")
        print(f"      Position: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")
        print(f"      Orientation (rad): roll={rpy[0]:.3f} pitch={rpy[1]:.3f} yaw={rpy[2]:.3f}")
        print(f"      Control mode: {mode} ({DOG_STATE_NAMES.get(mode, 'unknown')})")
        print("      [PASS]")
    except Exception as exc:
        print(f"      [FAIL] {exc}")
        return False

    print("\n[4/6] Stand up")
    if mode == 1:
        print("      Already standing")
    else:
        dog.standUp()
        time.sleep(3)
        new_mode = dog.getCurrentCtrlmode()
        if new_mode == 1:
            print("      [PASS] Standing")
        else:
            print(f"      [WARN] mode={new_mode} after standUp(), may need more time")

    print("\n[5/6] Move test (0.2 m/s forward, 1.5s)")
    input("      Clear the area around the robot, then press Enter...")
    dog.move(0.2, 0.0, 0.0)
    time.sleep(1.5)
    dog.move(0.0, 0.0, 0.0)
    time.sleep(0.5)
    pos2 = dog.getPosition()
    dx, dy = pos2[0] - pos[0], pos2[1] - pos[1]
    print(f"      Position change: dx={dx:.3f} dy={dy:.3f}")
    if abs(dx) > 0.05 or abs(dy) > 0.05:
        print("      [PASS] Robot moved")
    else:
        print("      [WARN] Position barely changed -- did it actually move?")

    print("\n[6/6] Lie down")
    input("      Press Enter to lie the robot down...")
    dog.lieDown()
    time.sleep(2)
    final_mode = dog.getCurrentCtrlmode()
    if final_mode == 0:
        print("      [PASS] Lying down (Damping)")
    else:
        print(f"      [WARN] mode={final_mode} after lieDown()")

    print("\n" + "=" * 60)
    print(" All tests complete")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ip", required=True, help="This machine's IP on the robot's network")
    parser.add_argument("--local-port", type=int, default=43988)
    parser.add_argument("--dog-ip", default="192.168.234.1")
    parser.add_argument("--sdk-model", choices=["zsl-1", "zsl-1w"], default="zsl-1w")
    args = parser.parse_args()

    ok = run(args.local_ip, args.local_port, args.dog_ip, args.sdk_model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
