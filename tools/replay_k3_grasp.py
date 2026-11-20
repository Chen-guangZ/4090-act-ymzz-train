"""Replay measured K3 arm motion against dynamic grasp contacts in Newton MJWarp."""

import argparse
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path

import newton
import numpy as np
from k3_gripper import AXIAL_TRAVEL_M, MAX_OPENING_M, axial_retraction_m, opening_m


DT = 1 / 180


def read_tracks(path, joints, start_ns, end_ns):
    tracks = {name: ([], []) for name in joints}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = int(row["bag_time_ns"])
            name = row["name"]
            if start_ns <= timestamp <= end_ns and name in tracks and row["driver_enabled_state"] != "-1.0":
                t, q = tracks[name]
                t.append((timestamp - start_ns) / 1e9)
                q.append(float(row["position_rad"]))
    return tracks


def event_time(event):
    return datetime.fromisoformat(event["time"]).timestamp()


def bottom_height(xyzquat, half_dims):
    x, y, z, w = xyzquat[3:]
    body_vertical = np.array([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)])
    return float(xyzquat[2] - np.abs(body_vertical) @ half_dims)


def longest_lift(values, threshold, time_per_sample):
    longest = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        longest = max(longest, current)
    return longest * time_per_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cone", choices=("elliptic", "pyramidal"), default="pyramidal")
    args = parser.parse_args()
    trial = json.loads((args.trial_dir / "replay.json").read_text())
    scene = json.loads(args.scene.with_suffix(".json").read_text())
    begin = next(c for c in trial["gripper_commands"] if c["ratio"] < 1)
    return_motion = next(e for e in trial["events"]
                         if e["event"] == "motion_start" and e["stage"] == "return_above_pickup")
    duration = event_time(return_motion) - begin["bag_time_ns"] / 1e9
    start_ns = begin["bag_time_ns"] - int(0.75e9)
    steps = int((duration + 0.75) / DT)
    clock = np.arange(steps) * DT

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(args.scene.resolve()), load_visual_shapes=False, enable_self_collisions=False)
    slots = {name.rsplit("/", 1)[-1]: joint for joint, name in enumerate(builder.joint_label)}
    positions = trial["snapshot_before_close_rad"]
    tracks = read_tracks(args.trial_dir / "joint_states.csv.gz", positions,
                         start_ns, begin["bag_time_ns"] + int((duration + 1) * 1e9))
    controls = np.zeros((steps, len(builder.joint_target_q)), dtype=np.float32)
    for name in positions:
        joint = slots[name]
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        t, values = tracks[name]
        controls[:, dof] = np.interp(clock, t, values)
        builder.joint_q[q] = controls[0, dof]
        if name.startswith("Body"):
            stiffness, damping, armature, effort = 6000, 300, 0.4, 250
        elif name.startswith("Neck"):
            stiffness, damping, armature, effort = 40, 4, 0.02, 100
        else:
            stiffness, damping, armature, effort = 3000, 140, 0.20, 100
        builder.joint_target_ke[dof] = stiffness
        builder.joint_target_kd[dof] = damping
        builder.joint_armature[dof] = armature
        builder.joint_effort_limit[dof] = effort
    fingers = []
    for name in ("LeftFinger", "RightFinger"):
        joint = slots[name + "Slide"]
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        depth_joint = slots[name + "Depth"]
        depth_q = builder.joint_q_start[depth_joint]
        depth_target = builder.joint_qd_start[depth_joint]
        fingers.append((q, depth_target))
        builder.joint_q[q] = MAX_OPENING_M / 2
        builder.joint_q[depth_q] = AXIAL_TRAVEL_M
        # Match the driver command, including its delay before the start frame.
        fraction = np.clip((clock - 0.85) / 0.55, 0.0, 1.0)
        controls[:, dof] = (MAX_OPENING_M + fraction * (opening_m(begin["ratio"]) - MAX_OPENING_M)) / 2
        controls[:, depth_target] = AXIAL_TRAVEL_M
    builder.joint_target_q[:] = controls[0].tolist()
    box = builder.body_label.index("/World/Box")
    right = next(i for i, label in enumerate(builder.body_label) if label.endswith("/Right_Arm_Link8"))
    model = builder.finalize(args.device)
    solver = newton.solvers.SolverMuJoCo(model, njmax=120, nconmax=120, integrator="implicitfast",
                                       cone=args.cone,
                                       use_mujoco_cpu=args.device == "cpu")
    state, following = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    samples = []
    half_dims = np.asarray(scene["box_dims_m_perception_estimate"]) / 2
    q = state.joint_q.numpy()
    for i, target in enumerate(controls):
        for width_q, depth_target in fingers:
            target[depth_target] = axial_retraction_m(q[width_q])
        control.joint_target_q.assign(target)
        state.clear_forces()
        solver.step(state, following, control, None, DT)
        state, following = following, state
        q = state.joint_q.numpy()
        if i % 6 == 0:
            pose = state.body_q.numpy()
            clearance = bottom_height(pose[box], half_dims) - scene["table_top_world_m"]
            samples.append([float(clock[i]), *pose[box, :3].tolist(), *pose[right, :3].tolist(),
                            float(sum(q[width_q] for width_q, _ in fingers)), clearance])
    result = np.array(samples)
    rest_z = scene["table_top_world_m"] + half_dims[2]
    raised = result[:, 3] - rest_z
    clearance = result[:, 8]
    sustained = longest_lift(clearance, 0.10, DT * 6)
    report = {
        "trial": trial["trial"], "close_command": begin["ratio"], "friction_cone": args.cone,
        "commanded_gap_m": opening_m(begin["ratio"]), "finite": bool(np.isfinite(result).all()),
        "peak_center_lift_m": float(raised.max()), "peak_bottom_clearance_m": float(clearance.max()),
        "continuous_10cm_bottom_clearance_s": sustained,
        "passed": bool(np.isfinite(result).all() and sustained >= 1.0),
        "sample_columns": ["elapsed_s", "box_x", "box_y", "box_z", "wrist_x", "wrist_y", "wrist_z",
                           "gap_m", "bottom_clearance_m"],
        "samples": samples,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "sample_columns")}, indent=2))


if __name__ == "__main__":
    main()
