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


def jaw_opening_track(clock, start_ns, commands):
    opening = np.full_like(clock, MAX_OPENING_M)
    previous = MAX_OPENING_M
    for command in commands:
        target = opening_m(command["ratio"])
        if target == previous:
            continue
        when = (command["bag_time_ns"] - start_ns) * 1e-9 + 0.10
        opening += (target - previous) * np.clip((clock - when) / 0.55, 0.0, 1.0)
        previous = target
    return opening


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


class BoxContacts:
    """Sample the solver's pad-to-box contact normal loads."""

    names = ("LeftFinger", "RightFinger")

    def __init__(self, builder, solver, device):
        self.solver = solver
        self.box = builder.shape_label.index("/World/Box/Collider")
        self.pads = [builder.shape_label.index(f"/World/Robot/GripperParts/{name}/Collider")
                     for name in self.names]
        self.buffer = newton.Contacts(solver.mjw_data.naconmax, 0, device=device, requested_attributes={"force"})

    def loads(self, state):
        buffer = self.buffer
        buffer.clear()
        self.solver.update_contacts(buffer, state)
        count = int(buffer.rigid_contact_count.numpy()[0])
        if count == 0:
            return [0.0, 0.0]
        a, b = buffer.rigid_contact_shape0.numpy()[:count], buffer.rigid_contact_shape1.numpy()[:count]
        normals = buffer.rigid_contact_normal.numpy()[:count]
        forces = buffer.force.numpy()[:count, :3]
        normal_loads = np.abs(np.einsum("ij,ij->i", normals, forces))
        return [float(normal_loads[((a == self.box) & (b == pad)) | ((a == pad) & (b == self.box))].sum())
                for pad in self.pads]


class ArmBiasCompensation:
    """Remove simulated gravity and Coriolis bias from measured-pose replay."""

    def __init__(self, solver, arm_dofs, effort_limits):
        self.solver = solver
        mapping = solver.mjc_dof_to_newton_dof.numpy()[0]
        self.mj_dofs = np.flatnonzero(np.isin(mapping, arm_dofs))
        self.newton_dofs = mapping[self.mj_dofs]
        self.effort = np.asarray(effort_limits)[self.newton_dofs]
        self.torques = np.zeros(solver.model.joint_dof_count, dtype=np.float32)

    def apply(self, control):
        bias = self.solver.mjw_data.qfrc_bias.numpy()[0, self.mj_dofs]
        self.torques[self.newton_dofs] = np.clip(bias, -self.effort, self.effort)
        control.joint_f.assign(self.torques)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cone", choices=("elliptic", "pyramidal"), default="pyramidal")
    parser.add_argument("--sample-contacts", action="store_true", help="Log the two pad-to-box normal loads.")
    parser.add_argument("--arm-bias-compensation", action="store_true",
                        help="Compensate simulated arm dynamics for measured-pose replay.")
    parser.add_argument("--through-release", action="store_true",
                        help="Continue through the recorded place, release and retreat.")
    args = parser.parse_args()
    if args.device == "cpu" and (args.sample_contacts or args.arm_bias_compensation):
        parser.error("Contact sampling and bias compensation are available on the MJWarp GPU path.")
    trial = json.loads((args.trial_dir / "replay.json").read_text())
    scene = json.loads(args.scene.with_suffix(".json").read_text())
    begin = next(c for c in trial["gripper_commands"] if c["ratio"] < 1)
    finish_event, finish_stage = ("motion_result", "retreat_from_box") if args.through_release else (
        "motion_start", "return_above_pickup")
    finish = next(e for e in trial["events"] if e["event"] == finish_event and e["stage"] == finish_stage)
    duration = event_time(finish) - begin["bag_time_ns"] / 1e9
    start_ns = begin["bag_time_ns"] - int(0.75e9)
    steps = int((duration + 0.75) / DT)
    clock = np.arange(steps) * DT
    opening = jaw_opening_track(clock, start_ns, trial["gripper_commands"])

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(args.scene.resolve()), load_visual_shapes=False, enable_self_collisions=False)
    slots = {name.rsplit("/", 1)[-1]: joint for joint, name in enumerate(builder.joint_label)}
    positions = trial["snapshot_before_close_rad"]
    tracks = read_tracks(args.trial_dir / "joint_states.csv.gz", positions,
                         start_ns, begin["bag_time_ns"] + int((duration + 1) * 1e9))
    controls = np.zeros((steps, len(builder.joint_target_q)), dtype=np.float32)
    arm_dofs = []
    for name in positions:
        joint = slots[name]
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        arm_dofs.append(dof)
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
        controls[:, dof] = opening / 2
        controls[:, depth_target] = AXIAL_TRAVEL_M
    builder.joint_target_q[:] = controls[0].tolist()
    box = builder.body_label.index("/World/Box")
    right = next(i for i, label in enumerate(builder.body_label) if label.endswith("/Right_Arm_Link8"))
    model = builder.finalize(args.device)
    solver = newton.solvers.SolverMuJoCo(model, njmax=120, nconmax=120, integrator="implicitfast",
                                       cone=args.cone,
                                       use_mujoco_cpu=args.device == "cpu")
    contacts = BoxContacts(builder, solver, args.device) if args.sample_contacts else None
    compensation = (ArmBiasCompensation(solver, arm_dofs, builder.joint_effort_limit)
                    if args.arm_bias_compensation else None)
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
        if compensation and i:
            compensation.apply(control)
        state.clear_forces()
        solver.step(state, following, control, None, DT)
        state, following = following, state
        q = state.joint_q.numpy()
        if i % 6 == 0:
            pose = state.body_q.numpy()
            clearance = bottom_height(pose[box], half_dims) - scene["table_top_world_m"]
            samples.append([float(clock[i]), *pose[box, :3].tolist(), *pose[right, :3].tolist(),
                            float(sum(q[width_q] for width_q, _ in fingers)), clearance])
            if contacts:
                samples[-1].extend(contacts.loads(state))
    result = np.array(samples)
    rest_z = scene["table_top_world_m"] + half_dims[2]
    raised = result[:, 3] - rest_z
    clearance = result[:, 8]
    sustained = longest_lift(clearance, 0.10, DT * 6)
    report = {
        "trial": trial["trial"], "close_command": begin["ratio"], "friction_cone": args.cone,
        "arm_bias_compensation": args.arm_bias_compensation,
        "replayed_through_release": args.through_release,
        "commanded_gap_m": opening_m(begin["ratio"]), "finite": bool(np.isfinite(result).all()),
        "peak_center_lift_m": float(raised.max()), "peak_bottom_clearance_m": float(clearance.max()),
        "continuous_10cm_bottom_clearance_s": sustained,
        "passed": bool(np.isfinite(result).all() and sustained >= 1.0),
        "sample_columns": ["elapsed_s", "box_x", "box_y", "box_z", "wrist_x", "wrist_y", "wrist_z",
                           "gap_m", "bottom_clearance_m"],
        "samples": samples,
    }
    if contacts:
        report["sample_columns"] += ["left_pad_normal_load_n", "right_pad_normal_load_n"]
        report["peak_pad_normal_load_n"] = result[:, 9:11].max(axis=0).tolist()
        both = np.min(result[:, 9:11], axis=1)
        report["longest_bilateral_contact_s"] = longest_lift(both, 0.1, DT * 6)
        report["continuous_bilateral_lift_s"] = longest_lift(
            (both >= 0.1) & (clearance >= 0.10), True, DT * 6,
        )
    if args.through_release:
        tail = result[result[:, 0] >= result[-1, 0] - 1.0]
        offset = tail[-1, 1:3] - np.asarray(scene["box_center_world_m"][:2])
        rest = np.ptp(tail[:, 1:4], axis=0).max()
        report["placement_after_retreat"] = {
            "box_bottom_clearance_m": float(tail[-1, 8]),
            "box_xy_offset_from_pickup_m": float(np.linalg.norm(offset)),
            "box_max_motion_last_second_m": float(rest),
            "gripper_opening_m": float(tail[-1, 7]),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "sample_columns")}, indent=2))


if __name__ == "__main__":
    main()
