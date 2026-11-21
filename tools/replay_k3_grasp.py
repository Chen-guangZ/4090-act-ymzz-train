"""Replay measured K3 arm motion against dynamic grasp contacts in Newton MJWarp."""

import argparse
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp
from k3_gripper import AXIAL_TRAVEL_M, MAX_OPENING_M, axial_retraction_m, opening_m


DT = 1 / 180
SAMPLE_HZ = 30


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
    """Sample the solver's normal loads on the box by contact counterpart."""

    names = ("LeftFinger", "RightFinger")
    columns = ("left_pad_normal_load_n", "right_pad_normal_load_n",
               "table_box_normal_load_n", "floor_box_normal_load_n", "other_robot_box_normal_load_n")

    def __init__(self, builder, solver, device):
        self.solver = solver
        self.box = builder.shape_label.index("/World/Box/Collider")
        self.pads = [
            [i for i, label in enumerate(builder.shape_label)
             if (label.startswith(f"/World/Robot/GripperParts/{name}/")
                 and builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES)]
            for name in self.names
        ]
        if not all(self.pads):
            raise ValueError("Both finger pad colliders must be present to sample their normal loads.")
        pads = set(self.pads[0] + self.pads[1])
        robot = [i for i, label in enumerate(builder.shape_label)
                 if label.startswith("/World/Robot/") and i not in pads]
        self.groups = [*self.pads, [builder.shape_label.index("/World/TableTop/Collider")],
                       [builder.shape_label.index("/World/Ground/Collider")], robot]
        if solver.use_mujoco_cpu:
            self.geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
            self.wrench = np.zeros(6, dtype=np.float64)
        else:
            self.buffer = newton.Contacts(solver.mjw_data.naconmax, 0, device=device, requested_attributes={"force"})

    def _gpu_loads(self, state):
        buffer = self.buffer
        buffer.clear()
        self.solver.update_contacts(buffer, state)
        count = int(buffer.rigid_contact_count.numpy()[0])
        a, b = buffer.rigid_contact_shape0.numpy()[:count], buffer.rigid_contact_shape1.numpy()[:count]
        normals = buffer.rigid_contact_normal.numpy()[:count]
        forces = buffer.force.numpy()[:count, :3]
        normal_loads = np.abs(np.einsum("ij,ij->i", normals, forces))
        touches_box = (a == self.box) | (b == self.box)
        counterpart = np.where(a == self.box, b, a)[touches_box]
        loads = normal_loads[touches_box]
        return counterpart, loads

    def _cpu_loads(self):
        m, d = self.solver.mj_model, self.solver.mj_data
        counterpart, loads = [], []
        for index in range(d.ncon):
            a, b = self.geom_to_shape[d.contact[index].geom]
            if a != self.box and b != self.box:
                continue
            mujoco.mj_contactForce(m, d, index, self.wrench)
            counterpart.append(b if a == self.box else a)
            loads.append(abs(self.wrench[0]))
        return np.asarray(counterpart), np.asarray(loads)

    def loads(self, state):
        counterpart, loads = self._cpu_loads() if self.solver.use_mujoco_cpu else self._gpu_loads(state)
        return [float(loads[np.isin(counterpart, group)].sum()) for group in self.groups]


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
        if self.solver.use_mujoco_cpu:
            bias = self.solver.mj_data.qfrc_bias[self.mj_dofs]
        else:
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
    parser.add_argument("--sample-contacts", action="store_true",
                        help="Log normal loads on the box from both pads, the table, the floor and other robot geometry.")
    parser.add_argument("--arm-bias-compensation", action="store_true",
                        help="Compensate simulated arm dynamics for measured-pose replay.")
    parser.add_argument("--through-release", action="store_true",
                        help="Continue through the recorded place, release and retreat.")
    parser.add_argument("--physics-hz", type=int, default=round(1 / DT),
                        help="Physics update rate. Contact and pose output remain at 30 Hz.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use MJWarp's run-to-run deterministic kernel mode.")
    args = parser.parse_args()
    if args.device == "cpu" and args.deterministic:
        parser.error("The deterministic kernel option applies to the MJWarp GPU path.")
    if args.physics_hz < SAMPLE_HZ or args.physics_hz % SAMPLE_HZ:
        parser.error("The physics rate must be a positive multiple of 30 Hz.")
    dt = 1 / args.physics_hz
    sample_stride = args.physics_hz // SAMPLE_HZ
    sample_dt = sample_stride * dt
    trial = json.loads((args.trial_dir / "replay.json").read_text())
    scene = json.loads(args.scene.with_suffix(".json").read_text())
    begin = next(c for c in trial["gripper_commands"] if c["ratio"] < 1)
    finish_event, finish_stage = ("motion_result", "retreat_from_box") if args.through_release else (
        "motion_start", "return_above_pickup")
    finish = next(e for e in trial["events"] if e["event"] == finish_event and e["stage"] == finish_stage)
    duration = event_time(finish) - begin["bag_time_ns"] / 1e9
    start_ns = begin["bag_time_ns"] - int(0.75e9)
    steps = int((duration + 0.75) / dt)
    clock = np.arange(steps) * dt
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
    passive_linkage = scene.get("passive_linkage", False)
    for name in ("LeftFinger", "RightFinger"):
        joint = slots[name + "Slide"]
        q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        depth_joint = slots[name + "Depth"]
        depth_q = builder.joint_q_start[depth_joint]
        depth_target = builder.joint_qd_start[depth_joint]
        fingers.append((q, depth_q, depth_target))
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
                                       deterministic=wp.DeterministicMode.RUN_TO_RUN if args.deterministic else None,
                                       disable_sensors=args.deterministic,
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
        if not passive_linkage:
            for width_q, _, depth_target in fingers:
                target[depth_target] = axial_retraction_m(q[width_q])
        control.joint_target_q.assign(target)
        if compensation and i:
            compensation.apply(control)
        state.clear_forces()
        solver.step(state, following, control, None, dt)
        state, following = following, state
        q = state.joint_q.numpy()
        if i % sample_stride == 0:
            pose = state.body_q.numpy()
            clearance = bottom_height(pose[box], half_dims) - scene["table_top_world_m"]
            samples.append([float(clock[i]), *pose[box, :3].tolist(), *pose[right, :3].tolist(),
                            float(sum(q[width_q] for width_q, _, _ in fingers)), clearance])
            if contacts:
                samples[-1].extend(contacts.loads(state))
            samples[-1].extend(pose[box, 3:].tolist())
            samples[-1].extend(pose[right, 3:].tolist())
            samples[-1].extend(float(q[depth_q]) for _, depth_q, _ in fingers)
    result = np.array(samples)
    rest_z = scene["table_top_world_m"] + half_dims[2]
    raised = result[:, 3] - rest_z
    clearance = result[:, 8]
    sustained = longest_lift(clearance, 0.10, sample_dt)
    report = {
        "trial": trial["trial"], "close_command": begin["ratio"], "friction_cone": args.cone,
        "arm_bias_compensation": args.arm_bias_compensation,
        "physics_hz": args.physics_hz, "sample_hz": SAMPLE_HZ,
        "solver_device": args.device,
        "deterministic_kernels": args.deterministic,
        "passive_linkage": passive_linkage,
        "pad_tiles_per_axis": scene.get("pad_tiles_per_axis", 1),
        "box_com_offset_local_m": scene.get("box_com_offset_local_m", [0.0, 0.0, 0.0]),
        "jaw_horizontal_drive_limit_per_pad_n": scene.get("jaw_horizontal_drive_limit_per_pad_n"),
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
        report["sample_columns"] += list(contacts.columns)
        report["peak_pad_normal_load_n"] = result[:, 9:11].max(axis=0).tolist()
        both = np.min(result[:, 9:11], axis=1)
        report["longest_bilateral_contact_s"] = longest_lift(both, 0.1, sample_dt)
        report["continuous_bilateral_lift_s"] = longest_lift(
            (both >= 0.1) & (clearance >= 0.10), True, sample_dt,
        )
    report["sample_columns"] += [
        f"{body}_quat_{axis}" for body in ("box", "wrist") for axis in ("x", "y", "z", "w")
    ]
    report["sample_columns"] += ["left_pad_axial_travel_m", "right_pad_axial_travel_m"]
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
