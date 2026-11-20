"""Isolate K3 finger-pad contact from arm tracking and vision alignment."""

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from k3_gripper import (
    DRIVE_DAMPING, DRIVE_FORCE_N, DRIVE_STIFFNESS, MAX_OPENING_M, PAD_FRICTION,
    PAD_LENGTH_M, PAD_MASS_KG, PAD_THICKNESS_M, PAD_WIDTH_M, opening_m,
)
from replay_k3_grasp import bottom_height


def transform(x, y, z):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def cuboid_inertia(mass, x, y, z):
    xx = mass * (y*y + z*z) / 12
    yy = mass * (x*x + z*z) / 12
    zz = mass * (x*x + y*y) / 12
    return wp.mat33(xx, 0, 0, 0, yy, 0, 0, 0, zz)


def run(command, depth_offset):
    width, depth, height = 0.0488, 0.150, 0.165
    box_mass = 0.06
    dt = 1 / 180
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane()
    palm = builder.add_link(mass=0.57, inertia=cuboid_inertia(0.57, 0.066, 0.09, 0.045),
                            xform=transform(0, 0, height / 2))
    lift = builder.add_joint_prismatic(
        -1, palm, axis=(0, 0, 1), parent_xform=transform(0, 0, height / 2),
        target_pos=0, limit_lower=0, limit_upper=0.20, target_ke=5000, target_kd=80,
    )
    joints = [lift]
    for direction in (-1, 1):
        center_x = direction * (MAX_OPENING_M + PAD_THICKNESS_M) / 2
        finger = builder.add_link(
            mass=PAD_MASS_KG,
            inertia=cuboid_inertia(PAD_MASS_KG, PAD_THICKNESS_M, PAD_LENGTH_M, PAD_WIDTH_M),
            xform=transform(center_x, depth_offset, height / 2),
        )
        joint = builder.add_joint_prismatic(
            palm, finger, axis=(direction, 0, 0),
            parent_xform=transform(direction * PAD_THICKNESS_M / 2, depth_offset, 0),
            target_pos=MAX_OPENING_M / 2, limit_lower=0, limit_upper=MAX_OPENING_M / 2,
            target_ke=DRIVE_STIFFNESS, target_kd=DRIVE_DAMPING, effort_limit=DRIVE_FORCE_N,
        )
        joints.append(joint)
        builder.joint_q[builder.joint_q_start[joint]] = MAX_OPENING_M / 2
        builder.add_shape_box(
            finger, hx=PAD_THICKNESS_M / 2, hy=PAD_LENGTH_M / 2, hz=PAD_WIDTH_M / 2,
            cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=PAD_FRICTION),
        )
    builder.add_articulation(joints)
    box = builder.add_body(xform=transform(0, 0, height / 2 + 0.001))
    builder.add_shape_box(
        box, hx=width / 2, hy=depth / 2, hz=height / 2,
        cfg=newton.ModelBuilder.ShapeConfig(density=box_mass / (width * depth * height)),
    )
    model = builder.finalize("cuda:0")
    solver = newton.solvers.SolverMuJoCo(model, integrator="implicitfast", njmax=80, nconmax=64)
    control = model.control()
    state, following = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    height_trace = []
    half_dims = np.asarray([width, depth, height]) / 2
    for tick in range(4 * 180):
        time = tick * dt
        lift_target = 0.15 * np.clip(time - 1.25, 0.0, 1.0)
        fraction = np.clip(2 * (time - 0.35), 0.0, 1.0)
        jaw = (MAX_OPENING_M + fraction * (opening_m(command) - MAX_OPENING_M)) / 2
        control.joint_target_q.assign(np.array([lift_target, jaw, jaw], dtype=np.float32))
        state.clear_forces()
        solver.step(state, following, control, None, dt)
        state, following = following, state
        if tick % 6 == 0:
            height_trace.append(bottom_height(state.body_q.numpy()[box], half_dims))
    held = all(clearance >= 0.10 for clearance in height_trace[-30:])
    return {"command": command, "depth_offset_m": depth_offset, "held_last_second": held,
            "max_bottom_clearance_m": max(height_trace)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = [run(0.50, 0.0), run(0.25, 0.0), run(0.25, -0.060)]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if results[0]["held_last_second"] or not results[1]["held_last_second"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
