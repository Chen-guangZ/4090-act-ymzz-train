"""Resettable K3 paper-box task using the calibrated Isaac Lab MJWarp scene."""

from contextlib import ExitStack
import json

import numpy as np

from k3_gripper import AXIAL_TRAVEL_M, MAX_OPENING_M, axial_retraction_m, opening_m
from k3_isaaclab_profile import make_sim_cfg, open_scene
from replay_k3_grasp import ArmBiasCompensation, BoxContacts, bottom_height


class K3BoxTask:
    """Step with seven absolute joint targets in rad and one jaw ratio in [0, 1]."""

    def __init__(self, scene, config, *, device="cuda:0", timeout_s=30):
        self.scene = scene
        self.config = config
        self.device = device
        self.physics_hz = config["physics_hz"]
        self.action_hz = config["action_hz"]
        if self.action_hz <= 0 or self.physics_hz <= 0 or self.physics_hz % self.action_hz or timeout_s <= 0:
            raise ValueError("Use an integral physics/action frequency ratio and a positive timeout.")
        self.substeps = self.physics_hz // self.action_hz
        self.arm_names = config["arm_joint_names"]
        self.observation_pose = config["observation_joint_rad"]
        self.start_arm = np.array([self.observation_pose[name] for name in self.arm_names])
        self.timeout_steps = round(timeout_s * self.action_hz)
        params = json.loads(scene.with_suffix(".json").read_text(encoding="utf-8"))
        self.table_z = params["table_top_world_m"]
        self.half_box = np.asarray(params.get("box_dimensions_m", params["box_dims_m_perception_estimate"])) / 2
        self.stack = ExitStack()
        self.termination = "needs_reset"

    def close(self):
        self.stack.close()
        self.termination = "needs_reset"

    def reset(self, *, seed=0, randomize=False):
        """Rebuild the physics context, clearing state, controls and solver history."""
        self.close()
        try:
            return self._reset(seed, randomize)
        except BaseException:
            self.close()
            raise

    def _reset(self, seed, randomize):
        cfg = make_sim_cfg(self.device, self.config["contact_impratio"], self.physics_hz)
        self.sim, builder = open_scene(self.stack, self.scene, cfg)
        joint = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(builder.joint_label)}
        def q_index(name):
            return builder.joint_q_start[joint[name]]

        def dof_index(name):
            return builder.joint_qd_start[joint[name]]
        self.arm_q = [q_index(name) for name in self.arm_names]
        self.arm_dofs = [dof_index(name) for name in self.arm_names]
        self.fixed_names = [name for name in self.observation_pose if name not in self.arm_names]
        self.fixed_q = [q_index(name) for name in self.fixed_names]
        dofs = []
        for name, value in self.observation_pose.items():
            dof = dof_index(name)
            dofs.append(dof)
            builder.joint_q[q_index(name)] = value
            builder.joint_target_q[dof] = value
            if name.startswith("Body") or name in self.fixed_names:
                ke, kd, armature, effort = 6000, 300, 0.4, 250
            else:
                ke, kd, armature, effort = 3000, 140, 0.20, 100
            builder.joint_target_ke[dof] = ke
            builder.joint_target_kd[dof] = kd
            builder.joint_armature[dof] = armature
            builder.joint_effort_limit[dof] = effort
        self.lower = np.asarray(builder.joint_limit_lower)[self.arm_dofs]
        self.upper = np.asarray(builder.joint_limit_upper)[self.arm_dofs]
        self.jaws = []
        for name in BoxContacts.names:
            slide_q, slide = q_index(name + "Slide"), dof_index(name + "Slide")
            depth_q, depth = q_index(name + "Depth"), dof_index(name + "Depth")
            builder.joint_q[slide_q] = builder.joint_target_q[slide] = MAX_OPENING_M / 2
            builder.joint_q[depth_q] = builder.joint_target_q[depth] = AXIAL_TRAVEL_M
            self.jaws.append((slide_q, slide, depth))

        self.box = builder.body_label.index("/World/Box")
        box_q = builder.joint_q_start[builder.joint_child.index(self.box)]
        offset = np.zeros(3)
        if randomize:
            rng = np.random.default_rng(seed)
            ranges = [*self.config["box_xy_half_range_m"], self.config["box_yaw_half_range_deg"]]
            offset = rng.uniform(-np.asarray(ranges), ranges)
            for i in range(2):
                builder.joint_q[box_q + i] += offset[i]
            yaw = 2 * np.arctan2(builder.joint_q[box_q + 5], builder.joint_q[box_q + 6]) + np.deg2rad(offset[2])
            builder.joint_q[box_q + 5:box_q + 7] = [float(np.sin(yaw / 2)), float(np.cos(yaw / 2))]

        self.labels = list(builder.body_label)
        self.base = next(i for i, label in enumerate(self.labels) if label.endswith("/base_link"))
        self.target = np.asarray(builder.joint_target_q, dtype=np.float32)
        self.sim.reset()
        self.manager = self.sim.physics_manager
        self.control = self.manager.get_control()
        self.contacts = BoxContacts(builder, self.manager._solver, self.device)
        self.bias = ArmBiasCompensation(self.manager._solver, dofs, builder.joint_effort_limit)
        self.physics_steps = self.steps = self.held_steps = 0
        self.ever_contacted = False
        self.termination = None
        self.jaw_ratio = 1.0
        self.jaw_gap = self.jaw_from = self.jaw_to = MAX_OPENING_M
        self.jaw_started = 0
        self.reset_info = {"seed": seed, "box_offset_xy_m_yaw_deg": offset.tolist()}
        return self._sample()

    def step(self, action):
        if self.termination is not None:
            raise RuntimeError("Reset the task before stepping a finished episode.")
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all() or not 0 <= action[-1] <= 1:
            raise ValueError("Expected seven finite joint angles and a jaw ratio in [0, 1].")
        stride = self.config["arm_target_rate_rad_s"] / self.action_hz
        previous = self.target[self.arm_dofs]
        self.target[self.arm_dofs] = previous + np.clip(np.clip(action[:7], self.lower, self.upper) - previous,
                                                       -stride, stride)
        if action[-1] != self.jaw_ratio:
            self.jaw_ratio = float(action[-1])
            self.jaw_from, self.jaw_to = self.jaw_gap, opening_m(self.jaw_ratio)
            self.jaw_started = self.physics_steps

        for _ in range(self.substeps):
            state = self.manager.get_state_0()
            q = state.joint_q.numpy()
            elapsed = (self.physics_steps - self.jaw_started) / self.physics_hz
            fraction = np.clip((elapsed - 0.10) / 0.55, 0, 1)
            self.jaw_gap = self.jaw_from + (self.jaw_to - self.jaw_from) * fraction
            for slide_q, slide, depth in self.jaws:
                self.target[slide] = self.jaw_gap / 2
                self.target[depth] = axial_retraction_m(q[slide_q])
            self.control.joint_target_q.assign(self.target)
            if self.physics_steps:
                self.bias.apply(self.control)
            state.clear_forces()
            self.sim.step(render=False)
            self.physics_steps += 1

        self.steps += 1
        observation, info = self._sample()
        left, right = info["pad_load_n"]
        both = min(left, right) >= 0.1
        self.ever_contacted |= both
        self.held_steps = (self.held_steps + 1) if both and info["box_bottom_clearance_m"] >= 0.10 else 0
        if self.held_steps >= self.action_hz:
            self.termination = "lifted_and_held"
        elif info["box_bottom_clearance_m"] < -0.035:
            self.termination = "box_fell"
        elif self.steps >= self.timeout_steps:
            self.termination = "timeout" if self.ever_contacted else "grasp_empty"
        info["requested_action"] = action.tolist()
        info["applied_arm_target_rad"] = self.target[self.arm_dofs].tolist()
        info["applied_jaw_gap_target_m"] = self.jaw_gap
        info["termination"] = self.termination
        return observation, info

    def _sample(self):
        state = self.manager.get_state_0()
        poses, q = state.body_q.numpy(), state.joint_q.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(q).all():
            raise RuntimeError("Nonfinite physics state.")
        info = {
            "elapsed_s": self.physics_steps / self.physics_hz,
            "box_pose_xyzw": poses[self.box].tolist(),
            "base_pose_xyzw": poses[self.base].tolist(),
            "fixed_joint_rad": dict(zip(self.fixed_names, q[self.fixed_q].tolist(), strict=True)),
            "jaw_gap_m": sum(float(q[jaw[0]]) for jaw in self.jaws),
            "box_bottom_clearance_m": bottom_height(poses[self.box], self.half_box) - self.table_z,
            "pad_load_n": self.contacts.loads(state)[:2] if self.physics_steps else [0.0, 0.0],
        }
        return [*q[self.arm_q].tolist(), self.jaw_ratio], info

    def body_poses(self):
        return self.manager.get_state_0().body_q.numpy()
