"""Create an Isaac Lab MJWarp configuration and check it against a K3 scene."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg
import numpy as np


def make_sim_cfg(device="cuda:0", contact_impratio=800.0, physics_hz=360, *, cone="elliptic", deterministic=False):
    """Configure the solver used for the subdivided K3 pad experiments."""
    if not device.startswith("cuda"):
        raise ValueError("Isaac Lab's internal MuJoCo contact manager requires a CUDA device.")
    if not all(math.isfinite(value) and value > 0 for value in (contact_impratio, physics_hz)):
        raise ValueError("Contact impedance and physics rate must be positive and finite.")
    return SimulationCfg(
        device=device,
        dt=1.0 / physics_hz,
        physics=NewtonCfg(
            num_substeps=1,
            deterministic_mode="run_to_run" if deterministic else "not_guaranteed",
            solver_cfg=MJWarpSolverCfg(
                njmax=120,
                nconmax=120,
                cone=cone,
                integrator="implicitfast",
                impratio=contact_impratio,
                use_mujoco_contacts=True,
                disable_sensors=deterministic,
            ),
        ),
    )


def open_scene(stack, scene_path, cfg):
    """Load the same K3 world into Isaac Lab's builder without starting physics."""
    cfg.physics.use_cuda_graph = False
    sim = stack.enter_context(build_simulation_context(sim_cfg=cfg))
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
    builder.add_usd(str(scene_path.resolve()), load_visual_shapes=False, enable_self_collisions=False)
    return sim, builder


def verify_scene(scene_path, cfg, steps):
    """Import the K3 scene and check the manager's compiled solver and state."""
    with ExitStack() as stack:
        sim, _ = open_scene(stack, scene_path, cfg)
        sim.reset()
        manager = sim.physics_manager
        solver = manager._solver
        opts = solver.mj_model.opt
        model = manager.get_model()
        box = model.body_label.index("/World/Box")
        pad_groups = {
            name: [i for i, label in enumerate(model.shape_label)
                   if label.startswith(f"/World/Robot/GripperParts/{name}/ContactPatch")]
            for name in ("LeftFinger", "RightFinger")
        }
        if any(len(indices) != 4 for indices in pad_groups.values()):
            raise ValueError("The profile requires the K3 scene with 2x2 contact patches on each pad.")
        expected = cfg.physics.solver_cfg.impratio
        applied = float(solver.mjw_model.opt.impratio_invsqrt.numpy().reshape(-1)[0]) ** -2
        if not math.isclose(applied, expected, rel_tol=1e-6) or int(opts.cone) != 1:
            raise RuntimeError("MJWarp did not apply the requested contact impedance and elliptic cone.")
        initial_z = float(manager.get_state_0().body_q.numpy()[box, 2])
        for _ in range(steps):
            sim.step(render=False)
        final = manager.get_state_0().body_q.numpy()
        if not np.isfinite(final).all():
            raise RuntimeError("The imported scene produced nonfinite body poses.")
        return {
            "scene": str(scene_path),
            "manager": manager.__name__,
            "device": str(sim.device),
            "steps": steps,
            "solver_dt_s": manager.get_solver_dt(),
            "compiled_cone": "elliptic",
            "compiled_contact_impratio": applied,
            "box_mass_kg": float(model.body_mass.numpy()[box]),
            "pad_patches": {name: len(indices) for name, indices in pad_groups.items()},
            "box_z_m_before_after": [initial_z, float(final[box, 2])],
            "finite_body_poses": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-impratio", type=float, default=800.0)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")
    cfg = make_sim_cfg(args.device, args.contact_impratio)
    report = verify_scene(args.scene, cfg, args.steps)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
