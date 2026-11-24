"""Run the Task 01 reset, motion and open-jaw acceptance cases."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from export_k3_animation import write_animation
from k3_box_task import K3BoxTask
from replay_k3_grasp import event_time, read_tracks


def run_case(task, output, name, actions, *, seed=0, randomize=False, animate=False):
    start, initial = task.reset(seed=seed, randomize=randomize)
    frames = [task.body_poses()] if animate else None
    steps = []
    for action in actions:
        obs, info = task.step(action)
        steps.append({"observation": obs, **info})
        if animate:
            frames.append(task.body_poses())
        if task.termination:
            break
    report = {"reset": task.reset_info, "initial": {"observation": start, **initial}, "steps": steps}
    (output / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if animate:
        write_animation(task.scene, output / f"{name}.usda", task.labels, frames, task.action_hz)
    return report


def open_jaw_actions(task, data, start, end):
    source = data / "derived/trial_21/joint_states.csv.gz"
    start_ns = int(round(start * 1e9))
    query_ns = start_ns - 200_000_000
    tracks = read_tracks(source, task.arm_names, query_ns, int(round((end + .2) * 1e9)))
    for tick in np.arange(1, round((end - start) * task.action_hz) + 1) / task.action_hz:
        yield [*(np.interp(tick + .2, *tracks[name]) for name in task.arm_names), 1.0]


def grade_motion(report):
    steps = report["steps"]
    initial = report["initial"]
    fixed = max(
        abs(value - initial["fixed_joint_rad"][name])
        for row in steps for name, value in row["fixed_joint_rad"].items()
    )
    base_drift = max(np.linalg.norm(np.asarray(row["base_pose_xyzw"][:3]) - initial["base_pose_xyzw"][:3])
                     for row in steps)
    base_attitude = max(np.linalg.norm(np.asarray(row["base_pose_xyzw"][3:]) - initial["base_pose_xyzw"][3:])
                        for row in steps)
    return {
        "max_fixed_joint_drift_rad": fixed,
        "max_base_translation_m": base_drift,
        "max_base_quaternion_delta": base_attitude,
        "fixed_pose_pass": bool(fixed < 0.002 and base_drift < 0.0001 and base_attitude < 0.0001),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/k3_box_task_01.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trial = json.loads((args.data / "derived/trial_21/replay.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(config["observation_timestamp"]).timestamp()
    end = next(event_time(e) for e in trial["events"]
               if e.get("stage") == "return_above_pickup" and e.get("event") == "motion_start")
    task = K3BoxTask(args.data / config["scene_relative_to_data"], config, timeout_s=end - start)
    try:
        # At one-second intervals: track a known 0.05 rad change, close, reopen and return.
        actions = []
        moved = task.start_arm.copy()
        moved[0] += 0.05
        for pose, jaw in ((moved, 1.0), (moved, 0.25), (task.start_arm, 1.0)):
            actions.extend([[*pose, jaw]] * task.action_hz)
        results = [run_case(task, args.output, f"motion_{i}", actions, seed=seed, randomize=True, animate=i == 0)
                   for i, seed in enumerate((7, 8, 7))]
        negative = run_case(task, args.output, "open_jaw", open_jaw_actions(task, args.data, start, end), animate=True)
    finally:
        task.close()

    n = task.action_hz
    motion = []
    for report in results:
        first, closed, reopened = [report["steps"][k * n - 1] for k in (1, 2, 3)]
        error = abs(first["observation"][0] - moved[0])
        motion.append({**grade_motion(report), "tracking_error_rad": error,
                       "closed_gap_m": closed["jaw_gap_m"], "reopened_gap_m": reopened["jaw_gap_m"],
                       "passed": bool(error < 0.005 and closed["jaw_gap_m"] < 0.031
                                      and reopened["jaw_gap_m"] > 0.083)})
    starts = np.asarray([r["initial"]["box_pose_xyzw"] for r in results])
    same_seed = bool(np.allclose(starts[0], starts[2], atol=1e-7, rtol=0))
    other_seed = bool(np.linalg.norm(starts[0, :2] - starts[1, :2]) > 0.001)
    repeat_error = float(np.max(np.abs(np.asarray([s["observation"] for s in results[0]["steps"]]) -
                                       np.asarray([s["observation"] for s in results[2]["steps"]]))))
    last = negative["steps"][-1]
    negative_summary = {**grade_motion(negative), "termination": last["termination"],
                        "minimum_jaw_gap_m": min(s["jaw_gap_m"] for s in negative["steps"]),
                        "max_clearance_m": max(s["box_bottom_clearance_m"] for s in negative["steps"])}
    negative_summary["passed"] = (negative_summary["termination"] == "grasp_empty"
                                  and negative_summary["minimum_jaw_gap_m"] > 0.083
                                  and negative_summary["max_clearance_m"] < 0.10
                                  and negative_summary["fixed_pose_pass"])
    report = {"same_seed_reset_matches": same_seed, "different_seed_changes_box_position": other_seed,
              "same_seed_observation_max_abs_diff": repeat_error, "motion": motion, "open_jaw": negative_summary}
    report["passed"] = (same_seed and other_seed and repeat_error < 0.002 and negative_summary["passed"]
                        and all(r["passed"] and r["fixed_pose_pass"] for r in motion))
    (args.output / "task_score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
