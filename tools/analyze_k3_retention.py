"""Compare box motion in the wrist frame against K3 replay phase boundaries."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def rotation_xyzw(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    xyz = q[:, :3]
    w = q[:, 3]
    identity = np.eye(3)[None, :, :]
    cross = np.cross(xyz[:, None, :], identity)
    matrix = identity + 2 * (w[:, None, None] * cross + np.cross(xyz[:, None, :], cross))
    return matrix.transpose(0, 2, 1)


def phase_result(samples, first, last):
    if first >= last:
        return None
    q = samples[first:last]
    displacement = q[:, 1:4] - q[0, 1:4]
    rotations = q[:, 4:13].reshape(-1, 3, 3)
    cosines = (np.sum(rotations[0] * rotations, axis=(1, 2)) - 1) / 2
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    return {
        "peak_relative_translation_mm": float(np.linalg.norm(displacement, axis=1).max() * 1000),
        "peak_relative_rotation_deg": float(angles.max()),
        "end_wrist_frame_displacement_mm": (displacement[-1] * 1000).tolist(),
    }


def analyze(trace, log):
    channels = dict(zip(trace["sample_columns"], np.asarray(trace["samples"]).T, strict=True))
    time = channels["elapsed_s"]
    position = {k: np.column_stack([channels[f"{k}_{a}"] for a in "xyz"]) for k in ("box", "wrist")}
    orientation = {
        k: rotation_xyzw(np.column_stack([channels[f"{k}_quat_{a}"] for a in "xyzw"]))
        for k in ("box", "wrist")
    }
    wrist_inverse = orientation["wrist"].transpose(0, 2, 1)
    wrist_xyz = np.einsum("nij,nj->ni", wrist_inverse, position["box"] - position["wrist"])
    relative_rotation = wrist_inverse @ orientation["box"]
    local = np.column_stack((time, wrist_xyz, relative_rotation.reshape(-1, 9)))
    load_fields = ["left_pad_normal_load_n", "right_pad_normal_load_n"]
    bilateral = (np.minimum(*[channels[name] for name in load_fields]) >= 0.1
                 if all(name in channels for name in load_fields) else None)
    close = next(c["bag_time_ns"] for c in log["gripper_commands"] if c["ratio"] < 1)
    release = next(c["bag_time_ns"] for c in log["gripper_commands"] if c["bag_time_ns"] > close and c["ratio"] == 1)
    elapsed = (release - close) * 1e-9 + 0.75
    starts = {
        e["stage"]: datetime.fromisoformat(e["time"]).timestamp() - close / 1e9 + 0.75
        for e in log["events"] if e["event"] == "motion_start"
    }
    stages = {
        "lift": (starts["lift_clear_table"], starts["carry_near_body"]),
        "carry": (starts["carry_near_body"], starts["return_above_pickup"]),
        "return": (starts["return_above_pickup"], starts["replace_at_saved_pickup"]),
    }
    phase_stats = {}
    for phase, (start, end) in stages.items():
        first, last = time.searchsorted((start, end))
        stats = phase_result(local, first, last)
        if stats is None:
            continue
        stats["peak_bottom_clearance_mm"] = float(channels["bottom_clearance_m"][first:last].max() * 1000)
        if bilateral is not None:
            stats["fraction_with_bilateral_contact"] = float(bilateral[first:last].mean())
            contact_samples = local[first:last][bilateral[first:last]]
            stats["while_bilateral"] = phase_result(contact_samples, 0, len(contact_samples))
        phase_stats[phase] = stats
    held = np.flatnonzero((time < elapsed) & bilateral) if bilateral is not None else []
    last_contact = int(held[-1]) if len(held) else None
    handoff = None
    if last_contact is not None and time[last_contact] >= starts["return_above_pickup"]:
        first_return = int(time.searchsorted(starts["return_above_pickup"]))
        handoff = phase_result(local, first_return, last_contact + 1)
    return {
        "trial": trace["trial"],
        "physics_hz": trace["physics_hz"],
        "passive_linkage": trace.get("passive_linkage", False),
        "phases": phase_stats,
        "return_until_final_bilateral_contact": handoff,
        "last_bilateral_contact_before_release_s": (
            float(elapsed - time[last_contact]) if last_contact is not None else None
        ),
        "contact_loads_sampled": bilateral is not None,
        "continuous_bilateral_lift_s": trace.get("continuous_bilateral_lift_s"),
        "placement_after_retreat": trace["placement_after_retreat"],
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    summary = {}
    for path in args.reports:
        trace = read(path)
        if not trace["finite"] or not trace["replayed_through_release"]:
            raise ValueError(f"Expected a finite replay through retreat: {path}")
        trial = trace["trial"]
        summary[str(path)] = analyze(trace, read(args.derived / f"trial_{trial:02}" / "replay.json"))
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
