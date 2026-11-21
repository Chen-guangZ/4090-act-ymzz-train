"""Score K3 grasp replays through retreat against explicit placement tolerances."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_k3_retention import read, recorded_release


CONTACT_N = 0.1
LIFT_M = 0.10
LIFT_S = 1.0
LOSS_S = 0.25
SUPPORT_S = 0.5


def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(edges[::2], edges[1::2])


def evaluate(trace, trial, label, xy_limit, motion_limit):
    if not trace["finite"] or not trace["replayed_through_release"]:
        raise ValueError("Expected a finite replay through retreat.")
    if trace["trial"] != trial["trial"]:
        raise ValueError("Replay and field trial identifiers differ.")
    c = dict(zip(trace["sample_columns"], np.asarray(trace["samples"]).T, strict=True))
    time = c["elapsed_s"]
    dt = 1.0 / trace["sample_hz"]
    if len(time) < 2 or not np.allclose(np.diff(time), dt):
        raise ValueError("Expected uniformly sampled replay telemetry.")
    bilateral = np.minimum(c["left_pad_normal_load_n"], c["right_pad_normal_load_n"]) >= CONTACT_N
    _, release = recorded_release(trial)
    end = int(time.searchsorted(release))
    if end == 0 or time[-1] - release < 1.0:
        raise ValueError("Replay must cover grasp and at least one second after the release command.")

    high = c["bottom_clearance_m"] >= LIFT_M
    lifted = next((b for a, b in intervals((bilateral & high)[:end]) if (b - a) * dt >= LIFT_S), None)
    if label == "failed":
        any_lift = any((b - a) * dt >= LIFT_S for a, b in intervals(high[:end]))
        return {"operator_hold_label": label, "checks": {"no_sustained_lift": not any_lift}}
    if label != "held":
        raise ValueError(f"Expected a reviewed held or failed label, got {label!r}.")

    lost = None
    if lifted is not None:
        lost = next((lifted + a for a, b in intervals(~bilateral[lifted:end])
                     if (b - a) * dt >= LOSS_S), None)
    held_to_release = lifted is not None and lost is None and bool(bilateral[end - 1])
    tail = time > time[-1] - SUPPORT_S
    on_table = bool(np.all((c["table_box_normal_load_n"][tail] >= CONTACT_N)
                           & (c["floor_box_normal_load_n"][tail] < CONTACT_N)))
    robot_load = np.maximum.reduce([c[name][tail] for name in
                                   ("left_pad_normal_load_n", "right_pad_normal_load_n",
                                    "other_robot_box_normal_load_n")])
    clear = bool(np.all(robot_load < CONTACT_N))
    placement = trace["placement_after_retreat"]
    xy = placement["box_xy_offset_from_pickup_m"]
    motion = placement["box_max_motion_last_second_m"]
    return {
        "operator_hold_label": label,
        "recorded_release_s": release,
        "sustained_loss_before_release_s": release - float(time[lost]) if lost is not None else None,
        "placement_after_retreat": placement,
        "checks": {
            "bilateral_lift": lifted is not None,
            "held_to_release": held_to_release,
            "supported_by_table_after_retreat": on_table,
            "clear_of_robot_after_retreat": clear,
            "within_pickup_xy_tolerance": xy <= xy_limit,
            "within_rest_motion_tolerance": motion <= motion_limit,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--max-placement-error-mm", type=float, required=True)
    parser.add_argument("--max-rest-motion-mm", type=float, required=True)
    parser.add_argument("--require-pass", action="store_true", help="Return 1 if a check fails, after writing the report.")
    args = parser.parse_args()
    limits = (args.max_placement_error_mm, args.max_rest_motion_mm)
    if not all(np.isfinite(value) and value > 0 for value in limits):
        parser.error("Placement and motion tolerances must be finite and positive.")
    labels = {row["trial"]: row["operator_hold_label"] for row in read(args.derived / "trial_index.json")}
    trials = {}
    for path in args.reports:
        trace = read(path)
        number = trace["trial"]
        if number in trials:
            parser.error(f"Duplicate trace for trial {number}.")
        try:
            result = evaluate(trace, read(args.derived / f"trial_{number:02}" / "replay.json"), labels[number],
                              limits[0] / 1000, limits[1] / 1000)
        except (KeyError, ValueError) as error:
            parser.error(f"{path}: {error}")
        trials[number] = {"trace": str(path), **result, "passed": all(result["checks"].values())}
    passed = all(trial["passed"] for trial in trials.values())
    report = {
        "passed": passed,
        "thresholds": {"lift_m": LIFT_M, "lift_s": LIFT_S, "contact_n": CONTACT_N,
                       "loss_s": LOSS_S, "table_support_s": SUPPORT_S,
                       "max_placement_error_mm": limits[0], "max_rest_motion_mm": limits[1]},
        "trials": trials,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return int(args.require_pass and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
