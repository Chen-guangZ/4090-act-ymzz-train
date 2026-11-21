"""Triangulate a marked carton edge before and after K3 placement."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

import newton
import numpy as np

from k3_scene import body_pose, joint_positions_at, read, set_fk_pose
from track_k3_box_crease import frame_time


def optical_rays(pixels, camera):
    k = np.asarray(camera["k"]).reshape(3, 3)
    distorted = (np.asarray(pixels) - k[:2, 2]) / k.diagonal()[:2]
    k1, k2, p1, p2, k3 = camera["d"]
    points = distorted.copy()
    for _ in range(8):
        x, y = points.T
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        tangential = np.column_stack((2*p1*x*y + p2*(r2 + 2*x*x), p1*(r2 + 2*y*y) + 2*p2*x*y))
        points = (distorted - tangential) / radial[:, None]
    return np.column_stack((points, np.ones(len(points))))


def triangulate(views, offsets):
    origins = np.asarray([poses[offset][:3, 3] for (_, _, _, poses, _), offset in zip(views, offsets, strict=True)])
    rotation = np.asarray([poses[offset][:3, :3] for (_, _, _, poses, _), offset in zip(views, offsets, strict=True)])
    rays = np.asarray([view[-1] for view in views])
    directions = np.einsum("vij,vpj->vpi", rotation, rays)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    projectors = np.eye(3) - directions[..., :, None] * directions[..., None, :]
    points = np.linalg.solve(projectors.sum(axis=0), np.einsum("vpij,vj->pi", projectors, origins)[..., None])[..., 0]
    relative = points[None] - origins[:, None]
    if np.any((relative * directions).sum(axis=-1) <= 0):
        raise ValueError("A triangulated point is behind a camera.")
    ray_error = np.linalg.norm(np.einsum("vpij,vpj->vpi", projectors, relative), axis=-1)
    return points, ray_error, origins


def relative_angle(a, b, normal):
    return float(np.degrees(np.arctan2(np.cross(a, b) @ normal, a @ b)))


def plane_intersections(pose, rays, plane, normal):
    origin = pose[:3, 3]
    directions = rays @ pose[:3, :3].T
    denominator = directions @ normal
    if np.any(np.abs(denominator) < 1e-5):
        raise ValueError("A marked ray is nearly parallel to its assumed plane.")
    depth = ((plane - origin) @ normal) / denominator
    if np.any(depth <= 0):
        raise ValueError("A marked ray intersects its assumed plane behind the camera.")
    return origin + directions * depth[:, None]


def height_anchored(frames, plane, normal, height):
    reference = next(frame for frame in frames if frame[0]["group"] == "before")
    samples = []
    for row, stamp, _, poses, rays in frames:
        if row is reference[0]:
            continue
        differences = []
        for height_offset in (-0.010, 0.0, 0.010):
            edge_plane = plane + (height + height_offset) * normal
            for ref_offset in (0.0, -0.10, 0.10):
                ref_edge = plane_intersections(reference[3][ref_offset], reference[4][:2], edge_plane, normal)
                for offset in (0.0, -0.10, 0.10):
                    edge = plane_intersections(poses[offset], rays[:2], edge_plane, normal)
                    differences.append(float(np.linalg.norm((edge - ref_edge).mean(axis=0))))
        edge_plane = plane + height * normal
        ref_edge = plane_intersections(reference[3][0], reference[4][:2], edge_plane, normal)
        edge = plane_intersections(poses[0], rays[:2], edge_plane, normal)
        ref_marker = plane_intersections(reference[3][0], reference[4][2:], plane, normal)
        marker = plane_intersections(poses[0], rays[2:], plane, normal)
        samples.append({
            "archive_member": row["archive_member"],
            "phase": row["phase"],
            "image_file_time_approx": stamp.isoformat(),
            "edge_midpoint_displacement_m": float(np.linalg.norm((edge - ref_edge).mean(axis=0))),
            "edge_midpoint_displacement_sensitivity_range_m": [min(differences), max(differences)],
            "edge_yaw_change_deg": relative_angle(ref_edge[1] - ref_edge[0], edge[1] - edge[0], normal),
            "fixed_marker_projected_displacement_m": float(np.linalg.norm(marker - ref_marker)),
        })
    return {"assumed_edge_height_m": height, "reference_archive_member": reference[0]["archive_member"],
            "height_sensitivity_m": 0.010, "frame_time_sensitivity_s": 0.10, "comparisons": samples}


def measure(backup, scene, annotation, robot_asset, archive):
    derived = backup / "derived"
    trial_dir = derived / f"trial_{scene['trial']:02}"
    trial = read(trial_dir / "replay.json")
    camera = next(c for c in read(derived / "preflight.json")["checks"] if c["check"] == "camera_info")
    wrist_optical = np.asarray(scene["wrist_optical_in_link8"])
    positions = trial["snapshot_before_close_rad"]
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(robot_asset), load_visual_shapes=False, enable_self_collisions=False)
    slots = {label.rsplit("/", 1)[-1]: builder.joint_q_start[i]
             for i, label in enumerate(builder.joint_label) if label.rsplit("/", 1)[-1] in positions}
    bodies = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(builder.body_label)}
    model = builder.finalize("cpu")
    state = model.state()
    set_fk_pose(model, state, slots, positions)
    body = body_pose(state, bodies["Body_Link5"])
    table = trial["table"]
    surface = table["table"]
    normal = body[:3, :3] @ surface["normal"]
    normal /= np.linalg.norm(normal)
    plane = (body @ [*table["box_center"][:2], surface["z_center_m"], 1])[:3]
    close_time = next(datetime.fromisoformat(row["time"]) for row in trial["events"]
                      if row["event"] == "pickup_measured_before_close")
    frames = []
    for row in annotation["frames"]:
        member = row["archive_member"]
        stamp = frame_time(member, close_time)
        pixels = [*row["carton_lower_edge_px"], row["fixed_marker_px"]]
        rays = optical_rays(pixels, camera)
        with archive.extractfile(member) as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()

        poses = {}
        for offset in (-0.10, 0.0, 0.10):
            q = joint_positions_at(trial_dir / "joint_states.csv.gz", stamp.timestamp() + offset)
            set_fk_pose(model, state, slots, {name: q[name] for name in slots})
            poses[offset] = body_pose(state, bodies["Right_Arm_Link8"]) @ wrist_optical
        frames.append((row, stamp, digest, poses, rays))

    groups = {name: [frame for frame in frames if frame[0]["group"] == name] for name in ("before", "after")}
    if not all(len(views) >= 2 for views in groups.values()):
        raise ValueError("At least two marked views are required per group.")
    points_by_group, diagnostics = {}, {}
    for name, views in groups.items():
        offsets = [0.0] * len(views)
        points, ray_error, origins = triangulate(views, offsets)
        points_by_group[name] = points
        midpoint = points[:2].mean(axis=0)
        time_shifts = []
        for i in range(len(views)):
            for offset in (-0.10, 0.10):
                offsets[i] = offset
                shifted, _, _ = triangulate(views, offsets)
                time_shifts.append(np.linalg.norm(shifted[:2].mean(axis=0) - midpoint))
            offsets[i] = 0.0
        diagnostics[name] = {
            "camera_baseline_m": float(np.linalg.norm(origins[:, None] - origins[None, :], axis=-1).max()),
            "edge_width_m": float(np.linalg.norm(points[1] - points[0])),
            "edge_endpoint_heights_above_fitted_table_m": ((points[:2] - plane) @ normal).tolist(),
            "edge_midpoint_in_model_root_m": midpoint.tolist(),
            "edge_ray_error_max_m": float(ray_error[:, :2].max()),
            "fixed_marker_ray_error_max_m": float(ray_error[:, 2].max()),
            "max_edge_midpoint_shift_if_one_frame_time_off_by_0_10s_m": float(max(time_shifts)),
        }
    before, after = points_by_group["before"], points_by_group["after"]
    displacement = (after[:2] - before[:2]).mean(axis=0)
    horizontal = displacement - (displacement @ normal) * normal
    before_edge, after_edge = before[1] - before[0], after[1] - after[0]
    return {
        "trial": scene["trial"],
        "table_fit_rms_m": surface["rms_m"],
        "estimated_carton_width_m": scene["box_dims_m_perception_estimate"][1],
        "edge_midpoint_horizontal_displacement_m": float(np.linalg.norm(horizontal)),
        "edge_midpoint_height_change_m": float(displacement @ normal),
        "edge_yaw_change_deg": relative_angle(before_edge, after_edge, normal),
        "fixed_marker_displacement_m": float(np.linalg.norm(after[2] - before[2])),
        "triangulation": diagnostics,
        "height_anchored": height_anchored(frames, plane, normal, scene["box_dims_m_perception_estimate"][2]),
        "frames": [{
            **row,
            "sha256": digest,
            "image_file_time_approx": stamp.isoformat(),
        } for row, stamp, digest, _, _ in frames],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("scenes", type=Path, help="Directory containing trial_N.json scene reports.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    robot_asset = Path(__file__).resolve().parents[1] / "assets/ymbot_d_rl/ymbot_d_rl.usda"
    labels = read(args.annotations)
    with tarfile.open(args.backup / "raw/calibration_and_labels.tar.gz") as archive:
        runs = [measure(args.backup, read(args.scenes / f"trial_{trial}.json"), annotation, robot_asset, archive)
                for trial, annotation in labels.items()]
    report = {
        "method": "Hand-marked lower edge and fixed landmark, D405 undistortion and least-squares triangulation "
                  "from recorded-pose FK across viewpoints.",
        "assumptions": "Free triangulation assumes the box is stationary in each group. The height-anchored method "
                       "uses the estimated carton height for the marked edge and the fitted table for the landmark. "
                       "Time uses approximate JPEG filenames. Displacement refers to the observed edge.",
        "trials": runs,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
