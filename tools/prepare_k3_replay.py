"""Index K3 grasp trials and extract synchronized ROS data from the raw backup."""

import argparse
import csv
from contextlib import closing
from datetime import datetime
import gzip
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile

import numpy as np
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


FIELD_SESSION = "logs/session-20261117-161643-box/runs/"
REPLAY_TRIALS = {16, 17, 18, 19, 20, 21}
RECORDINGS = {
    17: "eval/yolo26ft/grasp/wrist_trial17",
    18: "eval/yolo26ft/grasp/wrist_trial18",
    19: "eval/yolo26ft/grasp/wrist_trial19",
    20: "eval/grasp_20261120_evening/wrist_1855",
    21: "eval/grasp_20261120_evening/wrist_v040",
}
MOVEIT_HUMBLE = {
    "moveit_msgs/msg/DisplayTrajectory":
        "string model_id\nRobotTrajectory[] trajectory\nRobotState trajectory_start",
    "moveit_msgs/msg/RobotTrajectory":
        "trajectory_msgs/JointTrajectory joint_trajectory\ntrajectory_msgs/MultiDOFJointTrajectory multi_dof_joint_trajectory",
    "moveit_msgs/msg/RobotState":
        "sensor_msgs/JointState joint_state\nsensor_msgs/MultiDOFJointState multi_dof_joint_state\n"
        "AttachedCollisionObject[] attached_collision_objects\nbool is_diff",
    "moveit_msgs/msg/AttachedCollisionObject":
        "string link_name\nCollisionObject object\nstring[] touch_links\n"
        "trajectory_msgs/JointTrajectory detach_posture\nfloat64 weight",
    "moveit_msgs/msg/CollisionObject":
        "std_msgs/Header header\ngeometry_msgs/Pose pose\nstring id\nobject_recognition_msgs/ObjectType type\n"
        "shape_msgs/SolidPrimitive[] primitives\ngeometry_msgs/Pose[] primitive_poses\n"
        "shape_msgs/Mesh[] meshes\ngeometry_msgs/Pose[] mesh_poses\nshape_msgs/Plane[] planes\n"
        "geometry_msgs/Pose[] plane_poses\nstring[] subframe_names\ngeometry_msgs/Pose[] subframe_poses\nbyte operation",
    "object_recognition_msgs/msg/ObjectType": "string key\nstring db",
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_lines(archive, path):
    return [json.loads(line) for line in archive.extractfile(path)]


def seconds(stamp):
    return int(stamp.sec) + int(stamp.nanosec) * 1e-9


def read_bag(path, joint_csv):
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = {}
    for name, schema in MOVEIT_HUMBLE.items():
        definitions.update(get_types_from_msg(schema, name))
    store.register(definitions)
    query = (
        "SELECT timestamp, name, type, data FROM messages JOIN topics ON topics.id=topic_id "
        "WHERE name IN ('/joint_states','/tf_static','/display_planned_path','/gripper/command') "
        "ORDER BY timestamp"
    )
    tracks = []
    commands = []
    plans = []
    transforms = {}
    with (
        closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as conn,
        gzip.open(joint_csv, "wt", encoding="utf-8", newline="") as stream,
    ):
        topics = dict(conn.execute("SELECT name, count(*) FROM messages JOIN topics ON topics.id=topic_id GROUP BY name"))
        writer = csv.writer(stream)
        writer.writerow(("bag_time_ns", "header_time_s", "name", "position_rad", "velocity_fd_rad_s",
                         "driver_alarm_code", "driver_enabled_state"))
        previous = {}
        for ns, name, msg_type, data in conn.execute(query):
            msg = store.deserialize_cdr(data, msg_type)
            if name == "/joint_states":
                time = seconds(msg.header.stamp)
                tracks.append((ns, dict(zip(msg.name, map(float, msg.position))), list(map(float, msg.effort))))
                for idx, joint in enumerate(msg.name):
                    position = float(msg.position[idx])
                    alarm = msg.velocity[idx] if msg.velocity.size else ""
                    enabled = msg.effort[idx] if msg.effort.size else ""
                    velocity = ""
                    if enabled != -1:
                        if joint in previous and time > previous[joint][0]:
                            before, old_position = previous[joint]
                            velocity = (position - old_position) / (time - before)
                        previous[joint] = (time, position)
                    else:
                        previous.pop(joint, None)
                    writer.writerow((ns, f"{time:.9f}", joint, position, velocity, alarm, enabled))
            elif name == "/gripper/command":
                commands.append({"bag_time_ns": ns, "ratio": float(msg.data)})
            elif name == "/display_planned_path":
                for trajectory in msg.trajectory:
                    j = trajectory.joint_trajectory
                    plans.append({
                        "bag_time_ns": ns, "joint_names": j.joint_names,
                        "points": [{"time_from_start_s": seconds(p.time_from_start),
                                    "position_rad": p.positions.tolist(),
                                    "velocity_rad_s": p.velocities.tolist()} for p in j.points],
                    })
            elif name == "/tf_static":
                for tf in msg.transforms:
                    t, q = tf.transform.translation, tf.transform.rotation
                    transforms[tf.child_frame_id] = {
                        "parent": tf.header.frame_id, "xyz_m": [t.x, t.y, t.z], "xyzw": [q.x, q.y, q.z, q.w],
                    }
    return topics, tracks, commands, plans, transforms


def convert_trial(run_archive, prefix, trial, output):
    output.mkdir(exist_ok=True)
    events = json_lines(run_archive, prefix + "/events.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "bag.db3"
        with run_archive.extractfile(prefix + "/bag/bag_0.db3") as source, db.open("wb") as target:
            shutil.copyfileobj(source, target)
        topics, tracks, commands, plans, transforms = read_bag(db, output / "joint_states.csv.gz")
    close = next(row for row in commands if row["ratio"] < 1.0)
    close_ns = close["bag_time_ns"]
    snapshot = max((row for row in tracks if row[0] <= close_ns), key=lambda row: row[0])
    table = next(row for row in events if row["event"] == "table_scene" and row.get("action") == "applied")
    perception = [row for row in events if row["event"] == "perception"]
    at_table = next(i for i, row in enumerate(events) if row is table)
    candidate = [row for row in events[:at_table] if row["event"] == "perception"][-1]
    driver_enable = np.array([row[2] for row in tracks], dtype=float)
    write_json(output / "replay.json", {
        "trial": trial, "source_run": prefix, "frame": "Body_Link5",
        "ros_topics": topics,
        "joint_states_rate_hz": (len(tracks) - 1) * 1e9 / (tracks[-1][0] - tracks[0][0]),
        "driver_joint_enable_samples": {
            "enabled": int(np.count_nonzero(driver_enable == 1)),
            "disabled": int(np.count_nonzero(driver_enable == 0)),
            "missing": int(np.count_nonzero(driver_enable == -1)),
        },
        "snapshot_before_close_rad": snapshot[1],
        "snapshot_offset_from_close_s": (snapshot[0] - close_ns) * 1e-9,
        "gripper_commands": commands, "static_tf": transforms, "table": table,
        "selected_perception": candidate, "perception_frames": len(perception),
        "events": events,
    })
    write_json(output / "moveit_plans.json", plans)
    selected_frames = [
        name for name in run_archive.getnames()
        if name.startswith(prefix + "/frames/") and name.endswith("_selected.json")
    ]
    if len(selected_frames) == 1:
        (output / "selected_rgbd.json").write_bytes(run_archive.extractfile(selected_frames[0]).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    raw = args.backup / "raw"
    output = args.backup / "derived"
    output.mkdir(exist_ok=True)
    with (
        tarfile.open(raw / "calibration_and_labels.tar.gz") as calibration,
        tarfile.open(raw / "run_logs_and_bags.tar.gz") as runs,
    ):
        with io.TextIOWrapper(calibration.extractfile("logs/trials.csv"), encoding="utf-8-sig") as stream:
            rows = list(csv.reader(stream))
        index = []
        for row in rows[1:]:
            if not row:
                continue
            trial = int(row[0])
            run = Path(row[-1]).name
            manual = {"y": "held", "n": "failed"}.get(row[5].lower(), "unreviewed")
            index.append({
                "trial": trial, "start_local": datetime.fromisoformat(row[1]).isoformat(),
                "target": row[2], "software_status": row[3], "software_reason": row[4],
                "operator_hold_label": manual, "operator_notes": row[6],
                "run": run, "wrist_frames_archive_path": RECORDINGS.get(trial),
            })
            if trial in REPLAY_TRIALS:
                print(f"Decoding trial {trial}: {run}", flush=True)
                convert_trial(runs, FIELD_SESSION + run, trial, output / f"trial_{trial:02}")
        write_json(output / "trial_index.json", index)
        for member, name in (
            ("config/head_camera.json", "head_camera.json"),
            ("config/grasp_config.json", "grasp_config.json"),
            ("reference_model/ymbot_d.urdf", "k3_reference.urdf"),
        ):
            (output / name).write_bytes(calibration.extractfile(member).read())
        preflight = "logs/session-20261117-161643-box/preflight"
        for member in ("preflight.json", "snapshot_rgbd.npz", "robot_description.urdf"):
            (output / member).write_bytes(runs.extractfile(f"{preflight}/{member}").read())


if __name__ == "__main__":
    main()
