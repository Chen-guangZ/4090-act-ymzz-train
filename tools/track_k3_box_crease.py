"""Track two visible carton crease points during an archived K3 return."""

import argparse
from datetime import datetime, timedelta
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
from scipy.signal import correlate2d


ANCHORS = {20: ((258, 114), (432, 87)), 21: ((258, 102), (432, 66))}
PATCH_RADIUS = 13
SEARCH_RADIUS = 20
GOOD_CORRELATION = 0.60


def frame_time(member, start):
    clock, tenths = Path(member).stem.split("_")
    time = datetime.strptime(clock, "%H%M%S").time()
    return datetime.combine(start.date(), time, start.tzinfo) + timedelta(seconds=int(tenths) / 10)


def gray(archive, member):
    with archive.extractfile(member) as source:
        with Image.open(BytesIO(source.read())) as image:
            return np.asarray(image.convert("L"), dtype=np.float64)


def crop(image, anchor, radius):
    x, y = anchor
    return image[y-radius:y+radius+1, x-radius:x+radius+1]


def track(image, template, anchor):
    search = crop(image, anchor, PATCH_RADIUS + SEARCH_RADIUS)
    windows = np.lib.stride_tricks.sliding_window_view(search, template.shape)
    variance = (windows**2).sum(axis=(-2, -1)) - windows.sum(axis=(-2, -1))**2 / template.size
    score = correlate2d(search, template, mode="valid") / np.sqrt(np.maximum(variance, 1))
    y, x = np.unravel_index(score.argmax(), score.shape)
    return [int(x - SEARCH_RADIUS), int(y - SEARCH_RADIUS)], float(score[y, x])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", type=int, choices=sorted(ANCHORS), default=20)
    args = parser.parse_args()
    anchors = ANCHORS[args.trial]
    data = read(args.backup / "derived" / f"trial_{args.trial:02}" / "replay.json")
    trial = next(x for x in read(args.backup / "derived/trial_index.json") if x["trial"] == args.trial)
    starts = {
        event["stage"]: datetime.fromisoformat(event["time"])
        for event in data["events"] if event["event"] == "motion_start"
    }
    start, end = starts["return_above_pickup"], starts["replace_at_saved_pickup"]
    prefix = trial["wrist_frames_archive_path"] + "/"
    rows = []
    with tarfile.open(args.backup / "raw/calibration_and_labels.tar.gz") as archive:
        candidates = [name for name in archive.getnames() if name.startswith(prefix) and name.endswith(".jpg")]
        frames = sorted((frame_time(name, start), name) for name in candidates)
        frames = [(time, name) for time, name in frames if start <= time < end]
        reference = gray(archive, frames[0][1])
        templates = []
        for anchor in anchors:
            patch = crop(reference, anchor, PATCH_RADIUS).copy()
            patch -= patch.mean()
            templates.append(patch / np.linalg.norm(patch))
        for time, member in frames:
            image = gray(archive, member)
            matches = [track(image, template, anchor) for template, anchor in zip(templates, anchors)]
            rows.append({
                "image_file_time_approx": time.isoformat(),
                "source_archive_member": member,
                "anchor_displacement_px": [x[0] for x in matches],
                "anchor_correlation": [x[1] for x in matches],
            })
    shifts = np.asarray([row["anchor_displacement_px"] for row in rows])
    accepted = np.asarray([row["anchor_correlation"] for row in rows]) >= GOOD_CORRELATION
    anchors_vector = np.asarray(anchors[1]) - anchors[0]
    vector = anchors_vector + shifts[:, 1] - shifts[:, 0]
    rotation = np.degrees(np.arctan2(vector[:, 1], vector[:, 0]) -
                          np.arctan2(anchors_vector[1], anchors_vector[0]))
    result = {
        "trial": args.trial, "phase": "return_above_pickup", "start": start.isoformat(), "end": end.isoformat(),
        "anchors_px": anchors, "reference_archive_member": frames[0][1],
        "window_search_radius_px": SEARCH_RADIUS, "accepted_correlation_min": GOOD_CORRELATION,
        "accepted_per_anchor": accepted.sum(axis=0).tolist(),
        "peak_displacement_px_per_anchor": [
            float(np.linalg.norm(shifts[accepted[:, i], i], axis=1).max()) for i in range(2)
        ],
        "peak_crease_angle_change_deg": float(np.abs(rotation[accepted.all(axis=1)]).max()),
        "caveat": "2D patch displacement from wrist JPEGs. Filenames provide approximate timestamps. No 3D pose claim.",
        "frames": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
