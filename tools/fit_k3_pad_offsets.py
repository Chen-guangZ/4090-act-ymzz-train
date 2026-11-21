"""Measure the camera-visible K3 jaw pad center from the archived RGB-D probe."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from k3_camera import optical_rays


# Manually selected interiors of the two textured faces in each probe image.
# The image-left pad is on +X in Right_Arm_Link8.
IMAGE_POLYGONS = (
    (((132, 9), (184, 91), (186, 144), (152, 140), (105, 106), (88, 77)),
     ((567, 7), (610, 15), (592, 63), (512, 119), (507, 82), (523, 44))),
    (((238, 36), (252, 68), (260, 136), (248, 150), (229, 114), (218, 77)),
     ((459, 19), (479, 18), (481, 53), (454, 119), (433, 140), (429, 114), (438, 76))),
    (((270, 39), (279, 37), (293, 126), (293, 152), (278, 158), (259, 122), (255, 89)),
     ((416, 21), (439, 21), (439, 57), (425, 92), (401, 145), (397, 117), (399, 76))),
    (((288, 39), (301, 34), (309, 123), (308, 156), (292, 157), (279, 124), (275, 80)),
     ((400, 25), (421, 26), (418, 66), (407, 110), (381, 152), (380, 103), (387, 61))),
)
RATIOS = (1.0, 0.5, 0.3, 0.2)


def measure(depth, polygons, calibration, camera=None):
    fx, fy, cx, cy, width, height = calibration["wrist_rgbd_fx_fy_cx_cy_w_h"]
    transform = np.asarray(calibration["wrist_optical_in_link8"])
    if depth.shape != (height, width):
        raise ValueError("Probe depth and wrist intrinsics have different resolutions.")
    points = []
    normals, residuals = [], []
    for polygon in polygons:
        mask = Image.new("1", (width, height))
        ImageDraw.Draw(mask).polygon(polygon, fill=1)
        v, u = np.where(np.asarray(mask) & (depth > 0.06) & (depth < 0.145))
        if len(v) < 100:
            raise ValueError("Not enough aligned depth samples in the marked pad.")
        distance = depth[v, u]
        if camera is None:
            optical = np.column_stack(((u - cx) * distance / fx, (v - cy) * distance / fy, distance))
        else:
            optical = optical_rays(np.column_stack((u, v)), camera) * distance[:, None]
        wrist = optical @ transform[:3, :3].T + transform[:3, 3]
        points.append(np.quantile(wrist, [0.1, 0.5, 0.9], axis=0))
        centered = wrist - wrist.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        normal = vectors[-1]
        normals.append(normal if normal[0] >= 0 else -normal)
        residuals.append(float(np.linalg.norm(centered @ normal) / np.sqrt(len(centered))))
    return np.asarray(points), np.asarray(normals), residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("scene_calibration", type=Path)
    parser.add_argument("--undistort", action="store_true", help="Use archived D405 plumb_bob calibration.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.scene_calibration.read_text(encoding="utf-8"))
    camera = None
    if args.undistort:
        preflight = json.loads((args.backup / "derived/preflight.json").read_text(encoding="utf-8"))
        camera = next(check for check in preflight["checks"] if check["check"] == "camera_info")
    observations = []
    with tarfile.open(args.backup / "raw/calibration_and_labels.tar.gz") as archive:
        for i, (ratio, polygons) in enumerate(zip(RATIOS, IMAGE_POLYGONS)):
            stem = f"eval/yolo26ft/grasp/gripper_gap/{i}_r{ratio:.2f}"
            depth_path = stem + "_depth.npy"
            with archive.extractfile(depth_path) as source:
                depth = np.load(BytesIO(source.read()), allow_pickle=False)
            measurements, normals, residuals = measure(depth, polygons, calibration, camera)
            axis = normals.mean(axis=0)
            axis /= np.linalg.norm(axis)
            observations.append({
                "ratio": ratio, "depth_archive_member": depth_path, "image_archive_member": stem + ".jpg",
                "interior_polygons_px": polygons,
                "pad_point_quantiles_10_50_90_in_link8_m": measurements.tolist(),
                "face_normals_positive_x_in_link8": normals.tolist(),
                "face_fit_rms_m": residuals,
                "normal_axis_gap_m": float(abs((measurements[0, 1] - measurements[1, 1]) @ axis)),
                "midpoint_in_link8_m": measurements[:, 1, :].mean(axis=0).tolist(),
            })
    midpoints = np.asarray([observation["midpoint_in_link8_m"] for observation in observations])
    center = [*np.median(midpoints[:, :2], axis=0), (midpoints[2, 2] + midpoints[3, 2]) / 2]
    normals = np.asarray([observation["face_normals_positive_x_in_link8"] for observation in observations[1:]])
    yaw = np.degrees(np.arctan2(normals[..., 1], normals[..., 0]))
    result = {
        "frame": "Right_Arm_Link8", "reference_close_ratio": 0.25, "pad_midpoint_m": center,
        "pad_closing_axis_yaw_deg": float(np.median(yaw)),
        "method": ("Median of marked inner-face depth points, D405 plumb_bob undistortion, then wrist optical TF"
                   if camera else "Median of marked inner-face depth points, pinhole projection, then wrist optical TF"),
        "caveat": "Visible-face median estimates pad placement. Near-range depth and mask coverage limit absolute accuracy.",
        "observations": observations,
    }
    output = args.output if args.output is not None else args.backup / "derived/gripper_pad_calibration.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{output}: {np.asarray(center) * 1000} mm")


if __name__ == "__main__":
    main()
