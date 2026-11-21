"""Build a calibrated K3 scene layer and report FK error against a field trial."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import re

import newton
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from k3_gripper import (
    AXIAL_TRAVEL_M, DRIVE_DAMPING, DRIVE_FORCE_N, DRIVE_STIFFNESS, MAX_OPENING_M,
    PAD_FRICTION, PAD_LENGTH_M, PAD_MASS_KG, PAD_THICKNESS_M, PAD_WIDTH_M,
    axial_retraction_m, opening_m, passive_linkage_coefficients,
)


OPTICAL_FROM_USD = np.diag([1, -1, -1, 1])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_from_pose(xyz, xyzw):
    x, y, z, w = xyzw
    transform = Gf.Matrix4d(1)
    transform.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    transform.SetTranslateOnly(Gf.Vec3d(*xyz))
    return np.array(transform).T


def camera(stage, path, transform, intrinsics):
    sensor = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(sensor)
    xform.MakeMatrixXform().Set(Gf.Matrix4d(*transform.T.flatten().tolist()))
    fx, fy, cx, cy, width, height = intrinsics
    # Aperture and focal length use the same USD units. Keep the calibrated
    # off-center projection and unequal focal scales.
    focal = 20.0
    sensor.GetFocalLengthAttr().Set(focal)
    sensor.GetHorizontalApertureAttr().Set(width * focal / fx)
    sensor.GetVerticalApertureAttr().Set(height * focal / fy)
    sensor.GetHorizontalApertureOffsetAttr().Set((width / 2 - cx) * focal / fx)
    sensor.GetVerticalApertureOffsetAttr().Set((cy - height / 2) * focal / fy)
    sensor.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 10.0))
    prim = sensor.GetPrim()
    prim.SetCustomDataByKey("k3ImageWidth", width)
    prim.SetCustomDataByKey("k3ImageHeight", height)
    return prim


def body_pose(state, body):
    xyzquat = state.body_q.numpy()[body]
    return matrix_from_pose(xyzquat[:3].tolist(), xyzquat[3:].tolist())


def joint_positions_at(track, timestamp):
    nearest = {}
    with gzip.open(track, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["driver_enabled_state"] == "-1.0":
                continue
            joint = row["name"]
            distance = abs(float(row["header_time_s"]) - timestamp)
            if distance < nearest.get(joint, (float("inf"),))[0]:
                nearest[joint] = (distance, float(row["position_rad"]))
    if not nearest or max(sample[0] for sample in nearest.values()) > 0.05:
        raise ValueError(f"No synchronized arm pose for RGB-D stamp {timestamp}")
    return {name: sample[1] for name, sample in nearest.items()}


def set_fk_pose(model, state, joint_slots, positions):
    q = model.joint_q.numpy()
    for name, angle in positions.items():
        q[joint_slots[name]] = angle
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def box_faces(trial_dir, wrist_tf, model, state, joint_slots, bodies, level):
    observation = read(trial_dir / "replay.json")["selected_perception"]
    rgbd = read(trial_dir / "selected_rgbd.json")
    angles = joint_positions_at(trial_dir / "joint_states.csv.gz", rgbd["stamp"])
    set_fk_pose(model, state, joint_slots, angles)
    optical_world = level @ body_pose(state, bodies["Right_Arm_Link8"]) @ wrist_tf
    body_world = level @ body_pose(state, bodies["Body_Link5"])
    target = (body_world @ [*observation["center_m"], 1])[:3]
    surfaces = []
    for candidate in rgbd["candidates"]:
        geometry = candidate.get("geometry") or {}
        surface = geometry.get("surface") or {}
        points = surface.get("point_cloud_camera_m", [])
        normal = (geometry.get("plane") or {}).get("normal_camera")
        if len(points) >= 100 and normal is not None and surface.get("minor_length_m", 0) > 0:
            cloud = np.asarray(points) @ optical_world[:3, :3].T + optical_world[:3, 3]
            surfaces.append((cloud, optical_world[:3, :3] @ normal, surface["minor_length_m"]))
    side, normal, width = min(surfaces, key=lambda item: np.linalg.norm(item[0].mean(axis=0) - target))
    if abs(normal[2]) > 0.35:
        raise ValueError("Selected RGB-D plane is not a near-vertical box face.")
    if normal @ (optical_world[:3, 3] - side.mean(axis=0)) < 0:
        normal = -normal
    inward = -normal[:2] / np.linalg.norm(normal[:2])
    sideways = np.array([-inward[1], inward[0]])
    front = float(np.median(side[:, :2] @ inward))
    edge = np.quantile(side[:, :2] @ sideways, [0.01, 0.99])
    return side, surfaces, inward, sideways, front, edge, width


def observed_box_footprint(reference_dir, wrist_tf, model, state, joint_slots, bodies, level):
    side, surfaces, inward, sideways, front, edge, side_width = box_faces(
        reference_dir, wrist_tf, model, state, joint_slots, bodies, level,
    )
    top_z = np.quantile(side[:, 2], 0.99)
    candidates = []
    for points, normal, top_width in surfaces:
        if abs(normal[2]) < 0.85 or abs(np.median(points[:, 2]) - top_z) > 0.025:
            continue
        lateral = points[:, :2] @ sideways
        in_width = (lateral >= edge[0] - 0.008) & (lateral <= edge[1] + 0.008)
        if in_width.sum() >= 70:
            depth = float(np.quantile(points[in_width, :2] @ inward, 0.99) - front)
            if 0.05 < depth < 0.25 and abs(top_width - side_width) < 0.012:
                candidates.append((depth, (top_width + side_width) / 2))
    if not candidates:
        raise ValueError("Reference trial lacks a matching horizontal box top in RGB-D.")
    return max(candidates)


def author_observed_pose(stage, builder, state, level):
    cache = UsdGeom.XformCache()
    for index, label in sorted(enumerate(builder.body_label), key=lambda entry: entry[1].count("/")):
        path = label.replace("/ymbot_d/", "/World/Robot/", 1)
        prim = stage.GetPrimAtPath(path)
        parent_world = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
        local = np.linalg.solve(parent_world, level @ body_pose(state, index))
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*local.T.flatten().tolist()))
        cache.Clear()


def slide(stage, path, parent, child, origin, axis, limit, stiffness, damping, max_force,
          parent_yaw=0.0, child_yaw=0.0):
    slider = UsdPhysics.PrismaticJoint.Define(stage, path)
    slider.CreateBody0Rel().SetTargets([parent])
    slider.CreateBody1Rel().SetTargets([child])
    slider.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    for yaw, attr in ((parent_yaw, slider.CreateLocalRot0Attr), (child_yaw, slider.CreateLocalRot1Attr)):
        if yaw == 180.0:
            attr(Gf.Quatf(0.0, Gf.Vec3f(0, 0, 1)))
        elif yaw:
            attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    slider.CreateAxisAttr(axis)
    slider.CreateLowerLimitAttr(0.0)
    slider.CreateUpperLimitAttr(limit)
    slider.CreateCollisionEnabledAttr(False)
    if stiffness:
        drive = UsdPhysics.DriveAPI.Apply(slider.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(max_force)
        drive.CreateTargetPositionAttr(limit)
    return slider.GetPrim()


def pad_contact_shapes(stage, path, pad_tiles):
    pad = stage.GetPrimAtPath(path + "/Collider")
    tint = Gf.Vec3f(0.18, 0.18, 0.18)
    UsdGeom.Gprim(pad).CreateDisplayColorAttr([tint])
    if pad_tiles == 1:
        return [pad]

    # Cover the same pad envelope with separated patches. This represents
    # the distributed pad surface without adding a grasp constraint.
    UsdPhysics.CollisionAPI(pad).CreateCollisionEnabledAttr(False)
    patch_width = PAD_WIDTH_M / pad_tiles
    patch_length = PAD_LENGTH_M / pad_tiles
    patch_gap = 0.0002
    shapes = []
    for y in range(pad_tiles):
        for z in range(pad_tiles):
            cube = UsdGeom.Cube.Define(stage, f"{path}/ContactPatch{y}{z}")
            cube.GetSizeAttr().Set(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(
                0.0, (y + 0.5) * patch_width - PAD_WIDTH_M / 2,
                (z + 0.5) * patch_length - PAD_LENGTH_M / 2,
            ))
            cube.AddScaleOp().Set(Gf.Vec3d(PAD_THICKNESS_M, patch_width - patch_gap, patch_length - patch_gap))
            shape = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(shape)
            UsdGeom.Imageable(shape).MakeInvisible()
            shapes.append(shape)
    return shapes


def add_fingers(stage, wrist_path, wrist_world, pad_center, pad_yaw, friction, contact_dim, torsion,
                coupled_jaws, passive_linkage, pad_tiles, jaw_force):
    base = "/World/Robot/GripperParts"
    UsdGeom.Xform.Define(stage, base)
    material = UsdShade.Material.Define(stage, "/World/FingerPadMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction)
    material.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(torsion)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), pad_yaw).GetQuat()
    level = matrix_from_pose([0, 0, 0], [*rotation.GetImaginary(), rotation.GetReal()])
    closing_axis = level[:3, 0]
    for name, direction in (("LeftFinger", -1), ("RightFinger", 1)):
        path = f"{base}/{name}"
        carriage_path = path + "Carriage"
        closed_origin = np.asarray(pad_center) + direction * PAD_THICKNESS_M * closing_axis / 2
        closed_origin[2] -= axial_retraction_m(opening_m(0.25) / 2)
        open_local = level.copy()
        open_local[:3, 3] = closed_origin + direction * MAX_OPENING_M * closing_axis / 2
        carriage = UsdGeom.Xform.Define(stage, carriage_path)
        carriage.AddTransformOp().Set(Gf.Matrix4d(*(wrist_world @ open_local).T.flatten().tolist()))
        UsdPhysics.RigidBodyAPI.Apply(carriage.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(carriage.GetPrim())
        mass.CreateMassAttr(0.020)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        open_local[2, 3] += AXIAL_TRAVEL_M
        finger = apply_shape(stage, path, [0, 0, 0],
                             (PAD_THICKNESS_M, PAD_WIDTH_M, PAD_LENGTH_M), mass=PAD_MASS_KG)
        xf = UsdGeom.Xformable(finger)
        xf.MakeMatrixXform().Set(Gf.Matrix4d(*(wrist_world @ open_local).T.flatten().tolist()))
        for pad in pad_contact_shapes(stage, path, pad_tiles):
            UsdShade.MaterialBindingAPI.Apply(pad).Bind(material, materialPurpose="physics")
            pad.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(contact_dim)
        mirror = 180.0 if direction < 0 else 0.0
        slide(stage, path + "Slide", wrist_path, carriage_path, closed_origin, "X",
              MAX_OPENING_M / 2, DRIVE_STIFFNESS, DRIVE_DAMPING, jaw_force,
              parent_yaw=pad_yaw + mirror, child_yaw=mirror)
        depth = slide(stage, path + "Depth", carriage_path, path, (0, 0, 0), "Z",
                      AXIAL_TRAVEL_M, 0.0 if passive_linkage else DRIVE_STIFFNESS,
                      DRIVE_DAMPING, DRIVE_FORCE_N)
        if passive_linkage:
            depth.AddAppliedSchema("MjcEqualityJointAPI")
            depth.CreateRelationship("newton:mimicJoint").SetTargets([path + "Slide"])
            for index, coefficient in enumerate(passive_linkage_coefficients()):
                name = f"newton:mimicCoef{index}" if index < 2 else f"mjc:coef{index}"
                depth.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(coefficient)
    if coupled_jaws:
        # Both sliders measure positive travel outward. The real parallel
        # gripper has one motor and keeps these distances equal.
        follower = stage.GetPrimAtPath(base + "/LeftFingerSlide")
        follower.AddAppliedSchema("NewtonMimicAPI")
        follower.CreateRelationship("newton:mimicJoint").SetTargets([base + "/RightFingerSlide"])
        follower.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(1.0)
    # Remove the old whole-gripper USD instances, including the instance
    # proxy collider that includes fingers fixed in the mesh.
    wrist = stage.GetPrimAtPath(wrist_path)
    for child in wrist.GetChildren():
        if child.GetName().startswith("moveit_right_zhixing_gripper_link"):
            child.SetActive(False)
    housing = UsdGeom.Cube.Define(stage, wrist_path + "/GripperHousingCollider")
    housing.GetSizeAttr().Set(1.0)
    housing.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.045))
    housing.AddScaleOp().Set(Gf.Vec3d(0.066, 0.045, 0.09))
    UsdPhysics.CollisionAPI.Apply(housing.GetPrim())
    housing.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.18)])


def replace_wrist_camera_housing(stage, wrist_path, wrist_optical):
    old_housing = stage.GetPrimAtPath(wrist_path + "/Right_Hand_Camera_Link")
    UsdGeom.Imageable(old_housing).MakeInvisible()
    for old_prim in Usd.PrimRange(old_housing):
        if old_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(old_prim).CreateCollisionEnabledAttr(False)
    housing = UsdGeom.Xform.Define(stage, wrist_path + "/D405Housing")
    housing.MakeMatrixXform().Set(Gf.Matrix4d(*wrist_optical.T.flatten().tolist()))
    # Optical origin is the lens, so the housing volume sits behind it.
    enclosure = UsdGeom.Cube.Define(stage, housing.GetPath().AppendChild("Body"))
    enclosure.GetSizeAttr().Set(1.0)
    enclosure.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.021))
    enclosure.AddScaleOp().Set(Gf.Vec3d(0.042, 0.042, 0.026))
    enclosure.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.16)])
    UsdPhysics.CollisionAPI.Apply(enclosure.GetPrim())


def apply_shape(stage, path, center, dims, yaw=0.0, mass=None):
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*center))
    root.AddRotateZOp().Set(yaw)
    root_prim = root.GetPrim()
    if mass is not None:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        props = UsdPhysics.MassAPI.Apply(root_prim)
        props.CreateMassAttr(mass)
        x, y, z = dims
        props.CreateDiagonalInertiaAttr(Gf.Vec3f(
            mass * (y*y + z*z) / 12, mass * (x*x + z*z) / 12, mass * (x*x + y*y) / 12,
        ))
    collider = UsdGeom.Cube.Define(stage, f"{path}/Collider")
    collider.GetSizeAttr().Set(1.0)
    collider.AddScaleOp().Set(Gf.Vec3d(*dims))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    color = (0.60, 0.42, 0.24) if mass is not None else (0.86, 0.85, 0.82)
    collider.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return root_prim


def build(backup, trial, robot_asset, output, nominal_mass, pad_friction, pad_contact_dim, pad_torsion,
          coupled_jaws, nominal_pad_center, fit_pad_axis, passive_linkage, pad_tiles, box_com, jaw_force):
    data = read(backup / f"trial_{trial:02}" / "replay.json")
    head = read(backup / "head_camera.json")
    if not head["calibrated"]:
        raise ValueError("Head camera extrinsic is marked uncalibrated.")
    preflight = read(backup / "preflight.json")
    def find_camera_info(value):
        if isinstance(value, dict):
            if value.get("check") == "camera_info":
                return value["detail"]
            return next((v for child in value.values() if (v := find_camera_info(child))), None)
        if isinstance(value, list):
            return next((v for child in value if (v := find_camera_info(child))), None)
        return None
    wrist_info = find_camera_info(preflight)
    wrist_k = [float(re.search(rf"{key}=([\d.]+)", wrist_info)[1]) for key in ("fx", "fy", "cx", "cy")]
    with np.load(backup / "snapshot_rgbd.npz") as rgbd:
        wh = rgbd["color"].shape[1::-1]
    h = head["intrinsics_1280x720"]
    head_k = [h[k] for k in ("fx", "fy", "ppx", "ppy", "width", "height")]
    transforms = data["static_tf"]
    mount = transforms["camera_link"]
    optical = transforms["camera_optical_frame"]
    if mount["parent"] != "Right_Arm_Link8" or optical["parent"] != "camera_link":
        raise ValueError("Unexpected wrist optical TF chain.")
    wrist_tf = matrix_from_pose(mount["xyz_m"], mount["xyzw"]) @ matrix_from_pose(optical["xyz_m"], optical["xyzw"])
    head_tf = np.asarray(head["T_parent_camera"])

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(robot_asset), load_visual_shapes=False, enable_self_collisions=False)
    positions = data["snapshot_before_close_rad"]
    joint_slots = {name.rsplit("/", 1)[-1]: builder.joint_q_start[i]
                   for i, name in enumerate(builder.joint_label) if name.rsplit("/", 1)[-1] in positions}
    bodies = {name.rsplit("/", 1)[-1]: idx for idx, name in enumerate(builder.body_label)}
    model = builder.finalize("cpu")
    state = model.state()
    set_fk_pose(model, state, joint_slots, positions)
    table = data["table"]
    original_body = body_pose(state, bodies["Body_Link5"])
    support_normal = original_body[:3, :3] @ table["table"]["normal"]
    # The measured support plane defines up for this stationary replay. This
    # compensates for the difference between the URDF base and K3's TF frames.
    alignment = Gf.Rotation(Gf.Vec3d(*support_normal), Gf.Vec3d(0, 0, 1)).GetQuat()
    level = matrix_from_pose([0, 0, 0], [*alignment.GetImaginary(), alignment.GetReal()])
    side, _, inward, sideways, front, edge, _ = box_faces(
        backup / f"trial_{trial:02}", wrist_tf, model, state, joint_slots, bodies, level,
    )
    depth, width = observed_box_footprint(backup / "trial_21", wrist_tf, model, state, joint_slots, bodies, level)
    set_fk_pose(model, state, joint_slots, positions)
    body_world = level @ body_pose(state, bodies["Body_Link5"])
    wrist_world = level @ body_pose(state, bodies["Right_Arm_Link8"])
    measured = next(event for event in data["events"] if event["event"] == "pickup_measured_before_close")
    coordinates = read(backup / "grasp_config.json")
    tool_length = coordinates["robot"]["tool_len_m"]
    pad_file = backup / "gripper_pad_calibration.json"
    pad_center = [0.0, 0.0, -tool_length]
    pad_yaw = 0.0
    if not nominal_pad_center and pad_file.is_file():
        pad_calibration = read(pad_file)
        if pad_calibration["frame"] != "Right_Arm_Link8" or pad_calibration["reference_close_ratio"] != 0.25:
            raise ValueError("Unexpected pad calibration frame or reference ratio.")
        pad_center = pad_calibration["pad_midpoint_m"]
        if fit_pad_axis:
            pad_yaw = pad_calibration["pad_closing_axis_yaw_deg"]
    elif fit_pad_axis:
        raise ValueError("The measured pad axis requires the K3 pad calibration.")
    tcp = (np.linalg.inv(body_world) @ wrist_world @ [0, 0, -tool_length, 1])[:3]
    fk_error = np.linalg.norm(tcp - measured["tcp_m"])
    table_center = (body_world @ [*table["box_center"], 1])[:3]
    # The K3 planner uses an axis-aligned snapshot in Body_Link5. Place the
    # simulated static panel level at the fitted surface height near the box.
    x, y, _ = data["selected_perception"]["center_m"]
    normal = table["table"]["normal"]
    table_xy = table["box_center"][:2]
    table_top_body = table["table"]["z_center_m"] - (
        normal[0] * (x - table_xy[0]) + normal[1] * (y - table_xy[1])
    ) / normal[2]
    surface_world = (body_world @ [x, y, table_top_body, 1])[:3]
    height = float(np.quantile(side[:, 2], 0.99) - surface_world[2])
    box_xy = inward * (front + depth / 2) + sideways * np.mean(edge)
    box_center = [*box_xy, surface_world[2] + height / 2 + 0.001]
    box_yaw = float(np.degrees(np.arctan2(inward[1], inward[0])))
    table_center[2] = surface_world[2] - table["box_size"][2] / 2

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    robot = stage.DefinePrim("/World/Robot")
    robot.GetReferences().AddReference(str(robot_asset))
    author_observed_pose(stage, builder, state, level)
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in positions and prim.IsA(UsdPhysics.RevoluteJoint):
            attr = prim.GetAttribute("state:angular:physics:position")
            if attr:
                attr.Set(float(np.degrees(positions[name])))
        if name.endswith("Gripper_Center_Link") and name.startswith("Right_"):
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, -tool_length))
    robot_body_paths = {
        name: builder.body_label[bodies[name]].replace("/ymbot_d/", "/World/Robot/", 1)
        for name in ("Neck_Link2", "Right_Arm_Link8")
    }
    for prim in stage.Traverse():
        if prim.GetName() in ("HeadCamL", "HeadCamR", "WristCam"):
            prim.SetActive(False)
    head_prim = camera(stage, robot_body_paths["Neck_Link2"] + "/K3HeadRGBD", head_tf @ OPTICAL_FROM_USD, head_k)
    head_prim.SetCustomDataByKey("k3DistortionModel", h["distortion_model"])
    wrist_prim = camera(stage, robot_body_paths["Right_Arm_Link8"] + "/K3WristRGBD", wrist_tf @ OPTICAL_FROM_USD, (*wrist_k, *wh))
    wrist_prim.SetCustomDataByKey("k3DistortionSource", wrist_info)
    heading = np.degrees(np.arctan2(body_world[1, 0], body_world[0, 0]))
    apply_shape(stage, "/World/TableTop", table_center, table["box_size"], heading)
    box_dims = np.asarray([depth, width, height])
    if np.any(np.abs(box_com) >= box_dims / 2):
        raise ValueError("Box center of mass must lie inside the fitted box.")
    box_prim = apply_shape(stage, "/World/Box", box_center, box_dims, box_yaw, nominal_mass)
    if any(box_com):
        UsdPhysics.MassAPI(box_prim).CreateCenterOfMassAttr(Gf.Vec3f(*box_com))
    apply_shape(stage, "/World/Ground", [0, 0, -0.026], [12, 12, 0.05])
    add_fingers(stage, robot_body_paths["Right_Arm_Link8"], wrist_world, pad_center, pad_yaw,
                pad_friction, pad_contact_dim, pad_torsion, coupled_jaws, passive_linkage, pad_tiles, jaw_force)
    replace_wrist_camera_housing(stage, robot_body_paths["Right_Arm_Link8"], wrist_tf)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    stage.GetRootLayer().Save()
    report = {
        "trial": trial, "source": data["source_run"], "joint_positions_rad": positions,
        "tool_length_m": tool_length, "tcp_body_link5_m": tcp.tolist(), "tcp_fk_error_m": float(fk_error),
        "table_top_world_m": float(surface_world[2]), "table_fitting_rms_m": table["table"]["rms_m"],
        "base_alignment_deg": float(np.degrees(np.arccos(np.clip(support_normal[2], -1, 1)))),
        "box_dims_m_perception_estimate": [depth, width, height], "box_mass_kg_nominal": nominal_mass,
        "box_yaw_world_deg": box_yaw, "box_footprint_reference_trial": 21,
        "box_com_offset_local_m": box_com,
        "jaw_horizontal_drive_limit_per_pad_n": jaw_force,
        "pad_friction_nominal": pad_friction, "pad_contact_dim": pad_contact_dim,
        "pad_torsion_nominal_m": pad_torsion,
        "coupled_jaws": coupled_jaws,
        "passive_linkage": passive_linkage,
        "pad_tiles_per_axis": pad_tiles,
        "pad_midpoint_at_0_25_in_link8_m": pad_center,
        "pad_midpoint_from_rgbd": not nominal_pad_center and pad_file.is_file(),
        "pad_closing_axis_yaw_in_link8_deg": pad_yaw,
        "table_center_world_m": table_center.tolist(), "box_center_world_m": box_center,
        "head_rgbd_fx_fy_cx_cy_w_h": head_k, "wrist_rgbd_fx_fy_cx_cy_w_h": [*wrist_k, *wh],
        "head_rgbd_optical_in_neck": head_tf.tolist(), "wrist_optical_in_link8": wrist_tf.tolist(),
        "close_command": next(cmd["ratio"] for cmd in data["gripper_commands"] if cmd["ratio"] < 1),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", type=int, default=21)
    parser.add_argument("--mass", type=float, default=0.060)
    parser.add_argument("--box-com-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("DEPTH", "WIDTH", "HEIGHT"), help="Box-local COM offset from geometric center.")
    parser.add_argument("--jaw-force-per-pad-n", type=float, default=DRIVE_FORCE_N,
                        help="Horizontal actuator force bound for each pad in a sensitivity test.")
    parser.add_argument("--pad-friction", type=float, default=PAD_FRICTION)
    parser.add_argument("--pad-contact-dim", type=int, choices=(3, 4), default=3)
    parser.add_argument("--pad-torsion", type=float, default=0.005)
    parser.add_argument("--independent-jaws", action="store_true", help="Compare with the original jaw model.")
    parser.add_argument("--nominal-pad-center", action="store_true", help="Place the pad midpoint at the planner TCP.")
    parser.add_argument("--fit-pad-axis", action="store_true", help="Use RGB-D pad-face normals for the closing axis.")
    parser.add_argument("--passive-linkage", action="store_true",
                        help="Couple axial retraction to jaw travel in MJWarp without a separate axial drive.")
    parser.add_argument("--pad-tiles", type=int, choices=(1, 2), default=1,
                        help="Subdivide each pad collider into a 1x1 or 2x2 contact grid.")
    args = parser.parse_args()
    if args.mass <= 0 or args.pad_friction <= 0 or args.pad_torsion < 0:
        parser.error("Use a positive box mass and pad friction, and nonnegative torsional friction.")
    asset = Path(__file__).resolve().parents[1] / "assets/ymbot_d_rl/ymbot_d_rl.usda"
    box_com = [value / 1000 for value in args.box_com_mm]
    if not np.isfinite(box_com).all():
        parser.error("Box center of mass must be finite.")
    if not np.isfinite(args.jaw_force_per_pad_n) or args.jaw_force_per_pad_n <= 0:
        parser.error("Jaw force must be finite and positive.")
    build(args.derived.resolve(), args.trial, asset, args.output.resolve(), args.mass,
          args.pad_friction, args.pad_contact_dim, args.pad_torsion, not args.independent_jaws,
          args.nominal_pad_center, args.fit_pad_axis, args.passive_linkage, args.pad_tiles, box_com,
          args.jaw_force_per_pad_n)


if __name__ == "__main__":
    main()
