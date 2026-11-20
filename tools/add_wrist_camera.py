#!/usr/bin/env python3
"""Add an eye-in-hand WristCam to the grasp project's robot asset copy.

Authors one inert USD Camera prim under Right_Arm_Link8 of the repo-local
asset copy ``assets/ymbot_d_rl/ymbot_d_rl.usda`` (same authoring pattern as
ymbot_push 3a's add_head_cameras.py: root-layer over chain, payloads untouched).

Mount: +0.05 above the Link8 frame origin; USD camera default forward is local
-Z, which in this asset points from the wrist toward the fingers (the TCP
Right_Gripper_Center_Link sits at z = -0.18), so an unrotated camera looks
straight down the gripper axis at whatever the hand is approaching.

Run:
    /home/nvidia4090/IsaacLab/.venv/bin/python /home/nvidia4090/ymbot_grasp/tools/add_wrist_camera.py
"""

from pxr import Gf, Usd, UsdGeom

ASSET_ROOT = "/home/nvidia4090/ymbot_grasp/assets/ymbot_d_rl/ymbot_d_rl.usda"
LINK8_PATH = (
    "/ymbot_d/Geometry/base_footprint_link/base_link/Body_Link1/Body_Link2/Body_Link3/"
    "Body_Link4/Body_Link5/Right_Arm_Link1/Right_Arm_Link2/Right_Arm_Link3/Right_Arm_Link4/"
    "Right_Arm_Link5/Right_Arm_Link6/Right_Arm_Link7/Right_Arm_Link8"
)
FOCAL_CM = 4.5  # wide (~77 deg FOV with the 20.955 mm aperture) for close-range palm view
CLIPPING = (0.02, 5.0)


def main() -> None:
    stage = Usd.Stage.Open(ASSET_ROOT)
    stage.OverridePrim(LINK8_PATH)  # ensure over chain exists in the root layer
    cam = UsdGeom.Camera.Define(stage, LINK8_PATH + "/WristCam")
    cam.GetFocalLengthAttr().Set(FOCAL_CM)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*CLIPPING))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # forward -Z -> along fingers
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(ASSET_ROOT)
    print("WristCam composed:", check.GetPrimAtPath(LINK8_PATH + "/WristCam").IsValid())


if __name__ == "__main__":
    main()
