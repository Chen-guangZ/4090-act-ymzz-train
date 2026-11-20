"""Render check for the grasp scene: human view + wrist-cam view + bbox sanity."""
import math
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 96, "height": 96})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom

SCENE = "/home/nvidia4090/ymbot_grasp/assets/scene_table_box.usda"
OUT_HUMAN = "/tmp/grasp_scene_human.png"
OUT_WRIST = "/tmp/grasp_scene_wrist.png"

ctx = omni.usd.get_context()
assert ctx.open_stage(SCENE), "failed to open scene"
for _ in range(30):
    app.update()
stage = ctx.get_stage()

# bbox sanity for placement
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for name in ("Robot", "TableTop", "TableLegs", "Box"):
    prim = stage.GetPrimAtPath(f"/World/{name}")
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    print(f"[CHECK] {name}: min ({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}) max ({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})")

wrist = stage.GetPrimAtPath(
    "/World/Robot/Geometry/base_footprint_link/base_link/Body_Link1/Body_Link2/Body_Link3/"
    "Body_Link4/Body_Link5/Right_Arm_Link1/Right_Arm_Link2/Right_Arm_Link3/Right_Arm_Link4/"
    "Right_Arm_Link5/Right_Arm_Link6/Right_Arm_Link7/Right_Arm_Link8/WristCam"
)
print(f"[CHECK] WristCam valid: {wrist.IsValid()}")

# table top must sit at z = 1.0
top = stage.GetPrimAtPath("/World/TableTop")
rng = cache.ComputeWorldBound(top).ComputeAlignedRange()
print(f"[CHECK] TableTop z range: {rng.GetMin()[2]:.3f} .. {rng.GetMax()[2]:.3f} (want 0.96 .. 1.00)")

# create a temporary human-view camera looking at the workspace
import numpy as np  # noqa: E402
from pxr import Vt  # noqa: E402

POS = Gf.Vec3d(2.6, -2.6, 1.7)
TGT = Gf.Vec3d(0.9, 0.0, 1.0)
f = np.array([TGT[0] - POS[0], TGT[1] - POS[1], TGT[2] - POS[2]])
f /= np.linalg.norm(f)
up_w = np.array([0.0, 0.0, 1.0])
r = np.cross(f, up_w)
r /= np.linalg.norm(r)
u = np.cross(r, f)
R = np.column_stack([r, u, -f])  # USD camera: +X right, +Y up, -Z forward
t = np.trace(R)
if t > 0:
    s = math.sqrt(t + 1.0) * 2
    quat = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    quat = (0.25 * ((R[2, 1] - R[1, 2]) / s), 0.25 * s, 0.25 * (R[1, 0] + R[0, 1]) / s, 0.25 * (R[0, 2] + R[2, 0]) / s)
elif R[1, 1] > R[2, 2]:
    s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    quat = (0.25 * ((R[0, 2] - R[2, 0]) / s), 0.25 * (R[1, 0] + R[0, 1]) / s, 0.25 * s, 0.25 * (R[2, 1] + R[1, 2]) / s)
else:
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    quat = (0.25 * ((R[1, 0] - R[0, 1]) / s), 0.25 * (R[0, 2] + R[2, 0]) / s, 0.25 * (R[2, 1] + R[1, 2]) / s, 0.25 * s)

human_cam = UsdGeom.Camera.Define(stage, "/World/HumanCam")
human_cam.GetFocalLengthAttr().Set(18.0)
hxf = UsdGeom.Xformable(human_cam.GetPrim())
hxf.AddTranslateOp().Set(POS)
hxf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))


def snap(cam_path: str, res: tuple, out: str) -> None:
    rp = rep.create.render_product(cam_path, res)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    for _ in range(30):
        app.update()
    rep.orchestrator.step(rt_subframes=48)
    rgb = np.asarray(annot.get_data())[:, :, :3]
    Image.fromarray(rgb.astype(np.uint8)).save(out)
    print(f"[CHECK] saved {out}")
    annot.detach(rp)
    rp.destroy()


snap("/World/HumanCam", (1280, 720), OUT_HUMAN)
snap(str(wrist.GetPath()), (640, 480), OUT_WRIST)

app.close()
sys.exit(0)
