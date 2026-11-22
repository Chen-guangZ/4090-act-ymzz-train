"""Author a visual USD layer from sampled K3 simulation body poses."""

from os import link
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from k3_scene import matrix_from_pose


def write_animation(source, output, labels, samples, hz):
    if output.exists():
        raise FileExistsError(output)
    poses = np.asarray(samples, dtype=np.float64)
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError("The visual replay requires finite sampled body poses.")
    with TemporaryDirectory(prefix="k3-usd-", dir=output.resolve().parent) as directory:
        temporary = Path(directory) / output.name
        author_layer(source, temporary, labels, poses, hz)
        link(temporary, output)


def author_layer(source, output, labels, poses, hz):
    stage = Usd.Stage.CreateNew(str(output.resolve()))
    stage.GetRootLayer().subLayerPaths = [str(source.resolve())]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetFramesPerSecond(hz)
    stage.SetTimeCodesPerSecond(hz)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(poses) - 1)
    indices = {label: i for i, label in enumerate(labels)}
    cache = UsdGeom.XformCache()
    targets = []
    for label, i in indices.items():
        prim = stage.GetPrimAtPath(label)
        parent = prim.GetParent()
        ancestor = parent
        while ancestor and str(ancestor.GetPath()) not in indices:
            ancestor = ancestor.GetParent()
        parent_world = np.array(cache.GetLocalToWorldTransform(parent)).T
        if ancestor:
            ancestor_to_parent = np.linalg.solve(np.array(cache.GetLocalToWorldTransform(ancestor)).T, parent_world)
            parent_index = indices[str(ancestor.GetPath())]
        else:
            ancestor_to_parent, parent_index = parent_world, None
        targets.append((prim, i, parent_index, ancestor_to_parent))
    ops = []
    for prim, i, parent_index, ancestor_to_parent in targets:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp(opSuffix="k3Replay")
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        ops.append((op, i, parent_index, ancestor_to_parent))
    camera = UsdGeom.Camera.Define(stage, "/World/K3ReviewCamera")
    eye = Gf.Vec3d(2.5, -2.3, 1.65)
    aim = Gf.Vec3d(0.25, 0, 0.85)
    view = Gf.Matrix4d(1).SetLookAt(eye, aim, Gf.Vec3d(0, 0, 1))
    camera.MakeMatrixXform().Set(view.GetInverse())
    camera.GetFocalLengthAttr().Set(26.0)
    for frame, sample in enumerate(poses):
        matrices = [matrix_from_pose(body[:3], body[3:]) for body in sample]
        for op, i, parent, parent_to_world in ops:
            parent_world = matrices[parent] @ parent_to_world if parent is not None else parent_to_world
            local = np.linalg.solve(parent_world, matrices[i])
            op.Set(Gf.Matrix4d(*local.T.flatten().tolist()), frame)
    stage.GetRootLayer().Save()
