"""Render the K3-calibrated head and wrist camera views from a replay scene."""

import argparse
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera", choices=("both", "head", "wrist"), default="both")
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    result = 0
    try:
        import carb.settings
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import UsdGeom, UsdLux
        carb.settings.get_settings().set_int("/rtx/post/aa/op", 0)

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve())):
            raise RuntimeError(f"Could not open {args.scene}")
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        paths = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)}
        UsdLux.DomeLight.Define(stage, "/World/RenderLighting").CreateIntensityAttr(1500)
        args.output.mkdir(parents=True, exist_ok=True)
        names = {"head": ("K3HeadRGBD",), "wrist": ("K3WristRGBD",),
                 "both": ("K3HeadRGBD", "K3WristRGBD")}
        targets = []
        xforms = UsdGeom.XformCache()
        for name in names[args.camera]:
            prim = paths[name]
            resolution = tuple(prim.GetCustomDataByKey(f"k3Image{axis}") for axis in ("Width", "Height"))
            # Standalone Replicator cameras use the same optical transform.
            # The articulated camera prims are retained for runtime sensors.
            view_camera = UsdGeom.Camera.Define(stage, "/World/Render_" + name)
            source = UsdGeom.Camera(prim).GetCamera()
            source.transform = xforms.GetLocalToWorldTransform(prim)
            view_camera.SetFromCamera(source)
            factor = 2 if min(resolution) < 600 else 1
            render_resolution = tuple(d * factor for d in resolution)
            product = rep.create.render_product(str(view_camera.GetPath()), render_resolution)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
            targets.append((name, resolution, render_resolution, product, rgb))
        for _ in range(30):
            app.update()
        rep.orchestrator.step(rt_subframes=48)
        for name, resolution, render_resolution, _, rgb in targets:
            view = np.asarray(rgb.get_data())[:, :, :3].copy()
            if view.shape[:2] != render_resolution[::-1] or np.std(view) < 1:
                raise RuntimeError(f"{name} rendered invalid image {view.shape}, std={np.std(view):.3f}, "
                                   f"range=[{view.min()}, {view.max()}]")
            frame = Image.fromarray(view.astype(np.uint8))
            if render_resolution != resolution:
                frame = frame.resize(resolution, Image.Resampling.LANCZOS)
            frame.save(args.output / f"{name}.png")
            print(f"Rendered {name}: {frame.size}", flush=True)
        for _, _, _, product, rgb in targets:
            rgb.detach(product)
            product.destroy()
    except Exception:
        traceback.print_exc()
        result = 1
    finally:
        app.close(skip_cleanup=True, exit_code=result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
