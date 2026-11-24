"""Open and loop a visual K3 trajectory in a dedicated Isaac Sim window."""

import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--frame", type=Path, help="Render offscreen and continuously update this JPEG.")
    outputs.add_argument("--gif", type=Path, help="Record one offscreen pass of the USD animation.")
    args = parser.parse_args()
    from isaacsim import SimulationApp

    offscreen = args.frame is not None or args.gif is not None
    app = SimulationApp({"headless": offscreen, "window_width": 1340, "window_height": 850,
                         "width": 1280, "height": 720, "hide_ui": offscreen})
    try:
        import omni.timeline
        import omni.usd
        from pxr import UsdLux

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve())):
            raise RuntimeError(f"Unable to open {args.scene}")
        for _ in range(40):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdLux.DomeLight.Define(stage, "/World/K3ReviewLight").CreateIntensityAttr(1000)
        camera = "/World/K3ReviewCamera"
        if not offscreen:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = camera
        else:
            import carb.settings
            import numpy as np
            import omni.replicator.core as rep
            from PIL import Image

            carb.settings.get_settings().set_int("/rtx/post/aa/op", 0)
            product = rep.create.render_product(camera, (640, 440) if args.gif else (1100, 760))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
            if args.frame:
                temporary = args.frame.with_suffix(".part")
        timeline = omni.timeline.get_timeline_interface()
        duration = (stage.GetEndTimeCode() + 1) / stage.GetTimeCodesPerSecond()
        origin = time.monotonic()
        print(f"Ready: {args.scene.name}, {duration:.1f} s", flush=True)
        if args.gif:
            frames = []
            for timecode in np.arange(0, duration, 0.125):
                timeline.set_current_time(float(timecode))
                rep.orchestrator.step(rt_subframes=2, pause_timeline=True)
                pixels = np.asarray(rgb.get_data())
                if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels[:, :, :3].std() <= 1:
                    raise RuntimeError(f"Invalid rendered frame at {timecode:.3f} s")
                frames.append(Image.fromarray(pixels[:, :, :3].astype(np.uint8)).quantize(colors=128))
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=125, loop=0)
            return
        while app.is_running():
            timeline.set_current_time((time.monotonic() - origin) % duration)
            if args.frame is None:
                app.update()
                time.sleep(0.01)
            else:
                rep.orchestrator.step(rt_subframes=2, pause_timeline=True)
                pixels = np.asarray(rgb.get_data())
                if pixels.ndim == 3 and pixels.shape[2] >= 3 and pixels[:, :, :3].std() > 1:
                    Image.fromarray(pixels[:, :, :3].astype(np.uint8)).save(temporary, format="JPEG", quality=86)
                    temporary.replace(args.frame)
                time.sleep(0.16)
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
