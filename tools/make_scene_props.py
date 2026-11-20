#!/usr/bin/env python3
"""Generate scene prop variants: white table + cardboard shipping box.

Creates textures + asset copies in the grasp project (one-shot, idempotent):
  assets/textures/table_white.png      near-white with faint noise
  assets/textures/box_cardboard.png    kraft-brown + tape band + shipping label
  assets/table_white.usda              copy of ymbot_push table_wood, texture swapped
  assets/box_cardboard.usda            copy of ymbot_push box_checker, texture swapped

Run:
    /home/nvidia4090/IsaacLab/.venv/bin/python /home/nvidia4090/ymbot_grasp/tools/make_scene_props.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/nvidia4090/ymbot_grasp")
TEX_DIR = ROOT / "assets" / "textures"
TEX_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)


def make_white_table() -> Path:
    img = np.full((256, 256, 3), 244, dtype=np.int16)
    img += RNG.integers(-5, 6, img.shape, dtype=np.int16)
    out = TEX_DIR / "table_white.png"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    return out


def make_cardboard() -> Path:
    base = np.array([198, 160, 100], dtype=np.int16)
    img = np.tile(base, (512, 512, 1))
    img += RNG.integers(-14, 15, img.shape, dtype=np.int16)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    # 横向封箱胶带（中间一条，略亮略偏黄）
    tape = (222, 205, 160)
    d.rectangle([0, 216, 511, 296], fill=tape)
    d.line([(0, 216), (511, 216)], fill=(150, 120, 70), width=2)
    d.line([(0, 296), (511, 296)], fill=(150, 120, 70), width=2)
    # 中央白色快递面单
    d.rectangle([316, 340, 470, 470], fill=(245, 245, 240), outline=(120, 120, 120), width=2)
    for i in range(6):  # 面单上的条码/文字行
        y = 356 + i * 18
        d.rectangle([330, y, 330 + RNG.integers(60, 120), y + 8], fill=(70, 70, 70))
    # 边缘折痕暗线
    d.line([(0, 0), (0, 511)], fill=(140, 110, 66), width=6)
    d.line([(511, 0), (511, 511)], fill=(140, 110, 66), width=6)
    out = TEX_DIR / "box_cardboard.png"
    pil.save(out)
    return out


def clone_asset(src: Path, dst: Path, old_tex: str, new_tex: Path) -> None:
    text = src.read_text()
    assert old_tex in text, f"texture path not found in {src}"
    dst.write_text(text.replace(old_tex, str(new_tex)))


def main() -> None:
    white = make_white_table()
    cardboard = make_cardboard()
    clone_asset(
        Path("/home/nvidia4090/ymbot_push/assets/usd/table_wood.usda"),
        ROOT / "assets" / "table_white.usda",
        "/home/nvidia4090/ymbot_push/assets/textures/table_wood.png",
        white,
    )
    clone_asset(
        Path("/home/nvidia4090/ymbot_push/assets/usd/box_checker.usda"),
        ROOT / "assets" / "box_cardboard.usda",
        "/home/nvidia4090/ymbot_push/assets/textures/box_print.png",
        cardboard,
    )
    print("done:", white, cardboard)


if __name__ == "__main__":
    main()
