#!/usr/bin/env python3
"""Prepare synthetic demo data for smoke testing DiSIINet."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_demo_sample(idx: int, size: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(idx)
    image = rng.integers(30, 200, (size, size), dtype=np.uint8)

    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2 + rng.integers(-20, 20), size // 2 + rng.integers(-20, 20)
    r = size // 4 + rng.integers(-10, 10)
    yy, xx = np.ogrid[:size, :size]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    mask[circle] = min(1, num_classes - 1)

    if num_classes > 2:
        inner_r = r // 2
        inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= inner_r**2
        mask[inner] = min(2, num_classes - 1)

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=180, width=2)
    image = np.array(pil)
    return image, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/demo")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.output)
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for i in range(args.num_samples):
        name = f"sample_{i:04d}"
        names.append(name)
        image, mask = create_demo_sample(i, args.size, args.num_classes)
        Image.fromarray(image).save(img_dir / f"{name}.png")
        Image.fromarray(mask).save(mask_dir / f"{name}.png")

    n_train = int(args.num_samples * 0.7)
    n_val = int(args.num_samples * 0.15)
    (root / "train.txt").write_text("\n".join(names[:n_train]))
    (root / "val.txt").write_text("\n".join(names[n_train : n_train + n_val]))
    (root / "test.txt").write_text("\n".join(names[n_train + n_val :]))

    print(f"Created {args.num_samples} demo samples in {root}")


if __name__ == "__main__":
    main()
