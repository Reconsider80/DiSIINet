#!/usr/bin/env python3
"""Generate train/val/test split files for DiSIINet datasets."""

import argparse
import random
from pathlib import Path


def collect_stems(image_dir: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    stems = sorted(
        {
            p.stem
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        }
    )
    return stems


def write_splits(
    root: Path,
    stems: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    names = stems[:]
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train.txt": names[:n_train],
        "val.txt": names[n_train : n_train + n_val],
        "test.txt": names[n_train + n_val :],
    }
    for filename, items in splits.items():
        (root / filename).write_text("\n".join(items))
        print(f"Wrote {filename}: {len(items)} samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset split files")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    stems = collect_stems(image_dir)
    if not stems:
        raise RuntimeError(f"No images found in {image_dir}")

    write_splits(root, stems, args.train_ratio, args.val_ratio, args.seed)
    print(f"Prepared splits for {len(stems)} samples under {root}")


if __name__ == "__main__":
    main()
