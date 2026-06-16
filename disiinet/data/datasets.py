"""Data loading utilities and medical image datasets."""

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def degrade_image(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Downsample degradation following STS-SR (Sec. 4.1).
    Original image as HQ; bicubic downsample-upsample as LQ.
    """
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    small = pil.resize((w // scale, h // scale), Image.BICUBIC)
    degraded = small.resize((w, h), Image.BICUBIC)
    return np.array(degraded)


def mask_to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert label map to one-hot encoding."""
    onehot = np.zeros((num_classes, *mask.shape), dtype=np.float32)
    for c in range(num_classes):
        onehot[c] = (mask == c).astype(np.float32)
    return onehot


class MedicalJointDataset(Dataset):
    """
    Generic dataset for joint enhancement and segmentation.

    Expected directory structure:
        root/
          images/   # HQ images
          masks/    # segmentation masks (same filename stem)
          split.txt # optional: list of sample names
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 320,
        num_classes: int = 4,
        degrade_scale: int = 4,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.degrade_scale = degrade_scale
        self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[str, str]]:
        split_file = self.root / f"{self.split}.txt"
        if split_file.exists():
            names = [
                line.strip()
                for line in split_file.read_text().splitlines()
                if line.strip()
            ]
        else:
            img_dir = self.root / "images"
            names = [
                p.stem
                for p in sorted(img_dir.iterdir())
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".nii.gz"}
            ]

        samples = []
        for name in names:
            img_path = self._find_file(self.root / "images", name)
            mask_path = self._find_file(self.root / "masks", name)
            if img_path and mask_path:
                samples.append((str(img_path), str(mask_path)))
        return samples

    @staticmethod
    def _find_file(directory: Path, name: str) -> Optional[Path]:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".npg"]:
            p = directory / f"{name}{ext}"
            if p.exists():
                return p
        for p in directory.glob(f"{name}*"):
            return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path))

        image = np.array(
            Image.fromarray(image).resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        )
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
        )

        x_gt = image.astype(np.float32) / 255.0
        x_lq = degrade_image((x_gt * 255).astype(np.uint8), self.degrade_scale)
        x_lq = x_lq.astype(np.float32) / 255.0

        mask_onehot = mask_to_onehot(mask, self.num_classes)

        sample = {
            "x_lq": torch.from_numpy(x_lq).unsqueeze(0),
            "x_gt": torch.from_numpy(x_gt).unsqueeze(0),
            "mask": torch.from_numpy(mask_onehot),
            "name": Path(img_path).stem,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class ACDCDataset(MedicalJointDataset):
    """ACDC cardiac MRI dataset (150 patients, 320x320)."""

    def __init__(self, root: str, split: str = "train", **kwargs):
        kwargs.setdefault("image_size", 320)
        kwargs.setdefault("num_classes", 4)
        super().__init__(root, split, **kwargs)


class KiTS19Dataset(MedicalJointDataset):
    """KiTS19 kidney CT dataset (300 cases, 320x320)."""

    def __init__(self, root: str, split: str = "train", **kwargs):
        kwargs.setdefault("image_size", 320)
        kwargs.setdefault("num_classes", 3)
        super().__init__(root, split, **kwargs)


class TN3KDataset(MedicalJointDataset):
    """TN3K thyroid ultrasound dataset (3493 images, 256x256)."""

    def __init__(self, root: str, split: str = "train", **kwargs):
        kwargs.setdefault("image_size", 256)
        kwargs.setdefault("num_classes", 2)
        super().__init__(root, split, **kwargs)


DATASET_REGISTRY = {
    "acdc": ACDCDataset,
    "kits19": KiTS19Dataset,
    "tn3k": TN3KDataset,
}


def build_dataset(name: str, root: str, split: str, **kwargs) -> Dataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](root, split, **kwargs)
